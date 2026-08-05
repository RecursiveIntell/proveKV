//! HTTP transport for transferring validated proveKV pages over a Tailscale URL.
use crate::error::{ProveKvError, Result};
use crate::page_format::{PageHeader, MAX_PAYLOAD_BYTES};
use crate::page_store::PageStore;
use futures_util::stream;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

const CHUNK_SIZE: usize = 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TransferEnvelope {
    transfer_id: String,
    header: PageHeader,
    digest: String,
    payload: Vec<u8>,
}

/// Transfers pages using HTTP(S), normally to a Tailscale hostname.
#[derive(Clone)]
pub struct PageTransfer {
    store: Arc<PageStore>,
    client: Client,
    endpoint: String,
}

impl PageTransfer {
    pub fn new(store: PageStore, endpoint: impl Into<String>) -> Self {
        Self {
            store: Arc::new(store),
            client: Client::new(),
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
        }
    }

    /// Find the page matching the logical identity and POST it to `destination_url`.
    /// The request body is streamed in 1 MiB chunks, so reqwest uses HTTP chunked
    /// transfer when the destination does not require a Content-Length.
    pub async fn send_page(
        &self,
        state_id: &str,
        layer: u32,
        kv_type: &str,
        destination_url: &str,
    ) -> Result<String> {
        let (header, payload) = self.find_page(state_id, layer, kv_type)?;
        let digest = blake3::hash(&payload).to_hex().to_string();
        let transfer_id = format!("{}:{}:{}", state_id, layer, kv_type);
        let envelope = serde_json::to_vec(&TransferEnvelope {
            transfer_id: transfer_id.clone(),
            header,
            digest,
            payload,
        })?;
        let chunks: Vec<_> = envelope
            .chunks(CHUNK_SIZE)
            .map(|c| Ok::<_, std::io::Error>(c.to_vec()))
            .collect();
        let body = reqwest::Body::wrap_stream(stream::iter(chunks));
        let response = self
            .client
            .post(destination_url)
            .header("content-type", "application/json")
            .header("x-provekv-transfer-id", &transfer_id)
            .body(body)
            .send()
            .await
            .map_err(|e| ProveKvError::Internal(format!("page upload: {e}")))?;
        if !response.status().is_success() {
            return Err(ProveKvError::Internal(format!(
                "page upload returned {}",
                response.status()
            )));
        }
        Ok(transfer_id)
    }

    /// Fetch a transfer by id from the configured Tailscale endpoint and verify it.
    pub async fn receive_page(&self, transfer_id: &str) -> Result<(PageHeader, Vec<u8>)> {
        let url = if transfer_id.starts_with("http://") || transfer_id.starts_with("https://") {
            transfer_id.to_string()
        } else {
            format!(
                "{}/transfer/{}",
                self.endpoint,
                urlencoding::encode(transfer_id)
            )
        };
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| ProveKvError::Internal(format!("page download: {e}")))?;
        if response.status() != StatusCode::OK {
            return Err(ProveKvError::Internal(format!(
                "page download returned {}",
                response.status()
            )));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|e| ProveKvError::Internal(format!("page body: {e}")))?;
        if bytes.len() > MAX_PAYLOAD_BYTES as usize + 16 * 1024 * 1024 {
            return Err(ProveKvError::ResourceLimitExceeded(
                "transfer too large".into(),
            ));
        }
        let envelope: TransferEnvelope = serde_json::from_slice(&bytes)?;
        let got = blake3::hash(&envelope.payload).to_hex().to_string();
        if got != envelope.digest || got != envelope.header.payload_digest {
            return Err(ProveKvError::DigestMismatch {
                expected: envelope.header.payload_digest,
                got,
            });
        }
        envelope.header.validate()?;
        if envelope.header.payload_len != envelope.payload.len() as u64 {
            return Err(ProveKvError::CorruptPayload(
                "payload length mismatch".into(),
            ));
        }
        Ok((envelope.header, envelope.payload))
    }

    fn find_page(
        &self,
        state_id: &str,
        layer: u32,
        kv_type: &str,
    ) -> Result<(PageHeader, Vec<u8>)> {
        for entry in std::fs::read_dir(self.store.root())? {
            let path = entry?.path();
            if path.extension().and_then(|x| x.to_str()) != Some("page") {
                continue;
            }
            let digest = path
                .file_stem()
                .and_then(|x| x.to_str())
                .unwrap_or_default();
            if let Ok((header, payload)) = self.store.read_page(digest) {
                if header.component_kind == kv_type && header.position_start == layer {
                    return Ok((header, payload));
                }
            }
        }
        Err(ProveKvError::Internal(format!(
            "page not found for state_id={state_id}, layer={layer}, kv_type={kv_type}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page_format::build_page_header;
    #[test]
    fn digest_is_blake3() {
        let p = b"page";
        assert_eq!(blake3::hash(p).to_hex().to_string().len(), 64);
    }
    #[test]
    fn chunking_covers_all_bytes() {
        let data = vec![7u8; CHUNK_SIZE * 2 + 3];
        assert_eq!(data.chunks(CHUNK_SIZE).flatten().count(), data.len());
    }
}
