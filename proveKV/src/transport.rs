//! HTTP transport for transferring validated proveKV pages over a Tailscale URL.
use crate::error::{ProveKvError, Result};
use crate::page_format::{PageHeader, MAX_PAYLOAD_BYTES};
use crate::state_id::HybridStateId;
use crate::state_store::StateStore;
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
    state_store: Arc<StateStore>,
    client: Client,
    endpoint: String,
}

impl PageTransfer {
    /// Bind transfer lookup to the canonical immutable state owner.
    ///
    /// The caller transfers a page from a state the store has committed. A
    /// raw `PageStore` is deliberately insufficient: it knows payload bytes
    /// but not which immutable state manifest authorizes their reuse.
    pub fn new(state_store: StateStore, endpoint: impl Into<String>) -> Self {
        Self {
            state_store: Arc::new(state_store),
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
        let requested_state = HybridStateId::try_from(state_id.to_owned())?;
        let state = self.state_store.get(&requested_state).ok_or_else(|| {
            ProveKvError::InvalidManifest(format!("state {state_id} is not committed"))
        })?;
        requested_state.verify_manifest(&state.manifest)?;

        let mut matching_page = None;
        for page_ref in &state.manifest.page_refs {
            let (header, payload) = self.state_store.page_store.read_page(&page_ref.digest)?;
            if header.payload_digest != page_ref.digest {
                return Err(ProveKvError::DigestMismatch {
                    expected: page_ref.digest.clone(),
                    got: header.payload_digest,
                });
            }
            if header.component_kind == kv_type && header.position_start == layer {
                if matching_page.is_some() {
                    return Err(ProveKvError::InvalidManifest(format!(
                        "state {state_id} has multiple pages for layer={layer}, kv_type={kv_type}"
                    )));
                }
                matching_page = Some((header, payload));
            }
        }

        matching_page.ok_or_else(|| {
            ProveKvError::InvalidManifest(format!(
                "manifest-bound page not found for state_id={state_id}, layer={layer}, kv_type={kv_type}"
            ))
        })
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

    fn manifest_for_page(header: &PageHeader) -> crate::hybrid_manifest::HybridStateManifestV1 {
        use crate::hybrid_manifest::{HybridComponent, HybridPageRef, HybridStateManifestV1};
        use crate::shape::{AttentionType, KvTensorShape};

        HybridStateManifestV1::new(
            "model",
            "tokenizer",
            KvTensorShape {
                attention_type: AttentionType::MHA,
                num_layers: 1,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 1,
                hidden_size: 1,
            },
            vec![HybridComponent {
                name: "full_attn_k".into(),
                version: "v1".into(),
                digest: "component:v1".into(),
            }],
            vec![HybridPageRef {
                page_id: header.payload_digest.clone(),
                digest: header.payload_digest.clone(),
            }],
            vec![],
            "policy:v1",
            "version:v1",
        )
    }

    #[test]
    fn find_page_does_not_return_another_states_matching_page() {
        use std::sync::atomic::{AtomicU32, Ordering};

        static STORE_COUNTER: AtomicU32 = AtomicU32::new(0);
        let n = STORE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "provekv-transport-state-binding-{}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = StateStore::open(&dir).unwrap();

        let page_a = vec![1u8, 2, 3, 4];
        let header_a = build_page_header(
            "full_attn_k",
            &[1],
            "float32",
            b'l',
            "sha256:model",
            "sha256:layout",
            7,
            8,
            "raw_exact",
            &page_a,
        )
        .unwrap();
        store.page_store.write_page(&header_a, &page_a).unwrap();

        let page_b = vec![5u8, 6, 7, 8];
        let header_b = build_page_header(
            "full_attn_k",
            &[1],
            "float32",
            b'l',
            "sha256:model",
            "sha256:layout",
            7,
            8,
            "raw_exact",
            &page_b,
        )
        .unwrap();
        store.page_store.write_page(&header_b, &page_b).unwrap();

        store.commit_root(manifest_for_page(&header_a)).unwrap();
        let state_b = store.commit_root(manifest_for_page(&header_b)).unwrap();
        let transfer = PageTransfer::new(store, "http://unused.invalid");
        let (resolved, payload) = transfer
            .find_page(state_b.as_str(), 7, "full_attn_k")
            .unwrap();

        assert_eq!(resolved.payload_digest, header_b.payload_digest);
        assert_eq!(payload, page_b);
    }
}
