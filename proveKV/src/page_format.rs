//! Binary page format for hybrid state persistence.
//!
//! Every page has a self-describing header with typed bounds, component
//! identity, layout/model digests, codec profile, and dual payload/header
//! digests. Parsing validates bounds before any allocation. Unknown or
//! malformed pages are rejected without partial visibility.

use crate::error::{ProveKvError, Result};
use serde::{Deserialize, Serialize};

/// Magic bytes identifying a proveKV hybrid state page.
pub const PAGE_MAGIC: &[u8; 4] = b"PKVP";

/// Current page schema version.
pub const PAGE_SCHEMA_VERSION: u16 = 1;

/// Maximum header size in bytes to bound parse before allocation.
pub const MAX_HEADER_BYTES: usize = 4096;

/// Maximum payload size in bytes (1 GiB).
pub const MAX_PAYLOAD_BYTES: u64 = 1_073_741_824;

/// Maximum component kind string length.
pub const MAX_COMPONENT_KIND_LEN: usize = 64;

/// Maximum codec profile string length.
pub const MAX_CODEC_PROFILE_LEN: usize = 128;

/// Page header that prefixes every persisted page.
///
/// The header is written as a fixed-size binary structure followed by
/// variable-length string fields (component kind, codec profile), then the
/// raw payload. The header digest covers everything up to the payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageHeader {
    /// Always `PAGE_MAGIC`.
    pub magic: [u8; 4],
    /// Schema version for forward compatibility.
    pub schema_version: u16,
    /// Component kind (e.g. "full_attn_k", "conv_state").
    pub component_kind: String,
    /// Total header byte length including this field.
    pub header_len: u32,
    /// Number of axes.
    pub rank: u8,
    /// Dimension per axis.
    pub dims: Vec<u32>,
    /// Data type string (e.g. "float32").
    pub dtype: String,
    /// Endianness: 'l' for little, 'b' for big.
    pub endianness: u8,
    /// Model identifier digest.
    pub model_digest: String,
    /// Canonical layout digest.
    pub layout_digest: String,
    /// Position span: (start, exclusive_end) in token space.
    pub position_start: u32,
    pub position_end: u32,
    /// Codec profile name.
    pub codec_profile: String,
    /// Payload length in bytes.
    pub payload_len: u64,
    /// BLAKE3 digest of the payload bytes.
    pub payload_digest: String,
    /// BLAKE3 digest of all header fields (payload_len bytes of zeros
    /// substituted for the payload digest during computation).
    pub header_digest: String,
}

impl PageHeader {
    /// Validate header invariants before any allocation.
    pub fn validate(&self) -> Result<()> {
        if &self.magic != PAGE_MAGIC {
            return Err(ProveKvError::CorruptPayload(format!(
                "bad magic: expected {:?}, got {:?}",
                PAGE_MAGIC, self.magic
            )));
        }

        if self.schema_version != PAGE_SCHEMA_VERSION {
            return Err(ProveKvError::CorruptPayload(format!(
                "unsupported schema version: {}",
                self.schema_version
            )));
        }

        if self.component_kind.is_empty() || self.component_kind.len() > MAX_COMPONENT_KIND_LEN {
            return Err(ProveKvError::CorruptPayload(
                "component_kind empty or too long".into(),
            ));
        }

        if self.rank == 0 {
            return Err(ProveKvError::CorruptPayload("rank must be >= 1".into()));
        }

        if self.dims.len() != self.rank as usize {
            return Err(ProveKvError::CorruptPayload(format!(
                "dims length {} != rank {}",
                self.dims.len(),
                self.rank
            )));
        }

        for (i, &dim) in self.dims.iter().enumerate() {
            if dim == 0 {
                return Err(ProveKvError::CorruptPayload(format!("dim[{}] is zero", i)));
            }
            // Overflow guard: product of dims must fit u64
            let _: u64 = self
                .dims
                .iter()
                .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))
                .ok_or_else(|| ProveKvError::CorruptPayload("dimension product overflow".into()))?;
        }

        if self.dtype.is_empty() {
            return Err(ProveKvError::CorruptPayload("dtype is empty".into()));
        }

        if self.endianness != b'l' && self.endianness != b'b' {
            return Err(ProveKvError::CorruptPayload(format!(
                "bad endianness: {}",
                self.endianness
            )));
        }

        if self.model_digest.is_empty() || self.layout_digest.is_empty() {
            return Err(ProveKvError::CorruptPayload(
                "model_digest or layout_digest empty".into(),
            ));
        }

        if self.position_start >= self.position_end {
            return Err(ProveKvError::CorruptPayload(format!(
                "position_start {} >= position_end {}",
                self.position_start, self.position_end
            )));
        }

        if self.codec_profile.is_empty() || self.codec_profile.len() > MAX_CODEC_PROFILE_LEN {
            return Err(ProveKvError::CorruptPayload(
                "codec_profile empty or too long".into(),
            ));
        }

        if self.payload_len > MAX_PAYLOAD_BYTES {
            return Err(ProveKvError::ResourceLimitExceeded(format!(
                "payload {} exceeds max {}",
                self.payload_len, MAX_PAYLOAD_BYTES
            )));
        }

        if self.payload_digest.len() != 64 {
            return Err(ProveKvError::CorruptPayload(
                "payload_digest must be 64 hex chars".into(),
            ));
        }

        if self.header_digest.len() != 64 {
            return Err(ProveKvError::CorruptPayload(
                "header_digest must be 64 hex chars".into(),
            ));
        }

        if self.header_len as usize > MAX_HEADER_BYTES {
            return Err(ProveKvError::CorruptPayload(format!(
                "header_len {} exceeds max {}",
                self.header_len, MAX_HEADER_BYTES
            )));
        }

        Ok(())
    }

    /// Compute the element count from dims.
    pub fn element_count(&self) -> u64 {
        self.dims.iter().fold(1u64, |acc, &d| acc * d as u64)
    }

    /// Byte size of the payload given dtype.
    pub fn bytes_per_element(&self) -> Result<u64> {
        match self.dtype.as_str() {
            "float32" => Ok(4),
            "float16" => Ok(2),
            "bfloat16" => Ok(2),
            "int8" => Ok(1),
            "int4" => Ok(1), // packed, caller handles
            other => Err(ProveKvError::CorruptPayload(format!(
                "unsupported dtype: {}",
                other
            ))),
        }
    }

    /// Verify that payload_len matches dimensions × bytes_per_element.
    pub fn verify_payload_size(&self) -> Result<()> {
        let expected = self
            .element_count()
            .checked_mul(self.bytes_per_element()?)
            .ok_or_else(|| ProveKvError::CorruptPayload("payload size overflow".into()))?;
        if self.payload_len != expected {
            return Err(ProveKvError::CorruptPayload(format!(
                "payload_len {} != expected {} ({} elements × {} bytes/elem)",
                self.payload_len,
                expected,
                self.element_count(),
                self.bytes_per_element()?
            )));
        }
        Ok(())
    }
}

/// Serialize a header to canonical JSON bytes for digest computation.
///
/// The payload_digest field is zeroed before hashing so the header
/// digest covers all identity fields without circular dependency.
pub fn header_to_digest_input(header: &PageHeader) -> Result<Vec<u8>> {
    let mut h = header.clone();
    h.payload_digest = "0".repeat(64);
    h.header_digest = "0".repeat(64);
    let json = serde_json::to_vec(&h)?;
    Ok(json)
}

/// Compute the header digest from a PageHeader.
pub fn compute_header_digest(header: &PageHeader) -> String {
    let input = header_to_digest_input(header).expect("header serialization must not fail");
    blake3::hash(&input).to_hex().to_string()
}

/// Compute the payload digest from raw bytes.
pub fn compute_payload_digest(payload: &[u8]) -> String {
    blake3::hash(payload).to_hex().to_string()
}

/// Build a valid PageHeader with all digests filled in.
pub fn build_page_header(
    component_kind: &str,
    dims: &[u32],
    dtype: &str,
    endianness: u8,
    model_digest: &str,
    layout_digest: &str,
    position_start: u32,
    position_end: u32,
    codec_profile: &str,
    payload: &[u8],
) -> Result<PageHeader> {
    let payload_digest = compute_payload_digest(payload);
    let payload_len = payload.len() as u64;

    let mut header = PageHeader {
        magic: *PAGE_MAGIC,
        schema_version: PAGE_SCHEMA_VERSION,
        component_kind: component_kind.to_string(),
        header_len: 0, // filled below
        rank: dims.len() as u8,
        dims: dims.to_vec(),
        dtype: dtype.to_string(),
        endianness,
        model_digest: model_digest.to_string(),
        layout_digest: layout_digest.to_string(),
        position_start,
        position_end,
        codec_profile: codec_profile.to_string(),
        payload_len,
        payload_digest: payload_digest.clone(),
        header_digest: String::new(),
    };

    // Compute header digest, then measure actual JSON size
    header.header_digest = compute_header_digest(&header);
    let json_bytes = serde_json::to_vec(&header)?;
    header.header_len = json_bytes.len() as u32;

    // Recompute header digest with final header_len
    header.header_digest = compute_header_digest(&header);

    header.validate()?;
    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_payload(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 256) as u8).collect()
    }

    #[test]
    fn valid_header_passes_validation() {
        let payload = sample_payload(16384); // 32*128*4
        let header = build_page_header(
            "full_attn_k",
            &[32, 128],
            "float32",
            b'l',
            "sha256:aaaa",
            "sha256:bbbb",
            0,
            64,
            "raw_exact",
            &payload,
        )
        .unwrap();
        header.validate().unwrap();
        header.verify_payload_size().unwrap();
        assert_eq!(header.element_count(), 32 * 128);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut header = build_page_header(
            "full_attn_k",
            &[1],
            "float32",
            b'l',
            "sha256:aaaa",
            "sha256:bbbb",
            0,
            1,
            "raw_exact",
            &[0u8; 4],
        )
        .unwrap();
        header.magic = *b"BOGS";
        assert!(header.validate().is_err());
    }

    #[test]
    fn zero_dim_rejected() {
        let payload = sample_payload(4);
        let result = build_page_header(
            "full_attn_k",
            &[0, 128],
            "float32",
            b'l',
            "sha256:aaaa",
            "sha256:bbbb",
            0,
            1,
            "raw_exact",
            &payload,
        );
        assert!(result.is_err());
    }

    #[test]
    fn dim_product_overflow_rejected() {
        // u32::MAX^2 actually fits in u64, so use more dims
        let result = build_page_header(
            "full_attn_k",
            &[u32::MAX, u32::MAX, u32::MAX],
            "float32",
            b'l',
            "sha256:aaaa",
            "sha256:bbbb",
            0,
            1,
            "raw_exact",
            &[0u8; 4],
        );
        assert!(result.is_err());
    }

    #[test]
    fn empty_component_kind_rejected() {
        let result = build_page_header(
            "",
            &[1],
            "float32",
            b'l',
            "sha256:aaaa",
            "sha256:bbbb",
            0,
            1,
            "raw_exact",
            &[0u8; 4],
        );
        assert!(result.is_err());
    }

    #[test]
    fn bad_endianness_rejected() {
        let payload = sample_payload(4);
        let result = build_page_header(
            "full_attn_k",
            &[1],
            "float32",
            b'x',
            "sha256:aaaa",
            "sha256:bbbb",
            0,
            1,
            "raw_exact",
            &payload,
        );
        assert!(result.is_err());
    }

    #[test]
    fn position_inversion_rejected() {
        let payload = sample_payload(4);
        let result = build_page_header(
            "full_attn_k",
            &[1],
            "float32",
            b'l',
            "sha256:aaaa",
            "sha256:bbbb",
            10,
            5,
            "raw_exact",
            &payload,
        );
        assert!(result.is_err());
    }

    #[test]
    fn payload_size_mismatch_rejected() {
        let payload = sample_payload(8); // 8 bytes = 2 float32s
        let header = build_page_header(
            "full_attn_k",
            &[1], // 1 element × 4 bytes = 4 expected
            "float32",
            b'l',
            "sha256:aaaa",
            "sha256:bbbb",
            0,
            1,
            "raw_exact",
            &payload,
        );
        // payload is 8 bytes but header says 1 float32 element = 4 bytes
        // build_page_header sets payload_len from actual payload, so it'd be 8
        // but dims says 1 element × 4 bytes = 4. Mismatch.
        assert!(header.is_err() || header.unwrap().verify_payload_size().is_err());
    }

    #[test]
    fn header_digest_is_deterministic() {
        let payload = sample_payload(1024);
        let h1 = build_page_header(
            "full_attn_k",
            &[32, 128],
            "float32",
            b'l',
            "sha256:aaaa",
            "sha256:bbbb",
            0,
            64,
            "raw_exact",
            &payload,
        )
        .unwrap();
        let h2 = build_page_header(
            "full_attn_k",
            &[32, 128],
            "float32",
            b'l',
            "sha256:aaaa",
            "sha256:bbbb",
            0,
            64,
            "raw_exact",
            &payload,
        )
        .unwrap();
        assert_eq!(h1.header_digest, h2.header_digest);
        assert_eq!(h1.payload_digest, h2.payload_digest);
    }

    #[test]
    fn different_payload_different_digest() {
        let p1 = vec![0u8; 64];
        let p2 = vec![1u8; 64];
        // sanity check
        assert_ne!(
            blake3::hash(&p1).to_hex().to_string(),
            blake3::hash(&p2).to_hex().to_string()
        );

        let h1 = build_page_header(
            "full_attn_k",
            &[16],
            "float32",
            b'l',
            "sha256:aaaa",
            "sha256:bbbb",
            0,
            1,
            "raw_exact",
            &p1,
        )
        .unwrap();
        let h2 = build_page_header(
            "full_attn_k",
            &[16],
            "float32",
            b'l',
            "sha256:aaaa",
            "sha256:bbbb",
            0,
            1,
            "raw_exact",
            &p2,
        )
        .unwrap();
        // Different payload → different payload digest
        assert_ne!(h1.payload_digest, h2.payload_digest);
        // Same header identity fields → same header digest (payload_digest is zeroed during header hash)
        assert_eq!(h1.header_digest, h2.header_digest);
    }

    #[test]
    fn blake3_really_hashes_different_inputs() {
        let d1 = blake3::hash(&[0u8; 4]).to_hex().to_string();
        let d2 = blake3::hash(&[1u8; 4]).to_hex().to_string();
        assert_ne!(d1, d2, "blake3 should produce different hashes");
    }
}
