use serde::{Deserialize, Serialize};

use crate::{
    codebook::CODEBOOK_SCHEMA,
    codec::CODE_SCHEMA,
    profile::{FibQuantProfileV1, NormFormat, PROFILE_SCHEMA},
    rotation::ROTATION_ALGORITHM_VERSION,
};

pub const RECEIPT_SCHEMA: &str = "fib_quant_compression_receipt_v1";

/// Compression receipt emitted by `encode_with_receipt`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FibQuantCompressionReceiptV1 {
    /// Stable schema marker.
    pub schema_version: String,
    /// Profile digest.
    pub profile_digest: String,
    /// Codebook digest.
    pub codebook_digest: String,
    /// Rotation digest.
    pub rotation_digest: String,
    /// Digest over the canonical source vector bytes.
    pub source_vector_digest: String,
    /// Encoded payload digest.
    pub encoded_digest: String,
    /// Norm payload format used by the encoded artifact.
    pub norm_format: NormFormat,
    /// Encoded artifact schema marker.
    pub code_schema_version: String,
    /// Profile schema marker.
    pub profile_schema_version: String,
    /// Codebook schema marker.
    pub codebook_schema_version: String,
    /// Ambient dimension `d`.
    pub ambient_dim: u32,
    /// Block dimension `k`.
    pub block_dim: u32,
    /// Codebook size `N`.
    pub codebook_size: u32,
    /// Paper rate.
    pub paper_rate_bits_per_coord: f64,
    /// Wire index bits.
    pub wire_index_bits: u8,
    /// Wire rate.
    pub wire_bits_per_coord: f64,
    /// Rotation seed.
    pub rotation_seed: u64,
    /// Rotation algorithm identity.
    pub rotation_algorithm_version: String,
    /// Codebook seed.
    pub codebook_seed: u64,
    /// Optional reconstruction MSE if measured.
    pub mse: Option<f64>,
    /// Optional cosine similarity if measured.
    pub cosine_similarity: Option<f64>,
    /// Whether a fallback codec was available.
    pub fallback_available: bool,
    /// Unix timestamp seconds, supplied by caller path.
    pub recorded_unix_seconds: i64,
}

impl FibQuantCompressionReceiptV1 {
    pub(crate) fn new(
        profile: &FibQuantProfileV1,
        profile_digest: String,
        codebook_digest: String,
        rotation_digest: String,
        source_vector_digest: String,
        encoded_digest: String,
    ) -> Self {
        Self {
            schema_version: RECEIPT_SCHEMA.into(),
            profile_digest,
            codebook_digest,
            rotation_digest,
            source_vector_digest,
            encoded_digest,
            norm_format: profile.norm_format.clone(),
            code_schema_version: CODE_SCHEMA.into(),
            profile_schema_version: PROFILE_SCHEMA.into(),
            codebook_schema_version: CODEBOOK_SCHEMA.into(),
            ambient_dim: profile.ambient_dim,
            block_dim: profile.block_dim,
            codebook_size: profile.codebook_size,
            paper_rate_bits_per_coord: profile.paper_rate_bits_per_coord,
            wire_index_bits: profile.wire_index_bits,
            wire_bits_per_coord: profile.wire_bits_per_coord,
            rotation_seed: profile.rotation_seed,
            rotation_algorithm_version: ROTATION_ALGORITHM_VERSION.into(),
            codebook_seed: profile.codebook_seed,
            mse: None,
            cosine_similarity: None,
            fallback_available: false,
            recorded_unix_seconds: current_unix_seconds(),
        }
    }
}

fn current_unix_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}
