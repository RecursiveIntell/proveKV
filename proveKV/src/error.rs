use thiserror::Error;

/// Errors for the proveKV crate.
#[derive(Error, Debug)]
pub enum ProveKvError {
    #[error("shape validation failed: {0}")]
    InvalidShape(String),

    #[error("policy validation failed: {0}")]
    InvalidPolicy(String),

    #[error("manifest validation failed: {0}")]
    InvalidManifest(String),

    #[error("receipt validation failed: {0}")]
    InvalidReceipt(String),

    #[error("corrupt payload: {0}")]
    CorruptPayload(String),

    #[error("compression failed: {0}")]
    CompressionFailed(String),

    #[error("decompression failed: {0}")]
    DecompressionFailed(String),

    #[error("digest mismatch: expected {expected}, got {got}")]
    DigestMismatch { expected: String, got: String },

    #[error("dimensionality mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("layer index out of bounds: {index} of {total}")]
    LayerIndexOutOfBounds { index: u32, total: u32 },

    #[error("empty corpus: cannot build pool from zero tokens")]
    EmptyCorpus,

    #[error("codec unavailable: codec '{codec}' requires feature '{feature}'")]
    CodecUnavailable { codec: String, feature: String },

    #[error("internal error: {0}")]
    Internal(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type alias for proveKV.
pub type Result<T> = std::result::Result<T, ProveKvError>;
