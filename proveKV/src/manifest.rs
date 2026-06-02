use serde::{Deserialize, Serialize};

use crate::policy::{CodecId, CompressionPolicy};
use crate::shape::KvTensorShape;

/// Schema version for PoolManifest.
pub const POOL_MANIFEST_SCHEMA: &str = "pool_manifest_v1";
/// Schema version for ShellManifest.
pub const SHELL_MANIFEST_SCHEMA: &str = "shell_manifest_v1";

/// Manifest describing a built SharedKVPool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoolManifest {
    /// Stable schema marker.
    pub schema_version: String,
    /// Unique pool identifier (blake3 of canonical JSON).
    pub pool_id: String,
    /// Logical tensor shape of the underlying model.
    pub shape: KvTensorShape,
    /// The compression policy used.
    pub policy: CompressionPolicy,
    /// Number of shared tokens in the pool.
    pub num_shared_tokens: u32,
    /// Number of transformer layers.
    pub num_layers: u32,
    /// Total compressed pool size in bytes.
    pub pool_size_bytes: u64,
    /// Codec used for shared pool blocks.
    pub shared_codec: CodecId,
    /// Overall compression ratio: raw f32 bytes / compressed bytes.
    pub compression_ratio: f64,
    /// Unix timestamp when the pool was built.
    pub built_at_unix: i64,
    /// Seed used during pool construction.
    pub build_seed: u64,
}

impl PoolManifest {
    /// Create and validate a pool manifest.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pool_id: String,
        shape: KvTensorShape,
        policy: CompressionPolicy,
        num_shared_tokens: u32,
        num_layers: u32,
        pool_size_bytes: u64,
        raw_size_bytes: u64,
        build_seed: u64,
        built_at_unix: i64,
    ) -> crate::error::Result<Self> {
        let compression_ratio = if pool_size_bytes > 0 {
            raw_size_bytes as f64 / pool_size_bytes as f64
        } else {
            0.0
        };

        let manifest = Self {
            schema_version: POOL_MANIFEST_SCHEMA.into(),
            pool_id,
            shape,
            policy,
            num_shared_tokens,
            num_layers,
            pool_size_bytes,
            shared_codec: CODEC_IDENTIFIER_SHARED.into(),
            compression_ratio,
            built_at_unix,
            build_seed,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    /// Validate this manifest against the schema.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.schema_version != POOL_MANIFEST_SCHEMA {
            return Err(crate::error::ProveKvError::InvalidManifest(format!(
                "expected schema {}, got {}",
                POOL_MANIFEST_SCHEMA, self.schema_version
            )));
        }
        if self.pool_id.is_empty() {
            return Err(crate::error::ProveKvError::InvalidManifest(
                "pool_id is empty".into(),
            ));
        }
        self.shape.validate()?;
        self.policy.validate()?;
        Ok(())
    }

    /// Compute the canonical digest of this manifest.
    pub fn digest(&self) -> crate::error::Result<String> {
        let json = serde_json::to_string(self)?;
        Ok(blake3::hash(json.as_bytes()).to_hex().to_string())
    }
}

/// Manifest describing a materialized agent shell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShellManifest {
    /// Stable schema marker.
    pub schema_version: String,
    /// Agent identifier.
    pub agent_id: String,
    /// Digest of the parent pool.
    pub pool_digest: String,
    /// Number of unique tokens in this shell.
    pub num_unique_tokens: u32,
    /// Number of layers with unique tokens.
    pub num_unique_layers: u32,
    /// Total compressed shell size in bytes.
    pub shell_size_bytes: u64,
    /// Codec used for shell blocks.
    pub shell_codec: CodecId,
    /// Unix timestamp when the shell was materialized.
    pub materialized_at_unix: i64,
    /// Seed used during materialization.
    pub materialize_seed: u64,
}

impl ShellManifest {
    /// Create and validate a shell manifest.
    pub fn new(
        agent_id: String,
        pool_digest: String,
        num_unique_tokens: u32,
        num_unique_layers: u32,
        shell_size_bytes: u64,
        materialize_seed: u64,
        materialized_at_unix: i64,
    ) -> crate::error::Result<Self> {
        let manifest = Self {
            schema_version: SHELL_MANIFEST_SCHEMA.into(),
            agent_id,
            pool_digest,
            num_unique_tokens,
            num_unique_layers,
            shell_size_bytes,
            shell_codec: CODEC_IDENTIFIER_SHELL.into(),
            materialized_at_unix,
            materialize_seed,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    /// Validate this manifest against the schema.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.schema_version != SHELL_MANIFEST_SCHEMA {
            return Err(crate::error::ProveKvError::InvalidManifest(format!(
                "expected schema {}, got {}",
                SHELL_MANIFEST_SCHEMA, self.schema_version
            )));
        }
        if self.agent_id.is_empty() {
            return Err(crate::error::ProveKvError::InvalidManifest(
                "agent_id is empty".into(),
            ));
        }
        if self.pool_digest.is_empty() {
            return Err(crate::error::ProveKvError::InvalidManifest(
                "pool_digest is empty".into(),
            ));
        }
        Ok(())
    }

    /// Compute the canonical digest of this manifest.
    pub fn digest(&self) -> crate::error::Result<String> {
        let json = serde_json::to_string(self)?;
        Ok(blake3::hash(json.as_bytes()).to_hex().to_string())
    }
}

// Well-known codec identifiers (mirrored in policy.rs, referenced here for manifest usage).
const CODEC_IDENTIFIER_SHARED: &str = crate::policy::CODEC_FIB_K4_N32;
const CODEC_IDENTIFIER_SHELL: &str = crate::policy::CODEC_TURBO_8BIT;
