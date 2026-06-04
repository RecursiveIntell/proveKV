use serde::{Deserialize, Serialize};

use crate::policy::{CodecId, CompressionPolicy, TurboConfig};
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

        let shared_codec = if policy.shared_codec.is_empty() {
            crate::policy::CODEC_FIB_K4_N32.into()
        } else {
            policy.shared_codec.clone()
        };
        let manifest = Self {
            schema_version: POOL_MANIFEST_SCHEMA.into(),
            pool_id,
            shape,
            policy,
            num_shared_tokens,
            num_layers,
            pool_size_bytes,
            shared_codec,
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
    /// Head dimension of the model (needed for decompression).
    pub head_dim: usize,
    /// Number of KV heads in the model.
    pub num_kv_heads: u32,
    /// TurboQuant configuration used for this shell.
    pub turbo_config: TurboConfig,
}

impl ShellManifest {
    /// Create and validate a shell manifest.
    ///
    /// The `shell_codec` argument is the codec id actually written into the
    /// per-layer blocks (e.g. `turbo_4bit_batched`, `turbo_8bit_batched_lossy`).
    /// Callers MUST derive it from the same `TurboConfig` they pass in (see
    /// [`crate::policy::turbo_batched_codec_id`]); the manifest will not
    /// invent a codec. This was previously hardcoded to `turbo_8bit` for all
    /// shells, which made the manifest lie about the actual block codec
    /// (audit finding F4, fixed 2026-06-03).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        agent_id: String,
        pool_digest: String,
        num_unique_tokens: u32,
        num_unique_layers: u32,
        shell_size_bytes: u64,
        shell_codec: crate::policy::CodecId,
        materialize_seed: u64,
        materialized_at_unix: i64,
        head_dim: usize,
        num_kv_heads: u32,
        turbo_config: TurboConfig,
    ) -> crate::error::Result<Self> {
        // Invariant: the manifest's shell_codec must agree with the
        // TurboConfig's bits and radii_compression. Refuse to build a
        // manifest that lies about either. Tests in shell_manifest_test.rs
        // cover this for the known config matrix.
        let expected = crate::policy::turbo_batched_codec_id(
            turbo_config.bits,
            matches!(
                turbo_config.radii_compression,
                crate::policy::RadiiCompression::Lossy
            ),
        );
        if shell_codec.as_str() != expected {
            return Err(crate::error::ProveKvError::InvalidManifest(format!(
                "shell_codec '{}' does not match turbo_config (bits={}, lossy={}); expected '{}'",
                shell_codec,
                turbo_config.bits,
                matches!(
                    turbo_config.radii_compression,
                    crate::policy::RadiiCompression::Lossy
                ),
                expected,
            )));
        }
        let manifest = Self {
            schema_version: SHELL_MANIFEST_SCHEMA.into(),
            agent_id,
            pool_digest,
            num_unique_tokens,
            num_unique_layers,
            shell_size_bytes,
            shell_codec,
            materialized_at_unix,
            materialize_seed,
            head_dim,
            num_kv_heads,
            turbo_config,
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
        // The shell_codec must agree with the turbo_config (bits, lossy).
        // This is the post-F4 invariant: the manifest cannot lie about
        // what codec was actually used to build the per-layer blocks.
        let expected = crate::policy::turbo_batched_codec_id(
            self.turbo_config.bits,
            matches!(
                self.turbo_config.radii_compression,
                crate::policy::RadiiCompression::Lossy
            ),
        );
        if self.shell_codec.as_str() != expected {
            return Err(crate::error::ProveKvError::InvalidManifest(format!(
                "shell_codec '{}' does not match turbo_config (bits={}, lossy={}); expected '{}'",
                self.shell_codec,
                self.turbo_config.bits,
                matches!(
                    self.turbo_config.radii_compression,
                    crate::policy::RadiiCompression::Lossy
                ),
                expected,
            )));
        }
        Ok(())
    }

    /// Compute the canonical digest of this manifest.
    pub fn digest(&self) -> crate::error::Result<String> {
        let json = serde_json::to_string(self)?;
        Ok(blake3::hash(json.as_bytes()).to_hex().to_string())
    }
}

// The shared pool codec id lives in `crate::policy::CODEC_FIB_K4_N32` and
// is referenced directly where needed. The shell codec is no longer
// hardcoded — callers must pass it in derived from the actual TurboConfig
// (see ShellManifest::new).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{RadiiCompression, TurboConfig};

    fn cfg(bits: u8, lossy: bool) -> TurboConfig {
        TurboConfig {
            bits,
            projections: 32,
            radii_compression: if lossy {
                RadiiCompression::Lossy
            } else {
                RadiiCompression::Lossless
            },
        }
    }

    fn make(bits: u8, lossy: bool) -> ShellManifest {
        let turbo = cfg(bits, lossy);
        let codec = crate::policy::turbo_batched_codec_id(bits, lossy);
        ShellManifest::new(
            "agent_0".into(),
            "00".repeat(16),
            0,
            24,
            0,
            codec,
            42,
            0,
            64,
            32,
            turbo,
        )
        .expect("manifest should build")
    }

    #[test]
    fn b4_lossless_shell_manifest_codec_is_turbo_4bit_batched() {
        let m = make(4, false);
        assert_eq!(m.shell_codec.as_str(), "turbo_4bit_batched");
        assert!(m.validate().is_ok());
    }

    #[test]
    fn b4_lossy_shell_manifest_codec_is_turbo_4bit_batched_lossy() {
        let m = make(4, true);
        assert_eq!(m.shell_codec.as_str(), "turbo_4bit_batched_lossy");
        assert!(m.validate().is_ok());
    }

    #[test]
    fn b8_lossless_shell_manifest_codec_is_turbo_8bit_batched() {
        let m = make(8, false);
        assert_eq!(m.shell_codec.as_str(), "turbo_8bit_batched");
        assert!(m.validate().is_ok());
    }

    #[test]
    fn b8_lossy_shell_manifest_codec_is_turbo_8bit_batched_lossy() {
        let m = make(8, true);
        assert_eq!(m.shell_codec.as_str(), "turbo_8bit_batched_lossy");
        assert!(m.validate().is_ok());
    }

    #[test]
    fn shell_manifest_refuses_mismatched_codec() {
        // F4: a manifest that lies about its shell_codec must fail
        // validation, not silently pass. Construct a manifest where the
        // declared codec is `turbo_8bit_batched` but the turbo_config
        // is b=4, and confirm the new validate() invariant catches it.
        let turbo = cfg(4, false);
        let result = ShellManifest::new(
            "agent_0".into(),
            "00".repeat(16),
            0,
            24,
            0,
            "turbo_8bit_batched".into(), // wrong codec
            42,
            0,
            64,
            32,
            turbo,
        );
        assert!(
            result.is_err(),
            "manifest with mismatched codec must fail to build"
        );
        let err = format!("{:?}", result.unwrap_err());
        assert!(
            err.contains("shell_codec 'turbo_8bit_batched' does not match"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn shell_manifest_validate_catches_round_tripped_lie() {
        // Simulate an old (pre-F4) manifest that was deserialized from
        // disk with `shell_codec: "turbo_8bit"` but `turbo_config: {bits: 4}`.
        // The new validate() must reject it.
        let m = make(4, false);
        // Now mutate the codec to lie and re-validate.
        let mut bad = m.clone();
        bad.shell_codec = "turbo_8bit".into();
        let err = bad.validate().unwrap_err();
        let msg = format!("{:?}", err);
        assert!(
            msg.contains("shell_codec 'turbo_8bit' does not match"),
            "unexpected error: {}",
            msg
        );
    }
}
