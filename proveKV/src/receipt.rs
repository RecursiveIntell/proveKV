use serde::{Deserialize, Serialize};

use crate::policy::CompressionPolicy;

/// Schema version for receipts.
pub const RECEIPT_SCHEMA: &str = "prove_kv_receipt_v1";
/// Schema version for pool build receipts.
pub const POOL_BUILD_RECEIPT_SCHEMA: &str = "pool_build_receipt_v1";
/// Schema version for shell materialize receipts.
pub const SHELL_MATERIALIZE_RECEIPT_SCHEMA: &str = "shell_materialize_receipt_v1";
/// Schema version for injection receipts.
pub const INJECTION_RECEIPT_SCHEMA: &str = "injection_receipt_v1";

/// Receipt produced when building a SharedKVPool.
///
/// Content-addressed via blake3 digest of canonical JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoolBuildReceipt {
    /// Stable schema marker.
    pub schema_version: String,
    /// Blake3 digest of the entire pool.
    pub pool_digest: String,
    /// Per-layer blake3 digests.
    pub layer_digests: Vec<String>,
    /// Digest of the fib-quant codebook.
    pub codebook_digest: String,
    /// Digest of the fib-quant rotation.
    pub rotation_digest: String,
    /// Total number of tokens in the pool.
    pub total_tokens: u32,
    /// Milliseconds spent on fib-quant build.
    pub fib_build_ms: u64,
    /// Total compressed pool size in bytes.
    pub pool_size_bytes: u64,
    /// Total raw f32 size in bytes.
    pub raw_size_bytes: u64,
    /// Compression ratio: raw / compressed.
    pub compression_ratio: f64,
    /// Snapshot of the policy used.
    pub policy_snapshot: CompressionPolicy,
    /// Seed used for construction.
    pub seeded_with: u64,
    /// Unix timestamp when built.
    pub built_at_unix: i64,
    /// Backend used: "cpu" or "gpu".
    #[serde(default = "default_backend")]
    pub backend: String,
}

fn default_backend() -> String {
    "cpu".to_string()
}

impl PoolBuildReceipt {
    /// Create a new pool build receipt.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pool_digest: String,
        layer_digests: Vec<String>,
        codebook_digest: String,
        rotation_digest: String,
        total_tokens: u32,
        fib_build_ms: u64,
        pool_size_bytes: u64,
        raw_size_bytes: u64,
        policy_snapshot: CompressionPolicy,
        seeded_with: u64,
        built_at_unix: i64,
    ) -> Self {
        let compression_ratio = if pool_size_bytes > 0 {
            raw_size_bytes as f64 / pool_size_bytes as f64
        } else {
            0.0
        };

        Self {
            schema_version: POOL_BUILD_RECEIPT_SCHEMA.into(),
            pool_digest,
            layer_digests,
            codebook_digest,
            rotation_digest,
            total_tokens,
            fib_build_ms,
            pool_size_bytes,
            raw_size_bytes,
            compression_ratio,
            policy_snapshot,
            seeded_with,
            built_at_unix,
            backend: "cpu".to_string(),
        }
    }

    /// Set the backend used for this build.
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_string();
        self
    }

    /// Validates this receipt schema version.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.schema_version != POOL_BUILD_RECEIPT_SCHEMA {
            return Err(crate::error::ProveKvError::InvalidReceipt(format!(
                "expected schema {}, got {}",
                POOL_BUILD_RECEIPT_SCHEMA, self.schema_version
            )));
        }
        if self.pool_digest.is_empty() {
            return Err(crate::error::ProveKvError::InvalidReceipt(
                "pool_digest is empty".into(),
            ));
        }
        Ok(())
    }

    /// Compute the canonical digest of this receipt.
    pub fn digest(&self) -> crate::error::Result<String> {
        let json = serde_json::to_string(self)?;
        Ok(blake3::hash(json.as_bytes()).to_hex().to_string())
    }
}

/// Receipt produced when materializing an AgentShell from a pool.
///
/// Content-addressed via blake3 digest of canonical JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShellMaterializeReceipt {
    /// Stable schema marker.
    pub schema_version: String,
    /// Agent identifier.
    pub agent_id: String,
    /// Digest of the parent pool.
    pub pool_digest: String,
    /// Digest of the materialized shell.
    pub shell_digest: String,
    /// Number of unique tokens in this shell.
    pub num_unique_tokens: u32,
    /// Total compressed shell size in bytes.
    pub shell_size_bytes: u64,
    /// Milliseconds spent on materialization.
    pub materialize_ms: u64,
    /// Unix timestamp when materialized.
    pub materialized_at_unix: i64,
}

impl ShellMaterializeReceipt {
    /// Create a new shell materialize receipt.
    pub fn new(
        agent_id: String,
        pool_digest: String,
        shell_digest: String,
        num_unique_tokens: u32,
        shell_size_bytes: u64,
        materialize_ms: u64,
        materialized_at_unix: i64,
    ) -> Self {
        Self {
            schema_version: SHELL_MATERIALIZE_RECEIPT_SCHEMA.into(),
            agent_id,
            pool_digest,
            shell_digest,
            num_unique_tokens,
            shell_size_bytes,
            materialize_ms,
            materialized_at_unix,
        }
    }

    /// Validate this receipt.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.schema_version != SHELL_MATERIALIZE_RECEIPT_SCHEMA {
            return Err(crate::error::ProveKvError::InvalidReceipt(format!(
                "expected schema {}, got {}",
                SHELL_MATERIALIZE_RECEIPT_SCHEMA, self.schema_version
            )));
        }
        if self.agent_id.is_empty() {
            return Err(crate::error::ProveKvError::InvalidReceipt(
                "agent_id is empty".into(),
            ));
        }
        if self.pool_digest.is_empty() {
            return Err(crate::error::ProveKvError::InvalidReceipt(
                "pool_digest is empty".into(),
            ));
        }
        if self.shell_digest.is_empty() {
            return Err(crate::error::ProveKvError::InvalidReceipt(
                "shell_digest is empty".into(),
            ));
        }
        Ok(())
    }

    /// Compute the canonical digest of this receipt.
    pub fn digest(&self) -> crate::error::Result<String> {
        let json = serde_json::to_string(self)?;
        Ok(blake3::hash(json.as_bytes()).to_hex().to_string())
    }
}

/// Receipt produced when injecting a shell into a KV cache.
///
/// Traces the source of every injected block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InjectionReceipt {
    /// Stable schema marker.
    pub schema_version: String,
    /// Agent identifier.
    pub agent_id: String,
    /// Digest of the parent pool.
    pub pool_digest: String,
    /// Digest of the injected shell.
    pub shell_digest: String,
    /// Number of injected blocks.
    pub blocks_injected: u32,
    /// Per-block digest traces (source → target).
    pub block_traces: Vec<BlockInjectionTrace>,
    /// Unix timestamp when injection occurred.
    pub injected_at_unix: i64,
}

/// Traces one block's injection from source (pool or shell) to target cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockInjectionTrace {
    /// Layer index.
    pub layer: u32,
    /// Source: "pool" or "shell".
    pub source: String,
    /// Digest of the source block.
    pub source_digest: String,
    /// Position in the target cache.
    pub target_position: u32,
}

impl InjectionReceipt {
    /// Create a new injection receipt.
    pub fn new(
        agent_id: String,
        pool_digest: String,
        shell_digest: String,
        blocks_injected: u32,
        block_traces: Vec<BlockInjectionTrace>,
        injected_at_unix: i64,
    ) -> Self {
        Self {
            schema_version: INJECTION_RECEIPT_SCHEMA.into(),
            agent_id,
            pool_digest,
            shell_digest,
            blocks_injected,
            block_traces,
            injected_at_unix,
        }
    }

    /// Validate this receipt.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.schema_version != INJECTION_RECEIPT_SCHEMA {
            return Err(crate::error::ProveKvError::InvalidReceipt(format!(
                "expected schema {}, got {}",
                INJECTION_RECEIPT_SCHEMA, self.schema_version
            )));
        }
        if self.agent_id.is_empty() {
            return Err(crate::error::ProveKvError::InvalidReceipt(
                "agent_id is empty".into(),
            ));
        }
        if self.pool_digest.is_empty() {
            return Err(crate::error::ProveKvError::InvalidReceipt(
                "pool_digest is empty".into(),
            ));
        }
        if self.shell_digest.is_empty() {
            return Err(crate::error::ProveKvError::InvalidReceipt(
                "shell_digest is empty".into(),
            ));
        }
        Ok(())
    }

    /// Compute the canonical digest of this receipt.
    pub fn digest(&self) -> crate::error::Result<String> {
        let json = serde_json::to_string(self)?;
        Ok(blake3::hash(json.as_bytes()).to_hex().to_string())
    }
}

/// Get the current unix timestamp as i64.
pub fn now_unix() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::CompressionPolicy;

    #[test]
    fn test_pool_build_receipt_round_trip() {
        let receipt = PoolBuildReceipt::new(
            "abc123".into(),
            vec!["layer0_digest".into(), "layer1_digest".into()],
            "codebook_digest".into(),
            "rotation_digest".into(),
            100,
            42,
            10_000,
            500_000,
            CompressionPolicy::default_two_tier(),
            42,
            now_unix(),
        );
        assert!(receipt.validate().is_ok());

        let json = serde_json::to_string(&receipt).unwrap();
        let deser: PoolBuildReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(receipt.pool_digest, deser.pool_digest);
        assert_eq!(receipt.layer_digests, deser.layer_digests);
        assert_eq!(receipt.compression_ratio, deser.compression_ratio);
    }

    #[test]
    fn test_shell_receipt_round_trip() {
        let receipt = ShellMaterializeReceipt::new(
            "agent_1".into(),
            "pool_abc".into(),
            "shell_xyz".into(),
            50,
            5_000,
            10,
            now_unix(),
        );
        assert!(receipt.validate().is_ok());

        let json = serde_json::to_string(&receipt).unwrap();
        let deser: ShellMaterializeReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(receipt.shell_digest, deser.shell_digest);
    }

    #[test]
    fn test_injection_receipt_traces() {
        let traces = vec![
            BlockInjectionTrace {
                layer: 0,
                source: "pool".into(),
                source_digest: "abc".into(),
                target_position: 0,
            },
            BlockInjectionTrace {
                layer: 0,
                source: "shell".into(),
                source_digest: "def".into(),
                target_position: 1,
            },
        ];
        let receipt = InjectionReceipt::new(
            "agent_1".into(),
            "pool_abc".into(),
            "shell_xyz".into(),
            2,
            traces,
            now_unix(),
        );
        assert!(receipt.validate().is_ok());
        assert_eq!(receipt.blocks_injected, 2);
    }
}
