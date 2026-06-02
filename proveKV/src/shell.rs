use std::time::Instant;

use crate::codec::{create_codec, CompressedBlock};
use crate::error::{ProveKvError, Result};
use crate::manifest::ShellManifest;
use crate::policy::CODEC_TURBO_8BIT;
use crate::pool::SharedKVPool;
use crate::receipt::{now_unix, ShellMaterializeReceipt};

/// One layer's worth of agent-unique compressed KV blocks.
#[derive(Debug, Clone)]
pub struct ShellLayer {
    /// Zero-based layer index.
    pub layer_index: u32,
    /// Key blocks unique to this agent (turbo-quant compressed).
    pub key_blocks: Vec<CompressedBlock>,
    /// Value blocks unique to this agent (turbo-quant compressed).
    pub value_blocks: Vec<CompressedBlock>,
    /// Blake3 digest of all blocks in this layer.
    pub block_digest: String,
}

/// A per-agent compressed context shell.
///
/// AgentShell stores turbo-quant compressed KV vectors for tokens that are
/// unique to a specific agent. Shared tokens are referenced from the parent
/// pool by digest, not duplicated.
#[derive(Debug, Clone)]
pub struct AgentShell {
    /// Agent identifier.
    pub agent_id: String,
    /// Shell manifest.
    pub shell_manifest: ShellManifest,
    /// Per-layer unique token blocks (turbo-quant compressed).
    pub unique_layers: Vec<ShellLayer>,
    /// Reference to the parent pool digest.
    pub pool_digest: String,
}

impl ShellLayer {
    /// Compute a content digest over the blocks in this layer.
    fn compute_digest(&self) -> Result<String> {
        let key_digests: Vec<&str> = self
            .key_blocks
            .iter()
            .map(|b| b.payload_digest.as_str())
            .collect();
        let value_digests: Vec<&str> = self
            .value_blocks
            .iter()
            .map(|b| b.payload_digest.as_str())
            .collect();
        let payload = serde_json::json!({
            "layer_index": self.layer_index,
            "key_digests": key_digests,
            "value_digests": value_digests,
        });
        let json = serde_json::to_string(&payload)
            .map_err(|e| ProveKvError::Internal(format!("shell layer digest: {}", e)))?;
        Ok(blake3::hash(json.as_bytes()).to_hex().to_string())
    }
}

/// Materialize an AgentShell from a SharedKVPool and agent-specific tokens.
///
/// Agent tokens that are not found in the shared pool are compressed with
/// turbo-quant and stored in shell layers.
pub fn materialize_shell(
    pool: &SharedKVPool,
    agent_id: &str,
    agent_tokens: &[(String, Vec<f32>)],
    seed: u64,
) -> Result<(AgentShell, ShellMaterializeReceipt)> {
    let start = Instant::now();

    if agent_tokens.is_empty() {
        // Empty shell — agent uses only shared pool tokens
        let shell_digest =
            blake3::hash(format!("empty_shell:{}:{}", agent_id, pool.manifest.pool_id).as_bytes())
                .to_hex()
                .to_string();

        let shell_manifest = ShellManifest::new(
            agent_id.to_string(),
            pool.manifest.pool_id.clone(),
            0,
            pool.manifest.num_layers,
            0,
            seed,
            now_unix(),
        )?;

        let receipt = ShellMaterializeReceipt::new(
            agent_id.to_string(),
            pool.manifest.pool_id.clone(),
            shell_digest.clone(),
            0,
            0,
            start.elapsed().as_millis() as u64,
            now_unix(),
        );

        return Ok((
            AgentShell {
                agent_id: agent_id.to_string(),
                shell_manifest,
                unique_layers: Vec::new(),
                pool_digest: pool.manifest.pool_id.clone(),
            },
            receipt,
        ));
    }

    let num_layers = pool.manifest.num_layers as usize;
    let num_kv_heads = pool.manifest.shape.num_kv_heads as usize;
    let head_dim = pool.manifest.shape.head_dim;

    // Create the turbo-quant codec
    let shell_codec = create_codec(
        CODEC_TURBO_8BIT,
        head_dim,
        None,
        Some(&pool.policy.turbo_config),
    )?;

    // Validate agent token shapes
    let expected_len = num_layers * num_kv_heads * head_dim * 2;
    for (token_id, vec) in agent_tokens {
        if vec.len() != expected_len {
            return Err(ProveKvError::DimensionMismatch {
                expected: expected_len,
                got: vec.len(),
            });
        }
        if vec.iter().any(|v| !v.is_finite()) {
            return Err(ProveKvError::CorruptPayload(format!(
                "agent token {} contains non-finite values",
                token_id
            )));
        }
    }

    let mut unique_layers: Vec<ShellLayer> = Vec::with_capacity(num_layers);
    let mut total_shell_bytes: u64 = 0;
    let num_unique_tokens = agent_tokens.len() as u32;

    for layer_idx in 0..num_layers {
        let mut key_blocks: Vec<CompressedBlock> =
            Vec::with_capacity(num_unique_tokens as usize * num_kv_heads);
        let mut value_blocks: Vec<CompressedBlock> =
            Vec::with_capacity(num_unique_tokens as usize * num_kv_heads);

        for (_token_id, vec) in agent_tokens.iter() {
            for head_idx in 0..num_kv_heads {
                let base_offset = layer_idx * num_kv_heads * head_dim * 2 + head_idx * head_dim * 2;

                let key_start = base_offset;
                let key_end = key_start + head_dim;
                let key: Vec<f32> = vec[key_start..key_end].to_vec();

                let value_start = key_end;
                let value_end = value_start + head_dim;
                let value: Vec<f32> = vec[value_start..value_end].to_vec();

                let encoded_key = shell_codec.encode(&key, seed)?;
                let encoded_value = shell_codec.encode(&value, seed)?;

                key_blocks.push(CompressedBlock::new(
                    shell_codec.codec_id(),
                    encoded_key,
                    head_dim,
                ));
                value_blocks.push(CompressedBlock::new(
                    shell_codec.codec_id(),
                    encoded_value,
                    head_dim,
                ));

                total_shell_bytes += key_blocks.last().unwrap().compressed_bytes as u64;
                total_shell_bytes += value_blocks.last().unwrap().compressed_bytes as u64;
            }
        }

        let mut layer = ShellLayer {
            layer_index: layer_idx as u32,
            key_blocks,
            value_blocks,
            block_digest: String::new(),
        };
        layer.block_digest = layer.compute_digest()?;
        unique_layers.push(layer);
    }

    let materialize_ms = start.elapsed().as_millis() as u64;
    let materialized_at_unix = now_unix();

    // Compute shell digest
    let layer_digests: Vec<String> = unique_layers
        .iter()
        .map(|l| l.block_digest.clone())
        .collect();
    let shell_digest = blake3::hash(
        serde_json::to_string(&serde_json::json!({
            "agent_id": agent_id,
            "pool_digest": pool.manifest.pool_id,
            "layer_digests": layer_digests,
            "seed": seed,
        }))
        .map_err(|e| ProveKvError::Internal(format!("shell digest: {}", e)))?
        .as_bytes(),
    )
    .to_hex()
    .to_string();

    let shell_manifest = ShellManifest::new(
        agent_id.to_string(),
        pool.manifest.pool_id.clone(),
        num_unique_tokens,
        num_layers as u32,
        total_shell_bytes,
        seed,
        materialized_at_unix,
    )?;

    let receipt = ShellMaterializeReceipt::new(
        agent_id.to_string(),
        pool.manifest.pool_id.clone(),
        shell_digest.clone(),
        num_unique_tokens,
        total_shell_bytes,
        materialize_ms,
        materialized_at_unix,
    );

    Ok((
        AgentShell {
            agent_id: agent_id.to_string(),
            shell_manifest,
            unique_layers,
            pool_digest: pool.manifest.pool_id.clone(),
        },
        receipt,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::SharedKVPool;
    use crate::shape::{AttentionType, KvTensorShape};

    fn make_test_shape() -> KvTensorShape {
        KvTensorShape {
            attention_type: AttentionType::MHA,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 8,
            hidden_size: 32,
        }
    }

    fn make_test_corpus(n: usize) -> Vec<(String, Vec<f32>)> {
        use rand::Rng;
        use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let shape = make_test_shape();
        let vec_len = shape.num_layers as usize * shape.num_kv_heads as usize * shape.head_dim * 2;

        (0..n)
            .map(|i| {
                let vec: Vec<f32> = (0..vec_len).map(|_| rng.gen_range(-1.0..1.0)).collect();
                (format!("token_{}", i), vec)
            })
            .collect()
    }

    #[test]
    fn test_shell_materialize_empty() {
        let shape = make_test_shape();
        let corpus = make_test_corpus(4);
        let (pool, _receipt) = SharedKVPool::build(&corpus, &shape, 42).unwrap();

        let agent_tokens: Vec<(String, Vec<f32>)> = vec![];
        let (shell, mat_receipt) = materialize_shell(&pool, "agent_1", &agent_tokens, 42).unwrap();

        assert_eq!(shell.agent_id, "agent_1");
        assert_eq!(mat_receipt.num_unique_tokens, 0);
        assert_eq!(mat_receipt.shell_size_bytes, 0);
    }

    #[test]
    fn test_shell_materialize_basic() {
        let shape = make_test_shape();
        let corpus = make_test_corpus(4);
        let (pool, _receipt) = SharedKVPool::build(&corpus, &shape, 42).unwrap();

        let agent_tokens = make_test_corpus(2);
        let (shell, mat_receipt) = materialize_shell(&pool, "agent_x", &agent_tokens, 42).unwrap();

        assert_eq!(shell.agent_id, "agent_x");
        assert_eq!(shell.unique_layers.len(), 2);
        assert_eq!(mat_receipt.num_unique_tokens, 2);
        assert!(mat_receipt.shell_size_bytes > 0);
        assert_eq!(shell.pool_digest, pool.manifest.pool_id);
    }

    #[test]
    fn test_shell_materialize_deterministic() {
        let shape = make_test_shape();
        let corpus = make_test_corpus(4);
        let (pool, _receipt) = SharedKVPool::build(&corpus, &shape, 42).unwrap();
        let agent_tokens = make_test_corpus(2);

        let (shell1, receipt1) = materialize_shell(&pool, "agent_x", &agent_tokens, 42).unwrap();
        let (shell2, receipt2) = materialize_shell(&pool, "agent_x", &agent_tokens, 42).unwrap();

        assert_eq!(receipt1.shell_digest, receipt2.shell_digest);
        assert_eq!(
            shell1.unique_layers[0].block_digest,
            shell2.unique_layers[0].block_digest
        );
    }
}
