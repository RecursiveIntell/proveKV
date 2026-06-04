use std::time::Instant;

use crate::codec::{CompressedBlock, TurboQuantAdapter};
use crate::error::{ProveKvError, Result};
use crate::manifest::ShellManifest;
use crate::policy::{is_batched_turbo, is_batched_turbo_lossy, turbo_batched_codec_id};
use crate::pool::SharedKVPool;
use crate::receipt::{now_unix, ShellMaterializeReceipt};

/// One layer's worth of agent-unique compressed KV blocks.
///
/// Storage is dual-form: either batched (one `CompressedBlock` per K/V side
/// holding the whole layer's worth of TQB1 batched bytes) or per-vector
/// (one `CompressedBlock` per (token, head) for legacy receipts). The
/// `key_codec` and `value_codec` of the blocks are the source of truth —
/// batched blocks carry `turbo_8bit_batched`, legacy blocks carry
/// `turbo_8bit`. Decoding dispatches per block.
#[derive(Debug, Clone)]
pub struct ShellLayer {
    /// Zero-based layer index.
    pub layer_index: u32,
    /// Key blocks unique to this agent. Batched: 1 block. Legacy: per (token, head).
    pub key_blocks: Vec<CompressedBlock>,
    /// Value blocks unique to this agent. Same semantics as `key_blocks`.
    pub value_blocks: Vec<CompressedBlock>,
    /// Blake3 digest of all blocks in this layer.
    pub block_digest: String,
}

impl ShellLayer {
    /// True when this layer's K and V are stored as single batched TQB1 blocks.
    /// Recognizes both the lossless (`turbo_8bit_batched`) and lossy
    /// (`turbo_8bit_batched_lossy`) variants — both are single-blob
    /// batched storage; the difference is the radii codec inside the wire.
    pub fn is_batched(&self) -> bool {
        self.key_blocks.len() == 1
            && self.value_blocks.len() == 1
            && (is_batched_turbo(&self.key_blocks[0].codec)
                || is_batched_turbo_lossy(&self.key_blocks[0].codec))
    }

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

impl AgentShell {
    /// Decompress every layer in this shell and return the full K and V
    /// tensors in HuggingFace-friendly layout. The output per layer is
    /// `(K_flat, V_flat)` with shape `[num_unique_tokens * num_kv_heads * head_dim]`,
    /// laid out as `[t0_h0_d0.., t0_h1_d0.., ..., t1_h0_d0.., ...]`.
    pub fn decompress_all_layers_with_seed(
        &self,
        seed: u64,
    ) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        let head_dim = self.shell_manifest.head_dim;
        let num_kv_heads = self.shell_manifest.num_kv_heads as usize;
        let num_unique_tokens = self.shell_manifest.num_unique_tokens as usize;

        let turbo = TurboQuantAdapter::with_radii_compression(
            head_dim,
            self.shell_manifest.turbo_config.bits,
            self.shell_manifest.turbo_config.projections,
            self.shell_manifest.turbo_config.radii_compression,
        )?;

        // Re-layout a per-head flattened K/V (shape
        // `[num_kv_heads][num_tokens * head_dim]`) into a single flat buffer
        // of shape `[num_tokens, num_kv_heads, head_dim]` (so that the
        // Python bench's `.reshape(1, num_tokens, num_kv_heads, head_dim)
        // .transpose(1, 2)` yields `[1, num_kv_heads, num_tokens, head_dim]`).
        let mut flatten =
            |per_head: &[Vec<f32>]| -> Vec<f32> {
                let mut out = Vec::with_capacity(num_unique_tokens * num_kv_heads * head_dim);
                for t in 0..num_unique_tokens {
                    for h in 0..num_kv_heads {
                        let base = t * head_dim;
                        out.extend_from_slice(&per_head[h][base..base + head_dim]);
                    }
                }
                out
            };

        let mut out = Vec::with_capacity(self.unique_layers.len());
        for layer in &self.unique_layers {
            let k_decoded: Vec<Vec<f32>> =
                turbo.decode_batch_compact(&layer.key_blocks[0].encoded_payload, seed)?;
            let v_decoded: Vec<Vec<f32>> =
                turbo.decode_batch_compact(&layer.value_blocks[0].encoded_payload, seed)?;

            let mut k_per_head: Vec<Vec<f32>> =
                vec![Vec::with_capacity(num_unique_tokens * head_dim); num_kv_heads];
            let mut v_per_head: Vec<Vec<f32>> =
                vec![Vec::with_capacity(num_unique_tokens * head_dim); num_kv_heads];
            for t in 0..num_unique_tokens {
                for h in 0..num_kv_heads {
                    let idx = t * num_kv_heads + h;
                    k_per_head[h].extend_from_slice(&k_decoded[idx]);
                    v_per_head[h].extend_from_slice(&v_decoded[idx]);
                }
            }
            out.push((flatten(&k_per_head), flatten(&v_per_head)));
        }
        Ok(out)
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
            pool.manifest.policy.shell_codec.clone(),
            seed,
            now_unix(),
            pool.manifest.shape.head_dim,
            pool.manifest.shape.num_kv_heads,
            pool.manifest.policy.turbo_config.clone(),
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

    // Create the turbo-quant codec. We hold the concrete adapter so we can
    // call `encode_batch_compact` and produce a single TQB1 batched payload
    // per (layer, K/V) — much smaller on disk than one block per (token, head).
    // Build the turbo-quant codec. The radii compression policy controls
    // whether the wire format stores raw f32 radii (lossless, default) or
    // BlockLogU8 (lossy, opt-in for the 58.14x system number). The decoder
    // reads the radii_codec flag from the wire and applies the inverse, so
    // a lossless shell and a lossy shell with the same dim/projections are
    // both decodable from the same decoder.
    let turbo_adapter = TurboQuantAdapter::with_radii_compression(
        head_dim,
        pool.policy.turbo_config.bits,
        pool.policy.turbo_config.projections,
        pool.policy.turbo_config.radii_compression,
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

    // Build a closure that produces one ShellLayer. Each layer is
    // independent (different head/data ranges in the agent corpus), so
    // we can dispatch them in parallel via Rayon when the feature is
    // enabled. Layers are collected by index to preserve order in the
    // output (we collect by index, not by completion time).
    let build_layer = |layer_idx: usize| -> Result<(ShellLayer, u64)> {
        // Collect every (token, head) K and V vector for this layer up
        // front, then dispatch two batched encodes (one for K, one for V).
        // The TQB1 batched path produces a single bytes blob per side
        // holding the whole layer's worth of turbo codes, instead of one
        // block per (token, head). 1.34x smaller on disk.
        let mut key_inputs: Vec<Vec<f32>> =
            Vec::with_capacity(num_unique_tokens as usize * num_kv_heads);
        let mut value_inputs: Vec<Vec<f32>> =
            Vec::with_capacity(num_unique_tokens as usize * num_kv_heads);
        for (_token_id, vec) in agent_tokens.iter() {
            for head_idx in 0..num_kv_heads {
                let base_offset =
                    layer_idx * num_kv_heads * head_dim * 2 + head_idx * head_dim * 2;
                let key_end = base_offset + head_dim;
                let value_end = key_end + head_dim;
                key_inputs.push(vec[base_offset..key_end].to_vec());
                value_inputs.push(vec[key_end..value_end].to_vec());
            }
        }
        let key_refs: Vec<&[f32]> = key_inputs.iter().map(|v| v.as_slice()).collect();
        let value_refs: Vec<&[f32]> = value_inputs.iter().map(|v| v.as_slice()).collect();

        let encoded_keys = turbo_adapter.encode_batch_compact(&key_refs, seed)?;
        let encoded_values = turbo_adapter.encode_batch_compact(&value_refs, seed)?;

        // Pick the codec id for this shell based on the policy's
        // `bits` and `radii_compression` fields. Both ids share the same
        // on-disk structure (one batched block per K/V per layer); the
        // difference is in how radii are encoded INSIDE the block. The
        // decoder dispatches on the id and reads the wire's bits/radii
        // bytes.
        let lossy_radii = matches!(
            pool.policy.turbo_config.radii_compression,
            crate::policy::RadiiCompression::Lossy
        );
        let shell_codec_id = turbo_batched_codec_id(pool.policy.turbo_config.bits, lossy_radii);
        let key_block = CompressedBlock::new(shell_codec_id.to_string(), encoded_keys, head_dim);
        let value_block =
            CompressedBlock::new(shell_codec_id.to_string(), encoded_values, head_dim);

        let layer_bytes = key_block.compressed_bytes as u64 + value_block.compressed_bytes as u64;
        let key_blocks = vec![key_block];
        let value_blocks = vec![value_block];

        let mut layer = ShellLayer {
            layer_index: layer_idx as u32,
            key_blocks,
            value_blocks,
            block_digest: String::new(),
        };
        layer.block_digest = layer.compute_digest()?;
        Ok((layer, layer_bytes))
    };

    // Layer build: serial or parallel. Both paths preserve layer order
    // in the output (we collect by index, not by completion time).
    let layer_results: Vec<Result<(ShellLayer, u64)>> = {
        #[cfg(feature = "parallel_shell")]
        {
            use rayon::prelude::*;
            (0..num_layers).into_par_iter().map(build_layer).collect()
        }
        #[cfg(not(feature = "parallel_shell"))]
        {
            (0..num_layers).map(build_layer).collect()
        }
    };
    for r in layer_results {
        let (layer, layer_bytes) = r?;
        total_shell_bytes += layer_bytes;
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
        pool.manifest.policy.shell_codec.clone(),
        seed,
        materialized_at_unix,
        pool.manifest.shape.head_dim,
        pool.manifest.shape.num_kv_heads,
        pool.manifest.policy.turbo_config.clone(),
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
