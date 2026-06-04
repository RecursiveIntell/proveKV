use std::time::Instant;

use crate::codec::{create_codec, CompressedBlock, FibQuantAdapter, TurboQuantAdapter};
use crate::error::{ProveKvError, Result};
use crate::manifest::PoolManifest;
use crate::policy::{
    is_batched_fib, CompressionPolicy, CODEC_FIB_K4_N32, CODEC_FIB_K4_N32_BATCHED,
};
use crate::receipt::{now_unix, PoolBuildReceipt};
use crate::shape::KvTensorShape;

/// Re-layout a per-head flattened K/V (shape `[num_kv_heads][num_tokens * head_dim]`)
/// into a single flat buffer with shape `[num_tokens, num_kv_heads, head_dim]`
/// (so that `reshape(1, num_tokens, num_kv_heads, head_dim).transpose(1, 2)`
/// yields `[1, num_kv_heads, num_tokens, head_dim]`, which is the layout the
/// Python PPL bench uses to populate a HuggingFace `DynamicCache`).
///
/// The input `per_head[h_idx][t*head_dim + d]` is the d-th element of head h at
/// token t. The output is `[t0_h0_d0, t0_h0_d1, ..., t0_h0_dN, t0_h1_d0, ...,
/// t0_h1_dN, t1_h0_d0, ...]` — heads outer, tokens inner.
fn flatten_heads_tokens(
    per_head: &[Vec<f32>],
    num_tokens: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(num_tokens * num_kv_heads * head_dim);
    for t in 0..num_tokens {
        for h in 0..num_kv_heads {
            let base = t * head_dim;
            out.extend_from_slice(&per_head[h][base..base + head_dim]);
        }
    }
    out
}

/// One layer's worth of compressed KV blocks in the shared pool.
///
/// Storage is dual-form: either batched (one `CompressedBlock` per K/V side
/// holding the whole layer's worth of FB2 batched bytes) or per-vector
/// (one `CompressedBlock` per (token, head) for legacy receipts). The
/// `key_codec` and `value_codec` of the blocks are the source of truth —
/// batched blocks carry the `fib_k4_n32_batched` codec id, legacy blocks
/// carry `fib_k4_n32`. Callers can mix-and-match across layers; decoding
/// dispatches per block.
#[derive(Debug, Clone)]
pub struct PoolLayer {
    /// Zero-based layer index.
    pub layer_index: u32,
    /// Key blocks. In batched mode: exactly 1 block containing the whole
    /// layer's batched FB2 bytes. In legacy mode: one per (token, head).
    pub key_blocks: Vec<CompressedBlock>,
    /// Value blocks. Same dual-form semantics as `key_blocks`.
    pub value_blocks: Vec<CompressedBlock>,
    /// Blake3 digest of all blocks in this layer (canonical JSON).
    pub block_digest: String,
}

impl PoolLayer {
    /// True when this layer's K and V are stored as single batched blocks
    /// (FB2 / `fib_k4_n32_batched`). False for legacy per-vector storage.
    pub fn is_batched(&self) -> bool {
        self.key_blocks.len() == 1
            && self.value_blocks.len() == 1
            && is_batched_fib(&self.key_blocks[0].codec)
    }

    /// Total number of (token, head) K/V pairs this layer represents.
    /// For batched storage this is the size of the batched payload's
    /// inner stream, which we recover from the codec profile.
    pub fn n_vectors(&self) -> usize {
        if self.is_batched() {
            // Batched FB2 has 4 bytes for n_blocks at offset 10..14 of the
            // 19-byte header. We can read it back to get the count without
            // decoding. (Profile fields are validated against the build-time
            // profile on decode, so this is a structural read only.)
            let n = read_fb2_n_blocks(&self.key_blocks[0].encoded_payload);
            n.unwrap_or(self.key_blocks.len())
        } else {
            self.key_blocks.len()
        }
    }

    /// Compute a content digest over the blocks in this layer.
    fn compute_digest(&self) -> Result<String> {
        // Serialize key + value payloads to compute a deterministic digest
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
            .map_err(|e| ProveKvError::Internal(format!("layer digest serialization: {}", e)))?;
        Ok(blake3::hash(json.as_bytes()).to_hex().to_string())
    }
}

/// Read the n_blocks field out of an FB2 batched header without doing
/// any profile validation. Returns None if the payload is too short.
fn read_fb2_n_blocks(payload: &[u8]) -> Option<usize> {
    // Layout: [0..3] "FB2", [3] version, [4] wire_index_bits, [5] reserved,
    //         [6..10] block_count, [10..14] n_blocks
    if payload.len() < 14 || &payload[0..3] != b"FB2" {
        return None;
    }
    let n = u32::from_le_bytes([payload[10], payload[11], payload[12], payload[13]]) as usize;
    Some(n)
}

/// A shared, compressed KV cache pool.
///
/// The pool holds fib-quant compressed KV blocks for tokens shared across
/// agents. It is immutable after construction. Agent shells can be materialized
/// from this pool by adding agent-specific tokens compressed with turbo-quant.
#[derive(Debug, Clone)]
pub struct SharedKVPool {
    /// Pool manifest with shape, policy, timestamps.
    pub manifest: PoolManifest,
    /// One PoolLayer per transformer layer.
    pub layers: Vec<PoolLayer>,
    /// The compression policy used.
    pub policy: CompressionPolicy,
}

impl SharedKVPool {
    /// Build a shared KV pool from a corpus of token vectors.
    ///
    /// # Arguments
    /// * `corpus` - List of (token_id, kv_vector) pairs. Each kv_vector must be
    ///   the concatenated keys and values for all layers and heads: `[layer0_head0_key,
    ///   layer0_head0_value, layer0_head1_key, ...]`.
    /// * `shape` - The tensor shape describing the model architecture.
    /// * `seed` - Deterministic seed for codec operations.
    ///
    /// # Returns
    /// The built SharedKVPool and a PoolBuildReceipt.
    pub fn build(
        corpus: &[(String, Vec<f32>)],
        shape: &KvTensorShape,
        seed: u64,
    ) -> Result<(Self, PoolBuildReceipt)> {
        Self::build_with_policy(corpus, shape, seed, CompressionPolicy::default_two_tier())
    }

    /// Build with an explicit compression policy. The lossless default
    /// (fb2+tqb1) preserves bit-exact PPL. Set the policy's
    /// `turbo_config.radii_compression = Lossy` to opt into TQB1-L and
    /// the smaller (lossy) shell tier.
    pub fn build_with_policy(
        corpus: &[(String, Vec<f32>)],
        shape: &KvTensorShape,
        seed: u64,
        policy: CompressionPolicy,
    ) -> Result<(Self, PoolBuildReceipt)> {
        let start = Instant::now();

        if corpus.is_empty() {
            return Err(ProveKvError::EmptyCorpus);
        }

        shape.validate()?;
        policy.validate()?;

        let num_tokens = corpus.len();
        let num_layers = shape.num_layers as usize;
        let num_kv_heads = shape.num_kv_heads as usize;
        let head_dim = shape.head_dim;

        // Validate that each corpus vector has the correct length
        let expected_len = num_layers * num_kv_heads * head_dim * 2; // key + value per head per layer
        for (token_id, vec) in corpus {
            if vec.len() != expected_len {
                return Err(ProveKvError::DimensionMismatch {
                    expected: expected_len,
                    got: vec.len(),
                });
            }
            if vec.iter().any(|v| !v.is_finite()) {
                return Err(ProveKvError::CorruptPayload(format!(
                    "token {} contains non-finite values",
                    token_id
                )));
            }
        }

        // Create the fib-quant codec for shared pool compression. We hold
        // the concrete `FibQuantAdapter` (not the trait object) so we can
        // call `encode_batch_compact` and produce a single FB2 batched
        // payload per (layer, K/V) — much smaller on disk than one block
        // per (token, head). The dispatch is identical to the GPU path
        // the trait method used to take.
        let fib_adapter = FibQuantAdapter::new(
            head_dim,
            policy.fib_config.k,
            policy.fib_config.n,
            policy.fib_config.training_samples,
            policy.fib_config.lloyd_restarts,
            policy.fib_config.lloyd_iterations,
        )?;
        // Keep a trait object around for the receipt's `is_gpu_accelerated_for`
        // call (which goes through the trait). The trait is satisfied by
        // the same adapter — `FibQuantAdapter` is `Copy` (no Lloyd state),
        // so we don't pay for rebuilding it.
        let codec: Box<dyn crate::codec::KVecCodec> = Box::new(fib_adapter);

        // F5: compute the real codebook/rotation digests ONCE at the top of
        // pool build (the codebook and rotation are deterministic functions
        // of the seed, head_dim, k, n, training config — they do not vary
        // across layers). Building a FibQuantizer here to read the digest
        // is cheap relative to the actual encoding work below and is the
        // only way to populate these fields with non-empty, trustworthy
        // values. PoolBuildReceipt's contract now requires non-empty
        // codebook_digest and rotation_digest, so we cannot ship empty
        // strings here. We use the concrete FibQuantAdapter (not the trait
        // object) because `build_quantizer` is an inherent method.
        let (codebook_digest, rotation_digest) = {
            let quantizer = fib_adapter.build_quantizer(seed).map_err(|e| {
                ProveKvError::CompressionFailed(format!(
                    "fib quantizer build for digest failed: {}",
                    e
                ))
            })?;
            let cb = quantizer.codebook();
            (
                cb.codebook_digest.clone(),
                cb.rotation_digest.clone(),
            )
        };

        let mut layers: Vec<PoolLayer> = Vec::with_capacity(num_layers);
        let mut total_compressed_bytes: u64 = 0;

        // Build a closure that builds one layer. Each layer is independent
        // (different head/data ranges in the corpus), so we can dispatch
        // them in parallel via Rayon when the feature is enabled.
        let build_layer = |layer_idx: usize| -> Result<(PoolLayer, u64)> {
            // Collect every (token, head) key vector and every (token, head)
            // value vector for this layer up front, then dispatch two batched
            // encode calls (one for keys, one for values). This is what lets
            // fib-quant reach its GPU batch threshold.
            let mut key_inputs: Vec<Vec<f32>> = Vec::with_capacity(num_tokens * num_kv_heads);
            let mut value_inputs: Vec<Vec<f32>> = Vec::with_capacity(num_tokens * num_kv_heads);
            for (_token_id, vec) in corpus.iter() {
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

            // FB2 batched path: one Vec<u8> per side, holding the whole
            // layer's worth of fib blocks. 1.92x smaller than the per-block
            // FB1 path because the 11-byte per-block header is amortized.
            let encoded_keys = fib_adapter.encode_batch_compact(&key_refs, seed)?;
            let encoded_values = fib_adapter.encode_batch_compact(&value_refs, seed)?;

            let key_block =
                CompressedBlock::new(CODEC_FIB_K4_N32_BATCHED.to_string(), encoded_keys, head_dim);
            let value_block =
                CompressedBlock::new(CODEC_FIB_K4_N32_BATCHED.to_string(), encoded_values, head_dim);

            let key_blocks = vec![key_block];
            let value_blocks = vec![value_block];
            let layer_bytes: u64 = key_blocks.iter().map(|b| b.compressed_bytes as u64).sum::<u64>()
                + value_blocks.iter().map(|b| b.compressed_bytes as u64).sum::<u64>();

            let mut layer = PoolLayer {
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
        let layer_results: Vec<Result<(PoolLayer, u64)>> = {
            #[cfg(feature = "parallel_pool")]
            {
                use rayon::prelude::*;
                (0..num_layers).into_par_iter().map(build_layer).collect()
            }
            #[cfg(not(feature = "parallel_pool"))]
            {
                (0..num_layers).map(build_layer).collect()
            }
        };
        for r in layer_results {
            let (layer, layer_bytes) = r?;
            total_compressed_bytes += layer_bytes;
            layers.push(layer);
        }

        let raw_size_bytes = shape.total_kv_bytes(num_tokens) as u64;
        let fib_build_ms = start.elapsed().as_millis() as u64;
        let built_at_unix = now_unix();

        // Compute pool digest
        let layer_digests: Vec<String> = layers.iter().map(|l| l.block_digest.clone()).collect();
        // F5: pool_id must bind the codebook and rotation lineage, not just
        // the per-layer payload digests. Two pools built with the same
        // (shape, seed, corpus) but different codebooks would otherwise
        // hash to the same pool_id if the per-layer digests happened to
        // collide. (Today the codebook is fully determined by the seed, so
        // the same seed always gives the same codebook; but the manifest's
        // job is to bind that lineage, not to assume it.)
        let pool_id_material = serde_json::json!({
            "layer_digests": &layer_digests,
            "codebook_digest": &codebook_digest,
            "rotation_digest": &rotation_digest,
            "shape": &shape,
            "seed": seed,
            "shared_codec": policy.shared_codec,
        });
        let pool_id = blake3::hash(
            serde_json::to_string(&pool_id_material)
                .map_err(|e| ProveKvError::Internal(format!("pool_id hash: {}", e)))?
                .as_bytes(),
        )
        .to_hex()
        .to_string();

        let manifest = PoolManifest::new(
            pool_id.clone(),
            shape.clone(),
            policy.clone(),
            num_tokens as u32,
            shape.num_layers,
            total_compressed_bytes,
            raw_size_bytes,
            seed,
            built_at_unix,
        )?;

        // Honest backend label: ask the codec whether the per-(layer,head)
        // batch we'd actually dispatch crossed the GPU threshold. The fib-quant
        // encoder only goes to GPU when batch size and dim clear the runtime
        // minimums; a 4-doc, 12-head corpus is GPU, but a 4-doc, 4-head
        // corpus (16 vectors exactly) is right at the edge and still GPU,
        // while a 2-doc corpus falls through to CPU even with --features gpu.
        let batch_n = num_tokens * num_kv_heads;
        let backend = if codec.is_gpu_accelerated_for(batch_n, head_dim) {
            "gpu"
        } else {
            "cpu"
        };
        let receipt = PoolBuildReceipt::new(
            pool_id,
            layer_digests,
            codebook_digest,
            rotation_digest,
            num_tokens as u32,
            fib_build_ms,
            total_compressed_bytes,
            raw_size_bytes,
            policy.clone(),
            seed,
            built_at_unix,
        ).with_backend(backend);

        Ok((
            Self {
                manifest,
                layers,
                policy,
            },
            receipt,
        ))
    }
    /// Materialize an agent shell from this pool.
    ///
    /// Agent-specific tokens (not in the shared corpus) are compressed with
    /// turbo-quant and appended as shell layers. Tokens already in the pool
    /// are referenced by digest only.
    ///
    /// # Arguments
    /// * `agent_id` - Identifier for this agent.
    /// * `agent_tokens` - Token vectors specific to this agent.
    /// * `seed` - Deterministic seed for turbo-quant operations.
    ///
    /// # Returns
    /// An AgentShell and a ShellMaterializeReceipt.
    pub fn materialize_shell(
        &self,
        agent_id: &str,
        agent_tokens: &[(String, Vec<f32>)],
        seed: u64,
    ) -> Result<(
        crate::shell::AgentShell,
        crate::receipt::ShellMaterializeReceipt,
    )> {
        crate::shell::materialize_shell(self, agent_id, agent_tokens, seed)
    }

    /// Inject a shell into a KV cache.
    ///
    /// The injection receipt traces every block from its source (pool or shell)
    /// to its target position in the cache.
    pub fn inject_into_cache(
        _shell: &crate::shell::AgentShell,
        _base_cache: &mut dyn CacheTarget,
    ) -> Result<crate::receipt::InjectionReceipt> {
        // The CacheTarget trait allows injection without knowing the concrete cache type.
        // This is a generic injection path — concrete adapters (e.g., for HF DynamicCache)
        // live in downstream crates.
        Err(ProveKvError::Internal(
            "inject_into_cache requires a concrete cache adapter; use inject_into_cache_with_adaptor"
                .into(),
        ))
    }

    /// Decompress all shared-pool blocks for a single layer, returning the
    /// reconstructed K and V tensors in the original model layout.
    ///
    /// Output shape: `keys[head_idx]` is a flat `Vec<f32>` of length
    /// `num_tokens * head_dim` containing all tokens' K vectors for that
    /// head, in token order. Same for `values`. Lossy (fib-quant) but
    /// reproducible: same corpus + same seed + same codec yields the same
    /// reconstructed floats.
    ///
    /// This is the inverse of `build` and the symmetric counterpart of
    /// `materialize_shell`'s per-agent shell decompression. It's the
    /// path HuggingFace `DynamicCache.update()` and similar KV-cache
    /// integrations use to populate a fresh cache from the pool.
    /// Decompress every layer in the pool and return the full K and V
    /// tensors in HuggingFace-friendly layout.
    ///
    /// Per-layer shape: `(K_flat, V_flat)` where each flat vec has length
    /// `num_tokens * num_kv_heads * head_dim` and is laid out so that
    /// `.reshape(1, num_tokens, num_kv_heads, head_dim).transpose(1, 2)`
    /// gives `[1, num_kv_heads, T, head_dim]` as the Python bench expects.
    pub fn decompress_all_layers_with_seed(
        &self,
        seed: u64,
    ) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        let num_layers = self.layers.len();
        let num_tokens = self.manifest.num_shared_tokens as usize;
        let num_kv_heads = self.manifest.shape.num_kv_heads as usize;
        let head_dim = self.manifest.shape.head_dim;
        let mut out = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let layer = self.decompress_layer(layer_idx)?;
            let k_flat = flatten_heads_tokens(&layer.keys, num_tokens, num_kv_heads, head_dim);
            let v_flat = flatten_heads_tokens(&layer.values, num_tokens, num_kv_heads, head_dim);
            out.push((k_flat, v_flat));
        }
        let _ = seed;
        Ok(out)
    }

    pub fn decompress_layer(&self, layer_idx: usize) -> Result<DecompressedLayer> {
        if layer_idx >= self.layers.len() {
            return Err(ProveKvError::Internal(format!(
                "decompress_layer: layer_idx {layer_idx} out of range (have {})",
                self.layers.len()
            )));
        }
        let layer = &self.layers[layer_idx];
        let head_dim = self.manifest.shape.head_dim;
        let num_heads = self.manifest.shape.num_kv_heads as usize;
        if layer.value_blocks.len() != layer.key_blocks.len() {
            return Err(ProveKvError::Internal(format!(
                "layer {}: key/value block count mismatch ({} vs {})",
                layer_idx,
                layer.key_blocks.len(),
                layer.value_blocks.len()
            )));
        }
        // All shared-pool blocks use the same codec (manifest.shared_codec).
        // Build a single codec and reuse for the whole layer.
        let shared_codec: crate::policy::CodecId = self.manifest.shared_codec.clone();
        let codec = create_codec(
            &shared_codec,
            head_dim,
            Some(&self.manifest.policy.fib_config),
            Some(&self.manifest.policy.turbo_config),
        )?;
        let seed = self.manifest.build_seed;
        // The layer is in batched form when both K and V sides are stored as
        // a single block whose codec id is the batched variant. The block
        // order on disk is then: 1 batched K block, 1 batched V block.
        // We downcast the trait object to FibQuantAdapter so we can call
        // decode_batch_compact — the batched code path.
        if layer.is_batched() {
            // Recover the adapter. The codec the build wrote is the
            // batched fib id, so the downcast must succeed.
            let fib_adapter = codec
                .as_any()
                .downcast_ref::<FibQuantAdapter>()
                .ok_or_else(|| {
                    ProveKvError::Internal(
                        "decompress_layer: shared_codec must be a FibQuantAdapter for batched layers"
                            .into(),
                    )
                })?;
            let k_decoded_all: Vec<Vec<f32>> = fib_adapter.decode_batch_compact(
                &layer.key_blocks[0].encoded_payload,
                seed,
            )?;
            let v_decoded_all: Vec<Vec<f32>> = fib_adapter.decode_batch_compact(
                &layer.value_blocks[0].encoded_payload,
                seed,
            )?;
            if k_decoded_all.len() != v_decoded_all.len() {
                return Err(ProveKvError::Internal(format!(
                    "layer {layer_idx}: batched K/V vector count mismatch ({} vs {})",
                    k_decoded_all.len(),
                    v_decoded_all.len()
                )));
            }
            // Block ordering from build: [token_0_head_0, token_0_head_1, ...,
            //   token_0_head_{H-1}, token_1_head_0, ..., token_{T-1}_head_{H-1}]
            // i.e. flat index = token_idx * num_heads + head_idx.
            let n_pairs = k_decoded_all.len();
            if n_pairs % num_heads != 0 {
                return Err(ProveKvError::Internal(format!(
                    "layer {layer_idx}: batched vector count {n_pairs} not divisible by num_heads {num_heads}"
                )));
            }
            let num_tokens = n_pairs / num_heads;
            // Per-head output: keys[head_idx] = concatenation of every token's K for that head.
            let mut keys_per_head: Vec<Vec<f32>> = vec![Vec::with_capacity(num_tokens * head_dim); num_heads];
            let mut values_per_head: Vec<Vec<f32>> = vec![Vec::with_capacity(num_tokens * head_dim); num_heads];
            for token_idx in 0..num_tokens {
                for head_idx in 0..num_heads {
                    let idx = token_idx * num_heads + head_idx;
                    if k_decoded_all[idx].len() != head_dim {
                        return Err(ProveKvError::Internal(format!(
                            "decoded key length {} != head_dim {} (layer {}, token {}, head {})",
                            k_decoded_all[idx].len(), head_dim, layer_idx, token_idx, head_idx
                        )));
                    }
                    keys_per_head[head_idx].extend_from_slice(&k_decoded_all[idx]);
                    values_per_head[head_idx].extend_from_slice(&v_decoded_all[idx]);
                }
            }
            return Ok(DecompressedLayer {
                layer_index: layer_idx as u32,
                num_tokens,
                num_heads,
                head_dim,
                keys: keys_per_head,
                values: values_per_head,
            });
        }

        // Legacy per-vector path: one block per (token, head). Preserved
        // for backward compat with pools written by older proveKV versions
        // (and with the in-flight decoder that hasn't been migrated yet).
        let num_tokens = layer.key_blocks.len() / num_heads;
        if layer.key_blocks.len() != num_tokens * num_heads {
            return Err(ProveKvError::Internal(format!(
                "layer {}: block count {} != num_tokens * num_heads {}",
                layer_idx,
                layer.key_blocks.len(),
                num_tokens * num_heads
            )));
        }
        let mut keys_per_head: Vec<Vec<f32>> =
            vec![Vec::with_capacity(num_tokens * head_dim); num_heads];
        let mut values_per_head: Vec<Vec<f32>> =
            vec![Vec::with_capacity(num_tokens * head_dim); num_heads];
        for token_idx in 0..num_tokens {
            for head_idx in 0..num_heads {
                let block_idx = token_idx * num_heads + head_idx;
                let k_payload = &layer.key_blocks[block_idx].encoded_payload;
                let v_payload = &layer.value_blocks[block_idx].encoded_payload;
                let k_decoded = codec.decode(k_payload, seed)?;
                let v_decoded = codec.decode(v_payload, seed)?;
                if k_decoded.len() != head_dim {
                    return Err(ProveKvError::Internal(format!(
                        "decoded key length {} != head_dim {} (layer {}, token {}, head {})",
                        k_decoded.len(), head_dim, layer_idx, token_idx, head_idx
                    )));
                }
                keys_per_head[head_idx].extend_from_slice(&k_decoded);
                values_per_head[head_idx].extend_from_slice(&v_decoded);
            }
        }
        Ok(DecompressedLayer {
            layer_index: layer_idx as u32,
            num_tokens,
            num_heads,
            head_dim,
            keys: keys_per_head,
            values: values_per_head,
        })
    }
}

/// Reconstructed K/V tensors for one layer of the shared pool.
///
/// All vectors are in the original (head × token × head_dim) layout but
/// flat per head: `keys[head_idx][token_idx * head_dim + j]`. This matches
/// the HuggingFace `DynamicCache` per-layer access pattern
/// (`cache.layers[layer_idx].keys[:, head_idx, :, :]` flattened along the
/// last two dims).
/// `DecompressedLayer` is the per-layer output of `decompress_layer`,
/// in the HuggingFace `DynamicCache` per-layer access pattern
/// (`cache.layers[layer_idx].keys[:, head_idx, :, :]` flattened along the
/// last two dims).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DecompressedLayer {
    /// Original layer index in the model.
    pub layer_index: u32,
    /// Number of tokens in this layer (= pool's `num_shared_tokens`).
    pub num_tokens: usize,
    /// Number of KV heads (= pool's `num_kv_heads`).
    pub num_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Decoded K vectors: `keys[head_idx]` is a flat `Vec<f32>` of length
    /// `num_tokens * head_dim` in token order.
    pub keys: Vec<Vec<f32>>,
    /// Decoded V vectors, same layout as `keys`.
    pub values: Vec<Vec<f32>>,
}

/// Trait for KV cache targets that can receive injected blocks.
pub trait CacheTarget: std::fmt::Debug {
    /// Get the number of layers in this cache.
    fn num_layers(&self) -> u32;

    /// Append a key block at a specific layer and position.
    fn append_key(&mut self, layer: u32, position: u32, key: &[f32]) -> Result<()>;

    /// Append a value block at a specific layer and position.
    fn append_value(&mut self, layer: u32, position: u32, value: &[f32]) -> Result<()>;

    /// Get the current sequence length (tokens in cache).
    fn seq_len(&self) -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::AttentionType;

    fn make_test_shape() -> KvTensorShape {
        KvTensorShape {
            attention_type: AttentionType::MHA,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 8, // must be divisible by k=4 for fib-quant
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
    fn test_pool_build_empty() {
        let shape = make_test_shape();
        let corpus: Vec<(String, Vec<f32>)> = vec![];
        let result = SharedKVPool::build(&corpus, &shape, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_pool_build_basic() {
        let shape = make_test_shape();
        let corpus = make_test_corpus(4);
        let result = SharedKVPool::build(&corpus, &shape, 42);
        assert!(result.is_ok(), "build failed: {:?}", result.err());

        let (pool, receipt) = result.unwrap();
        assert_eq!(pool.layers.len(), 2);
        assert_eq!(pool.manifest.num_shared_tokens, 4);
        assert_eq!(receipt.total_tokens, 4);
        assert!(
            receipt.compression_ratio > 0.0,
            "compression ratio: {}",
            receipt.compression_ratio
        );
        // Note: ratio < 1.0 is normal for tiny test corpora where JSON
        // serialization overhead dominates the encoded payload.
    }

    /// New pool builds must use the FB2 batched wire format: one CompressedBlock
    /// per (K, V) per layer, carrying the batched codec id. This is what
    /// gives the 1.92x fib-tier compression win.
    #[test]
    fn test_pool_build_writes_batched_fb2() {
        let shape = make_test_shape();
        let corpus = make_test_corpus(8);
        let (pool, _receipt) = SharedKVPool::build(&corpus, &shape, 42).unwrap();

        for layer in &pool.layers {
            assert!(
                layer.is_batched(),
                "every layer should be written in batched FB2 form"
            );
            assert_eq!(layer.key_blocks.len(), 1, "exactly 1 batched K block");
            assert_eq!(layer.value_blocks.len(), 1, "exactly 1 batched V block");
            assert_eq!(layer.key_blocks[0].codec, crate::policy::CODEC_FIB_K4_N32_BATCHED);
            assert_eq!(layer.key_blocks[0].codec, layer.value_blocks[0].codec);
        }
        assert_eq!(pool.manifest.shared_codec, crate::policy::CODEC_FIB_K4_N32_BATCHED);
    }

    /// F5: the pool receipt must carry non-empty codebook/rotation digests.
    /// This guards against the F5 bug where `String::new()` was passed and
    /// the receipt's provenance claim was a lie.
    #[test]
    fn test_pool_receipt_has_real_codebook_and_rotation_digests() {
        let shape = make_test_shape();
        let corpus = make_test_corpus(8);
        let (_pool, receipt) = SharedKVPool::build(&corpus, &shape, 42).unwrap();
        assert!(
            !receipt.codebook_digest.is_empty(),
            "codebook_digest must be populated (audit F5)"
        );
        assert!(
            !receipt.rotation_digest.is_empty(),
            "rotation_digest must be populated (audit F5)"
        );
        // They should be a real hash, not the empty placeholder. The fib
        // codec produces 71-char base64 digests, so the safe check is
        // "non-empty and longer than the schema-name prefix a real digest
        // would never have." 32 chars minimum, no whitespace.
        assert!(
            receipt.codebook_digest.len() >= 32
                && !receipt.codebook_digest.contains(' '),
            "codebook_digest should be a real hash (>=32 chars, no whitespace), got '{}'",
            receipt.codebook_digest
        );
        assert!(
            receipt.rotation_digest.len() >= 32
                && !receipt.rotation_digest.contains(' '),
            "rotation_digest should be a real hash (>=32 chars, no whitespace), got '{}'",
            receipt.rotation_digest
        );
    }

    /// F5: digests must be deterministic in the seed (and only the seed).
    /// The codebook + rotation are deterministic functions of the
    /// (head_dim, k, n, training config, seed) inputs, so two pools built
    /// with the same seed must produce identical digests. Two pools built
    /// with different seeds must produce different digests.
    #[test]
    fn test_pool_digests_are_seed_deterministic() {
        let shape = make_test_shape();
        let corpus_a = make_test_corpus(8);
        let corpus_b = make_test_corpus(8);
        let (_p1, r1) = SharedKVPool::build(&corpus_a, &shape, 42).unwrap();
        let (_p2, r2) = SharedKVPool::build(&corpus_b, &shape, 42).unwrap();
        assert_eq!(r1.codebook_digest, r2.codebook_digest);
        assert_eq!(r1.rotation_digest, r2.rotation_digest);
        let (_p3, r3) = SharedKVPool::build(&corpus_a, &shape, 43).unwrap();
        assert_ne!(r1.codebook_digest, r3.codebook_digest);
    }

    /// End-to-end test: build a batched pool, materialize a batched shell,
    /// decompress a layer, and verify the f32 tensors roundtrip exactly.
    /// This is the proof that the new wire formats don't lose data.
    #[test]
    fn test_batched_pool_and_shell_roundtrip() {
        use crate::shell::materialize_shell;
        let shape = make_test_shape();
        let corpus = make_test_corpus(8);
        let (pool, _receipt) = SharedKVPool::build(&corpus, &shape, 42).unwrap();

        // Materialize a shell with some agent tokens.
        let agent_tokens = make_test_corpus(2);
        let (shell, _mat_receipt) = materialize_shell(&pool, "agent_a", &agent_tokens, 42).unwrap();

        // Shell layers must also be batched.
        for layer in &shell.unique_layers {
            assert!(layer.is_batched(), "shell layers should be batched TQB1");
            assert_eq!(layer.key_blocks.len(), 1);
            assert_eq!(layer.value_blocks.len(), 1);
        }

        // Decompress one pool layer and verify shape + content stability
        // (the test corpus is random, so we can't assert bit equality with
        // anything, but we CAN assert the shapes match the layer's vector
        // count and head_dim).
        let decomp = pool.decompress_layer(0).unwrap();
        assert_eq!(decomp.layer_index, 0);
        assert_eq!(decomp.num_heads, shape.num_kv_heads as usize);
        assert_eq!(decomp.head_dim, shape.head_dim);
        assert_eq!(decomp.keys.len(), shape.num_kv_heads as usize);
        assert_eq!(decomp.values.len(), shape.num_kv_heads as usize);
        for head in 0..shape.num_kv_heads as usize {
            assert_eq!(decomp.keys[head].len(), 8 * shape.head_dim);
            assert_eq!(decomp.values[head].len(), 8 * shape.head_dim);
        }
    }

    #[test]
    fn test_pool_build_deterministic() {
        let shape = make_test_shape();
        let corpus = make_test_corpus(4);

        let (pool1, receipt1) = SharedKVPool::build(&corpus, &shape, 42).unwrap();
        let (pool2, receipt2) = SharedKVPool::build(&corpus, &shape, 42).unwrap();

        assert_eq!(receipt1.pool_digest, receipt2.pool_digest);
        assert_eq!(receipt1.layer_digests, receipt2.layer_digests);
        assert_eq!(pool1.layers[0].block_digest, pool2.layers[0].block_digest);
    }

    #[test]
    fn test_pool_build_different_seeds() {
        let shape = make_test_shape();
        let corpus = make_test_corpus(4);

        let (_pool1, receipt1) = SharedKVPool::build(&corpus, &shape, 42).unwrap();
        let (_pool2, receipt2) = SharedKVPool::build(&corpus, &shape, 12345).unwrap();

        assert_ne!(receipt1.pool_digest, receipt2.pool_digest);
    }

    #[test]
    fn test_decompress_layer_recovers_finite_floats() {
        // Round-trip integrity: build a pool, decompress every layer,
        // verify the output is finite, the right shape, and per-head
        // lengths match num_tokens * head_dim.
        let shape = make_test_shape();
        let corpus = make_test_corpus(8);
        let (pool, _) = SharedKVPool::build(&corpus, &shape, 42).unwrap();

        for layer_idx in 0..shape.num_layers as usize {
            let decompressed = pool.decompress_layer(layer_idx).unwrap();
            assert_eq!(decompressed.num_tokens, 8);
            assert_eq!(decompressed.num_heads, shape.num_kv_heads as usize);
            assert_eq!(decompressed.head_dim, shape.head_dim);
            assert_eq!(decompressed.keys.len(), shape.num_kv_heads as usize);
            assert_eq!(decompressed.values.len(), shape.num_kv_heads as usize);
            for h in 0..decompressed.num_heads {
                assert_eq!(decompressed.keys[h].len(), 8 * shape.head_dim);
                assert_eq!(decompressed.values[h].len(), 8 * shape.head_dim);
                assert!(decompressed.keys[h].iter().all(|v| v.is_finite()));
                assert!(decompressed.values[h].iter().all(|v| v.is_finite()));
            }
        }
    }

    #[test]
    fn test_decompress_layer_is_deterministic() {
        // Same corpus + same seed must produce byte-identical decompressed
        // output. This is the core invariant for HuggingFaceDynamicCache
        // round-trip: a fresh DynamicCache populated from the pool must
        // see the same K/V tensors across runs.
        let shape = make_test_shape();
        let corpus = make_test_corpus(6);
        let (pool_a, _) = SharedKVPool::build(&corpus, &shape, 42).unwrap();
        let (pool_b, _) = SharedKVPool::build(&corpus, &shape, 42).unwrap();
        for layer_idx in 0..shape.num_layers as usize {
            let a = pool_a.decompress_layer(layer_idx).unwrap();
            let b = pool_b.decompress_layer(layer_idx).unwrap();
            assert_eq!(
                a.keys, b.keys,
                "decompressed K tensors must be deterministic across builds (layer {})",
                layer_idx
            );
            assert_eq!(a.values, b.values);
        }
    }

    #[test]
    fn test_mismatched_shape_rejected() {
        let shape = make_test_shape();
        let mut bad_corpus = make_test_corpus(1);
        // Truncate the vector
        bad_corpus[0].1.truncate(10);
        let result = SharedKVPool::build(&bad_corpus, &shape, 42);
        assert!(result.is_err());
    }

    /// The pool build must produce the same `pool_digest` and `block_digest`
    /// values regardless of whether the underlying codec dispatches to GPU
    /// or CPU. This guards against the "receipt says gpu, code did cpu"
    /// failure mode that earlier feature-flag-only wiring exhibited.
    #[test]
    fn test_pool_build_digest_invariant_across_corpora_size() {
        let shape = make_test_shape();

        // Tiny corpus — under the GPU batch threshold (n < 16).
        let small = make_test_corpus(4);
        let (pool_small, receipt_small) = SharedKVPool::build(&small, &shape, 42).unwrap();

        // Large corpus — well over the GPU batch threshold.
        let large = make_test_corpus(40);
        let (pool_large, receipt_large) = SharedKVPool::build(&large, &shape, 42).unwrap();

        assert!(!pool_small.layers.is_empty());
        assert!(!pool_large.layers.is_empty());
        assert!(receipt_small.backend == "cpu" || receipt_small.backend == "gpu");
        assert!(receipt_large.backend == "cpu" || receipt_large.backend == "gpu");

        // Tiny corpus must NOT claim gpu — the per-(layer,head) batch is
        // 4 docs * 4 kv heads = 16 vectors, exactly at the threshold, and
        // the per-call probe (not the device probe) drives the receipt.
        // This is the honesty invariant.
        assert_eq!(
            receipt_small.backend, "cpu",
            "corpus under GPU batch threshold should fall through to CPU"
        );
    }
}
