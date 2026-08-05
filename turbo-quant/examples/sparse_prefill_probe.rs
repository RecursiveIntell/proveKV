use turbo_quant::{
    compare_cache_sparse_prefill, compare_cache_sparse_prefill_with_options, AttentionScale,
    AttentionScoreOptions, KvCacheCompressor, KvQuantPolicy, KvRuntimeConfig, SparsePrefillConfig,
    SparsePrefillPattern,
};

fn main() -> turbo_quant::Result<()> {
    let dim = 64;
    let tokens = 96;
    let mut cache = KvCacheCompressor::new_runtime(KvRuntimeConfig {
        head_dim: dim,
        key_policy: KvQuantPolicy::quantized(8, 16),
        value_policy: KvQuantPolicy::Exact,
        seed: 42,
        keep_exact_shadow: true,
    })?;

    for token in 0..tokens {
        let key = (0..dim)
            .map(|index| {
                let phase = (token * dim + index) as f32;
                (phase * 0.017).sin()
                    + if token == 0 || token > tokens - 8 {
                        0.25
                    } else {
                        0.0
                    }
            })
            .collect::<Vec<_>>();
        let value = (0..dim)
            .map(|index| ((token * dim + index) as f32 * 0.019).cos())
            .collect::<Vec<_>>();
        cache.compress_token(&key, &value)?;
    }

    let query = (0..dim)
        .map(|index| (index as f32 * 0.023).sin() + 0.25)
        .collect::<Vec<_>>();
    let config = SparsePrefillConfig {
        anchor_count: 4,
        recent_window: 16,
        vertical_stride: 12,
        block_size: 16,
        max_blocks: 3,
        max_tokens: 32,
        top_k: 8,
        ..SparsePrefillConfig::default()
    };
    let score_options = AttentionScoreOptions {
        scale: AttentionScale::ByHeadDim,
    };

    let exact_receipts = [
        (
            "a_shape",
            compare_cache_sparse_prefill(
                &cache,
                &query,
                SparsePrefillPattern::AShape,
                config.clone(),
            )?,
        ),
        (
            "vertical_slash",
            compare_cache_sparse_prefill(
                &cache,
                &query,
                SparsePrefillPattern::VerticalSlash,
                config.clone(),
            )?,
        ),
        (
            "block_sparse",
            compare_cache_sparse_prefill(
                &cache,
                &query,
                SparsePrefillPattern::BlockSparse,
                config.clone(),
            )?,
        ),
        (
            "hybrid_anchor_recent_blocks",
            compare_cache_sparse_prefill(
                &cache,
                &query,
                SparsePrefillPattern::HybridAnchorRecentBlocks,
                config.clone(),
            )?,
        ),
    ];
    let compressed_block_sparse = compare_cache_sparse_prefill_with_options(
        &cache,
        &query,
        score_options,
        SparsePrefillPattern::BlockSparse,
        config,
    )?;

    println!(
        "{}",
        serde_json::json!({
            "schema": "KvSparsePrefillProbeV1",
            "tokens": cache.len(),
            "head_dim": dim,
            "compressed_bytes": cache.compressed_bytes(),
            "uncompressed_bytes": cache.uncompressed_bytes(),
            "exact_score_receipts": exact_receipts,
            "compressed_block_sparse_receipt": compressed_block_sparse,
            "warnings": [
                "probe compares sparse score selection only; it is not a fused-kernel benchmark",
                "softmax mass and top-k recall are local score-vector checks, not PPL validation"
            ]
        })
    );
    Ok(())
}
