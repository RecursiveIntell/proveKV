use std::{env, fs};

use turbo_quant::{
    benchmark_sparse_prefill_traces, SparsePrefillConfig, SparsePrefillGateConfig,
    SparsePrefillPattern, SparsePrefillTraceBenchmarkV1, SparsePrefillTraceV1,
};

fn main() -> turbo_quant::Result<()> {
    let traces = match env::args().nth(1) {
        Some(path) => read_traces(&path)?,
        None => synthetic_traces(),
    };
    let patterns = [
        SparsePrefillPattern::AShape,
        SparsePrefillPattern::VerticalSlash,
        SparsePrefillPattern::BlockSparse,
        SparsePrefillPattern::HybridAnchorRecentBlocks,
        SparsePrefillPattern::AdaptiveMass,
    ];
    let gate = SparsePrefillGateConfig::default();
    let receipts = candidate_configs()
        .into_iter()
        .map(|config| benchmark_sparse_prefill_traces(&traces, &patterns, config, gate))
        .collect::<turbo_quant::Result<Vec<_>>>()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "SparsePrefillTraceBenchRunV1",
            "trace_count": traces.len(),
            "receipts": receipts,
            "best_kernel_candidates": best_kernel_candidates(&receipts),
            "warnings": [
                "candidate configs are a small first-pass sweep, not an exhaustive autotuner",
                "use real layer/head traces before committing to CUDA work"
            ]
        }))
        .unwrap()
    );
    Ok(())
}

fn read_traces(path: &str) -> turbo_quant::Result<Vec<SparsePrefillTraceV1>> {
    let payload = fs::read_to_string(path).map_err(|error| {
        turbo_quant::TurboQuantError::ProfileMismatch {
            reason: format!("failed to read sparse prefill trace file {path}: {error}"),
        }
    })?;
    serde_json::from_str(&payload).map_err(|error| turbo_quant::TurboQuantError::ProfileMismatch {
        reason: format!("failed to parse sparse prefill trace JSON {path}: {error}"),
    })
}

fn synthetic_traces() -> Vec<SparsePrefillTraceV1> {
    vec![
        SparsePrefillTraceV1 {
            trace_id: "recent-and-anchor-heavy".into(),
            layer: Some(0),
            head: Some(0),
            scores: make_scores(&[(0, 7.0), (1, 6.0), (92, 8.0), (93, 9.0)]),
        },
        SparsePrefillTraceV1 {
            trace_id: "middle-block-plus-recent".into(),
            layer: Some(1),
            head: Some(3),
            scores: make_scores(&[(0, 6.0), (46, 7.0), (47, 8.0), (93, 9.0)]),
        },
        SparsePrefillTraceV1 {
            trace_id: "vertical-column-plus-tail".into(),
            layer: Some(2),
            head: Some(5),
            scores: make_scores(&[(12, 7.0), (36, 6.5), (72, 7.5), (93, 8.0)]),
        },
    ]
}

fn candidate_configs() -> Vec<SparsePrefillConfig> {
    vec![
        SparsePrefillConfig {
            anchor_count: 4,
            recent_window: 16,
            vertical_stride: 12,
            block_size: 16,
            max_blocks: 3,
            max_tokens: 32,
            top_k: 8,
            ..SparsePrefillConfig::default()
        },
        SparsePrefillConfig {
            anchor_count: 4,
            recent_window: 16,
            vertical_stride: 12,
            block_size: 16,
            max_blocks: 4,
            max_tokens: 48,
            top_k: 8,
            ..SparsePrefillConfig::default()
        },
        SparsePrefillConfig {
            anchor_count: 4,
            recent_window: 24,
            vertical_stride: 12,
            block_size: 16,
            max_blocks: 4,
            max_tokens: 48,
            top_k: 8,
            ..SparsePrefillConfig::default()
        },
        SparsePrefillConfig {
            anchor_count: 4,
            recent_window: 16,
            vertical_stride: 12,
            block_size: 8,
            max_blocks: 5,
            max_tokens: 48,
            top_k: 8,
            ..SparsePrefillConfig::default()
        },
        SparsePrefillConfig {
            max_tokens: 64,
            top_k: 8,
            adaptive_target_mass: 0.995,
            adaptive_min_score_reads_saved_ratio: 0.50,
            adaptive_include_top_k: true,
            ..SparsePrefillConfig::default()
        },
        SparsePrefillConfig {
            max_tokens: 96,
            top_k: 8,
            adaptive_target_mass: 0.995,
            adaptive_min_score_reads_saved_ratio: 0.25,
            adaptive_include_top_k: true,
            ..SparsePrefillConfig::default()
        },
    ]
}

fn best_kernel_candidates(receipts: &[SparsePrefillTraceBenchmarkV1]) -> Vec<serde_json::Value> {
    receipts
        .iter()
        .flat_map(|receipt| {
            receipt
                .patterns
                .iter()
                .filter(|summary| summary.kernel_candidate)
                .map(|summary| {
                    serde_json::json!({
                        "pattern": summary.pattern,
                        "config": receipt.config,
                        "min_top_k_recall": summary.min_top_k_recall,
                        "min_softmax_mass_coverage": summary.min_softmax_mass_coverage,
                        "min_score_reads_saved_ratio": summary.min_score_reads_saved_ratio
                    })
                })
        })
        .collect()
}

fn make_scores(peaks: &[(usize, f32)]) -> Vec<f32> {
    let mut scores = (0..96)
        .map(|index| ((index as f32) * 0.037).sin() * 0.05)
        .collect::<Vec<_>>();
    for &(index, value) in peaks {
        scores[index] = value;
    }
    scores
}
