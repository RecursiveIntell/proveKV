//! Sparse-prefill planning receipts for KV-cache experiments.
//!
//! This module is an executable bridge between MInference-style sparse
//! attention patterns and turbo-quant/proveKV KV shadow receipts. It does not
//! implement a fused attention kernel; it creates deterministic token/block
//! selection plans and compares them against full attention logits so callers
//! can decide whether a kernel path is worth building.

use std::collections::BTreeSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    error::{Result, TurboQuantError},
    kv::{AttentionScoreOptions, KvCacheCompressor},
};

/// Sparse prefill pattern families used by MInference-style long-context
/// attention routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SparsePrefillPattern {
    /// Anchor tokens plus a recent slash/tail window.
    AShape,
    /// Periodic vertical columns plus a recent slash/tail window.
    VerticalSlash,
    /// Contiguous blocks chosen by block score energy.
    BlockSparse,
    /// Prefix anchors plus recent tokens plus high-softmax-mass blocks.
    HybridAnchorRecentBlocks,
    /// Select highest-mass tokens until an explicit softmax-mass target is met.
    AdaptiveMass,
}

/// Tunables for deterministic sparse-prefill probes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SparsePrefillConfig {
    /// Prefix anchors retained for `AShape`.
    pub anchor_count: usize,
    /// Recent tokens retained for `AShape` and `VerticalSlash`.
    pub recent_window: usize,
    /// Stride for vertical-token retention in `VerticalSlash`.
    pub vertical_stride: usize,
    /// Block width for `BlockSparse`.
    pub block_size: usize,
    /// Maximum scored blocks retained in `BlockSparse`.
    pub max_blocks: usize,
    /// Hard cap for selected tokens after pattern expansion.
    pub max_tokens: usize,
    /// Top-k logits used by comparison receipts.
    pub top_k: usize,
    /// Target softmax mass for adaptive sparse selection.
    pub adaptive_target_mass: f32,
    /// Minimum score-read savings the adaptive selector must preserve.
    pub adaptive_min_score_reads_saved_ratio: f32,
    /// Keep the full-attention top-k candidates visible in adaptive selection.
    pub adaptive_include_top_k: bool,
}

impl Default for SparsePrefillConfig {
    fn default() -> Self {
        Self {
            anchor_count: 4,
            recent_window: 32,
            vertical_stride: 16,
            block_size: 16,
            max_blocks: 4,
            max_tokens: 128,
            top_k: 8,
            adaptive_target_mass: 0.995,
            adaptive_min_score_reads_saved_ratio: 0.50,
            adaptive_include_top_k: true,
        }
    }
}

/// Half-open token block selected by a sparse prefill plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SparseBlockRange {
    pub start: usize,
    pub end: usize,
}

/// Deterministic sparse prefill plan for one attention-head score vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct KvSparsePrefillPlanV1 {
    pub schema: String,
    pub pattern: SparsePrefillPattern,
    pub seq_len: usize,
    pub selected_indices: Vec<usize>,
    pub selected_blocks: Vec<SparseBlockRange>,
    pub selected_token_count: usize,
    pub coverage_ratio: f32,
    pub config: SparsePrefillConfig,
    pub warnings: Vec<String>,
}

/// Receipt comparing a sparse prefill plan against full attention scores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct KvSparsePrefillComparisonV1 {
    pub schema: String,
    pub plan: KvSparsePrefillPlanV1,
    pub top_k: usize,
    pub full_top_indices: Vec<usize>,
    pub sparse_top_indices: Vec<usize>,
    pub selected_full_top_k_hits: usize,
    pub top_k_recall: f32,
    pub softmax_mass_coverage: f32,
    pub estimated_score_reads_saved_ratio: f32,
    pub warnings: Vec<String>,
}

/// One captured attention-score trace for sparse prefill benchmarking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SparsePrefillTraceV1 {
    pub trace_id: String,
    pub layer: Option<usize>,
    pub head: Option<usize>,
    pub scores: Vec<f32>,
}

/// Gate thresholds for deciding whether a sparse pattern is worth kernel work.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SparsePrefillGateConfig {
    pub min_softmax_mass_coverage: f32,
    pub min_top_k_recall: f32,
    pub min_score_reads_saved_ratio: f32,
}

impl Default for SparsePrefillGateConfig {
    fn default() -> Self {
        Self {
            min_softmax_mass_coverage: 0.995,
            min_top_k_recall: 0.75,
            min_score_reads_saved_ratio: 0.50,
        }
    }
}

/// Per-pattern aggregate across many attention traces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SparsePrefillPatternSummaryV1 {
    pub pattern: SparsePrefillPattern,
    pub trace_count: usize,
    pub mean_top_k_recall: f32,
    pub min_top_k_recall: f32,
    pub mean_softmax_mass_coverage: f32,
    pub min_softmax_mass_coverage: f32,
    pub mean_score_reads_saved_ratio: f32,
    pub min_score_reads_saved_ratio: f32,
    pub pass_count: usize,
    pub pass_rate: f32,
    pub kernel_candidate: bool,
}

/// Multi-trace sparse-prefill benchmark receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SparsePrefillTraceBenchmarkV1 {
    pub schema: String,
    pub trace_count: usize,
    pub patterns: Vec<SparsePrefillPatternSummaryV1>,
    pub gate: SparsePrefillGateConfig,
    pub config: SparsePrefillConfig,
    pub warnings: Vec<String>,
}

/// Build a sparse plan from a full attention score vector.
pub fn build_sparse_prefill_plan(
    scores: &[f32],
    pattern: SparsePrefillPattern,
    config: SparsePrefillConfig,
) -> Result<KvSparsePrefillPlanV1> {
    validate_scores(scores)?;
    validate_config(&config)?;

    let seq_len = scores.len();
    let mut selected = BTreeSet::new();
    let mut selected_blocks = Vec::new();

    match pattern {
        SparsePrefillPattern::AShape => {
            insert_range(&mut selected, 0, config.anchor_count.min(seq_len));
            insert_range(
                &mut selected,
                seq_len.saturating_sub(config.recent_window),
                seq_len,
            );
        }
        SparsePrefillPattern::VerticalSlash => {
            for index in (0..seq_len).step_by(config.vertical_stride) {
                selected.insert(index);
            }
            insert_range(
                &mut selected,
                seq_len.saturating_sub(config.recent_window),
                seq_len,
            );
        }
        SparsePrefillPattern::BlockSparse => {
            let mut blocks = block_scores(scores, config.block_size);
            blocks.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
            for (start, end, _) in blocks.into_iter().take(config.max_blocks) {
                insert_range(&mut selected, start, end);
                selected_blocks.push(SparseBlockRange { start, end });
            }
            selected_blocks.sort_by_key(|block| block.start);
        }
        SparsePrefillPattern::HybridAnchorRecentBlocks => {
            insert_range(&mut selected, 0, config.anchor_count.min(seq_len));
            insert_range(
                &mut selected,
                seq_len.saturating_sub(config.recent_window),
                seq_len,
            );
            let mut blocks = block_scores_excluding(scores, config.block_size, &selected);
            blocks.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
            for (start, end, _) in blocks.into_iter().take(config.max_blocks) {
                insert_range(&mut selected, start, end);
                selected_blocks.push(SparseBlockRange { start, end });
            }
            selected_blocks.sort_by_key(|block| block.start);
        }
        SparsePrefillPattern::AdaptiveMass => {
            selected = adaptive_mass_selection(scores, &config)?;
        }
    }

    if selected.len() > config.max_tokens {
        selected = cap_selected_by_score(selected, scores, config.max_tokens);
    }

    let selected_indices = selected.into_iter().collect::<Vec<_>>();
    let selected_token_count = selected_indices.len();
    let coverage_ratio = if seq_len == 0 {
        0.0
    } else {
        selected_token_count as f32 / seq_len as f32
    };
    let mut warnings = Vec::new();
    if selected_token_count == seq_len {
        warnings.push("plan selects the full score vector; no sparse score-read savings".into());
    }
    if selected_token_count == 0 && seq_len > 0 {
        warnings.push("plan selected no tokens for a non-empty score vector".into());
    }

    Ok(KvSparsePrefillPlanV1 {
        schema: "KvSparsePrefillPlanV1".into(),
        pattern,
        seq_len,
        selected_indices,
        selected_blocks,
        selected_token_count,
        coverage_ratio,
        config,
        warnings,
    })
}

/// Compare a sparse prefill plan against full attention scores.
pub fn compare_sparse_prefill(
    scores: &[f32],
    pattern: SparsePrefillPattern,
    config: SparsePrefillConfig,
) -> Result<KvSparsePrefillComparisonV1> {
    let plan = build_sparse_prefill_plan(scores, pattern, config)?;
    let top_k = plan.config.top_k.min(scores.len());
    let full_top_indices = top_indices(scores.iter().copied().enumerate(), top_k);
    let selected_set = plan
        .selected_indices
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let sparse_top_indices = top_indices(
        plan.selected_indices
            .iter()
            .copied()
            .map(|index| (index, scores[index])),
        top_k,
    );
    let selected_full_top_k_hits = full_top_indices
        .iter()
        .filter(|index| selected_set.contains(index))
        .count();
    let top_k_recall = if top_k == 0 {
        0.0
    } else {
        selected_full_top_k_hits as f32 / top_k as f32
    };
    let softmax_mass_coverage = softmax_mass_for_indices(scores, &selected_set);
    let estimated_score_reads_saved_ratio = if scores.is_empty() {
        0.0
    } else {
        1.0 - plan.selected_token_count as f32 / scores.len() as f32
    };
    let mut warnings = plan.warnings.clone();
    if selected_full_top_k_hits < top_k {
        warnings.push("sparse plan omitted at least one full-attention top-k score".into());
    }

    Ok(KvSparsePrefillComparisonV1 {
        schema: "KvSparsePrefillComparisonV1".into(),
        plan,
        top_k,
        full_top_indices,
        sparse_top_indices,
        selected_full_top_k_hits,
        top_k_recall,
        softmax_mass_coverage,
        estimated_score_reads_saved_ratio,
        warnings,
    })
}

/// Run a sparse-prefill comparison against exact retained KV shadows.
pub fn compare_cache_sparse_prefill(
    cache: &KvCacheCompressor,
    query: &[f32],
    pattern: SparsePrefillPattern,
    config: SparsePrefillConfig,
) -> Result<KvSparsePrefillComparisonV1> {
    let scores = cache.exact_attention_scores(query)?;
    compare_sparse_prefill(&scores, pattern, config)
}

/// Run a sparse-prefill comparison against the cache's configured attention
/// scoring path, including compressed key scoring when enabled.
pub fn compare_cache_sparse_prefill_with_options(
    cache: &KvCacheCompressor,
    query: &[f32],
    score_options: AttentionScoreOptions,
    pattern: SparsePrefillPattern,
    config: SparsePrefillConfig,
) -> Result<KvSparsePrefillComparisonV1> {
    let scores = cache.attention_scores_with_options(query, score_options)?;
    compare_sparse_prefill(&scores, pattern, config)
}

/// Benchmark sparse-prefill patterns across captured score traces.
pub fn benchmark_sparse_prefill_traces(
    traces: &[SparsePrefillTraceV1],
    patterns: &[SparsePrefillPattern],
    config: SparsePrefillConfig,
    gate: SparsePrefillGateConfig,
) -> Result<SparsePrefillTraceBenchmarkV1> {
    if !gate.min_softmax_mass_coverage.is_finite()
        || !gate.min_top_k_recall.is_finite()
        || !gate.min_score_reads_saved_ratio.is_finite()
    {
        return Err(TurboQuantError::ProfileMismatch {
            reason: "sparse prefill gate thresholds must be finite".into(),
        });
    }

    let mut summaries = Vec::with_capacity(patterns.len());
    for &pattern in patterns {
        let mut receipts = Vec::with_capacity(traces.len());
        for trace in traces {
            receipts.push(compare_sparse_prefill(
                &trace.scores,
                pattern,
                config.clone(),
            )?);
        }
        summaries.push(summarize_pattern(pattern, &receipts, gate));
    }

    let mut warnings = Vec::new();
    if traces.is_empty() {
        warnings.push("benchmark has no traces; no kernel decision should be made".into());
    }
    if summaries.iter().all(|summary| !summary.kernel_candidate) && !traces.is_empty() {
        warnings.push("no sparse prefill pattern passed the configured kernel gate".into());
    }

    Ok(SparsePrefillTraceBenchmarkV1 {
        schema: "SparsePrefillTraceBenchmarkV1".into(),
        trace_count: traces.len(),
        patterns: summaries,
        gate,
        config,
        warnings,
    })
}

fn validate_scores(scores: &[f32]) -> Result<()> {
    for (index, score) in scores.iter().enumerate() {
        if !score.is_finite() {
            return Err(TurboQuantError::NonFiniteInput { index });
        }
    }
    Ok(())
}

fn validate_config(config: &SparsePrefillConfig) -> Result<()> {
    if config.vertical_stride == 0 {
        return Err(TurboQuantError::ProfileMismatch {
            reason: "vertical_stride must be non-zero".into(),
        });
    }
    if config.block_size == 0 {
        return Err(TurboQuantError::ProfileMismatch {
            reason: "block_size must be non-zero".into(),
        });
    }
    if config.max_blocks == 0 {
        return Err(TurboQuantError::ProfileMismatch {
            reason: "max_blocks must be non-zero".into(),
        });
    }
    if config.max_tokens == 0 {
        return Err(TurboQuantError::ProfileMismatch {
            reason: "max_tokens must be non-zero".into(),
        });
    }
    if !config.adaptive_target_mass.is_finite()
        || config.adaptive_target_mass <= 0.0
        || config.adaptive_target_mass > 1.0
    {
        return Err(TurboQuantError::ProfileMismatch {
            reason: "adaptive_target_mass must be finite and in (0.0, 1.0]".into(),
        });
    }
    if !config.adaptive_min_score_reads_saved_ratio.is_finite()
        || !(0.0..1.0).contains(&config.adaptive_min_score_reads_saved_ratio)
    {
        return Err(TurboQuantError::ProfileMismatch {
            reason: "adaptive_min_score_reads_saved_ratio must be finite and in [0.0, 1.0)".into(),
        });
    }
    Ok(())
}

fn insert_range(selected: &mut BTreeSet<usize>, start: usize, end: usize) {
    for index in start..end {
        selected.insert(index);
    }
}

fn block_scores(scores: &[f32], block_size: usize) -> Vec<(usize, usize, f32)> {
    block_scores_excluding(scores, block_size, &BTreeSet::new())
}

fn block_scores_excluding(
    scores: &[f32],
    block_size: usize,
    excluded: &BTreeSet<usize>,
) -> Vec<(usize, usize, f32)> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    scores
        .chunks(block_size)
        .enumerate()
        .map(|(block_index, chunk)| {
            let start = block_index * block_size;
            let end = start + chunk.len();
            let score = chunk
                .iter()
                .enumerate()
                .filter(|(offset, _)| !excluded.contains(&(start + *offset)))
                .map(|(_, value)| value)
                .map(|value| (*value - max_score).exp())
                .sum::<f32>();
            (start, end, score)
        })
        .collect()
}

fn cap_selected_by_score(
    selected: BTreeSet<usize>,
    scores: &[f32],
    max_tokens: usize,
) -> BTreeSet<usize> {
    let mut ranked = selected.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]).then_with(|| a.cmp(b)));
    ranked.truncate(max_tokens);
    ranked.into_iter().collect()
}

fn adaptive_mass_selection(
    scores: &[f32],
    config: &SparsePrefillConfig,
) -> Result<BTreeSet<usize>> {
    let seq_len = scores.len();
    if seq_len == 0 {
        return Ok(BTreeSet::new());
    }
    let max_by_savings =
        ((seq_len as f32) * (1.0 - config.adaptive_min_score_reads_saved_ratio)).floor() as usize;
    let token_cap = config.max_tokens.min(max_by_savings).max(1);
    let top_k = config.top_k.min(seq_len);
    if config.adaptive_include_top_k && top_k > token_cap {
        return Err(TurboQuantError::ProfileMismatch {
            reason: "adaptive config cannot include top_k while preserving max_tokens/min savings"
                .into(),
        });
    }

    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights = scores
        .iter()
        .map(|score| (*score - max_score).exp())
        .collect::<Vec<_>>();
    let total_weight = weights.iter().sum::<f32>();
    if total_weight <= 0.0 || !total_weight.is_finite() {
        return Err(TurboQuantError::MalformedCode {
            reason: "adaptive sparse prefill saw invalid softmax weight total".into(),
        });
    }

    let mut ranked = weights.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut selected = BTreeSet::new();
    let mut selected_weight = 0.0f32;

    if config.adaptive_include_top_k {
        for (index, weight) in ranked.iter().take(top_k).copied() {
            if selected.insert(index) {
                selected_weight += weight;
            }
        }
    }

    for (index, weight) in ranked {
        if selected_weight / total_weight >= config.adaptive_target_mass {
            break;
        }
        if selected.len() >= token_cap {
            break;
        }
        if selected.insert(index) {
            selected_weight += weight;
        }
    }

    Ok(selected)
}

fn top_indices<I>(items: I, top_k: usize) -> Vec<usize>
where
    I: IntoIterator<Item = (usize, f32)>,
{
    let mut ranked = items.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked
        .into_iter()
        .take(top_k)
        .map(|(index, _)| index)
        .collect()
}

fn softmax_mass_for_indices(scores: &[f32], selected: &BTreeSet<usize>) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0f32;
    let mut selected_total = 0.0f32;
    for (index, score) in scores.iter().enumerate() {
        let weight = (*score - max_score).exp();
        total += weight;
        if selected.contains(&index) {
            selected_total += weight;
        }
    }
    if total == 0.0 {
        0.0
    } else {
        selected_total / total
    }
}

fn summarize_pattern(
    pattern: SparsePrefillPattern,
    receipts: &[KvSparsePrefillComparisonV1],
    gate: SparsePrefillGateConfig,
) -> SparsePrefillPatternSummaryV1 {
    let trace_count = receipts.len();
    let mut pass_count = 0usize;
    let mut sum_top_k_recall = 0.0f32;
    let mut sum_mass = 0.0f32;
    let mut sum_saved = 0.0f32;
    let mut min_top_k_recall = f32::INFINITY;
    let mut min_mass = f32::INFINITY;
    let mut min_saved = f32::INFINITY;

    for receipt in receipts {
        sum_top_k_recall += receipt.top_k_recall;
        sum_mass += receipt.softmax_mass_coverage;
        sum_saved += receipt.estimated_score_reads_saved_ratio;
        min_top_k_recall = min_top_k_recall.min(receipt.top_k_recall);
        min_mass = min_mass.min(receipt.softmax_mass_coverage);
        min_saved = min_saved.min(receipt.estimated_score_reads_saved_ratio);
        if receipt.softmax_mass_coverage >= gate.min_softmax_mass_coverage
            && receipt.top_k_recall >= gate.min_top_k_recall
            && receipt.estimated_score_reads_saved_ratio >= gate.min_score_reads_saved_ratio
        {
            pass_count += 1;
        }
    }

    let denominator = trace_count.max(1) as f32;
    let empty_min = |value: f32| if trace_count == 0 { 0.0 } else { value };
    SparsePrefillPatternSummaryV1 {
        pattern,
        trace_count,
        mean_top_k_recall: sum_top_k_recall / denominator,
        min_top_k_recall: empty_min(min_top_k_recall),
        mean_softmax_mass_coverage: sum_mass / denominator,
        min_softmax_mass_coverage: empty_min(min_mass),
        mean_score_reads_saved_ratio: sum_saved / denominator,
        min_score_reads_saved_ratio: empty_min(min_saved),
        pass_count,
        pass_rate: pass_count as f32 / denominator,
        kernel_candidate: trace_count > 0 && pass_count == trace_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertical_slash_retains_strided_columns_and_recent_tail() {
        let scores = (0..16).map(|index| index as f32).collect::<Vec<_>>();
        let config = SparsePrefillConfig {
            recent_window: 3,
            vertical_stride: 5,
            max_tokens: 16,
            ..SparsePrefillConfig::default()
        };
        let plan = build_sparse_prefill_plan(&scores, SparsePrefillPattern::VerticalSlash, config)
            .unwrap();
        assert_eq!(plan.selected_indices, vec![0, 5, 10, 13, 14, 15]);
    }

    #[test]
    fn block_sparse_selects_high_energy_blocks() {
        let scores = vec![0.1, 0.2, 0.3, 0.4, 9.0, 8.0, 7.0, 6.0, 0.5, 0.6];
        let config = SparsePrefillConfig {
            block_size: 4,
            max_blocks: 1,
            max_tokens: 8,
            ..SparsePrefillConfig::default()
        };
        let plan =
            build_sparse_prefill_plan(&scores, SparsePrefillPattern::BlockSparse, config).unwrap();
        assert_eq!(
            plan.selected_blocks,
            vec![SparseBlockRange { start: 4, end: 8 }]
        );
        assert_eq!(plan.selected_indices, vec![4, 5, 6, 7]);
    }

    #[test]
    fn hybrid_keeps_anchor_recent_and_high_mass_block() {
        let mut scores = vec![0.0f32; 32];
        scores[1] = 7.0;
        scores[14] = 8.0;
        scores[30] = 9.0;
        let config = SparsePrefillConfig {
            anchor_count: 2,
            recent_window: 3,
            block_size: 4,
            max_blocks: 1,
            max_tokens: 16,
            ..SparsePrefillConfig::default()
        };
        let plan = build_sparse_prefill_plan(
            &scores,
            SparsePrefillPattern::HybridAnchorRecentBlocks,
            config,
        )
        .unwrap();
        assert!(plan.selected_indices.contains(&0));
        assert!(plan.selected_indices.contains(&1));
        assert!(plan.selected_indices.contains(&14));
        assert!(plan.selected_indices.contains(&30));
    }

    #[test]
    fn comparison_reports_top_k_recall_and_mass_coverage() {
        let mut scores = vec![0.0f32; 32];
        scores[0] = 7.0;
        scores[30] = 6.0;
        let config = SparsePrefillConfig {
            anchor_count: 1,
            recent_window: 4,
            top_k: 2,
            max_tokens: 8,
            ..SparsePrefillConfig::default()
        };
        let receipt =
            compare_sparse_prefill(&scores, SparsePrefillPattern::AShape, config).unwrap();
        assert_eq!(receipt.full_top_indices, vec![0, 30]);
        assert_eq!(receipt.selected_full_top_k_hits, 2);
        assert_eq!(receipt.top_k_recall, 1.0);
        assert!(receipt.softmax_mass_coverage > 0.97);
    }

    #[test]
    fn adaptive_mass_selects_just_enough_high_mass_tokens() {
        let mut scores = vec![-10.0f32; 64];
        scores[7] = 5.0;
        scores[18] = 4.0;
        scores[52] = 3.0;
        let config = SparsePrefillConfig {
            top_k: 3,
            max_tokens: 16,
            adaptive_target_mass: 0.99,
            adaptive_min_score_reads_saved_ratio: 0.5,
            adaptive_include_top_k: true,
            ..SparsePrefillConfig::default()
        };
        let receipt =
            compare_sparse_prefill(&scores, SparsePrefillPattern::AdaptiveMass, config).unwrap();
        assert_eq!(receipt.top_k_recall, 1.0);
        assert!(receipt.softmax_mass_coverage >= 0.99);
        assert!(receipt.estimated_score_reads_saved_ratio >= 0.5);
        assert!(receipt.plan.selected_token_count <= 16);
    }

    #[test]
    fn adaptive_mass_respects_savings_when_mass_target_is_impossible() {
        let scores = vec![0.0f32; 64];
        let config = SparsePrefillConfig {
            top_k: 8,
            max_tokens: 16,
            adaptive_target_mass: 0.995,
            adaptive_min_score_reads_saved_ratio: 0.75,
            adaptive_include_top_k: true,
            ..SparsePrefillConfig::default()
        };
        let receipt =
            compare_sparse_prefill(&scores, SparsePrefillPattern::AdaptiveMass, config).unwrap();
        assert_eq!(receipt.plan.selected_token_count, 16);
        assert!(receipt.estimated_score_reads_saved_ratio >= 0.75);
        assert!(receipt.softmax_mass_coverage < 0.995);
    }

    #[test]
    fn trace_benchmark_marks_only_patterns_that_pass_all_traces() {
        let traces = vec![
            SparsePrefillTraceV1 {
                trace_id: "recent-heavy".into(),
                layer: Some(0),
                head: Some(0),
                scores: {
                    let mut scores = vec![0.0f32; 32];
                    scores[0] = 6.0;
                    scores[29] = 7.0;
                    scores
                },
            },
            SparsePrefillTraceV1 {
                trace_id: "middle-block".into(),
                layer: Some(0),
                head: Some(1),
                scores: {
                    let mut scores = vec![0.0f32; 32];
                    scores[0] = 6.0;
                    scores[13] = 7.0;
                    scores[30] = 8.0;
                    scores
                },
            },
        ];
        let config = SparsePrefillConfig {
            anchor_count: 2,
            recent_window: 4,
            block_size: 4,
            max_blocks: 1,
            max_tokens: 16,
            top_k: 2,
            ..SparsePrefillConfig::default()
        };
        let receipt = benchmark_sparse_prefill_traces(
            &traces,
            &[
                SparsePrefillPattern::AShape,
                SparsePrefillPattern::HybridAnchorRecentBlocks,
                SparsePrefillPattern::AdaptiveMass,
            ],
            config,
            SparsePrefillGateConfig {
                min_softmax_mass_coverage: 0.95,
                min_top_k_recall: 1.0,
                min_score_reads_saved_ratio: 0.5,
            },
        )
        .unwrap();
        assert_eq!(receipt.trace_count, 2);
        assert!(!receipt.patterns[0].kernel_candidate);
        assert!(receipt.patterns[1].kernel_candidate);
    }
}
