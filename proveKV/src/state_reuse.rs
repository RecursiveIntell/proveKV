//! Policy and accounting helpers for reusing immutable hybrid states.

use std::collections::HashSet;

use crate::hybrid_manifest::HybridStateManifestV1;
use crate::lease::LeaseStatus;

/// Return whether a leased state is eligible for reuse.
///
/// A state must have an active lease and be no older than `max_age_ms`.
/// Callers should calculate `age_ms` from their trusted clock.
pub fn reuse_policy(status: LeaseStatus, age_ms: u64, max_age_ms: u64) -> bool {
    matches!(status, LeaseStatus::Active) && age_ms <= max_age_ms
}

/// Check whether a requested manifest can reuse an existing state as a fork.
///
/// Model, tokenizer, shape, policy, version, and component inventory must
/// match exactly. `tolerance` is the permitted fraction of page references
/// that may differ (0.0 requires an exact page set; 1.0 permits all pages).
pub fn detect_fork_opportunity(
    requested: &HybridStateManifestV1,
    existing: &HybridStateManifestV1,
    tolerance: f32,
) -> bool {
    if !(0.0..=1.0).contains(&tolerance)
        || requested.model_id != existing.model_id
        || requested.tokenizer_id != existing.tokenizer_id
        || requested.shape != existing.shape
        || requested.policy_digest != existing.policy_digest
        || requested.version_digest != existing.version_digest
        || requested.component_inventory != existing.component_inventory
    {
        return false;
    }

    let requested_pages: HashSet<&str> = requested
        .page_refs
        .iter()
        .map(|p| p.digest.as_str())
        .collect();
    let existing_pages: HashSet<&str> = existing
        .page_refs
        .iter()
        .map(|p| p.digest.as_str())
        .collect();
    let total = requested_pages.len().max(existing_pages.len());
    if total == 0 {
        return true;
    }
    let shared = requested_pages.intersection(&existing_pages).count();
    let difference = total - shared;
    (difference as f32 / total as f32) <= tolerance
}

/// Storage accounting for state reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DedupStats {
    /// Number of requested state materializations.
    pub total_states: usize,
    /// Number of physically unique state payloads.
    pub unique_states: usize,
    /// Bytes that would have been stored without reuse.
    pub logical_bytes: u64,
    /// Bytes stored after reuse.
    pub physical_bytes: u64,
    /// Bytes saved by reuse.
    pub saved_bytes: u64,
}

/// Report storage savings from reusing `unique_states` payloads.
pub fn dedup_stats(total_states: usize, unique_states: usize, bytes_per_state: u64) -> DedupStats {
    let unique_states = unique_states.min(total_states);
    let logical_bytes = (total_states as u64).saturating_mul(bytes_per_state);
    let physical_bytes = (unique_states as u64).saturating_mul(bytes_per_state);
    DedupStats {
        total_states,
        unique_states,
        logical_bytes,
        physical_bytes,
        saved_bytes: logical_bytes.saturating_sub(physical_bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid_manifest::HybridStateManifestV1;
    use crate::shape::{AttentionType, KvTensorShape};

    fn manifest(pages: &[&str]) -> HybridStateManifestV1 {
        HybridStateManifestV1::new(
            "model",
            "tokenizer",
            KvTensorShape {
                attention_type: AttentionType::MHA,
                num_layers: 1,
                num_heads: 2,
                num_kv_heads: 2,
                head_dim: 4,
                hidden_size: 8,
            },
            vec![],
            pages
                .iter()
                .map(|d| crate::hybrid_manifest::HybridPageRef {
                    page_id: (*d).into(),
                    digest: (*d).into(),
                })
                .collect(),
            vec![],
            "policy",
            "version",
        )
    }

    #[test]
    fn reuse_requires_active_fresh_lease() {
        assert!(reuse_policy(LeaseStatus::Active, 10, 10));
        assert!(!reuse_policy(LeaseStatus::Expired, 0, 10));
        assert!(!reuse_policy(LeaseStatus::Active, 11, 10));
    }

    #[test]
    fn fork_tolerance_matches_shared_pages() {
        let a = manifest(&["a", "b"]);
        assert!(detect_fork_opportunity(&a, &manifest(&["a", "c"]), 0.5));
        assert!(!detect_fork_opportunity(&a, &manifest(&["a", "c"]), 0.25));
    }

    #[test]
    fn stats_report_saved_storage() {
        let stats = dedup_stats(10, 3, 100);
        assert_eq!(stats.logical_bytes, 1000);
        assert_eq!(stats.physical_bytes, 300);
        assert_eq!(stats.saved_bytes, 700);
    }
}
