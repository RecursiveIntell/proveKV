//! Memory-cache reuse adapter.
//!
//! This module keeps the application-level episode identity alongside the
//! canonical hybrid-state manifest and delegates compatibility decisions to
//! [`crate::state_reuse`].

use crate::hybrid_manifest::HybridStateManifestV1;
use crate::state_reuse;

/// A captured memory state that may be reused by a subsequent episode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryCache {
    /// Application/episode identifier associated with this capture.
    pub episode_id: String,
    /// Canonical state description captured for the episode.
    pub captured_context: HybridStateManifestV1,
}

impl MemoryCache {
    /// Create a cache entry for an episode and its captured state.
    pub fn new(episode_id: impl Into<String>, captured_context: HybridStateManifestV1) -> Self {
        Self {
            episode_id: episode_id.into(),
            captured_context,
        }
    }

    /// Return whether an incoming request can reuse this captured state.
    ///
    /// The comparison uses the existing state-reuse policy: identity fields
    /// must match and the differing page fraction must be within `tolerance`.
    pub fn similarity_check(
        &self,
        incoming_request: &HybridStateManifestV1,
        tolerance: f32,
    ) -> bool {
        state_reuse::detect_fork_opportunity(incoming_request, &self.captured_context, tolerance)
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
                .map(|digest| crate::hybrid_manifest::HybridPageRef {
                    page_id: (*digest).into(),
                    digest: (*digest).into(),
                })
                .collect(),
            vec![],
            "policy",
            "version",
        )
    }

    #[test]
    fn reuses_similar_captured_context() {
        let cache = MemoryCache::new("episode-1", manifest(&["a", "b"]));
        assert!(cache.similarity_check(&manifest(&["a", "c"]), 0.5));
        assert!(!cache.similarity_check(&manifest(&["a", "c"]), 0.25));
    }

    #[test]
    fn rejects_incompatible_request() {
        let cache = MemoryCache::new("episode-1", manifest(&["a"]));
        let mut request = manifest(&["a"]);
        request.model_id = "other-model".into();
        assert!(!cache.similarity_check(&request, 1.0));
    }
}
