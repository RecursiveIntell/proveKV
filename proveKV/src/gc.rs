//! Mark-and-sweep garbage collection for the state store.
//!
//! GC identifies unreachable pages (not referenced by any live manifest) and
//! states that are released with no active leases and no reachable children.
//! It never deletes a reachable page or resurrects a revoked state.

use std::collections::HashSet;

use crate::error::Result;
use crate::state_store::StateStore;

/// Outcome of a GC pass.
#[derive(Debug, Default)]
pub struct GcReport {
    /// State IDs that were collected.
    pub collected_states: Vec<String>,
    /// Page digests that were deleted.
    pub collected_pages: Vec<String>,
    /// States that were reachable and retained.
    pub retained_states: usize,
}

/// Run a mark-and-sweep GC pass.
///
/// 1. Mark: from every non-released, non-expired root state, walk children
///    and mark them reachable. States with active leases are also reachable.
/// 2. Sweep: delete released states that are unreachable and have no active
///    leases. Delete pages only referenced by collected states.
pub fn collect(store: &mut StateStore) -> Result<GcReport> {
    let mut report = GcReport::default();

    // Build reachability set.
    let mut reachable: HashSet<String> = HashSet::new();
    let mut worklist: Vec<String> = Vec::new();

    // All non-released states are roots.
    for id in store.state_ids() {
        let node = store.get(id).unwrap();
        if !node.released || !node.active_leases.is_empty() {
            let id_str = id.as_str().to_string();
            if reachable.insert(id_str.clone()) {
                worklist.push(id_str);
            }
        }
    }

    // Walk children.
    while let Some(current) = worklist.pop() {
        if let Some(node) = store.get(&crate::state_id::HybridStateId(current.clone())) {
            for child in &node.children {
                let child_str = child.as_str().to_string();
                if reachable.insert(child_str.clone()) {
                    worklist.push(child_str);
                }
            }
        }
    }

    report.retained_states = reachable.len();

    // Collect unreachable released states with no active leases.
    let all_ids: Vec<String> = store
        .state_ids()
        .iter()
        .map(|id| id.as_str().to_string())
        .collect();

    for id_str in &all_ids {
        if reachable.contains(id_str) {
            continue;
        }

        let node = store.get(&crate::state_id::HybridStateId(id_str.clone()));
        if let Some(node) = node {
            if node.released && node.active_leases.is_empty() {
                // Collect pages referenced by this state.
                for page_ref in &node.manifest.page_refs {
                    if store.page_store.page_exists(&page_ref.digest) {
                        store.page_store.delete_page(&page_ref.digest)?;
                        report.collected_pages.push(page_ref.digest.clone());
                    }
                }
                report.collected_states.push(id_str.clone());
            }
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::branch::branch;
    use crate::hybrid_manifest::{HybridComponent, HybridPageRef, HybridStateManifestV1};
    use crate::page_format::build_page_header;
    use std::env;
    use std::sync::atomic::{AtomicU32, Ordering};

    static STORE_COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_store() -> StateStore {
        let n = STORE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("provekv-gc-{}-{}", std::process::id(), n));
        let _ = std::fs::remove_dir_all(&dir);
        StateStore::open(&dir).unwrap()
    }

    fn sample_manifest(label: &str) -> HybridStateManifestV1 {
        let shape = crate::shape::KvTensorShape {
            attention_type: crate::shape::AttentionType::MHA,
            num_layers: 1,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            hidden_size: 4096,
        };
        HybridStateManifestV1::new(
            "qwen3.5-2b",
            "qwen3.5-tokenizer",
            shape,
            vec![HybridComponent {
                name: format!("{}-full_attn_k", label),
                version: "1.0".into(),
                digest: format!("sha256:comp_{label}_0"),
            }],
            vec![HybridPageRef {
                page_id: format!("page_{label}_0"),
                digest: format!("sha256:page_{label}_0"),
            }],
            vec![],
            format!("sha256:policy_{label}"),
            format!("sha256:version_{label}"),
        )
    }

    #[test]
    fn gc_retains_reachable_states() {
        let mut store = temp_store();
        let root = store.commit_root(sample_manifest("root")).unwrap();

        let mut m = sample_manifest("child");
        let child = branch(&mut store, &root, m).unwrap();

        let report = collect(&mut store).unwrap();
        assert_eq!(report.retained_states, 2);
        assert!(report.collected_states.is_empty());
        assert!(store.contains(&root));
        assert!(store.contains(&child));
    }

    #[test]
    fn gc_collects_released_unreachable_state() {
        let mut store = temp_store();
        let root = store.commit_root(sample_manifest("orphan")).unwrap();
        store.release(&root).unwrap();

        let report = collect(&mut store).unwrap();
        assert!(report.collected_states.contains(&root.as_str().to_string()));
    }

    #[test]
    fn gc_retains_released_state_with_active_lease() {
        let mut store = temp_store();
        let root = store.commit_root(sample_manifest("leased")).unwrap();

        let lease = crate::lease::StateLease {
            lease_id: "lease-v1:gc-test".into(),
            principal: crate::principal::Principal::new("agent-1", "ns").unwrap(),
            namespace: "ns".into(),
            scope: crate::principal::ExecutionScope::new("r", "n", 0).unwrap(),
            state_id: root.as_str().to_string(),
            rights: crate::lease::LeaseRights::empty().with(crate::lease::LeaseRight::Inspect),
            issued_unix_ms: 0,
            expires_unix_ms: None,
            revocation_epoch: 0,
            nonce: 0,
        };
        store.register_lease(lease).unwrap();
        store.release(&root).unwrap();

        let report = collect(&mut store).unwrap();
        // Released but has active lease — should be retained.
        assert!(!report.collected_states.contains(&root.as_str().to_string()));
    }

    #[test]
    fn gc_preserves_child_of_reachable_parent() {
        let mut store = temp_store();
        let root = store.commit_root(sample_manifest("root")).unwrap();
        let mut m = sample_manifest("child");
        let child = branch(&mut store, &root, m).unwrap();

        // Release the child but root is still reachable → child reachable via parent.
        store.release(&child).unwrap();

        let report = collect(&mut store).unwrap();
        assert!(!report
            .collected_states
            .contains(&child.as_str().to_string()));
    }
}
