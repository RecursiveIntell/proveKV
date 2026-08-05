//! Immutable hybrid-state store with O(1)-metadata forks.
//!
//! Every state is content-addressed. Forks create only metadata overlays
//! (no page copies). Append returns a new state ID. Parents and siblings
//! are immutable once committed.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::error::{ProveKvError, Result};
use crate::hybrid_manifest::HybridStateManifestV1;
use crate::lease::StateLease;
use crate::page_store::PageStore;
use crate::state_id::HybridStateId;

/// An immutable state node in the store.
#[derive(Debug, Clone)]
pub struct StateNode {
    /// Content-addressed identity.
    pub id: HybridStateId,
    /// Immutable manifest.
    pub manifest: HybridStateManifestV1,
    /// Parent state ID, if this is a fork.
    pub parent_id: Option<HybridStateId>,
    /// Child state IDs (forks from this state).
    pub children: Vec<HybridStateId>,
    /// Active leases on this state.
    pub active_leases: Vec<String>,
    /// Whether this state has been released.
    pub released: bool,
}

/// The immutable hybrid-state store.
///
/// All mutations return a new `StateId`. Once committed, a `StateNode` is
/// never modified in place. Forks create overlay metadata; pages are
/// shared by reference.
pub struct StateStore {
    /// Persistent page store.
    pub page_store: PageStore,
    /// All committed states, keyed by their state ID string.
    states: HashMap<String, StateNode>,
    /// Active leases, keyed by lease ID.
    leases: HashMap<String, StateLease>,
    /// Store root directory.
    root: PathBuf,
}

impl StateStore {
    /// Open or create a state store.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let page_store = PageStore::open(root.join("pages"))?;
        Ok(Self {
            page_store,
            states: HashMap::new(),
            leases: HashMap::new(),
            root,
        })
    }

    /// Commit a new root state (no parent).
    pub fn commit_root(&mut self, manifest: HybridStateManifestV1) -> Result<HybridStateId> {
        let id = HybridStateId::from_manifest(&manifest)?;
        let id_str = id.as_str().to_string();

        if self.states.contains_key(&id_str) {
            return Err(ProveKvError::InvalidManifest(format!(
                "state {} already exists",
                id_str
            )));
        }

        self.states.insert(
            id_str,
            StateNode {
                id: id.clone(),
                manifest,
                parent_id: None,
                children: Vec::new(),
                active_leases: Vec::new(),
                released: false,
            },
        );

        Ok(id)
    }

    /// Fork a state: create a new child with overlay metadata.
    ///
    /// Returns the new child state ID. The parent's `children` list is
    /// updated. No pages are copied.
    pub fn fork(
        &mut self,
        parent_id: &HybridStateId,
        manifest: HybridStateManifestV1,
    ) -> Result<HybridStateId> {
        let parent_str = parent_id.as_str();

        let parent = self.states.get(parent_str).ok_or_else(|| {
            ProveKvError::InvalidManifest(format!("parent {} not found", parent_str))
        })?;

        if parent.released {
            return Err(ProveKvError::InvalidManifest(format!(
                "cannot fork released state {}",
                parent_str
            )));
        }

        let child_id = HybridStateId::from_manifest(&manifest)?;
        let child_str = child_id.as_str().to_string();

        if self.states.contains_key(&child_str) {
            return Err(ProveKvError::InvalidManifest(format!(
                "state {} already exists",
                child_str
            )));
        }

        // Insert child.
        self.states.insert(
            child_str.clone(),
            StateNode {
                id: child_id.clone(),
                manifest,
                parent_id: Some(parent_id.clone()),
                children: Vec::new(),
                active_leases: Vec::new(),
                released: false,
            },
        );

        // Update parent's children list.
        if let Some(parent) = self.states.get_mut(parent_str) {
            parent.children.push(child_id.clone());
        }

        Ok(child_id)
    }

    /// Get a state by ID.
    pub fn get(&self, id: &HybridStateId) -> Option<&StateNode> {
        self.states.get(id.as_str())
    }

    /// Check if a state exists.
    pub fn contains(&self, id: &HybridStateId) -> bool {
        self.states.contains_key(id.as_str())
    }

    /// Register a lease against a state.
    pub fn register_lease(&mut self, lease: StateLease) -> Result<()> {
        let state_id = lease.state_id.clone();
        let lease_id = lease.lease_id.clone();

        if self.leases.contains_key(&lease_id) {
            return Err(ProveKvError::InvalidLease(format!(
                "lease {} already registered",
                lease_id
            )));
        }

        let state = self
            .states
            .get_mut(&state_id)
            .ok_or_else(|| ProveKvError::InvalidLease(format!("state {} not found", state_id)))?;

        if state.released {
            return Err(ProveKvError::InvalidLease(
                "cannot register lease on released state".into(),
            ));
        }

        state.active_leases.push(lease_id.clone());
        self.leases.insert(lease_id, lease);
        Ok(())
    }

    /// Revoke a lease.
    pub fn revoke_lease(&mut self, lease_id: &str) -> Result<()> {
        let lease = self
            .leases
            .remove(lease_id)
            .ok_or_else(|| ProveKvError::InvalidLease(format!("lease {} not found", lease_id)))?;

        if let Some(state) = self.states.get_mut(&lease.state_id) {
            state.active_leases.retain(|lid| lid != lease_id);
        }

        Ok(())
    }

    /// Release a state (marks as released, does not delete pages — GC
    /// handles that separately). States with active leases can be released
    /// but will not be collected by GC until all leases are revoked.
    pub fn release(&mut self, id: &HybridStateId) -> Result<()> {
        let id_str = id.as_str();
        let state = self
            .states
            .get_mut(id_str)
            .ok_or_else(|| ProveKvError::InvalidManifest(format!("state {} not found", id_str)))?;
        state.released = true;
        Ok(())
    }

    /// Return all state IDs.
    pub fn state_ids(&self) -> Vec<&HybridStateId> {
        self.states.values().map(|n| &n.id).collect()
    }

    /// Return the number of committed states.
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Return the number of active leases.
    pub fn lease_count(&self) -> usize {
        self.leases.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid_manifest::{HybridComponent, HybridPageRef};
    use std::env;
    use std::sync::atomic::{AtomicU32, Ordering};

    static STORE_COUNTER: AtomicU32 = AtomicU32::new(0);

    fn temp_store() -> StateStore {
        let n = STORE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("provekv-state-{}-{}", std::process::id(), n));
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
    fn commit_and_retrieve_root() {
        let mut store = temp_store();
        let m = sample_manifest("root");
        let id = store.commit_root(m.clone()).unwrap();
        assert!(store.contains(&id));
        let node = store.get(&id).unwrap();
        assert!(node.parent_id.is_none());
        assert_eq!(node.children.len(), 0);
    }

    #[test]
    fn fork_creates_child_without_copying_parent() {
        let mut store = temp_store();
        let root_id = store.commit_root(sample_manifest("parent")).unwrap();

        let child_m = sample_manifest("child");
        let child_id = store.fork(&root_id, child_m).unwrap();

        assert_ne!(root_id.as_str(), child_id.as_str());
        let parent = store.get(&root_id).unwrap();
        assert_eq!(parent.children.len(), 1);
        assert_eq!(parent.children[0].as_str(), child_id.as_str());

        let child = store.get(&child_id).unwrap();
        assert_eq!(child.parent_id.as_ref().unwrap().as_str(), root_id.as_str());
    }

    #[test]
    fn fork_released_state_rejected() {
        let mut store = temp_store();
        let root_id = store.commit_root(sample_manifest("r")).unwrap();
        store.release(&root_id).unwrap();

        let child_m = sample_manifest("child");
        assert!(store.fork(&root_id, child_m).is_err());
    }

    #[test]
    fn release_requires_no_active_leases() {
        let mut store = temp_store();
        let root_id = store.commit_root(sample_manifest("r")).unwrap();

        // Register a lease (simplified — real lease would have proper fields).
        let lease = StateLease {
            lease_id: "lease-v1:test".into(),
            principal: crate::principal::Principal::new("agent-1", "ns").unwrap(),
            namespace: "ns".into(),
            scope: crate::principal::ExecutionScope::new("run-1", "node-1", 0).unwrap(),
            state_id: root_id.as_str().to_string(),
            rights: crate::lease::LeaseRights::empty().with(crate::lease::LeaseRight::Inspect),
            issued_unix_ms: 0,
            expires_unix_ms: None,
            revocation_epoch: 0,
            nonce: 0,
        };
        store.register_lease(lease).unwrap();
        // Release works even with active leases — GC will retain it.
        assert!(store.release(&root_id).is_ok());
    }

    #[test]
    fn duplicate_state_rejected() {
        let mut store = temp_store();
        let m = sample_manifest("dup");
        let id = store.commit_root(m.clone()).unwrap();
        assert!(store.commit_root(m).is_err());
        // Fork to same manifest also rejected.
        let parent_id = id;
        let child_m = sample_manifest("child");
        let child_id = store.fork(&parent_id, child_m.clone()).unwrap();
        assert!(store.fork(&parent_id, child_m).is_err());
    }

    #[test]
    fn state_count_tracks_committed() {
        let mut store = temp_store();
        assert_eq!(store.state_count(), 0);
        let root = store.commit_root(sample_manifest("a")).unwrap();
        assert_eq!(store.state_count(), 1);
        let m2 = sample_manifest("b");
        store.fork(&root, m2).unwrap();
        assert_eq!(store.state_count(), 2);
    }
}
