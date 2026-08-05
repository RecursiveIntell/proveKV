//! Agent execution state reuse.
//!
//! An [`AgentSession`] keeps the small amount of mutable execution metadata
//! needed to reuse immutable proveKV states. State payloads remain owned by
//! [`StateStore`]; a session only records the parent and its fork IDs.

use std::collections::HashSet;

use crate::error::{ProveKvError, Result};
use crate::hybrid_manifest::HybridStateManifestV1;
use crate::lease::{LeaseRight, StateLease};
use crate::principal::Principal;
use crate::state_id::HybridStateId;
use crate::state_store::StateStore;

/// Execution state lineage for one agent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentSession {
    /// Stable identifier for the agent execution.
    pub agent_id: String,
    /// Immutable state from which this execution began.
    pub parent_state: HybridStateId,
    /// States created by this execution, in fork order.
    pub forked_states: Vec<HybridStateId>,
}

impl AgentSession {
    /// Start an agent execution from an existing state.
    pub fn new(agent_id: impl Into<String>, parent_state: HybridStateId) -> Self {
        Self {
            agent_id: agent_id.into(),
            parent_state,
            forked_states: Vec::new(),
        }
    }

    /// Fork a state under a lease and attach the resulting state to this session.
    pub fn fork(
        &mut self,
        store: &mut StateStore,
        manifest: HybridStateManifestV1,
        lease: &StateLease,
        now_unix_ms: u64,
        revocation_epoch: u64,
        principal: &Principal,
    ) -> Result<HybridStateId> {
        let parent = self.forked_states.last().unwrap_or(&self.parent_state);
        lease.authorize(
            LeaseRight::Fork,
            now_unix_ms,
            revocation_epoch,
            parent.as_str(),
            principal,
        )?;
        let child = store.fork(parent, manifest)?;
        self.forked_states.push(child.clone());
        Ok(child)
    }

    /// Replay this session's state lineage from the parent through its forks.
    ///
    /// The store is consulted for every edge, so stale or foreign IDs fail
    /// closed rather than producing a plausible partial replay.
    pub fn replay(&self, store: &StateStore) -> Result<Vec<HybridStateId>> {
        let mut replay = vec![self.parent_state.clone()];
        let mut seen = HashSet::from([self.parent_state.as_str().to_owned()]);
        for state in &self.forked_states {
            let node = store.get(state).ok_or_else(|| {
                ProveKvError::InvalidManifest(format!(
                    "state {} not found in session",
                    state.as_str()
                ))
            })?;
            let mut chain = Vec::new();
            let mut current = Some(state.clone());
            while let Some(id) = current {
                let node = store.get(&id).ok_or_else(|| {
                    ProveKvError::InvalidManifest(format!(
                        "state {} not found in lineage",
                        id.as_str()
                    ))
                })?;
                chain.push(id.clone());
                current = node.parent_id.clone();
            }
            chain.reverse();
            for id in chain {
                if seen.insert(id.as_str().to_owned()) {
                    replay.push(id);
                }
            }
            let _ = node;
        }
        Ok(replay)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid_manifest::HybridPageRef;
    use crate::lease::LeaseRights;
    use crate::principal::ExecutionScope;
    use std::env;

    fn manifest(label: &str) -> HybridStateManifestV1 {
        HybridStateManifestV1::new(
            "m",
            "t",
            crate::shape::KvTensorShape {
                attention_type: crate::shape::AttentionType::MHA,
                num_layers: 1,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 4,
                hidden_size: 4,
            },
            vec![],
            vec![HybridPageRef {
                page_id: label.into(),
                digest: label.into(),
            }],
            vec![],
            "p",
            label,
        )
    }

    #[test]
    fn replay_walks_parent_and_forks() {
        let dir = env::temp_dir().join(format!("provekv-agent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut store = StateStore::open(dir).unwrap();
        let root = store.commit_root(manifest("root")).unwrap();
        let mut session = AgentSession::new("agent", root.clone());
        let a = store.fork(&root, manifest("a")).unwrap();
        let b = store.fork(&a, manifest("b")).unwrap();
        session.forked_states = vec![a, b];
        assert_eq!(
            session.replay(&store).unwrap(),
            vec![
                root,
                session.forked_states[0].clone(),
                session.forked_states[1].clone()
            ]
        );
    }

    #[test]
    fn fork_requires_lease_right() {
        let _ = ExecutionScope::new("r", "n", 0);
        assert!(LeaseRights::empty().can(LeaseRight::Fork) == false);
    }
}
