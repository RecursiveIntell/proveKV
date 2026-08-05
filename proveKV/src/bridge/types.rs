//! Serializable bridge representations of proveKV state transitions.

use serde::{Deserialize, Serialize};

use crate::state_id::HybridStateId;

/// A point-in-time summary of the states currently committed in a store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateSnapshot {
    /// IDs of all states visible at capture time.
    pub state_ids: Vec<HybridStateId>,
    /// Number of active leases at capture time.
    pub lease_count: usize,
}

/// The state-ID changes between two captured snapshots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateDelta {
    pub from: StateSnapshot,
    pub to: StateSnapshot,
    pub added: Vec<HybridStateId>,
    pub removed: Vec<HybridStateId>,
}

/// The parent/child relationship created by a fork.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForkPoint {
    pub parent_id: HybridStateId,
    pub child_id: HybridStateId,
}
