//! Capture helpers for exporting store state to bridge consumers.

use super::types::StateSnapshot;
use crate::error::Result;
use crate::state_store::StateStore;

/// Owns a state store while exposing a stable, serializable capture boundary.
pub struct CaptureContext {
    pub store: StateStore,
}

impl CaptureContext {
    pub fn new(store: StateStore) -> Self {
        Self { store }
    }

    /// Capture the committed state IDs and lease count without mutating the store.
    pub fn capture_snapshot(&self) -> Result<StateSnapshot> {
        let mut state_ids = self
            .store
            .state_ids()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        state_ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        Ok(StateSnapshot {
            state_ids,
            lease_count: self.store.lease_count(),
        })
    }

    pub fn into_store(self) -> StateStore {
        self.store
    }
}
