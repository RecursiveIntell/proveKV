//! Exactly-once stream processing over immutable [`StateStore`] snapshots.
use std::collections::{HashSet, VecDeque};
use std::time::{Duration, Instant};

use crate::error::{ProveKvError, Result};
use crate::gc;
use crate::lease::{LeaseRight, LeaseRights, StateLease};
use crate::principal::{ExecutionScope, Principal};
use crate::state_id::HybridStateId;
use crate::state_store::StateStore;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    pub sequence: u64,
    pub state_id: HybridStateId,
    pub digest: String,
    pub lease_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delivery<T> {
    pub sequence: u64,
    pub payload: T,
}

/// A bounded stream processor. State snapshots are immutable and delivery
/// sequence numbers are monotonic; a sequence is acknowledged at most once.
pub struct StreamProcessor<T> {
    pub store: StateStore,
    interval: Duration,
    lease_ttl: Option<u64>,
    principal: Principal,
    scope: ExecutionScope,
    next_sequence: u64,
    next_snapshot: Instant,
    latest: Option<Snapshot>,
    delivered: HashSet<u64>,
    pending: VecDeque<Delivery<T>>,
}

impl<T> StreamProcessor<T> {
    pub fn new(
        store: StateStore,
        interval: Duration,
        lease_ttl: Option<u64>,
        principal: Principal,
        scope: ExecutionScope,
    ) -> Self {
        Self {
            store,
            interval,
            lease_ttl,
            principal,
            scope,
            next_sequence: 0,
            next_snapshot: Instant::now() + interval,
            latest: None,
            delivered: HashSet::new(),
            pending: VecDeque::new(),
        }
    }

    pub fn snapshot_scheduler(&mut self, state_id: &HybridStateId) -> Result<Option<Snapshot>> {
        if Instant::now() < self.next_snapshot {
            return Ok(None);
        }
        let snapshot = self.capture_snapshot(state_id)?;
        self.next_snapshot = Instant::now() + self.interval;
        Ok(Some(snapshot))
    }

    pub fn capture_snapshot(&mut self, state_id: &HybridStateId) -> Result<Snapshot> {
        if !self.store.contains(state_id) {
            return Err(ProveKvError::InvalidManifest(
                "snapshot state not found".into(),
            ));
        }
        self.next_sequence = self.next_sequence.saturating_add(1);
        let lease = StateLease::new(
            self.principal.clone(),
            self.scope.clone(),
            state_id,
            LeaseRights::empty().with(LeaseRight::Inspect),
            self.lease_ttl,
            0,
            self.next_sequence,
        )?;
        let digest = blake3::hash(state_id.as_str().as_bytes())
            .to_hex()
            .to_string();
        self.store.register_lease(lease.clone())?;
        let snapshot = Snapshot {
            sequence: self.next_sequence,
            state_id: state_id.clone(),
            digest,
            lease_id: lease.lease_id,
        };
        self.latest = Some(snapshot.clone());
        Ok(snapshot)
    }

    pub fn standby_resume(&self) -> Result<HybridStateId> {
        let snapshot = self.latest.as_ref().ok_or_else(|| {
            ProveKvError::InvalidManifest("no verified snapshot available".into())
        })?;
        let expected = blake3::hash(snapshot.state_id.as_str().as_bytes())
            .to_hex()
            .to_string();
        if expected != snapshot.digest || !self.store.contains(&snapshot.state_id) {
            return Err(ProveKvError::InvalidManifest(
                "latest snapshot failed verification".into(),
            ));
        }
        Ok(snapshot.state_id.clone())
    }

    /// Queue a delivery. Replaying an already acknowledged sequence is a no-op.
    pub fn deliver(&mut self, sequence: u64, payload: T) -> bool {
        if self.delivered.contains(&sequence) || self.pending.iter().any(|d| d.sequence == sequence)
        {
            return false;
        }
        self.pending.push_back(Delivery { sequence, payload });
        true
    }

    pub fn receive(&mut self) -> Option<Delivery<T>> {
        self.pending.pop_front()
    }

    /// Acknowledge only once; callers should persist this result with their sink.
    pub fn acknowledge(&mut self, sequence: u64) -> bool {
        self.delivered.insert(sequence)
    }

    pub fn is_delivered(&self, sequence: u64) -> bool {
        self.delivered.contains(&sequence)
    }
    pub fn latest_snapshot(&self) -> Option<&Snapshot> {
        self.latest.as_ref()
    }

    /// Release old snapshot leases and run the existing collector.
    pub fn collect(&mut self) -> Result<gc::GcReport> {
        gc::collect(&mut self.store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid_manifest::{HybridComponent, HybridPageRef, HybridStateManifestV1};
    use crate::shape::{AttentionType, KvTensorShape};

    fn store() -> (StateStore, HybridStateId) {
        let root = std::env::temp_dir().join(format!("provekv-stream-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let mut s = StateStore::open(root).unwrap();
        let m = HybridStateManifestV1::new(
            "m",
            "t",
            KvTensorShape {
                attention_type: AttentionType::MHA,
                num_layers: 1,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 1,
                hidden_size: 1,
            },
            vec![HybridComponent {
                name: "c".into(),
                version: "1".into(),
                digest: "c".into(),
            }],
            vec![HybridPageRef {
                page_id: "p".into(),
                digest: "p".into(),
            }],
            vec![],
            "p",
            "v",
        );
        let id = s.commit_root(m).unwrap();
        (s, id)
    }
    fn processor() -> (StreamProcessor<String>, HybridStateId) {
        let (s, id) = store();
        (
            StreamProcessor::new(
                s,
                Duration::from_millis(0),
                Some(60_000),
                Principal::new("a", "n").unwrap(),
                ExecutionScope::new("r", "node", 0).unwrap(),
            ),
            id,
        )
    }

    #[test]
    fn snapshot_and_resume_verified_state() {
        let (mut p, id) = processor();
        let snap = p.snapshot_scheduler(&id).unwrap().unwrap();
        assert_eq!(p.standby_resume().unwrap(), id);
        assert_eq!(snap.sequence, 1);
    }
    #[test]
    fn duplicate_delivery_is_suppressed() {
        let (mut p, _) = processor();
        assert!(p.deliver(7, "x".into()));
        assert!(!p.deliver(7, "duplicate".into()));
        let d = p.receive().unwrap();
        assert_eq!(d.payload, "x");
        assert!(p.acknowledge(7));
        assert!(!p.acknowledge(7));
        assert!(!p.deliver(7, "again".into()));
    }
}
