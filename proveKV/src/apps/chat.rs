//! Conversation state reuse adapter.
//!
//! Chat sessions keep the conversation identity alongside a point-in-time
//! snapshot of the store. The snapshot is metadata-only: the underlying KV
//! pages remain content-addressed and shared by the store.

use crate::bridge::StateSnapshot;
use crate::error::Result;
use crate::state_store::StateStore;

/// Capture the current KV-cache state for a conversation.
///
/// `conversation_id` is accepted as part of the application boundary so the
/// caller can associate the returned snapshot with its chat. The ID is not
/// persisted in `StateSnapshot`, which is deliberately a reusable store-level
/// representation.
pub fn capture_conversation_state(
    store: &StateStore,
    conversation_id: impl AsRef<str>,
) -> Result<StateSnapshot> {
    if conversation_id.as_ref().is_empty() {
        return Err(crate::error::ProveKvError::InvalidManifest(
            "conversation ID must not be empty".into(),
        ));
    }

    let mut state_ids = store.state_ids().into_iter().cloned().collect::<Vec<_>>();
    state_ids.sort_by(|a, b| a.as_str().cmp(b.as_str()));
    Ok(StateSnapshot {
        state_ids,
        lease_count: store.lease_count(),
    })
}

/// Application state for one chat conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatSession {
    pub conversation_id: String,
    pub captured_snapshot: StateSnapshot,
    pub fork_count: usize,
}

impl ChatSession {
    /// Start a session by capturing the store's current KV-cache state.
    pub fn capture(store: &StateStore, conversation_id: impl Into<String>) -> Result<Self> {
        let conversation_id = conversation_id.into();
        let captured_snapshot = capture_conversation_state(store, &conversation_id)?;
        Ok(Self {
            conversation_id,
            captured_snapshot,
            fork_count: 0,
        })
    }

    /// Record a conversation fork performed by the application layer.
    pub fn record_fork(&mut self) {
        self.fork_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hybrid_manifest::{HybridComponent, HybridPageRef, HybridStateManifestV1};
    use crate::shape::{AttentionType, KvTensorShape};
    use std::env;

    fn store() -> StateStore {
        let dir = env::temp_dir().join(format!("provekv-chat-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        StateStore::open(dir).unwrap()
    }

    fn manifest(label: &str) -> HybridStateManifestV1 {
        HybridStateManifestV1::new(
            "model",
            "tokenizer",
            KvTensorShape {
                attention_type: AttentionType::MHA,
                num_layers: 1,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 4,
                hidden_size: 4,
            },
            vec![HybridComponent {
                name: label.into(),
                version: "1".into(),
                digest: label.into(),
            }],
            vec![HybridPageRef {
                page_id: label.into(),
                digest: label.into(),
            }],
            vec![],
            "policy",
            "version",
        )
    }

    #[test]
    fn captures_current_cache_and_session_metadata() {
        let mut state_store = store();
        state_store.commit_root(manifest("root")).unwrap();
        let snapshot = capture_conversation_state(&state_store, "conversation-1").unwrap();
        assert_eq!(snapshot.state_ids.len(), 1);
        assert_eq!(snapshot.lease_count, 0);

        let mut session = ChatSession::capture(&state_store, "conversation-1").unwrap();
        assert_eq!(session.conversation_id, "conversation-1");
        assert_eq!(session.captured_snapshot, snapshot);
        assert_eq!(session.fork_count, 0);
        session.record_fork();
        assert_eq!(session.fork_count, 1);
    }

    #[test]
    fn rejects_empty_conversation_id() {
        let state_store = store();
        assert!(capture_conversation_state(&state_store, "").is_err());
    }
}
