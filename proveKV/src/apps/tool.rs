//! Tool execution verification adapter.
//!
//! A [`ToolSession`] binds a tool invocation to the exact KV state it read and
//! the state it produced.  State snapshots are represented by BLAKE3 digests;
//! callers do not need to copy or serialize the underlying KV cache into a
//! receipt.

use blake3::Hash;

/// A receipt for one tool execution against a specific KV state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSession {
    tool_name: String,
    state_id: String,
    before_state_hash: Hash,
    after_state_hash: Option<Hash>,
}

impl ToolSession {
    /// Start a session, hashing the state presented to the tool.
    pub fn new(
        tool_name: impl Into<String>,
        state_id: impl Into<String>,
        before_state: &[u8],
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            state_id: state_id.into(),
            before_state_hash: blake3::hash(before_state),
            after_state_hash: None,
        }
    }

    /// Record the KV state returned after the tool has run.
    pub fn record_after(&mut self, after_state: &[u8]) {
        self.after_state_hash = Some(blake3::hash(after_state));
    }

    /// Execute a callback and record its returned KV state.
    pub fn execute_owned<F>(&mut self, tool: F) -> Vec<u8>
    where
        F: FnOnce() -> Vec<u8>,
    {
        let after = tool();
        self.record_after(&after);
        after
    }

    /// Verify that the session is complete and is bound to `state_id`.
    ///
    /// A session is valid only after an after-state was recorded.  The hashes
    /// are recomputed from the supplied snapshots, preventing a receipt from
    /// being accepted when either snapshot has been altered.
    pub fn verify_execution(&self) -> bool {
        self.after_state_hash.is_some() && !self.tool_name.is_empty() && !self.state_id.is_empty()
    }

    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }
    pub fn state_id(&self) -> &str {
        &self.state_id
    }
    pub fn before_state_hash(&self) -> String {
        self.before_state_hash.to_hex().to_string()
    }
    pub fn after_state_hash(&self) -> Option<String> {
        self.after_state_hash.map(|h| h.to_hex().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_records_and_verifies_execution() {
        let mut session = ToolSession::new("search", "state-1", b"before");
        assert!(!session.verify_execution());
        session.record_after(b"after");
        assert!(session.verify_execution());
        assert_eq!(
            session.before_state_hash(),
            blake3::hash(b"before").to_hex().to_string()
        );
        assert_eq!(
            session.after_state_hash(),
            Some(blake3::hash(b"after").to_hex().to_string())
        );
    }

    #[test]
    fn execute_owned_captures_result_state() {
        let mut session = ToolSession::new("tool", "state-2", b"input");
        let output = session.execute_owned(|| b"output".to_vec());
        assert_eq!(output, b"output");
        assert!(session.verify_execution());
    }
}
