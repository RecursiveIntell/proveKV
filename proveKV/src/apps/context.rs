//! Sliding context-window adapter for independently verifiable KV state.

/// A captured KV state at a context-window boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowCapture {
    /// Zero-based window number.
    pub window_index: usize,
    /// Number of tokens represented by this capture.
    pub token_count: usize,
    /// The independently captured KV payload.
    pub state: Vec<u8>,
    /// Content digest of `state`.
    pub digest: [u8; 32],
}

impl WindowCapture {
    /// Verify that the captured payload still matches its recorded identity.
    pub fn verify(&self) -> bool {
        *blake3::hash(&self.state).as_bytes() == self.digest
    }
}

/// Maintains the active context window and captures it when it slides.
#[derive(Debug, Clone)]
pub struct ContextWindow {
    /// Maximum number of tokens in one window.
    pub max_tokens: usize,
    /// KV state accumulated in the active window.
    pub current_state: Vec<u8>,
    captures: Vec<WindowCapture>,
    token_count: usize,
    next_window: usize,
}

impl ContextWindow {
    /// Create an empty context window with the given token capacity.
    pub fn new(max_tokens: usize) -> Self {
        assert!(
            max_tokens > 0,
            "context window must hold at least one token"
        );
        Self {
            max_tokens,
            current_state: Vec::new(),
            captures: Vec::new(),
            token_count: 0,
            next_window: 0,
        }
    }

    /// Add one token's KV payload. A full window is captured automatically.
    pub fn push(&mut self, kv: &[u8]) -> Option<WindowCapture> {
        self.current_state.extend_from_slice(kv);
        self.token_count += 1;
        (self.token_count == self.max_tokens).then(|| self.slide())
    }

    /// Capture the active window and begin a fresh independent window.
    pub fn slide(&mut self) -> WindowCapture {
        let capture = WindowCapture {
            window_index: self.next_window,
            token_count: self.token_count,
            state: self.current_state.clone(),
            digest: *blake3::hash(&self.current_state).as_bytes(),
        };
        self.next_window += 1;
        self.captures.push(capture.clone());
        self.current_state.clear();
        self.token_count = 0;
        capture
    }

    /// Captures produced so far, in window order.
    pub fn captures(&self) -> &[WindowCapture] {
        &self.captures
    }

    /// Number of tokens in the active window.
    pub fn token_count(&self) -> usize {
        self.token_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_window_is_captured_and_active_state_reset() {
        let mut window = ContextWindow::new(2);
        assert!(window.push(b"a").is_none());
        let capture = window.push(b"b").expect("window boundary");
        assert_eq!(capture.state, b"ab");
        assert!(capture.verify());
        assert!(window.current_state.is_empty());
        assert_eq!(window.token_count(), 0);
    }

    #[test]
    fn windows_are_independent_and_verifiable() {
        let mut window = ContextWindow::new(1);
        let first = window.push(b"one").unwrap();
        let second = window.push(b"two").unwrap();
        assert_eq!(first.window_index, 0);
        assert_eq!(second.window_index, 1);
        assert_ne!(first.digest, second.digest);
        assert!(window.captures().iter().all(WindowCapture::verify));
    }

    #[test]
    fn partial_window_can_be_slid_explicitly() {
        let mut window = ContextWindow::new(4);
        window.push(b"x");
        let capture = window.slide();
        assert_eq!(capture.token_count, 1);
        assert_eq!(window.captures().len(), 1);
    }
}
