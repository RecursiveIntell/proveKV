//! Application-level adapters built on proveKV primitives.

pub mod agent;
#[cfg(feature = "bridge")]
pub mod chat;
pub mod context;
pub mod memory;
pub mod rag;
pub mod tool;
pub mod workflow;

#[cfg(feature = "bridge")]
pub use chat::{capture_conversation_state, ChatSession};
pub use context::{ContextWindow, WindowCapture};
pub use tool::ToolSession;
