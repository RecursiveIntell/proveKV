//! Bridge module for serializable state capture and transition types.

mod capture;
mod types;

pub use capture::CaptureContext;
pub use types::{ForkPoint, StateDelta, StateSnapshot};
