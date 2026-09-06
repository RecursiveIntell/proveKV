#![allow(
    unused_imports,
    unused_variables,
    unused_mut,
    dead_code,
    clippy::too_many_arguments,
    clippy::derivable_impls,
    clippy::bool_comparison,
    clippy::useless_conversion,
    clippy::needless_range_loop
)]
//! proveKV: Shared compressed KV-cache pool for multi-agent context.
//!
//! ## Architecture
//!
//! A two-tier compression strategy derived from empirical benchmarks.
//! Current N=8 size ratios use eight independent contexts of 800 shared +
//! 28 unique tokens. PPL is separately checked on one aggregate 1024-token
//! SmolLM2-1.7B + WikiText-2 fixture over exact targets [800, 1024). That
//! cache-aligned shared-prefix continuation check degraded from 7.203125 to
//! 24.234375 (+236.44%). It does not consume reconstructed shell K/V, so it
//! is neither a PPL-neutrality claim nor independent-agent quality evidence.
//!
//! - **Shared pool (cold tier):** fib-quant at k=4, N=32 (FB2 batched
//!   wire format). Measured 21.33x ratio on the shared prefix alone,
//!   vs an f32-raw KV baseline. The fib codec is a codebook-based
//!   vector quantizer; reconstruction error per vector is bounded by the
//!   k=4 codebook resolution, not lossless. Historical single-pool PPL
//!   receipts are scoped separately; the current N=8 two-tier defaults do
//!   not pass the PPL-neutrality gate. See CLAIMS.json for the exact scope.
//! - **Agent shells (hot tier):** turbo-quant at b=4 (TQB1 batched wire
//!   format), 32 projections. Measured 160 B/vec at the lossless
//!   profile (f32 radii) and 72 B/vec at the lossy profile (BlockLogU8
//!   radii). At the b=4 default, the byte-derived N=8 size result is
//!   40.50x lossless and 76.54x lossy versus the explicit f32 baseline.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use provekv::{SharedKVPool, KvTensorShape, AttentionType};
//!
//! let shape = KvTensorShape {
//!     attention_type: AttentionType::MHA,
//!     num_layers: 32,
//!     num_heads: 32,
//!     num_kv_heads: 32,
//!     head_dim: 128,
//!     hidden_size: 4096,
//! };
//!
//! let corpus: Vec<(String, Vec<f32>)> = vec![
//!     ("tok_0".into(), vec![0.1; shape.num_layers as usize * 32 * 128 * 2]),
//! ];
//!
//! let (pool, receipt) = SharedKVPool::build(&corpus, &shape, 42).unwrap();
//! let (shell, mat_receipt) = pool.materialize_shell("agent_1", &[], 42).unwrap();
//! ```

pub mod apps;
pub mod branch;
#[cfg(feature = "bridge")]
pub mod bridge;
pub mod codec;
pub mod error;
pub mod fallback;
pub mod gc;
pub mod hybrid_manifest;
pub mod lease;
pub mod limits;
pub mod manifest;
#[cfg(feature = "mcp-server")]
pub mod mcp_server;
pub mod page_format;
pub mod page_store;
pub mod policy;
pub mod pool;
pub mod principal;
pub mod receipt;
pub mod recovery;
pub mod shape;
pub mod shell;
pub mod state_id;
pub mod state_policy;
pub mod state_reuse;
pub mod state_store;
pub mod transport;

pub use transport::PageTransfer;
pub mod stream;

pub use stream::{Delivery, Snapshot, StreamProcessor};

// Re-export core types
pub use codec::{create_codec, CompressedBlock, KVecCodec};
pub use error::{ProveKvError, Result};
pub use hybrid_manifest::{
    HybridComponent, HybridPageRef, HybridStateManifestV1, HYBRID_MANIFEST_SCHEMA,
};
pub use lease::{LeaseRight, LeaseRights, LeaseStatus, StateLease};
pub use limits::ResourceLimits;
pub use manifest::{
    PoolManifest, ShellComponentKind, ShellManifest, POOL_MANIFEST_SCHEMA, SHELL_MANIFEST_SCHEMA,
};
pub use policy::{
    CodecId, CompressionPolicy, FibConfig, TurboConfig, CODEC_EXACT_FALLBACK, CODEC_FIB_K4_N32,
    CODEC_TURBO_8BIT,
};
pub use pool::{CacheTarget, DecompressedLayer, PoolLayer, SharedKVPool};
pub use principal::{ExecutionScope, Principal};
pub use receipt::{
    BlockInjectionTrace, InjectionReceipt, PoolBuildReceipt, ShellMaterializeReceipt,
    INJECTION_RECEIPT_SCHEMA, POOL_BUILD_RECEIPT_SCHEMA, RECEIPT_SCHEMA,
    SHELL_MATERIALIZE_RECEIPT_SCHEMA,
};
pub use shape::{AttentionType, KvTensorShape};
pub use shell::{AgentShell, ShellLayer};
pub use state_id::HybridStateId;
pub use state_store::StateStore;
