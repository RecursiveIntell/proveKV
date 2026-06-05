//! proveKV: Shared compressed KV-cache pool for multi-agent context.
//!
//! ## Architecture
//!
//! A two-tier compression strategy derived from empirical benchmarks.
//! All ratios below are PPL-validated on SmolLM2-1.7B + WikiText-2 at
//! N=8 agents, 800 shared + 28×8 unique tokens, 1024 tokens total
//! (msi 2026-06-03). `delta_ppl_pct = +0.00%` in every run.
//!
//! - **Shared pool (cold tier):** fib-quant at k=4, N=32 (FB2 batched
//!   wire format). Measured 21.33x ratio on the shared prefix alone,
//!   vs an f32-raw KV baseline. The fib codec is a codebook-based
//!   vector quantizer; reconstruction error per vector is bounded by the
//!   k=4 codebook resolution, not lossless. The PPL-validated claim is
//!   "PPL-neutral on the measured configurations" — see CLAIMS.json
//!   for the per-baseline ratio breakdown.
//! - **Agent shells (hot tier):** turbo-quant at b=4 (TQB1 batched wire
//!   format), 32 projections. Measured 160 B/vec at the lossless
//!   profile (f32 radii) and 72 B/vec at the lossy profile (BlockLogU8
//!   radii). At the b=4 default, the system achieves 36.00x lossless
//!   and 68.04x lossy PPL-validated system-level compression at N=8.
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

pub mod codec;
pub mod error;
pub mod fallback;
pub mod manifest;
pub mod policy;
pub mod pool;
pub mod receipt;
pub mod shape;
pub mod shell;

// Re-export core types
pub use codec::{create_codec, CompressedBlock, KVecCodec};
pub use error::{ProveKvError, Result};
pub use manifest::{PoolManifest, ShellManifest, POOL_MANIFEST_SCHEMA, SHELL_MANIFEST_SCHEMA};
pub use policy::{
    CodecId, CompressionPolicy, FibConfig, TurboConfig, CODEC_EXACT_FALLBACK, CODEC_FIB_K4_N32,
    CODEC_TURBO_8BIT,
};
pub use pool::{CacheTarget, DecompressedLayer, PoolLayer, SharedKVPool};
pub use receipt::{
    BlockInjectionTrace, InjectionReceipt, PoolBuildReceipt, ShellMaterializeReceipt,
    INJECTION_RECEIPT_SCHEMA, POOL_BUILD_RECEIPT_SCHEMA, RECEIPT_SCHEMA,
    SHELL_MATERIALIZE_RECEIPT_SCHEMA,
};
pub use shape::{AttentionType, KvTensorShape};
pub use shell::{AgentShell, ShellLayer};
