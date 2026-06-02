#![warn(rustdoc::broken_intra_doc_links)]

//! Experimental paper-core FibQuant math crate.
//!
//! This crate implements the normalize, deterministic rotation,
//! spherical-Beta block source, radial-angular codebook, Lloyd-Max refinement,
//! and fixed-rate codec path described in `FibQuant: Universal Vector
//! Quantization for Random-Access KV-Cache Compression`.
//!
//! The `0.1.0-alpha.1` surface is deliberately narrow. It is not a production
//! KV-cache compressor, not a benchmark reproduction package, and not
//! integrated with any parent workspace memory crate. Profiles are validated
//! against explicit alpha resource limits before allocation-heavy paths run.
//!
//! ```
//! use fib_quant::{FibQuantProfileV1, FibQuantizer};
//!
//! # fn main() -> fib_quant::Result<()> {
//! let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 7)?;
//! profile.training_samples = 128;
//! profile.lloyd_restarts = 1;
//! profile.lloyd_iterations = 2;
//! let quantizer = FibQuantizer::new(profile)?;
//! let input = vec![0.25, -0.5, 0.75, 1.0, -1.25, 0.5, 0.125, -0.875];
//! let code = quantizer.encode(&input)?;
//! let decoded = quantizer.decode(&code)?;
//! assert_eq!(decoded.len(), input.len());
//! # Ok(())
//! # }
//! ```

pub mod beta_inv;
pub mod bitpack;
pub mod codebook;
pub mod codec;
pub mod digest;
pub mod directions;
pub mod error;
#[cfg(feature = "kv")]
pub mod kv;
pub mod lloyd;
pub mod metrics;
pub mod profile;
pub mod receipt;
pub mod rotation;
pub mod spherical_beta;

pub use codebook::{build_initial_codebook, FibCodebookV1};
pub use codec::{FibCodeV1, FibQuantizer, GpuStepReport, COMPACT_MAGIC, COMPACT_VERSION};
pub use directions::{fibonacci_sphere_3d, fibonacci_spiral_2d, roberts_kronecker};
pub use error::{FibQuantError, Result};
pub use lloyd::{LloydRepairEventV1, LloydReportV1};
pub use profile::{
    DirectionMethod, EmptyCellPolicy, FibQuantProfileV1, NormFormat, RadiusMethod, SourceMode,
    MAX_AMBIENT_DIM, MAX_BLOCK_DIM, MAX_CODEBOOK_SIZE, MAX_CODEBOOK_VALUES, MAX_PACKED_INDEX_BITS,
    MAX_ROTATION_MATRIX_VALUES, MAX_TRAINING_SAMPLES,
};
pub use receipt::FibQuantCompressionReceiptV1;
pub use rotation::{StoredRotation, ROTATION_ALGORITHM_VERSION, ROTATION_SCHEMA};
pub use spherical_beta::{
    beta_d_k, radius_quantile, radius_quantile_k2_closed_form, sample_reference_projection,
    sample_spherical_beta,
};
