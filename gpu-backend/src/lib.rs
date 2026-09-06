//! GPU backend crate for vector quantization acceleration.
//!
//! Provides CUDA kernels for:
//! - Fast Walsh-Hadamard Transform (WHT) — shared by fib-quant and turbo-quant
//! - Lloyd-Max scalar quantization — per-coordinate codebook quantization
//! - Bit-packing — compact index storage
//!
//! Feature-gated: `gpu` feature enables CUDA via cudarc.
//! Without it, this crate is a stub — all operations return `GpuUnavailable`.

use std::sync::OnceLock;

#[cfg(feature = "gpu")]
pub mod cuda;
pub mod error;
pub mod fallback;
pub mod simd_nearest;

pub use error::{GpuAvailability, GpuError};

/// Result type for GPU operations.
pub type Result<T> = std::result::Result<T, GpuError>;

/// Canonical checked-in CUDA source for the only kernel activated in this pass.
pub(crate) const CODEBOOK_LOOKUP_CU: &str = include_str!("../kernels/codebook_lookup.cu");

/// Global GPU initialization result — initialized once and shared across crates.
/// The error is retained so callers can distinguish missing hardware from a
/// missing compiler, failed source compilation, or incomplete module.
static GPU_CTX: OnceLock<std::result::Result<GpuContext, GpuAvailability>> = OnceLock::new();

/// GPU context holding device, stream, and compiled kernels.
#[derive(Debug)]
pub struct GpuContext {
    /// CUDA device index
    pub device_index: u32,
    /// Available device memory in bytes
    pub memory_bytes: usize,
    /// Device name
    pub device_name: String,
}

impl GpuContext {
    /// Initialize the codebook lookup CUDA context, if it is fully ready.
    pub fn init() -> Option<&'static GpuContext> {
        GPU_CTX
            .get_or_init(|| {
                #[cfg(feature = "gpu")]
                {
                    cuda::init_context()
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Err(GpuAvailability::FeatureDisabled)
                }
            })
            .as_ref()
            .ok()
    }

    /// Return the exact readiness state for the activated codebook kernel.
    pub fn availability() -> GpuAvailability {
        let _ = Self::init();
        match GPU_CTX
            .get()
            .expect("GPU initialization result must be cached")
        {
            Ok(_) => GpuAvailability::Ready,
            Err(availability) => availability.clone(),
        }
    }

    /// True only when [`Self::availability`] is [`GpuAvailability::Ready`].
    pub fn is_available() -> bool {
        matches!(Self::availability(), GpuAvailability::Ready)
    }

    /// Minimum batch size for GPU to be worth the launch overhead.
    pub const GPU_MIN_BATCH_SIZE: usize = 16;
    /// Minimum dimension for GPU acceleration (small dims are faster on CPU).
    pub const GPU_MIN_DIM: usize = 64;
}

/// Batched Hadamard Walsh-Hadamard Transform.
///
/// Applies in-place WHT to `n` vectors of length `dim`.
/// `dim` must be a power of 2. Pad input before calling.
/// CUDA Hadamard is not activated until its complete parity contract is
/// independently re-established; this pass always selects the CPU reference.
pub fn hadamard_batch(data: &mut [f32], n: usize, dim: usize, seed: u64) -> Result<()> {
    if data.len() != n * dim {
        return Err(GpuError::DimensionMismatch {
            expected: n * dim,
            got: data.len(),
        });
    }

    fallback::hadamard_batch_cpu(data, n, dim, seed)
}

/// Batched Lloyd-Max quantization.
///
/// Quantizes a block of `n` vectors, each of `dim` scalars, into
/// `n_levels` codebook entries per block of size `k`.
///
/// Returns (indices, norms) where:
/// - indices: flat u8 array of length n * (dim / k) — codebook indices
/// - norms: flat f32 array of length n * (dim / k) — per-block L2 norms
pub fn lloyd_max_batch(
    vectors: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    n_levels: usize,
    seed: u64,
) -> Result<(Vec<u8>, Vec<f32>)> {
    if vectors.len() != n * dim {
        return Err(GpuError::DimensionMismatch {
            expected: n * dim,
            got: vectors.len(),
        });
    }
    if dim % k != 0 {
        return Err(GpuError::InvalidConfig(format!(
            "dim ({}) must be divisible by k ({})",
            dim, k
        )));
    }

    // CUDA Lloyd-Max is deliberately not activated in this pass.
    fallback::lloyd_max_batch_cpu(vectors, n, dim, k, n_levels, seed)
}

/// Batched Lloyd-Max decode.
///
/// Reconstructs approximate f32 vectors from quantized indices and norms.
pub fn lloyd_max_decode_batch(
    indices: &[u8],
    norms: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    n_levels: usize,
    seed: u64,
) -> Result<Vec<f32>> {
    let blocks_per_vector = dim / k;
    if indices.len() != n * blocks_per_vector * k {
        return Err(GpuError::DimensionMismatch {
            expected: n * blocks_per_vector * k,
            got: indices.len(),
        });
    }

    // CUDA Lloyd-Max decode is deliberately not activated in this pass.
    fallback::lloyd_max_decode_batch_cpu(indices, norms, n, dim, k, n_levels, seed)
}

/// Bit-pack quantized indices into compact byte array.
///
/// Input: flat u8 array where each byte is a codebook index (0..n_levels-1).
/// Output: packed bytes using `bits_per_index` bits per index.
pub fn bitpack(indices: &[u8], bits_per_index: usize) -> Result<Vec<u8>> {
    if bits_per_index == 0 || bits_per_index > 8 {
        return Err(GpuError::InvalidConfig(format!(
            "bits_per_index must be 1-8, got {}",
            bits_per_index
        )));
    }

    // CUDA bitpacking is deliberately not activated in this pass.
    fallback::bitpack_cpu(indices, bits_per_index)
}

/// Nearest-codeword index lookup for fib-quant / vector quantization.
///
/// For each (vector, sub-block) pair in `input` (shape `[n × d]`, row-major
/// f32), finds the index of the codeword in `codebook` (shape `[N × k]`,
/// row-major f32) that minimizes the squared L2 distance. Returns
/// `n * (d / k)` u32 indices in row-major (vector, sub-block) order.
///
/// Uses GPU only for the proved kernel contract: `k == 4` and exactly 32
/// codewords in one warp. Other valid shapes use the CPU reference.
///
/// This is the operation that dominates fib-quant's `encode_batch` after
/// the Hadamard rotation — for k=4, N=32, d=128, n=80 it runs ~1.5M
/// argmin computations and is the bottleneck of the proveKV pool build.
pub fn codebook_lookup_batch(
    input: &[f32],
    codebook: &[f32],
    n: usize,
    d: usize,
    k: usize,
) -> Result<Vec<u32>> {
    let expected_input_len = n
        .checked_mul(d)
        .ok_or_else(|| GpuError::InvalidConfig("n * d overflows usize".into()))?;
    if input.len() != expected_input_len {
        return Err(GpuError::DimensionMismatch {
            expected: expected_input_len,
            got: input.len(),
        });
    }
    if k == 0 {
        return Err(GpuError::InvalidConfig(
            "k must be greater than zero".into(),
        ));
    }
    if d % k != 0 {
        return Err(GpuError::InvalidConfig(format!(
            "dim ({}) must be divisible by k ({})",
            d, k
        )));
    }
    if codebook.is_empty() || codebook.len() % k != 0 {
        return Err(GpuError::InvalidConfig(format!(
            "codebook length ({}) must be a non-zero multiple of k ({})",
            codebook.len(),
            k
        )));
    }
    if codebook.iter().any(|value| !value.is_finite()) {
        return Err(GpuError::InvalidConfig(
            "codebook values must all be finite".into(),
        ));
    }
    // The CUDA reduction does not define a stable ordering for NaN distances.
    // Preserve the established CPU behavior instead of treating a non-finite
    // input as eligible for the narrowly proved GPU contract.
    if input.iter().any(|value| !value.is_finite()) {
        return fallback::codebook_lookup_cpu(input, codebook, n, d, k);
    }
    #[cfg(feature = "gpu")]
    {
        let n_codewords = codebook.len() / k;
        if codebook_lookup_supports_gpu(n, d, k, n_codewords) {
            let ctx = GpuContext::init().expect("ready availability must expose a CUDA context");
            return cuda::codebook_lookup_batch_gpu(ctx, input, codebook, n, d, k);
        }
    }

    fallback::codebook_lookup_cpu(input, codebook, n, d, k)
}

/// True if a specific call to [`codebook_lookup_batch`] would dispatch to GPU.
/// The checked-in kernel currently proves only the `k=4`, exactly-32-codeword
/// warp contract; other valid shapes retain the CPU reference path.
pub fn codebook_lookup_supports_gpu(n: usize, d: usize, k: usize, n_codewords: usize) -> bool {
    k == 4
        && n_codewords == 32
        && n >= GpuContext::GPU_MIN_BATCH_SIZE
        && d >= GpuContext::GPU_MIN_DIM
        && i32::try_from(n).is_ok()
        && i32::try_from(d).is_ok()
        && n.checked_mul(d / k)
            .is_some_and(|total_blocks| u32::try_from(total_blocks).is_ok())
        && GpuContext::is_available()
}

/// Blake3 digest of the checked-in source compiled by NVRTC when readiness is
/// `Ready`. This is evidence linkage, not a substitute for the HIL parity gate.
pub fn codebook_kernel_source_digest() -> String {
    blake3::hash(CODEBOOK_LOOKUP_CU.as_bytes())
        .to_hex()
        .to_string()
}

#[cfg(all(test, feature = "gpu"))]
mod gpu_parity_tests {
    use super::*;
    use crate::fallback::codebook_lookup_cpu;

    /// When the GPU is available and the kernel can be reached, the GPU
    /// result must be byte-identical to the CPU fallback. This is the
    /// receipt-honesty test: if it fails, the GPU path is silently
    /// producing different codebook indices and pool digests will drift.
    #[test]
    fn test_codebook_lookup_gpu_matches_cpu() {
        if !GpuContext::is_available() {
            // Skip — GPU not present in this build environment
            eprintln!("GPU not available, skipping parity test");
            return;
        }

        // Build a deterministic test case.
        let n: usize = 32;
        let d: usize = 128;
        let k: usize = 4;
        let n_codewords: usize = 32;
        let mut seed: u64 = 0xDEAD_BEEF;
        let next = |s: &mut u64| {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *s
        };

        // Random codebook and inputs.
        let codebook: Vec<f32> = (0..n_codewords * k)
            .map(|_| {
                let v = (next(&mut seed) >> 32) as i32;
                (v as f32 / i32::MAX as f32) * 0.5
            })
            .collect();
        let input: Vec<f32> = (0..n * d)
            .map(|_| {
                let v = (next(&mut seed) >> 32) as i32;
                (v as f32 / i32::MAX as f32) * 0.5
            })
            .collect();

        let cpu = codebook_lookup_cpu(&input, &codebook, n, d, k).expect("cpu fallback failed");
        let gpu = codebook_lookup_batch(&input, &codebook, n, d, k).expect("gpu dispatch failed");

        assert_eq!(
            cpu.len(),
            gpu.len(),
            "GPU returned {} indices, CPU returned {}",
            gpu.len(),
            cpu.len()
        );
        let mismatches = cpu.iter().zip(gpu.iter()).filter(|(a, b)| a != b).count();
        assert_eq!(
            mismatches,
            0,
            "GPU codebook lookup produced {} differing indices out of {}",
            mismatches,
            cpu.len()
        );
    }
}

/// AVX2-accelerated nearest-codeword index lookup (CPU).
///
/// Returns the index of the codeword in `codebook` (row-major f32, shape
/// `[N × k]`) that minimizes the squared L2 distance from `sample`.
///
/// On x86_64 with AVX2+FMA, this runs ~4-8× faster than a naive scalar
/// loop for the k=4 case. Falls back to a scalar loop on other platforms
/// or when AVX2 isn't available at runtime.
///
/// The result is byte-identical to a scalar f32 reference within f32
/// precision. fib-quant's parity tests assert that this matches the
/// canonical f64 reference for trained Lloyd-Max codebooks.
pub fn nearest_codeword_f32(sample: &[f32], codebook: &[f32], k: usize) -> usize {
    simd_nearest::nearest_codeword_f32(sample, codebook, k)
}

#[cfg(test)]
mod readiness_tests {
    use super::{codebook_lookup_batch, GpuAvailability, GpuContext, GpuError};
    use crate::fallback::codebook_lookup_cpu;

    #[test]
    #[cfg(not(feature = "gpu"))]
    fn disabled_feature_reports_feature_disabled() {
        assert_eq!(GpuContext::availability(), GpuAvailability::FeatureDisabled);
        assert!(!GpuContext::is_available());
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn gpu_feature_availability_does_not_panic_when_cuda_is_unavailable() {
        let availability = std::panic::catch_unwind(GpuContext::availability)
            .expect("unavailable CUDA runtime must return typed availability, not panic");

        assert_eq!(
            GpuContext::is_available(),
            matches!(availability, GpuAvailability::Ready),
        );
    }

    #[test]
    fn non_finite_codebook_is_rejected_before_cuda_dispatch() {
        let input = vec![0.0; 4];
        let mut codebook = vec![0.0; 32 * 4];
        codebook[0] = f32::NAN;

        let error = codebook_lookup_batch(&input, &codebook, 1, 4, 4)
            .expect_err("non-finite codebook must not reach CPU or CUDA lookup");

        assert!(matches!(error, GpuError::InvalidConfig(message) if message.contains("finite")));
    }

    #[test]
    fn non_finite_input_uses_cpu_reference_before_cuda_dispatch() {
        let n = 16;
        let d = 64;
        let k = 4;
        let mut input = vec![0.0; n * d];
        input[0] = f32::NAN;
        let codebook = vec![0.0; 32 * k];

        let expected = codebook_lookup_cpu(&input, &codebook, n, d, k)
            .expect("CPU reference must accept the non-finite input fixture");
        let actual = codebook_lookup_batch(&input, &codebook, n, d, k)
            .expect("non-finite input must preserve CPU fallback behavior");

        assert_eq!(actual, expected);
    }

    #[test]
    fn overflowing_input_shape_is_rejected_before_length_comparison() {
        let error = codebook_lookup_batch(&[], &[0.0; 4], usize::MAX, 2, 4)
            .expect_err("overflowing input shape must be rejected");

        assert!(matches!(error, GpuError::InvalidConfig(message) if message.contains("overflows")));
    }
}
