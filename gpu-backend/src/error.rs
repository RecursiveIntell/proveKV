use std::fmt;

/// Observed readiness for the checked-in CUDA codebook lookup kernel.
///
/// `Ready` proves the CUDA driver and NVRTC are loadable, the canonical
/// codebook source compiled, its module loaded, and the kernel symbol resolved.
/// It does not certify unrelated CUDA kernels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuAvailability {
    /// The crate was compiled without the `gpu` feature.
    FeatureDisabled,
    /// CUDA driver loading or device context creation failed.
    DriverUnavailable { detail: String },
    /// NVRTC could not be loaded for checked-in source compilation.
    NvrtcUnavailable { detail: String },
    /// NVRTC rejected the checked-in codebook CUDA source.
    SourceCompileFailed { detail: String },
    /// The compiled PTX could not be loaded as a CUDA module.
    ModuleLoadFailed { detail: String },
    /// The loaded module does not export the required CUDA symbol.
    KernelMissing {
        symbol: &'static str,
        detail: String,
    },
    /// The codebook lookup CUDA path is ready for its documented contract.
    Ready,
}

impl GpuAvailability {
    /// Stable machine-readable category for diagnostics and receipts.
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::FeatureDisabled => "feature_disabled",
            Self::DriverUnavailable { .. } => "driver_unavailable",
            Self::NvrtcUnavailable { .. } => "nvrtc_unavailable",
            Self::SourceCompileFailed { .. } => "source_compile_failed",
            Self::ModuleLoadFailed { .. } => "module_load_failed",
            Self::KernelMissing { .. } => "kernel_missing",
            Self::Ready => "ready",
        }
    }

    /// Optional diagnostic detail. `Ready` carries no failure detail.
    pub fn detail(&self) -> Option<&str> {
        match self {
            Self::DriverUnavailable { detail }
            | Self::NvrtcUnavailable { detail }
            | Self::SourceCompileFailed { detail }
            | Self::ModuleLoadFailed { detail }
            | Self::KernelMissing { detail, .. } => Some(detail),
            Self::FeatureDisabled | Self::Ready => None,
        }
    }
}

/// GPU backend errors.
#[derive(Debug)]
pub enum GpuError {
    /// No CUDA device available.
    GpuUnavailable,
    /// Device memory insufficient.
    OutOfMemory { requested: usize, available: usize },
    /// Input dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// CUDA runtime error.
    CudaError(String),
    /// Internal error.
    Internal(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GpuUnavailable => write!(
                f,
                "GPU unavailable — CUDA device not found or feature disabled"
            ),
            Self::OutOfMemory {
                requested,
                available,
            } => {
                write!(
                    f,
                    "GPU out of memory: requested {} bytes, available {} bytes",
                    requested, available
                )
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::InvalidConfig(msg) => write!(f, "invalid config: {}", msg),
            Self::CudaError(msg) => write!(f, "CUDA error: {}", msg),
            Self::Internal(msg) => write!(f, "internal error: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}
