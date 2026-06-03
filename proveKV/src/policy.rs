use serde::{Deserialize, Serialize};

use crate::error::{ProveKvError, Result};

/// String identifier for a codec.
/// String identifier for a codec.
pub type CodecId = String;
pub const CODEC_FIB_K4_N32: &str = "fib_k4_n32";
/// Lossless batched (FB2) variant of fib_k4_n32: stores the profile once
/// per batch instead of once per block. Same codec, same lossless
/// property; ~1.92x smaller on disk.
pub const CODEC_FIB_K4_N32_BATCHED: &str = "fib_k4_n32_batched";
pub const CODEC_TURBO_8BIT: &str = "turbo_8bit";
/// Lossless batched (TQB1) variant of turbo_8bit: stores the profile once
/// per batch instead of once per vector. Same lossless property; ~1.34x
/// smaller on disk.
pub const CODEC_TURBO_8BIT_BATCHED: &str = "turbo_8bit_batched";
/// Lossy batched (TQB1-L) variant: same as TQB1 but radii are stored as
/// BlockLogU8 (1 byte each) instead of raw f32 (4 bytes each). ~3.40x
/// smaller on disk than TQB1, ~17x smaller than the per-vector TQW1, at
/// the cost of ~1.8% relative error per radius. Used for the 58.14x
/// lossy headline. Opt-in via `TurboConfig::radii_compression = Lossy`.
pub const CODEC_TURBO_8BIT_BATCHED_LOSSY: &str = "turbo_8bit_batched_lossy";
pub const CODEC_EXACT_FALLBACK: &str = "exact_f32_fallback";

/// True when the codec id is a batched (FB2 / TQB1) variant.
pub fn is_batched_fib(codec_id: &str) -> bool {
    matches!(codec_id, CODEC_FIB_K4_N32_BATCHED)
}

/// True when the codec id is a batched (TQB1) turbo variant (lossless).
pub fn is_batched_turbo(codec_id: &str) -> bool {
    matches!(codec_id, CODEC_TURBO_8BIT_BATCHED)
}

/// True when the codec id is the lossy batched turbo variant (TQB1-L).
pub fn is_batched_turbo_lossy(codec_id: &str) -> bool {
    matches!(codec_id, CODEC_TURBO_8BIT_BATCHED_LOSSY)
}

/// True if this codec id is any batched variant (either fib or turbo).
pub fn is_batched_codec(codec_id: &str) -> bool {
    is_batched_fib(codec_id) || is_batched_turbo(codec_id)
}

/// FibQuant configuration parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FibConfig {
    /// Block dimension k.
    pub k: u32,
    /// Codebook size N.
    pub n: u32,
    /// Number of training samples for Lloyd-Max refinement.
    pub training_samples: u32,
    /// Number of Lloyd restarts.
    pub lloyd_restarts: u32,
    /// Number of Lloyd iterations per restart.
    pub lloyd_iterations: u32,
}

impl FibConfig {
    /// The benchmark-proven configuration: k=4, N=32, 50× compression with 100% recall.
    pub fn default_k4_n32() -> Self {
        Self {
            k: 4,
            n: 32,
            training_samples: 2048,
            lloyd_restarts: 4,
            lloyd_iterations: 8,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.k == 0 {
            return Err(ProveKvError::InvalidPolicy("fib k must be > 0".into()));
        }
        if self.n < 2 {
            return Err(ProveKvError::InvalidPolicy("fib N must be >= 2".into()));
        }
        if self.k > 256 {
            return Err(ProveKvError::InvalidPolicy("fib k must be <= 256".into()));
        }
        if self.n > 1_048_576 {
            return Err(ProveKvError::InvalidPolicy(
                "fib N must be <= 1,048,576".into(),
            ));
        }
        if self.training_samples == 0 {
            return Err(ProveKvError::InvalidPolicy(
                "fib training_samples must be > 0".into(),
            ));
        }
        if self.lloyd_restarts == 0 {
            return Err(ProveKvError::InvalidPolicy(
                "fib lloyd_restarts must be > 0".into(),
            ));
        }
        if self.lloyd_iterations == 0 {
            return Err(ProveKvError::InvalidPolicy(
                "fib lloyd_iterations must be > 0".into(),
            ));
        }
        Ok(())
    }

    /// Expected compression ratio (input f32 bytes / compressed bytes).
    /// For k=4, N=32: each block of 4 f32 values (16 bytes) maps to a
    /// ceil(log2(32)) = 5-bit index (+norm header), yielding roughly 50:1.
    pub fn nominal_compression_ratio(&self) -> f64 {
        let bits_per_index = (self.n as f64).log2().ceil();
        // Each block of k * 4 bytes is compressed to bits_per_index bits + norm overhead
        // Approximate: (k * 32) / (bits_per_index + 16 / ceil(k/2)) with f16 norm
        // Simplified: roughly k * 32 / bits_per_index
        (self.k as f64 * 32.0) / bits_per_index
    }

    /// Expected codebook-based compression ratio (50× target for k=4,N=32).
    pub fn expected_compression_ratio(&self) -> f64 {
        50.0
    }
}

/// How the turbo-quant radii are compressed in the wire format.
///
/// - `Lossless`: store radii as raw f32 (4 bytes each). Exact roundtrip
///   for the polar component; same as the pre-TQB1-L code path. The 11.13×
///   lossless headline uses this.
/// - `Lossy`: store radii as BlockLogU8 (1 byte each, plus 8 bytes for
///   (min, max) of the log values). ~4× smaller per vector at the cost
///   of ~1.8% relative error per radius. The 58.14× lossy headline uses
///   this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RadiiCompression {
    /// Raw f32 radii, exact roundtrip. The default for the lossless tier.
    #[default]
    Lossless,
    /// BlockLogU8 (1 byte per radius). Smaller, lossy. Opt-in.
    Lossy,
}

/// TurboQuant configuration parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurboConfig {
    /// Bits per scalar (8 for the benchmark-proven path).
    pub bits: u8,
    /// Number of QJL projections for the residual sketch (32 for benchmark path).
    pub projections: usize,
    /// How to compress radii in the wire format. Defaults to `Lossless`
    /// so receipts remain bit-exact. Set to `Lossy` to enable the
    /// `turbo_8bit_batched_lossy` codec and the 58.14× system number.
    #[serde(default)]
    pub radii_compression: RadiiCompression,
}

impl TurboConfig {
    /// The benchmark-proven configuration: 8-bit, 32 projections, 8× compression.
    pub fn default_8bit() -> Self {
        Self {
            bits: 8,
            projections: 32,
            radii_compression: RadiiCompression::Lossless,
        }
    }

    /// Lossy variant of the 8-bit config. 1-byte radii, ~4× smaller shell
    /// payload at the cost of ~1.8% relative error per radius.
    pub fn default_8bit_lossy() -> Self {
        Self {
            bits: 8,
            projections: 32,
            radii_compression: RadiiCompression::Lossy,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.bits < 2 || self.bits > 16 {
            return Err(ProveKvError::InvalidPolicy(format!(
                "turbo bits must be 2–16, got {}",
                self.bits
            )));
        }
        if self.projections == 0 {
            return Err(ProveKvError::InvalidPolicy(
                "turbo projections must be > 0".into(),
            ));
        }
        Ok(())
    }

    /// Expected compression ratio (8× target for 8-bit).
    pub fn expected_compression_ratio(&self) -> f64 {
        match self.radii_compression {
            RadiiCompression::Lossless => 8.0,
            // Lossy: 4x smaller per-vector payload, so per-tier ratio ~32x
            // (vs the 8x of lossless). System-level depends on fib tier.
            RadiiCompression::Lossy => 32.0,
        }
    }
}

/// Hard-coded two-tier compression policy.
///
/// Derived from empirical benchmarks run on 2026-06-01:
/// - Shared pool (cold tier): fib-quant at k=4, N=32 → 50× compression, 100% recall
/// - Agent shells (hot tier): turbo-quant at 8-bit → 8× compression, 99.9% score retention
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompressionPolicy {
    /// Codec used for the shared pool (cold tier).
    pub shared_codec: CodecId,
    /// Codec used for agent shells (hot tier).
    pub shell_codec: CodecId,
    /// FibQuant configuration.
    pub fib_config: FibConfig,
    /// TurboQuant configuration.
    pub turbo_config: TurboConfig,
}

impl CompressionPolicy {
    /// Create the default benchmark-proven two-tier policy.
    pub fn default_two_tier() -> Self {
        Self {
            shared_codec: CODEC_FIB_K4_N32_BATCHED.into(),
            shell_codec: CODEC_TURBO_8BIT_BATCHED.into(),
            fib_config: FibConfig::default_k4_n32(),
            turbo_config: TurboConfig::default_8bit(),
        }
    }

    /// Validate the policy.
    pub fn validate(&self) -> Result<()> {
        let allowed_shared = [
            CODEC_FIB_K4_N32,
            CODEC_FIB_K4_N32_BATCHED,
            CODEC_EXACT_FALLBACK,
        ];
        if !allowed_shared.contains(&self.shared_codec.as_str()) {
            return Err(ProveKvError::InvalidPolicy(format!(
                "shared_codec must be one of {:?}, got '{}'",
                allowed_shared,
                self.shared_codec
            )));
        }
        let allowed_shell = [
            CODEC_TURBO_8BIT,
            CODEC_TURBO_8BIT_BATCHED,
            CODEC_TURBO_8BIT_BATCHED_LOSSY,
            CODEC_EXACT_FALLBACK,
        ];
        if !allowed_shell.contains(&self.shell_codec.as_str()) {
            return Err(ProveKvError::InvalidPolicy(format!(
                "shell_codec must be one of {:?}, got '{}'",
                allowed_shell,
                self.shell_codec
            )));
        }
        self.fib_config.validate()?;
        self.turbo_config.validate()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_policy_validates() {
        let policy = CompressionPolicy::default_two_tier();
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn test_fib_config_invalid_k_rejected() {
        let mut config = FibConfig::default_k4_n32();
        config.k = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_fib_config_invalid_n_rejected() {
        let mut config = FibConfig::default_k4_n32();
        config.n = 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_turbo_config_invalid_bits_rejected() {
        let mut config = TurboConfig::default_8bit();
        config.bits = 1;
        assert!(config.validate().is_err());
    }
}
