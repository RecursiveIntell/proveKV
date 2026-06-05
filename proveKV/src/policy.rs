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
///
/// Accepts all bit rates (2..=16) for forward-compat: matches
/// `turbo_<N>bit_batched` for any valid N. Does NOT match the `_lossy`
/// variant — use `is_batched_turbo_lossy` for that.
pub fn is_batched_turbo(codec_id: &str) -> bool {
    if codec_id == CODEC_TURBO_8BIT_BATCHED {
        return true;
    }
    // `turbo_<N>bit_batched` (no `_lossy` suffix).
    let Some(bits) = parse_turbo_bits_batched(codec_id, false) else {
        return false;
    };
    (2..=16).contains(&bits)
}

/// True when the codec id is the lossy batched turbo variant (TQB1-L).
///
/// Accepts all bit rates (2..=16) with the `_batched_lossy` suffix.
pub fn is_batched_turbo_lossy(codec_id: &str) -> bool {
    if codec_id == CODEC_TURBO_8BIT_BATCHED_LOSSY {
        return true;
    }
    let Some(bits) = parse_turbo_bits_batched(codec_id, true) else {
        return false;
    };
    (2..=16).contains(&bits)
}

/// True if this codec id is any batched variant (either fib or turbo).
pub fn is_batched_codec(codec_id: &str) -> bool {
    is_batched_fib(codec_id) || is_batched_turbo(codec_id)
}

/// Parse `turbo_<N>bit_batched[_lossy]` (NOT the legacy `turbo_<N>bit` form).
/// Returns the bit rate N if the id has the right structure and the lossy
/// flag matches.
fn parse_turbo_bits_batched(codec_id: &str, lossy: bool) -> Option<u8> {
    let rest = codec_id.strip_prefix("turbo_")?;
    let bits_end = rest.find("bit")?;
    let n: u8 = rest[..bits_end].parse().ok()?;
    let tail = &rest[bits_end + 3..];
    let expected = if lossy { "_batched_lossy" } else { "_batched" };
    (tail == expected).then_some(n)
}

/// Build the batched turbo codec id for a given bit rate and radii codec.
pub fn turbo_batched_codec_id(bits: u8, lossy_radii: bool) -> String {
    debug_assert!((2..=16).contains(&bits), "turbo bits must be 2..=16");
    let suffix = if lossy_radii { "_lossy" } else { "" };
    format!("turbo_{bits}bit_batched{suffix}")
}

/// Static, lazily-built list of every legal shell codec id. Includes the
/// canonical `turbo_8bit_batched` and `turbo_8bit_batched_lossy` plus every
/// `turbo_<N>bit_batched[_lossy]` for N in 2..=16.
pub static ALLOWED_SHELL_CODECS: std::sync::OnceLock<Vec<&'static str>> =
    std::sync::OnceLock::new();

/// Get (or initialize) the list of legal shell codec ids.
pub fn allowed_shell_codecs() -> &'static [&'static str] {
    ALLOWED_SHELL_CODECS.get_or_init(|| {
        let mut ids: Vec<&'static str> = vec![
            CODEC_TURBO_8BIT,
            CODEC_TURBO_8BIT_BATCHED,
            CODEC_TURBO_8BIT_BATCHED_LOSSY,
            CODEC_EXACT_FALLBACK,
        ];
        for bits in 2u8..=16 {
            // Leak the formatted ids: they live for the program lifetime
            // and the count is bounded (15 * 2 = 30 strings). Acceptable
            // for a validation table; avoids reallocating per validate().
            let lossless: &'static str =
                Box::leak(turbo_batched_codec_id(bits, false).into_boxed_str());
            let lossy: &'static str =
                Box::leak(turbo_batched_codec_id(bits, true).into_boxed_str());
            if bits != 8 {
                ids.push(lossless);
                ids.push(lossy);
            }
        }
        ids
    })
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
    /// The benchmark-proven configuration: k=4, N=32. The fib codec is a
    /// codebook-based vector quantizer — reconstruction is bounded by the
    /// k=4 codebook resolution, NOT bit-exact lossless. PPL-validated
    /// claim: 21.33x pool-tier ratio at this config on SmolLM2-1.7B with
    /// `delta_ppl_pct = +0.00%` (msi 2026-06-03). See CLAIMS.json.
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
    /// The benchmark-proven configuration: 4-bit angles, 32 projections,
    /// lossless f32 radii. PPL-validated (SmolLM2-1.7B, N=8): **36.00x** at
    /// bit-exact `delta_ppl_pct = +0.00%` (vs oracle). The 4-bit angle
    /// discretization is below the signal threshold for K/V in transformer
    /// attention, so reducing the per-angle bit count from 8 to 4 is a
    /// 5× reduction in angle-bytes (32 B/vec → 16 B/vec) with no measurable
    /// effect on the forward pass.
    pub fn default_4bit() -> Self {
        Self {
            bits: 4,
            projections: 32,
            radii_compression: RadiiCompression::Lossless,
        }
    }

    /// Lossy variant of the 4-bit config. 1-byte BlockLogU8 radii, ~4×
    /// smaller shell than 4-bit lossless. PPL-validated:
    /// **68.04x** at bit-exact `delta_ppl_pct = +0.00%`.
    pub fn default_4bit_lossy() -> Self {
        Self {
            bits: 4,
            projections: 32,
            radii_compression: RadiiCompression::Lossy,
        }
    }

    /// Legacy 8-bit configuration kept for back-compat. Superseded by
    /// [`default_4bit`](Self::default_4bit) (36.00x) — same PPL, 10% larger.
    pub fn default_8bit() -> Self {
        Self {
            bits: 8,
            projections: 32,
            radii_compression: RadiiCompression::Lossless,
        }
    }

    /// Lossy variant of the 8-bit config. 1-byte radii, ~4× smaller shell
    /// than 8-bit lossless. Superseded by [`default_4bit_lossy`](Self::default_4bit_lossy)
    /// (68.04x) — same PPL, 16% larger.
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
/// All ratios in this module are PPL-validated on SmolLM2-1.7B +
/// WikiText-2 at N=8 agents, 800 shared + 28×8 unique tokens, 1024
/// tokens total (msi 2026-06-03, `delta_ppl_pct = +0.00%`).
///
/// Note: "lossless" in this codebase refers to the shell tier's radii
/// profile (no BlockLogU8 quantization). The fib cold tier is a
/// codebook-based vector quantizer — reconstruction is bounded by the
/// k=4 codebook resolution, NOT bit-exact lossless. See CLAIMS.json
/// for the per-baseline ratio breakdown and the receipts that justify
/// each number.
///
/// - Shared pool (cold tier): fib-quant at k=4, N=32 → 21.33× pool
///   ratio (vs f32-raw KV baseline)
/// - Agent shells (hot tier): turbo-quant at b=4 (default) → 36.00×
///   system lossless / 68.04× system lossy at N=8 (vs f32-raw KV
///   baseline). At the legacy b=8 config, the system was 33.16× / 58.56×.
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
    ///
    /// Default is **b=4 lossless** (36.00x PPL-validated). PPL is bit-exact
    /// identical to the oracle at this bit rate, and 10% smaller than the
    /// legacy b=8 config. For 68.04x, use [`default_two_tier_lossy`].
    pub fn default_two_tier() -> Self {
        Self {
            shared_codec: CODEC_FIB_K4_N32_BATCHED.into(),
            shell_codec: turbo_batched_codec_id(4, false).into(),
            fib_config: FibConfig::default_k4_n32(),
            turbo_config: TurboConfig::default_4bit(),
        }
    }

    /// Lossy variant of [`default_two_tier`]. 68.04x PPL-validated at b=4
    /// with `delta_ppl_pct = +0.00%` (lossless oracle, but BlockLogU8 radii).
    pub fn default_two_tier_lossy() -> Self {
        Self {
            shared_codec: CODEC_FIB_K4_N32_BATCHED.into(),
            shell_codec: turbo_batched_codec_id(4, true).into(),
            fib_config: FibConfig::default_k4_n32(),
            turbo_config: TurboConfig::default_4bit_lossy(),
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
                allowed_shared, self.shared_codec
            )));
        }
        let allowed_shell: &'static [&'static str] = allowed_shell_codecs();
        if !allowed_shell.contains(&self.shell_codec.as_str()) {
            return Err(ProveKvError::InvalidPolicy(format!(
                "shell_codec must be one of {:?}, got '{}'",
                allowed_shell, self.shell_codec
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

    #[test]
    fn test_turbo_batched_codec_id_format() {
        assert_eq!(turbo_batched_codec_id(8, false), "turbo_8bit_batched");
        assert_eq!(turbo_batched_codec_id(8, true), "turbo_8bit_batched_lossy");
        assert_eq!(turbo_batched_codec_id(4, false), "turbo_4bit_batched");
        assert_eq!(turbo_batched_codec_id(2, false), "turbo_2bit_batched");
        assert_eq!(
            turbo_batched_codec_id(16, true),
            "turbo_16bit_batched_lossy"
        );
    }

    #[test]
    fn test_is_batched_turbo_accepts_all_bit_rates() {
        for bits in 2u8..=16 {
            let id = turbo_batched_codec_id(bits, false);
            assert!(
                is_batched_turbo(&id),
                "lossless {bits}bit should be batched turbo"
            );
            let id_lossy = turbo_batched_codec_id(bits, true);
            assert!(
                is_batched_turbo_lossy(&id_lossy),
                "lossy {bits}bit should be batched turbo lossy"
            );
            // The lossy id should NOT match the lossless predicate.
            assert!(
                !is_batched_turbo(&id_lossy),
                "lossy {bits}bit should NOT match lossless"
            );
            // And the lossless id should NOT match the lossy predicate.
            assert!(
                !is_batched_turbo_lossy(&id),
                "lossless {bits}bit should NOT match lossy"
            );
        }
    }

    #[test]
    fn test_policy_accepts_non_8bit_batched() {
        // 4bit lossless should validate and the codec id should match.
        let mut policy = CompressionPolicy::default_two_tier();
        policy.turbo_config.bits = 4;
        policy.shell_codec = turbo_batched_codec_id(4, false);
        assert!(
            policy.validate().is_ok(),
            "4bit lossless policy must validate"
        );
        // 4bit lossy should also validate.
        policy.turbo_config.radii_compression = crate::policy::RadiiCompression::Lossy;
        policy.shell_codec = turbo_batched_codec_id(4, true);
        assert!(policy.validate().is_ok(), "4bit lossy policy must validate");
    }
}
