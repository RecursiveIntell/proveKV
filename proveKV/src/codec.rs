use std::any::Any;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::policy::CodecId;

/// A codec compresses and decompresses KV vectors.
pub trait KVecCodec: Send + Sync {
    /// Return the codec identifier ("fib_k4_n32", "turbo_8bit", etc.).
    fn codec_id(&self) -> CodecId;

    /// Downcast hook for callers that need the concrete adapter (e.g. to
    /// call `encode_batch_compact` / `decode_batch_compact` which are
    /// inherent methods on the adapter, not on the trait). Returns `&dyn Any`
    /// so callers can `downcast_ref::<ConcreteType>()` to recover the type.
    fn as_any(&self) -> &dyn Any;

    /// Encode a vector of f32 values into a compressed byte payload.
    fn encode(&self, vector: &[f32], seed: u64) -> Result<Vec<u8>>;

    /// Encode a batch of vectors in one call.
    ///
    /// The default implementation calls `encode` in a loop. Codecs that can
    /// exploit batch-level parallelism (e.g. fib-quant with `gpu-backend`)
    /// override this to issue a single batched dispatch.
    ///
    /// The returned byte payloads are in the same order as `vectors`.
    fn encode_batch(&self, vectors: &[&[f32]], seed: u64) -> Result<Vec<Vec<u8>>> {
        vectors.iter().map(|v| self.encode(v, seed)).collect()
    }

    /// Decode a compressed byte payload back into a vector of f32 values.
    fn decode(&self, payload: &[u8], seed: u64) -> Result<Vec<f32>>;

    /// Decode a batch of compressed payloads. Default loops over `decode`.
    fn decode_batch(&self, payloads: &[&[u8]], seed: u64) -> Result<Vec<Vec<f32>>> {
        payloads.iter().map(|p| self.decode(p, seed)).collect()
    }

    /// The expected dimension of input/output vectors.
    fn dim(&self) -> usize;

    /// Expected compression ratio (nominal).
    fn compression_ratio(&self) -> f64;

    /// True if this adapter has access to GPU acceleration at runtime.
    ///
    /// This is distinct from the `gpu` feature being compiled in: a corpus
    /// too small for the GPU's batch threshold will fall through to CPU even
    /// when the feature is on. The pool build receipt uses this to set the
    /// `backend` field honestly.
    ///
    /// The default probes device availability only. Codecs that gate on
    /// batch size / dim should override [`Self::is_gpu_accelerated_for`].
    fn is_gpu_accelerated(&self) -> bool {
        false
    }

    /// True if a batch of `n` vectors at dimension `d` would actually
    /// dispatch to GPU. Default falls back to the device-availability probe;
    /// codecs with a runtime threshold should override.
    fn is_gpu_accelerated_for(&self, n: usize, d: usize) -> bool {
        let _ = (n, d);
        self.is_gpu_accelerated()
    }
}

/// A serialized compressed block with codec metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompressedBlock {
    /// Codec identifier.
    pub codec: CodecId,
    /// The compressed payload bytes.
    pub encoded_payload: Vec<u8>,
    /// Blake3 digest of the encoded payload.
    pub payload_digest: String,
    /// Original (uncompressed) vector dimension.
    pub original_dim: usize,
    /// Size of the compressed payload in bytes.
    pub compressed_bytes: usize,
}

impl CompressedBlock {
    /// Create a new CompressedBlock from encoded payload.
    pub fn new(codec: CodecId, encoded_payload: Vec<u8>, original_dim: usize) -> Self {
        let compressed_bytes = encoded_payload.len();
        let payload_digest = blake3::hash(&encoded_payload).to_hex().to_string();
        Self {
            codec,
            encoded_payload,
            payload_digest,
            original_dim,
            compressed_bytes,
        }
    }

    /// Compression ratio: original f32 bytes / compressed bytes.
    pub fn compression_ratio(&self) -> f64 {
        let raw_bytes = self.original_dim * 4; // 4 bytes per f32
        if self.compressed_bytes == 0 {
            return f64::INFINITY;
        }
        raw_bytes as f64 / self.compressed_bytes as f64
    }
}

// ── Exact fallback codec (no compression) ──

/// Exact fallback codec: stores raw f32 bytes with no compression.
pub struct ExactFallbackCodec {
    dim: usize,
}

impl ExactFallbackCodec {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl KVecCodec for ExactFallbackCodec {
    fn codec_id(&self) -> CodecId {
        crate::policy::CODEC_EXACT_FALLBACK.into()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn encode(&self, vector: &[f32], _seed: u64) -> Result<Vec<u8>> {
        if vector.len() != self.dim {
            return Err(crate::error::ProveKvError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        // Store raw f32 bytes in little-endian
        let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(bytes)
    }

    fn decode(&self, payload: &[u8], _seed: u64) -> Result<Vec<f32>> {
        let expected_len = self.dim * 4;
        if payload.len() != expected_len {
            return Err(crate::error::ProveKvError::CorruptPayload(format!(
                "exact fallback payload size {} != expected {}",
                payload.len(),
                expected_len
            )));
        }
        let mut vec = Vec::with_capacity(self.dim);
        for chunk in payload.chunks_exact(4) {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            vec.push(f32::from_le_bytes(arr));
        }
        Ok(vec)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn compression_ratio(&self) -> f64 {
        1.0
    }
}

// ── TurboQuant adapter ──

/// Adapter for the turbo-quant crate (8-bit, 32 projections).
#[cfg(feature = "turbo")]
#[derive(Clone, Copy)]
pub struct TurboQuantAdapter {
    dim: usize,
    bits: u8,
    projections: usize,
    /// How to compress radii in the wire format. Defaults to `Lossless`
    /// so receipts remain bit-exact; set to `Lossy` for the 58.14x
    /// two-tier system number.
    radii_compression: crate::policy::RadiiCompression,
}

#[cfg(feature = "turbo")]
impl TurboQuantAdapter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(dim: usize, bits: u8, projections: usize) -> Result<Self> {
        if dim == 0 {
            return Err(crate::error::ProveKvError::InvalidPolicy(
                "turbo dim must be > 0".into(),
            ));
        }
        if dim % 2 != 0 {
            // Pad to even — turbo-quant requires even dimensions
            return Err(crate::error::ProveKvError::InvalidPolicy(format!(
                "turbo requires even dimension, got {}",
                dim
            )));
        }
        Ok(Self {
            dim,
            bits,
            projections,
            radii_compression: crate::policy::RadiiCompression::Lossless,
        })
    }

    /// Construct with an explicit radii compression policy. The default
    /// `new` constructor uses `Lossless` (backward compat).
    #[allow(clippy::too_many_arguments)]
    pub fn with_radii_compression(
        dim: usize,
        bits: u8,
        projections: usize,
        radii_compression: crate::policy::RadiiCompression,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(crate::error::ProveKvError::InvalidPolicy(
                "turbo dim must be > 0".into(),
            ));
        }
        if dim % 2 != 0 {
            return Err(crate::error::ProveKvError::InvalidPolicy(format!(
                "turbo requires even dimension, got {}",
                dim
            )));
        }
        Ok(Self {
            dim,
            bits,
            projections,
            radii_compression,
        })
    }
}

#[cfg(feature = "turbo")]
impl TurboQuantAdapter {
    /// Batched encode using the TQB1 wire format. Builds a single quantizer,
    /// encodes N vectors, and writes them with a single shared header (one
    /// profile header for the whole batch instead of one per vector).
    ///
    /// For a batch of N vectors at head_dim=64, projections=32, b=2:
    /// - TQW1 (per-vector header): 46 + 136 = 182 bytes/vector, total 182*N
    /// - TQB1 (batched header):     38 + 136*N bytes total
    /// - Savings: 1.34x per batch
    ///
    /// Most useful when the caller has a full layer's worth of vectors
    /// (num_tokens * num_kv_heads * 2 K+V) to encode at once.
    pub fn encode_batch_compact(&self, vectors: &[&[f32]], seed: u64) -> Result<Vec<u8>> {
        let quantizer = turbo_quant::TurboQuantizer::new(self.dim, self.bits, self.projections, seed)
            .map_err(|e| {
                crate::error::ProveKvError::CompressionFailed(format!(
                    "turbo quantizer init failed: {e}"
                ))
            })?;

        let codes: Vec<turbo_quant::TurboCode> = vectors
            .iter()
            .map(|v| {
                quantizer.encode(v).map_err(|e| {
                    crate::error::ProveKvError::CompressionFailed(format!(
                        "turbo encode failed: {e}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Dispatch to the right wire encoder based on the configured
        // radii compression. The decoder reads the radii_codec flag from
        // the wire and applies the inverse, so a lossless-written batch
        // and a lossy-written batch with the same seed/dim/projections
        // are both decodable.
        let wire_result = match self.radii_compression {
            crate::policy::RadiiCompression::Lossless => {
                turbo_quant::TurboCodeWireV1::encode_batch(&codes, &quantizer)
            }
            crate::policy::RadiiCompression::Lossy => {
                turbo_quant::TurboCodeWireV1::encode_batch_with_radii(
                    &codes,
                    &quantizer,
                    turbo_quant::radius::RadiusCodecProfileV1::BlockLogU8,
                )
            }
        };
        wire_result.map_err(|e| {
            crate::error::ProveKvError::CompressionFailed(format!(
                "turbo batched wire encode failed: {e}"
            ))
        })
    }

    /// Batched decode using the TQB1 wire format. Returns a Vec<f32> per
    /// input vector (in the same order).
    pub fn decode_batch_compact(&self, payload: &[u8], seed: u64) -> Result<Vec<Vec<f32>>> {
        let quantizer = turbo_quant::TurboQuantizer::new(self.dim, self.bits, self.projections, seed)
            .map_err(|e| {
                crate::error::ProveKvError::DecompressionFailed(format!(
                    "turbo quantizer init failed: {e}"
                ))
            })?;

        let codes = turbo_quant::TurboCodeWireV1::decode_batch(payload, &quantizer).map_err(|e| {
            crate::error::ProveKvError::DecompressionFailed(format!(
                "turbo batched wire decode failed: {e}"
            ))
        })?;

        // Use the quantizer's batch decode (PolarQuantizer::decode_batch
        // → RotationBackend::apply_inverse_batch). Same numerical output
        // as the per-vec loop, but amortizes the per-call branch /
        // lookup overhead and keeps the rotation's signs (or matrix) hot
        // in cache across the whole batch. The quantizer is constructed
        // with the right mode from `self.bits`/`self.projections`, so
        // the polar-code bit count it produces is correct for both
        // PolarOnly and PolarWithQjl modes (avoiding the previous
        // hard-coded `bits - 1` which only matched PolarWithQjl).
        quantizer.decode_approximate_batch(&codes).map_err(|e| {
            crate::error::ProveKvError::DecompressionFailed(format!(
                "turbo batch decode failed: {e}"
            ))
        })
    }
}

#[cfg(feature = "turbo")]
impl KVecCodec for TurboQuantAdapter {
    fn codec_id(&self) -> CodecId {
        crate::policy::CODEC_TURBO_8BIT.into()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn encode(&self, vector: &[f32], seed: u64) -> Result<Vec<u8>> {
        let quantizer =
            turbo_quant::TurboQuantizer::new(self.dim, self.bits, self.projections, seed).map_err(
                |e| {
                    crate::error::ProveKvError::CompressionFailed(format!(
                        "turbo quantizer init failed: {}",
                        e
                    ))
                },
            )?;

        let code = quantizer.encode(vector).map_err(|e| {
            crate::error::ProveKvError::CompressionFailed(format!("turbo encode failed: {}", e))
        })?;

        // Use the compact binary wire format from turbo-quant's TurboCodeWireV1
        // (TQW1 magic, 46-byte header + packed polar/QJL data). The JSON envelope
        // was 472 bytes/block around 26 bytes of codec data; the compact format
        // is ~206 bytes/block for b=8 / head_dim=64 / 32 projections — 2.3x
        // smaller per block, 2.4x system-level memory reduction on multi-agent
        // benchmarks. See results/bench/multi_agent_compact/.
        turbo_quant::TurboCodeWireV1::encode(&code, &quantizer).map_err(|e| {
            crate::error::ProveKvError::CompressionFailed(format!("turbo wire encode failed: {}", e))
        })
    }

    fn decode(&self, payload: &[u8], seed: u64) -> Result<Vec<f32>> {
        // Compact binary format is preferred (header starts with TURBO_CODE_WIRE_MAGIC
        // = "TQW1", 4 bytes), but fall back to JSON for backward compat with shells
        // written by older proveKV versions.
        let code: turbo_quant::TurboCode = if payload.len() >= 4
            && &payload[0..4] == turbo_quant::TURBO_CODE_WIRE_MAGIC
        {
            let quantizer = turbo_quant::TurboQuantizer::new(
                self.dim,
                self.bits,
                self.projections,
                seed,
            )
            .map_err(|e| {
                crate::error::ProveKvError::DecompressionFailed(format!(
                    "turbo quantizer init failed: {}",
                    e
                ))
            })?;
            turbo_quant::TurboCodeWireV1::decode(payload, &quantizer).map_err(|e| {
                crate::error::ProveKvError::DecompressionFailed(format!(
                    "turbo wire decode failed: {}",
                    e
                ))
            })?
        } else {
            serde_json::from_slice(payload).map_err(|e| {
                crate::error::ProveKvError::DecompressionFailed(format!(
                    "turbo code deserialize failed: {}",
                    e
                ))
            })?
        };

        // Reconstruct from polar component via independent PolarQuantizer.
        // QJL residual is lossy and not invertible, so we return the polar
        // approximation.
        let polar_quant =
            turbo_quant::PolarQuantizer::new(self.dim, self.bits - 1, seed).map_err(|e| {
                crate::error::ProveKvError::DecompressionFailed(format!(
                    "turbo polar quantizer init failed: {}",
                    e
                ))
            })?;

        let reconstructed = polar_quant.decode(&code.polar_code).map_err(|e| {
            crate::error::ProveKvError::DecompressionFailed(format!("turbo decode failed: {}", e))
        })?;

        Ok(reconstructed)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn compression_ratio(&self) -> f64 {
        8.0
    }
}

// ── FibQuant adapter ──

/// Adapter for the fib-quant crate (k=4, N=32, paper core path).
#[cfg(feature = "fib")]
#[derive(Clone, Copy)]
pub struct FibQuantAdapter {
    dim: usize,
    k: u32,
    n: u32,
    training_samples: u32,
    lloyd_restarts: u32,
    lloyd_iterations: u32,
}

#[cfg(feature = "fib")]
impl FibQuantAdapter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        k: u32,
        n: u32,
        training_samples: u32,
        lloyd_restarts: u32,
        lloyd_iterations: u32,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(crate::error::ProveKvError::InvalidPolicy(
                "fib dim must be > 0".into(),
            ));
        }
        if dim % k as usize != 0 {
            return Err(crate::error::ProveKvError::InvalidPolicy(format!(
                "fib ambient dim ({}) must be divisible by k ({})",
                dim, k
            )));
        }
        Ok(Self {
            dim,
            k,
            n,
            training_samples,
            lloyd_restarts,
            lloyd_iterations,
        })
    }

    /// Build a FibQuantizer for the given seed.
    pub fn build_quantizer(
        &self,
        seed: u64,
    ) -> std::result::Result<fib_quant::FibQuantizer, crate::error::ProveKvError> {
        let mut profile = fib_quant::FibQuantProfileV1::paper_default(
            self.dim,
            self.k as usize,
            self.n as usize,
            seed,
        )
        .map_err(|e| {
            crate::error::ProveKvError::CompressionFailed(format!("fib profile build failed: {}", e))
        })?;

        // Override training parameters
        profile.training_samples = self.training_samples;
        profile.lloyd_restarts = self.lloyd_restarts;
        profile.lloyd_iterations = self.lloyd_iterations;

        fib_quant::FibQuantizer::new(profile).map_err(|e| {
            crate::error::ProveKvError::CompressionFailed(format!(
                "fib quantizer build failed: {}",
                e
            ))
        })
    }

    /// Batched encode using the FB2 wire format. Builds a single quantizer,
    /// encodes N vectors, and writes them with a single shared profile
    /// header (one profile header for the whole batch instead of one per
    /// block).
    ///
    /// For fib_k4_n32 with head_dim=64:
    /// - FB1 (per-block header): 11 + 12 = 23 bytes/block, total 23*N
    /// - FB2 (batched header):   19 + 12*N bytes total
    /// - Savings: 1.92x per block (47.8% reduction on the fib tier)
    ///
    /// Most useful when the caller has a full layer's worth of vectors
    /// to encode at once.
    pub fn encode_batch_compact(
        &self,
        vectors: &[&[f32]],
        seed: u64,
    ) -> Result<Vec<u8>> {
        let quantizer = self.build_quantizer(seed)?;
        let codes = quantizer.encode_batch(vectors).map_err(|e| {
            crate::error::ProveKvError::CompressionFailed(format!(
                "fib encode_batch failed: {}",
                e
            ))
        })?;
        let profile = quantizer.profile().clone();
        fib_quant::FibCodeV1::encode_batch(&codes, &profile).map_err(|e| {
            crate::error::ProveKvError::CompressionFailed(format!(
                "fib batched wire encode failed: {}",
                e
            ))
        })
    }

    /// Batched decode using the FB2 wire format. Returns a Vec<f32> per
    /// input vector (in the same order). Falls back to per-block
    /// `from_compact_bytes` if the payload uses the older FB1 format.
    pub fn decode_batch_compact(
        &self,
        payload: &[u8],
        seed: u64,
    ) -> Result<Vec<Vec<f32>>> {
        let quantizer = self.build_quantizer(seed)?;
        let profile = quantizer.profile().clone();
        // Detect format: FB2 = "FB2" magic, FB1 = "FB1" magic, else JSON.
        let codes: Vec<fib_quant::FibCodeV1> = if payload.len() >= 3
            && &payload[0..3] == fib_quant::BATCHED_MAGIC.as_slice()
        {
            fib_quant::FibCodeV1::decode_batch(payload, &profile).map_err(|e| {
                crate::error::ProveKvError::DecompressionFailed(format!(
                    "fib batched wire decode failed: {}",
                    e
                ))
            })?
        } else if payload.len() >= 3 && payload[0..3] == fib_quant::COMPACT_MAGIC {
            // Fallback to single-block FB1 decode (one block only).
            let code = fib_quant::FibCodeV1::from_compact_bytes(payload, &profile).map_err(|e| {
                crate::error::ProveKvError::DecompressionFailed(format!(
                    "fib compact decode failed: {}",
                    e
                ))
            })?;
            vec![code]
        } else {
            // Backward compat with JSON-encoded pools from older proveKV
            // versions.
            let code: fib_quant::FibCodeV1 =
                serde_json::from_slice(payload).map_err(|e| {
                    crate::error::ProveKvError::DecompressionFailed(format!(
                        "fib code deserialize failed: {}",
                        e
                    ))
                })?;
            vec![code]
        };
        quantizer.decode_batch_fast(&codes).map_err(|e| {
            crate::error::ProveKvError::DecompressionFailed(format!(
                "fib decode_batch_fast failed: {}",
                e
            ))
        })
    }
}

#[cfg(feature = "fib")]
impl KVecCodec for FibQuantAdapter {
    fn codec_id(&self) -> CodecId {
        crate::policy::CODEC_FIB_K4_N32.into()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn encode(&self, vector: &[f32], seed: u64) -> Result<Vec<u8>> {
        let quantizer = self.build_quantizer(seed)?;
        let code = quantizer.encode(vector).map_err(|e| {
            crate::error::ProveKvError::CompressionFailed(format!("fib encode failed: {}", e))
        })?;

        // Use the compact binary wire format (23 bytes vs 472 bytes JSON
        // for fib_k4_n32 with head_dim=64 — 20.5x smaller). The compact
        // format drops profile_digest, codebook_digest, rotation_digest,
        // ambient_dim, block_dim, norm_format — all of which the decoder
        // re-derives from its own profile. See fib_quant::FibCodeV1::to_compact_bytes.
        Ok(code.to_compact_bytes())
    }

    fn encode_batch(&self, vectors: &[&[f32]], seed: u64) -> Result<Vec<Vec<u8>>> {
        // Build a single quantizer shared across the batch. The codebook and
        // rotation are deterministic functions of the profile, so a single
        // quantizer is byte-identical to one built per-vector.
        let quantizer = self.build_quantizer(seed)?;
        let codes = quantizer.encode_batch(vectors).map_err(|e| {
            crate::error::ProveKvError::CompressionFailed(format!(
                "fib encode_batch failed: {}",
                e
            ))
        })?;
        let mut out = Vec::with_capacity(codes.len());
        for code in codes {
            out.push(code.to_compact_bytes());
        }
        Ok(out)
    }

    fn decode(&self, payload: &[u8], seed: u64) -> Result<Vec<f32>> {
        let quantizer = self.build_quantizer(seed)?;
        // Compact binary format is preferred (the pool always writes this
        // now), but fall back to JSON for backward compat with pools written
        // by older proveKV versions.
        let code = if payload.len() >= 3 && payload[0..3] == fib_quant::COMPACT_MAGIC {
            let profile = quantizer.profile().clone();
            fib_quant::FibCodeV1::from_compact_bytes(payload, &profile).map_err(|e| {
                crate::error::ProveKvError::DecompressionFailed(format!(
                    "fib compact decode failed: {}",
                    e
                ))
            })?
        } else {
            serde_json::from_slice(payload).map_err(|e| {
                crate::error::ProveKvError::DecompressionFailed(format!(
                    "fib code deserialize failed: {}",
                    e
                ))
            })?
        };

        let decoded = quantizer.decode(&code).map_err(|e| {
            crate::error::ProveKvError::DecompressionFailed(format!("fib decode failed: {}", e))
        })?;

        Ok(decoded)
    }

    fn decode_batch(&self, payloads: &[&[u8]], seed: u64) -> Result<Vec<Vec<f32>>> {
        let quantizer = self.build_quantizer(seed)?;
        let profile = quantizer.profile().clone();
        let mut codes = Vec::with_capacity(payloads.len());
        for p in payloads {
            let code = if p.len() >= 3 && p[0..3] == fib_quant::COMPACT_MAGIC {
                fib_quant::FibCodeV1::from_compact_bytes(p, &profile).map_err(|e| {
                    crate::error::ProveKvError::DecompressionFailed(format!(
                        "fib compact decode failed: {}",
                        e
                    ))
                })?
            } else {
                serde_json::from_slice(p).map_err(|e| {
                    crate::error::ProveKvError::DecompressionFailed(format!(
                        "fib code deserialize failed: {}",
                        e
                    ))
                })?
            };
            codes.push(code);
        }
        quantizer
            .decode_batch_fast(&codes)
            .map_err(|e| {
                crate::error::ProveKvError::DecompressionFailed(format!(
                    "fib decode_batch_fast failed: {}",
                    e
                ))
            })
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn compression_ratio(&self) -> f64 {
        50.0
    }

    fn is_gpu_accelerated(&self) -> bool {
        // Device-level probe — kept for trait compatibility. The pool build
        // uses is_gpu_accelerated_for(n, d) for honest per-batch reporting.
        match self.build_quantizer(0) {
            Ok(q) => q.is_gpu_accelerated(),
            Err(_) => false,
        }
    }

    fn is_gpu_accelerated_for(&self, n: usize, d: usize) -> bool {
        match self.build_quantizer(0) {
            Ok(q) => q.is_gpu_accelerated_for(n, d),
            Err(_) => false,
        }
    }
}

/// Create a codec from a policy and vector dimension.
///
/// Returns the appropriate codec based on the codec_id in the policy.
/// If the required compression crate is unavailable, returns an error.
#[allow(clippy::too_many_arguments)]
pub fn create_codec(
    codec_id: &str,
    dim: usize,
    fib_config: Option<&crate::policy::FibConfig>,
    turbo_config: Option<&crate::policy::TurboConfig>,
) -> Result<Box<dyn KVecCodec>> {
    match codec_id {
        crate::policy::CODEC_FIB_K4_N32 | crate::policy::CODEC_FIB_K4_N32_BATCHED => {
            #[cfg(feature = "fib")]
            {
                let fc = fib_config.ok_or_else(|| {
                    crate::error::ProveKvError::InvalidPolicy("fib codec requires fib_config".into())
                })?;
                let adapter = FibQuantAdapter::new(
                    dim,
                    fc.k,
                    fc.n,
                    fc.training_samples,
                    fc.lloyd_restarts,
                    fc.lloyd_iterations,
                )?;
                Ok(Box::new(adapter))
            }
            #[cfg(not(feature = "fib"))]
            {
                Err(crate::error::ProveKvError::CodecUnavailable {
                    codec: codec_id.into(),
                    feature: "fib".into(),
                })
            }
        }
        crate::policy::CODEC_TURBO_8BIT | crate::policy::CODEC_TURBO_8BIT_BATCHED => {
            #[cfg(feature = "turbo")]
            {
                let tc = turbo_config.ok_or_else(|| {
                    crate::error::ProveKvError::InvalidPolicy(
                        "turbo codec requires turbo_config".into(),
                    )
                })?;
                let adapter = TurboQuantAdapter::new(dim, tc.bits, tc.projections)?;
                Ok(Box::new(adapter))
            }
            #[cfg(not(feature = "turbo"))]
            {
                Err(crate::error::ProveKvError::CodecUnavailable {
                    codec: codec_id.into(),
                    feature: "turbo".into(),
                })
            }
        }
        crate::policy::CODEC_EXACT_FALLBACK => Ok(Box::new(ExactFallbackCodec::new(dim))),
        other => Err(crate::error::ProveKvError::InvalidPolicy(format!(
            "unknown codec id: {}",
            other
        ))),
    }
}
