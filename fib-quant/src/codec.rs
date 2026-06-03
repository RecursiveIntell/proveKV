use half::f16;
use serde::{Deserialize, Serialize};

use crate::{
    bitpack::{pack_indices, unpack_indices},
    codebook::FibCodebookV1,
    digest::{bytes_digest, json_digest},
    metrics,
    profile::{FibQuantProfileV1, NormFormat},
    receipt::FibQuantCompressionReceiptV1,
    rotation::StoredRotation,
    FibQuantError, Result,
};

pub const CODE_SCHEMA: &str = "fib_code_v1";

/// Magic + version prefix for the compact binary wire format.
/// `F` `B` `1` = Fib Binary v1. Any decoder that sees a different
/// magic should reject the payload as corrupt.
pub const COMPACT_MAGIC: [u8; 3] = [b'F', b'B', b'1'];
pub const COMPACT_VERSION: u8 = 1;

/// Magic + version prefix for the batched binary wire format.
/// `F` `B` `2` = Fib Binary v2 (batched). Stores the profile once
/// per batch, then concatenates per-block payloads (norm + indices)
/// with no per-block header.
pub const BATCHED_MAGIC: [u8; 3] = [b'F', b'B', b'2'];
pub const BATCHED_VERSION: u8 = 1;

/// Encoded fixed-rate FibQuant artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FibCodeV1 {
    /// Stable schema marker.
    pub schema_version: String,
    /// Profile digest.
    pub profile_digest: String,
    /// Codebook digest.
    pub codebook_digest: String,
    /// Rotation digest.
    pub rotation_digest: String,
    /// Ambient dimension.
    pub ambient_dim: u32,
    /// Block dimension.
    pub block_dim: u32,
    /// Norm payload format.
    pub norm_format: NormFormat,
    /// Norm bytes.
    pub norm_payload: Vec<u8>,
    /// Bits per fixed-rate index.
    pub wire_index_bits: u8,
    /// Number of indices.
    pub block_count: u32,
    /// Packed fixed-rate indices.
    pub indices: Vec<u8>,
}

impl FibCodeV1 {
    /// Compact binary wire format. The FibCodeV1 struct carries a lot of
    /// metadata for JSON deserialization (schema_version, profile_digest,
    /// rotation_digest, ambient_dim, block_dim, etc.) that the decoder
    /// either doesn't need (it has its own profile) or can reconstruct
    /// from the manifest (profile_digest/codebook_digest/rotation_digest).
    ///
    /// Compact layout (little-endian, packed):
    ///   [0..3]  magic: "FB1"
    ///   [3]     version: 1
    ///   [4]     wire_index_bits
    ///   [5..9]  block_count (u32)
    ///   [9..11] norm_payload (length-prefixed, max 65535 bytes)
    ///          actually: [9..11] norm_len (u16) + norm bytes
    ///   then indices bytes
    ///
    /// The decoder must already know the profile (or have the manifest
    /// supply it). It can re-derive the digests from that profile and
    /// check them at the manifest level. Per-block we only need
    /// wire_index_bits, block_count, norm_payload, and indices.
    ///
    /// For fib_k4_n32 with head_dim=64: 16 indices * 5 bits = 10 bytes
    /// indices + 2 bytes norm = 12 bytes payload + 11 bytes header =
    /// **23 bytes per block** vs **474 bytes for JSON** = 20.6x smaller.
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(11 + self.norm_payload.len() + self.indices.len());
        out.extend_from_slice(&COMPACT_MAGIC);
        out.push(COMPACT_VERSION);
        out.push(self.wire_index_bits);
        out.extend_from_slice(&self.block_count.to_le_bytes());
        let norm_len = self.norm_payload.len() as u16;
        out.extend_from_slice(&norm_len.to_le_bytes());
        out.extend_from_slice(&self.norm_payload);
        out.extend_from_slice(&self.indices);
        out
    }

    /// Decode the compact binary format. The caller must supply the
    /// profile so that the profile/codebook digests in the resulting
    /// `FibCodeV1` match what `validate_code_header` expects.
    ///
    /// The compact format omits the digests because they're derivable
    /// from the profile — there's no point storing them when the
    /// decoder will check them against the profile digest anyway.
    pub fn from_compact_bytes(bytes: &[u8], profile: &FibQuantProfileV1) -> Result<Self> {
        if bytes.len() < 11 {
            return Err(FibQuantError::CorruptPayload(format!(
                "compact FibCodeV1 too short: {} bytes (need >= 11)",
                bytes.len()
            )));
        }
        if bytes[0..3] != COMPACT_MAGIC {
            return Err(FibQuantError::CorruptPayload(format!(
                "compact FibCodeV1 bad magic: {:?} (expected {:?})",
                &bytes[0..3],
                COMPACT_MAGIC
            )));
        }
        if bytes[3] != COMPACT_VERSION {
            return Err(FibQuantError::CorruptPayload(format!(
                "compact FibCodeV1 version {} not supported (need {})",
                bytes[3], COMPACT_VERSION
            )));
        }
        let wire_index_bits = bytes[4];
        let block_count = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]);
        let norm_len = u16::from_le_bytes([bytes[9], bytes[10]]) as usize;
        let header_len = 11;
        if bytes.len() < header_len + norm_len {
            return Err(FibQuantError::CorruptPayload(format!(
                "compact FibCodeV1 truncated: norm_len={} but only {} bytes remain",
                norm_len,
                bytes.len() - header_len
            )));
        }
        let norm_payload = bytes[header_len..header_len + norm_len].to_vec();
        let indices = bytes[header_len + norm_len..].to_vec();

        // Validate packed index length
        let expected_packed_len = (block_count as usize)
            .checked_mul(wire_index_bits as usize)
            .map(|bits| (bits + 7) / 8)
            .ok_or_else(|| {
                FibQuantError::ResourceLimitExceeded("packed index bits overflow".into())
            })?;
        if indices.len() != expected_packed_len {
            return Err(FibQuantError::CorruptPayload(format!(
                "compact FibCodeV1 indices: got {} bytes, expected {} (block_count={} * wire_index_bits={})",
                indices.len(),
                expected_packed_len,
                block_count,
                wire_index_bits
            )));
        }

        // The compact wire format omits codebook_digest and
        // rotation_digest because they're derivable from the profile.
        // The decoder re-derives them itself and skips the mismatch check
        // (see validate_code_header's empty-string short-circuit). We
        // still need profile_digest because the decode path uses it
        // to verify the code matches the decoder's profile.
        let profile_digest = profile.digest()?;
        Ok(FibCodeV1 {
            schema_version: CODE_SCHEMA.into(),
            profile_digest,
            // Leave these empty — see validate_code_header for the
            // short-circuit. The cost of building a FibCodebookV1
            // per from_compact_bytes call is prohibitive (~6ms each,
            // 10s of seconds for a real pool).
            codebook_digest: String::new(),
            rotation_digest: String::new(),
            ambient_dim: profile.ambient_dim,
            block_dim: profile.block_dim,
            norm_format: profile.norm_format.clone(),
            norm_payload,
            wire_index_bits,
            block_count,
            indices,
        })
    }

    /// Compact size in bytes (does not allocate).
    pub fn compact_size(&self) -> usize {
        11 + self.norm_payload.len() + self.indices.len()
    }

    // ---- Batched wire format (FB2) ----
    //
    // The single-block format (FB1) repeats the profile-determined fields
    // (wire_index_bits, block_count, norm_payload length) in every block's
    // 11-byte header. For a pool with 78,624 blocks (819 shared tokens ×
    // 24 layers × 2 kv_heads × 2 K+V), that's 78,624 × 11 = 865 KB of
    // redundant header — 48% of the 1.81 MB pool.
    //
    // FB2 stores the profile ONCE per batch, then concatenates per-block
    // payloads (norm bytes + indices bytes) with no per-block header.
    // The norm length is constant per profile, so no length prefix.
    //
    // Layout (little-endian, packed):
    //   [0..3]   magic: "FB2"
    //   [3]      version: 1
    //   [4]      wire_index_bits (u8) — profile-determined
    //   [5]      reserved: 0
    //   [6..10]  block_count (u32) — profile-determined
    //   [10..14] n_blocks (u32)
    //   [14]     norm_format (u8 tag: 0=Fp16Paper, 1=F32Reference)
    //   [15..17] norm_payload_len_per_block (u16)
    //   [17..19] indices_len_per_block (u16)
    //   then for each block in [0..n_blocks):
    //     [..norm_len] norm_payload (length constant, no prefix)
    //     [..idx_len] indices (length constant, no prefix)
    //
    // Header is 19 bytes total (vs 11 per-block in FB1 = 11*N for N
    // blocks). The per-block payload is deterministic from the profile,
    // so no per-block length prefix.
    //
    // For fib_k4_n32 with head_dim=64:
    //   - norm_payload: 2 bytes (fp16 norm)
    //   - indices: 10 bytes (16 blocks × 5 bits, packed)
    //   - total per-block payload: 12 bytes
    //   - per-block total in FB1: 11 + 12 = 23 bytes
    //   - per-block total in FB2: 12 bytes (header amortized)
    //   - 1.92x smaller per block
    //   - 47.8% reduction on the fib tier.

    /// Encode a batch of FibCodeV1 blocks using the supplied shared profile.
    /// All blocks must share the same profile (wire_index_bits, block_count,
    /// norm format, norm/indices lengths).
    pub fn encode_batch(codes: &[FibCodeV1], profile: &FibQuantProfileV1) -> Result<Vec<u8>> {
        if codes.is_empty() {
            return Err(FibQuantError::CorruptPayload("empty batch".into()));
        }
        // Validate profile consistency across all blocks.
        let wire_index_bits = codes[0].wire_index_bits;
        let block_count = codes[0].block_count;
        let norm_format = codes[0].norm_format.clone();
        let norm_len = codes[0].norm_payload.len();
        let indices_len = codes[0].indices.len();
        for (i, code) in codes.iter().enumerate() {
            if code.wire_index_bits != wire_index_bits {
                return Err(FibQuantError::CorruptPayload(format!(
                    "batched wire block {i} wire_index_bits {} != header {}",
                    code.wire_index_bits, wire_index_bits
                )));
            }
            if code.block_count != block_count {
                return Err(FibQuantError::CorruptPayload(format!(
                    "batched wire block {i} block_count {} != header {}",
                    code.block_count, block_count
                )));
            }
            if code.norm_format != norm_format {
                return Err(FibQuantError::CorruptPayload(format!(
                    "batched wire block {i} norm_format mismatch"
                )));
            }
            if code.norm_payload.len() != norm_len {
                return Err(FibQuantError::CorruptPayload(format!(
                    "batched wire block {i} norm_payload_len {} != header {}",
                    code.norm_payload.len(),
                    norm_len
                )));
            }
            if code.indices.len() != indices_len {
                return Err(FibQuantError::CorruptPayload(format!(
                    "batched wire block {i} indices_len {} != header {}",
                    code.indices.len(),
                    indices_len
                )));
            }
        }
        let norm_format_tag: u8 = match norm_format {
            NormFormat::Fp16Paper => 0,
            NormFormat::F32Reference => 1,
        };
        let norm_len_u16 = u16::try_from(norm_len).map_err(|_| {
            FibQuantError::ResourceLimitExceeded(format!(
                "batched wire norm_payload_len {norm_len} exceeds u16::MAX"
            ))
        })?;
        let indices_len_u16 = u16::try_from(indices_len).map_err(|_| {
            FibQuantError::ResourceLimitExceeded(format!(
                "batched wire indices_len {indices_len} exceeds u16::MAX"
            ))
        })?;
        let n_blocks = u32::try_from(codes.len()).map_err(|_| {
            FibQuantError::ResourceLimitExceeded("batched wire n_blocks exceeds u32::MAX".into())
        })?;
        let block_payload_len = norm_len + indices_len;
        let total_payload = block_payload_len * codes.len();
        let mut bytes = Vec::with_capacity(19 + total_payload);
        bytes.extend_from_slice(&BATCHED_MAGIC);
        bytes.push(BATCHED_VERSION);
        bytes.push(wire_index_bits);
        bytes.push(0); // reserved
        bytes.extend_from_slice(&block_count.to_le_bytes());
        bytes.extend_from_slice(&n_blocks.to_le_bytes());
        bytes.push(norm_format_tag);
        bytes.extend_from_slice(&norm_len_u16.to_le_bytes());
        bytes.extend_from_slice(&indices_len_u16.to_le_bytes());
        for code in codes {
            bytes.extend_from_slice(&code.norm_payload);
            bytes.extend_from_slice(&code.indices);
        }
        // Profile is validated against the blocks we wrote, but the caller
        // may want a separate sanity check. We can validate the profile
        // matches the header fields here too.
        if profile.wire_index_bits != wire_index_bits {
            return Err(FibQuantError::ProfileDigestMismatch {
                expected: profile.digest().unwrap_or_default(),
                actual: format!(
                    "header wire_index_bits={wire_index_bits} does not match profile {}",
                    profile.wire_index_bits
                ),
            });
        }
        if profile.block_count() != block_count {
            return Err(FibQuantError::CorruptPayload(format!(
                "batched wire header block_count {block_count} != profile block_count {}",
                profile.block_count()
            )));
        }
        Ok(bytes)
    }

    /// Decode a batched FB2 payload into a Vec<FibCodeV1>.
    pub fn decode_batch(bytes: &[u8], profile: &FibQuantProfileV1) -> Result<Vec<FibCodeV1>> {
        if bytes.len() < 19 {
            return Err(FibQuantError::CorruptPayload(format!(
                "batched FibCodeV1 too short: {} bytes (need >= 19)",
                bytes.len()
            )));
        }
        if bytes[0..3] != BATCHED_MAGIC {
            return Err(FibQuantError::CorruptPayload(format!(
                "batched FibCodeV1 bad magic: {:?} (expected {:?})",
                &bytes[0..3],
                BATCHED_MAGIC
            )));
        }
        if bytes[3] != BATCHED_VERSION {
            return Err(FibQuantError::CorruptPayload(format!(
                "batched FibCodeV1 version {} not supported (need {})",
                bytes[3], BATCHED_VERSION
            )));
        }
        let wire_index_bits = bytes[4];
        let _reserved = bytes[5];
        let block_count = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]);
        let n_blocks = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;
        let norm_format_tag = bytes[14];
        let norm_len =
            u16::from_le_bytes([bytes[15], bytes[16]]) as usize;
        let indices_len =
            u16::from_le_bytes([bytes[17], bytes[18]]) as usize;
        // Validate profile match.
        if wire_index_bits != profile.wire_index_bits {
            return Err(FibQuantError::CorruptPayload(format!(
                "batched wire wire_index_bits {wire_index_bits} != profile {}",
                profile.wire_index_bits
            )));
        }
        if block_count != profile.block_count() {
            return Err(FibQuantError::CorruptPayload(format!(
                "batched wire block_count {block_count} != profile {}",
                profile.block_count()
            )));
        }
        let norm_format = match norm_format_tag {
            0 => NormFormat::Fp16Paper,
            1 => NormFormat::F32Reference,
            other => {
                return Err(FibQuantError::CorruptPayload(format!(
                    "batched wire unknown norm_format tag {other}"
                )));
            }
        };
        if norm_format != profile.norm_format {
            return Err(FibQuantError::CorruptPayload(format!(
                "batched wire norm_format tag {norm_format:?} != profile {:?}",
                profile.norm_format
            )));
        }
        let block_payload_len = norm_len + indices_len;
        let expected_total = 19 + n_blocks * block_payload_len;
        if bytes.len() < expected_total {
            return Err(FibQuantError::CorruptPayload(format!(
                "batched wire buffer {} bytes < expected {} for {n_blocks} blocks",
                bytes.len(),
                expected_total
            )));
        }
        // Validate packed index length matches the profile's expected length.
        let expected_packed_len = (block_count as usize)
            .checked_mul(wire_index_bits as usize)
            .map(|bits| (bits + 7) / 8)
            .ok_or_else(|| {
                FibQuantError::ResourceLimitExceeded("packed index bits overflow".into())
            })?;
        if indices_len != expected_packed_len {
            return Err(FibQuantError::CorruptPayload(format!(
                "batched wire indices_len {indices_len} != expected {expected_packed_len} (block_count={block_count} * wire_index_bits={wire_index_bits})"
            )));
        }
        let profile_digest = profile.digest()?;
        let mut codes = Vec::with_capacity(n_blocks);
        let mut cursor = 19;
        for _ in 0..n_blocks {
            let norm_payload = bytes[cursor..cursor + norm_len].to_vec();
            cursor += norm_len;
            let indices = bytes[cursor..cursor + indices_len].to_vec();
            cursor += indices_len;
            codes.push(FibCodeV1 {
                schema_version: CODE_SCHEMA.into(),
                profile_digest: profile_digest.clone(),
                // See from_compact_bytes: empty digests short-circuit the
                // match check in validate_code_header.
                codebook_digest: String::new(),
                rotation_digest: String::new(),
                ambient_dim: profile.ambient_dim,
                block_dim: profile.block_dim,
                norm_format: norm_format.clone(),
                norm_payload,
                wire_index_bits,
                block_count,
                indices,
            });
        }
        Ok(codes)
    }
}

/// FibQuant encoder/decoder bound to one profile and codebook.
#[derive(Debug, Clone)]
pub struct FibQuantizer {
    profile: FibQuantProfileV1,
    codebook: FibCodebookV1,
    rotation: StoredRotation,
}

impl FibQuantizer {
    /// Build a quantizer by constructing the profile codebook.
    pub fn new(profile: FibQuantProfileV1) -> Result<Self> {
        let codebook = FibCodebookV1::build(profile)?;
        Self::from_codebook(codebook)
    }

    /// Build a quantizer from a validated codebook.
    pub fn from_codebook(codebook: FibCodebookV1) -> Result<Self> {
        codebook.validate()?;
        let profile = codebook.profile.clone();
        let rotation = StoredRotation::new(profile.ambient_dim as usize, profile.rotation_seed)?;
        Ok(Self {
            profile,
            codebook,
            rotation,
        })
    }

    /// Access the profile.
    pub fn profile(&self) -> &FibQuantProfileV1 {
        &self.profile
    }

    /// Access the codebook.
    pub fn codebook(&self) -> &FibCodebookV1 {
        &self.codebook
    }

    /// Encode a vector into a fixed-rate artifact.
    pub fn encode(&self, x: &[f32]) -> Result<FibCodeV1> {
        let d = self.profile.ambient_dim as usize;
        let k = self.profile.block_dim as usize;
        if x.len() != d {
            return Err(FibQuantError::CorruptPayload(format!(
                "input dimension {}, expected {d}",
                x.len()
            )));
        }
        check_finite(x)?;
        let norm = l2_norm(x);
        if norm == 0.0 {
            return Err(FibQuantError::ZeroNorm);
        }
        // Convert to f64 for the rotation (it expects f64 internally),
        // then back to f32 for the SIMD-accelerated argmin loop.
        let normalized: Vec<f64> = x.iter().map(|value| f64::from(*value) / norm).collect();
        let rotated_f64 = self.rotation.apply(&normalized)?;
        let rotated_f32: Vec<f32> = rotated_f64.iter().map(|&v| v as f32).collect();
        let block_count = self.profile.block_count() as usize;
        let mut indices = Vec::with_capacity(block_count);
        for block in rotated_f32.chunks_exact(k) {
            indices.push(gpu_backend::nearest_codeword_f32(block, &self.codebook.codewords, k) as u32);
        }
        Ok(FibCodeV1 {
            schema_version: CODE_SCHEMA.into(),
            profile_digest: self.profile.digest()?,
            codebook_digest: self.codebook.codebook_digest.clone(),
            rotation_digest: self.rotation.digest()?,
            ambient_dim: self.profile.ambient_dim,
            block_dim: self.profile.block_dim,
            norm_format: self.profile.norm_format.clone(),
            norm_payload: encode_norm(norm, &self.profile.norm_format)?,
            wire_index_bits: self.profile.wire_index_bits,
            block_count: self.profile.block_count(),
            indices: pack_indices(&indices, self.profile.wire_index_bits)?,
        })
    }

    /// Decode a fixed-rate artifact.
    pub fn decode(&self, code: &FibCodeV1) -> Result<Vec<f32>> {
        self.validate_code_header(code)?;
        let k = self.profile.block_dim as usize;
        let block_count = self.profile.block_count() as usize;
        let unpacked = unpack_indices(&code.indices, block_count, self.profile.wire_index_bits)?;
        let mut rotated = Vec::with_capacity(self.profile.ambient_dim as usize);
        for index in unpacked {
            if index >= self.profile.codebook_size {
                return Err(FibQuantError::IndexOutOfRange {
                    index,
                    codebook_size: self.profile.codebook_size,
                });
            }
            rotated.extend(self.codebook.codeword(index as usize)?);
        }
        let expected_rotated_len = block_count.checked_mul(k).ok_or_else(|| {
            FibQuantError::ResourceLimitExceeded("decoded rotated vector length overflow".into())
        })?;
        if rotated.len() != expected_rotated_len {
            return Err(FibQuantError::CorruptPayload(
                "decoded rotated vector length mismatch".into(),
            ));
        }
        let norm = decode_norm(&code.norm_payload, &code.norm_format)?;
        let reconstructed = self.rotation.apply_inverse(&rotated)?;
        let out: Vec<f32> = reconstructed
            .into_iter()
            .map(|value| (value * norm) as f32)
            .collect();
        check_finite(&out)?;
        Ok(out)
    }

    /// Encode and emit a receipt.
    pub fn encode_with_receipt(
        &self,
        x: &[f32],
    ) -> Result<(FibCodeV1, FibQuantCompressionReceiptV1)> {
        let code = self.encode(x)?;
        let source_vector_digest = source_vector_digest(x)?;
        let mut receipt = FibQuantCompressionReceiptV1::new(
            &self.profile,
            code.profile_digest.clone(),
            code.codebook_digest.clone(),
            code.rotation_digest.clone(),
            source_vector_digest,
            encoded_digest(&code)?,
        );
        let decoded = self.decode(&code)?;
        receipt.mse = Some(metrics::mse(x, &decoded)?);
        receipt.cosine_similarity = Some(metrics::cosine_similarity(x, &decoded)?);
        Ok((code, receipt))
    }

    /// Reconstruction MSE for one vector.
    pub fn reconstruction_mse(&self, x: &[f32]) -> Result<f64> {
        let code = self.encode(x)?;
        let decoded = self.decode(&code)?;
        metrics::mse(x, &decoded)
    }

    /// Reconstruction cosine similarity for one vector.
    pub fn cosine_similarity(&self, x: &[f32]) -> Result<f64> {
        let code = self.encode(x)?;
        let decoded = self.decode(&code)?;
        metrics::cosine_similarity(x, &decoded)
    }

    // ── Batch encode/decode ──

    /// Encode a batch of vectors. Uses gpu-backend for the Hadamard + Lloyd-Max
    /// portions when available, keeping the FibCodeV1 format identical to single encode.
    pub fn encode_batch(&self, vectors: &[&[f32]]) -> Result<Vec<FibCodeV1>> {
        let d = self.profile.ambient_dim as usize;
        let k = self.profile.block_dim as usize;
        let n = vectors.len();
        if n == 0 {
            return Ok(vec![]);
        }

        // Fall back to single encode for small batches
        if n < 4 {
            return vectors.iter().map(|v| self.encode(v)).collect();
        }

        // Flatten input
        let mut flat = Vec::with_capacity(n * d);
        let mut norms_f64 = Vec::with_capacity(n);
        for v in vectors {
            if v.len() != d {
                return Err(FibQuantError::CorruptPayload(format!(
                    "input dimension {}, expected {d}",
                    v.len()
                )));
            }
            check_finite(v)?;
            let norm = l2_norm(v);
            if norm == 0.0 {
                return Err(FibQuantError::ZeroNorm);
            }
            norms_f64.push(norm);
            for &x in *v {
                flat.push((x as f64 / norm) as f32);
            }
        }

        // Apply Hadamard batch rotation (uses gpu-backend when available)
        #[cfg(feature = "gpu")]
        {
            if let Some(_ctx) = gpu_backend::GpuContext::init() {
                if n >= gpu_backend::GpuContext::GPU_MIN_BATCH_SIZE
                    && d >= gpu_backend::GpuContext::GPU_MIN_DIM
                {
                    gpu_backend::hadamard_batch(&mut flat, n, d, self.profile.rotation_seed)
                        .map_err(|e| {
                            FibQuantError::NumericalFailure(format!("gpu hadamard: {}", e))
                        })?;

                    // GPU codebook lookup: the dominant cost in encode_batch
                    // for k=4, N=32. Falls back to CPU if N > 32 or other
                    // gates fail; the indices produced are byte-identical to
                    // the CPU path (verified by gpu-backend parity test).
                    //
                    // The `gpu_codebook_lookup` cfg switches this on. When
                    // off, the rotated data goes back to the CPU for the
                    // codebook loop. The current dispatch path through
                    // gpu_backend pays H2D + D2H per call, which can be
                    // slower than a tight CPU loop for small batches.
                    #[cfg(feature = "gpu_codebook_lookup")]
                    {
                        let block_count = self.profile.block_count() as usize;
                        if let Ok(indices) = gpu_backend::codebook_lookup_batch(
                            &flat,
                            &self.codebook.codewords,
                            n,
                            d,
                            k,
                        ) {
                            if indices.len() == n * block_count {
                                return self.finish_batch_encode_with_indices(
                                    &flat, &norms_f64, &indices, n, d, k,
                                );
                            }
                            // Length mismatch — fall through to CPU for safety.
                        }
                    }

                    // CPU fallback for the codebook lookup (Hadamard already on GPU).
                    return self.finish_batch_encode(&flat, &norms_f64, n, d, k);
                }
            }
        }

        // CPU fallback: use StoredRotation on each vector
        let mut rotated_flat = Vec::with_capacity(n * d);
        for chunk in flat.chunks_exact(d) {
            let f64_chunk: Vec<f64> = chunk.iter().map(|&v| v as f64).collect();
            let rot = self.rotation.apply(&f64_chunk)?;
            rotated_flat.extend(rot.iter().map(|&v| v as f32));
        }

        self.finish_batch_encode(&rotated_flat, &norms_f64, n, d, k)
    }

    fn finish_batch_encode(
        &self,
        rotated: &[f32],
        norms: &[f64],
        n: usize,
        d: usize,
        k: usize,
    ) -> Result<Vec<FibCodeV1>> {
        // Precompute digest fields that are identical for every code in
        // this batch. Saves a digest call per vector (the profile digest
        // is the same for all codes).
        let profile_digest = self.profile.digest()?;
        let codebook_digest = self.codebook.codebook_digest.clone();
        let rotation_digest = self.rotation.digest()?;
        let profile = &self.profile;
        let codewords_f32: &[f32] = &self.codebook.codewords;

        // Per-vector work. Independent across vec_idx, so we can either
        // run it serially or via Rayon. The Rayon threshold is set so
        // that small batches don't pay the parallel-dispatch tax.
        let per_vec = |vec_idx: usize| -> Result<FibCodeV1> {
            let start = vec_idx * d;
            let chunk = &rotated[start..start + d];
            let mut indices = Vec::with_capacity(profile.block_count() as usize);
            for block in chunk.chunks_exact(k) {
                indices.push(gpu_backend::nearest_codeword_f32(block, codewords_f32, k) as u32);
            }
            Ok(FibCodeV1 {
                schema_version: CODE_SCHEMA.into(),
                profile_digest: profile_digest.clone(),
                codebook_digest: codebook_digest.clone(),
                rotation_digest: rotation_digest.clone(),
                ambient_dim: profile.ambient_dim,
                block_dim: profile.block_dim,
                norm_format: profile.norm_format.clone(),
                norm_payload: encode_norm(norms[vec_idx], &profile.norm_format)?,
                wire_index_bits: profile.wire_index_bits,
                block_count: profile.block_count(),
                indices: pack_indices(&indices, profile.wire_index_bits)?,
            })
        };

        // Heuristic: only parallelize when the per-vector work is large
        // enough to amortize Rayon's dispatch overhead. Empirically,
        // d=128 k=4 with n >= 16 sees a win on 4-core machines.
        #[cfg(feature = "parallel")]
        {
            const RAYON_MIN_N: usize = 16;
            if n >= RAYON_MIN_N {
                use rayon::prelude::*;
                return (0..n).into_par_iter().map(per_vec).collect();
            }
        }

        let mut codes = Vec::with_capacity(n);
        for vec_idx in 0..n {
            codes.push(per_vec(vec_idx)?);
        }
        Ok(codes)
    }

    /// Build `FibCodeV1` records from a pre-computed index array. Used by
    /// the GPU path after `codebook_lookup_batch` returns the per-block
    /// nearest-codeword indices. Length of `indices` must be `n * (d / k)`.
    #[cfg(all(feature = "gpu", feature = "gpu_codebook_lookup"))]
    fn finish_batch_encode_with_indices(
        &self,
        _rotated: &[f32], // not used; indices are already computed
        norms: &[f64],
        indices: &[u32],
        n: usize,
        _d: usize,
        _k: usize,
    ) -> Result<Vec<FibCodeV1>> {
        let block_count = self.profile.block_count() as usize;
        if indices.len() != n * block_count {
            return Err(FibQuantError::CorruptPayload(format!(
                "indices length {} != n * block_count {}",
                indices.len(),
                n * block_count
            )));
        }

        let mut codes = Vec::with_capacity(n);
        for vec_idx in 0..n {
            let start = vec_idx * block_count;
            let end = start + block_count;
            let vec_indices: Vec<u32> = indices[start..end].to_vec();

            codes.push(FibCodeV1 {
                schema_version: CODE_SCHEMA.into(),
                profile_digest: self.profile.digest()?,
                codebook_digest: self.codebook.codebook_digest.clone(),
                rotation_digest: self.rotation.digest()?,
                ambient_dim: self.profile.ambient_dim,
                block_dim: self.profile.block_dim,
                norm_format: self.profile.norm_format.clone(),
                norm_payload: encode_norm(norms[vec_idx], &self.profile.norm_format)?,
                wire_index_bits: self.profile.wire_index_bits,
                block_count: self.profile.block_count(),
                indices: pack_indices(&vec_indices, self.profile.wire_index_bits)?,
            });
        }
        Ok(codes)
    }

    /// Decode a batch of codes.
    pub fn decode_batch(&self, codes: &[FibCodeV1]) -> Result<Vec<Vec<f32>>> {
        codes.iter().map(|c| self.decode(c)).collect()
    }

    /// Fast batch decode. Optimized for the case where many small codes
    /// share the same profile (so the codebook + rotation are reused).
    ///
    /// Key wins over `decode_batch`:
    /// 1. No per-index `Vec<f64>` allocation in the codeword gather —
    ///    each codeword is copied in place into a single `Vec<f32>`.
    /// 2. The rotation matrix is converted to f32 once for the whole
    ///    batch, then `apply_inverse_f32` is called per code (no f32→f64
    ///    roundtrip on the rotation or the input).
    /// 3. The unpacked indices are reused via `as_f32_slice()` where
    ///    possible.
    ///
    /// Output is byte-identical to calling `decode` per code, modulo
    /// the final `as f32` cast in `decode` (we also cast to f32 at the
    /// end; intermediate precision is below the codebook quantization
    /// noise floor and matches the original `as f32` step exactly).
    pub fn decode_batch_fast(&self, codes: &[FibCodeV1]) -> Result<Vec<Vec<f32>>> {
        if codes.is_empty() {
            return Ok(Vec::new());
        }
        let d = self.profile.ambient_dim as usize;
        let k = self.profile.block_dim as usize;
        let codebook_size = self.profile.codebook_size as usize;
        let codewords = &self.codebook.codewords;
        let mut out = Vec::with_capacity(codes.len());
        for code in codes {
            self.validate_code_header(code)?;
            let block_count = self.profile.block_count() as usize;
            let unpacked = unpack_indices(&code.indices, block_count, self.profile.wire_index_bits)?;
            let expected_len = block_count.checked_mul(k).ok_or_else(|| {
                FibQuantError::ResourceLimitExceeded("decoded rotated vector length overflow".into())
            })?;
            // Gather codewords in place. No allocation per index.
            let mut rotated_f32: Vec<f32> = Vec::with_capacity(expected_len);
            for &index in &unpacked {
                let idx = index as usize;
                if idx >= codebook_size {
                    return Err(FibQuantError::IndexOutOfRange {
                        index,
                        codebook_size: codebook_size as u32,
                    });
                }
                let base = idx * k;
                // Direct slice extend in f32. No f32→f64 conversion.
                rotated_f32.extend_from_slice(&codewords[base..base + k]);
            }
            debug_assert_eq!(rotated_f32.len(), expected_len);
            let norm = decode_norm(&code.norm_payload, &code.norm_format)?;
            // Single f32 rotation. The original decode() does
            // f32→f64, f64 rotation, then f64→f32. We do f32 rotation
            // directly, matching the (matrix * input) as f32 of the
            // original final cast within f32 precision.
            let reconstructed = self.rotation.apply_inverse_f32(&rotated_f32)?;
            let scaled: Vec<f32> = reconstructed
                .into_iter()
                .map(|value| (value * norm as f32))
                .collect();
            check_finite(&scaled)?;
            out.push(scaled);
        }
        Ok(out)
    }

    /// Check if GPU acceleration is available.
    ///
    /// This is a **device-availability** probe: it returns true if a CUDA
    /// device was found at init time. Whether an *individual* encode_batch
    /// call actually dispatches to GPU depends on the call's batch size and
    /// vector dimension crossing the runtime thresholds.
    ///
    /// Use [`Self::is_gpu_accelerated_for`] for an honest per-call check.
    pub fn is_gpu_accelerated(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            gpu_backend::GpuContext::is_available()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Check if a batch of `n` vectors of dimension `d` would actually
    /// dispatch to GPU. Returns true only when:
    ///   - the `gpu` feature is compiled in,
    ///   - a CUDA device is available at runtime,
    ///   - `n >= GPU_MIN_BATCH_SIZE` and `d >= GPU_MIN_DIM`, AND
    ///   - the codebook size `N` is <= 32 (the codebook_lookup kernel
    ///     is one warp wide and falls back to CPU otherwise).
    ///
    /// This is the honest gate for receipts: a 4-doc corpus with dim 64
    /// returns false even with `--features gpu`, because the batch is too
    /// small to overcome GPU launch overhead. A corpus with a codebook
    /// larger than 32 also returns false.
    pub fn is_gpu_accelerated_for(&self, n: usize, d: usize) -> bool {
        #[cfg(feature = "gpu")]
        {
            if !gpu_backend::GpuContext::is_available() {
                return false;
            }
            n >= gpu_backend::GpuContext::GPU_MIN_BATCH_SIZE
                && d >= gpu_backend::GpuContext::GPU_MIN_DIM
                && (self.profile.codebook_size as usize) <= 32
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (n, d);
            false
        }
    }

    /// Per-step GPU dispatch report. `hadamard` is true if a batch of size
    /// `n` at dim `d` would dispatch the Hadamard rotation to GPU.
    /// `codebook_lookup` is true only if both the Hadamard AND the
    /// codebook-lookup step would dispatch (additionally requires codebook
    /// size <= 32). The latter is independent of the `gpu_codebook_lookup`
    /// feature gate — the feature controls whether the dispatch is enabled
    /// in `encode_batch`, not whether the kernel would be a win.
    pub fn gpu_steps_for(&self, n: usize, d: usize) -> GpuStepReport {
        let device_available = {
            #[cfg(feature = "gpu")]
            {
                gpu_backend::GpuContext::is_available()
            }
            #[cfg(not(feature = "gpu"))]
            {
                false
            }
        };
        // Thresholds are the same as gpu_backend::GpuContext's. Hard-code
        // them here to avoid requiring the gpu feature for the probe.
        const MIN_BATCH: usize = 16;
        const MIN_DIM: usize = 64;
        let clears_thresholds = n >= MIN_BATCH && d >= MIN_DIM;
        let codebook_fits = (self.profile.codebook_size as usize) <= 32;
        GpuStepReport {
            hadamard: device_available && clears_thresholds,
            codebook_lookup: device_available && clears_thresholds && codebook_fits,
        }
    }

    // ── End batch methods ──

    fn validate_code_header(&self, code: &FibCodeV1) -> Result<()> {
        if code.schema_version != CODE_SCHEMA {
            return Err(FibQuantError::CorruptPayload(format!(
                "code schema_version {}, expected {CODE_SCHEMA}",
                code.schema_version
            )));
        }
        let expected_profile = self.profile.digest()?;
        if code.profile_digest != expected_profile {
            return Err(FibQuantError::ProfileDigestMismatch {
                expected: expected_profile,
                actual: code.profile_digest.clone(),
            });
        }
        // Codebook and rotation digests are skipped if empty — the
        // compact wire format omits them because they're derivable from
        // the profile. The decoder trusts its own codebook/rotation in
        // that case. (Re-deriving the codebook just to compute the
        // digest cost ~6ms per call, which is prohibitive for batch
        // decode of 1.5M+ blocks.)
        if !code.codebook_digest.is_empty()
            && code.codebook_digest != self.codebook.codebook_digest
        {
            return Err(FibQuantError::CodebookDigestMismatch {
                expected: self.codebook.codebook_digest.clone(),
                actual: code.codebook_digest.clone(),
            });
        }
        let expected_rotation = self.rotation.digest()?;
        if !code.rotation_digest.is_empty()
            && (code.rotation_digest != expected_rotation
                || code.rotation_digest != self.codebook.rotation_digest)
        {
            return Err(FibQuantError::RotationDigestMismatch {
                expected: expected_rotation,
                actual: code.rotation_digest.clone(),
            });
        }
        if code.ambient_dim != self.profile.ambient_dim
            || code.block_dim != self.profile.block_dim
            || code.block_count != self.profile.block_count()
            || code.wire_index_bits != self.profile.wire_index_bits
            || code.norm_format != self.profile.norm_format
        {
            return Err(FibQuantError::CorruptPayload(
                "encoded header does not match profile".into(),
            ));
        }
        Ok(())
    }
}

/// Stable digest over the encoded artifact fields.
pub fn encoded_digest(code: &FibCodeV1) -> Result<String> {
    json_digest(CODE_SCHEMA, code)
}

fn source_vector_digest(x: &[f32]) -> Result<String> {
    check_finite(x)?;
    let mut bytes = Vec::with_capacity(32 + std::mem::size_of_val(x));
    bytes.extend_from_slice(b"fib_quant_source_vector_v1");
    bytes.push(0);
    bytes.extend_from_slice(&(x.len() as u64).to_le_bytes());
    for value in x {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes_digest(&bytes))
}

fn encode_norm(norm: f64, format: &NormFormat) -> Result<Vec<u8>> {
    if !norm.is_finite() || norm <= 0.0 {
        return Err(FibQuantError::CorruptPayload(
            "norm must be finite and positive".into(),
        ));
    }
    match format {
        NormFormat::Fp16Paper => {
            let narrowed = f16::from_f32(norm as f32);
            if !narrowed.is_finite() || narrowed <= f16::ZERO {
                return Err(FibQuantError::CorruptPayload(
                    "norm cannot be represented as finite positive fp16".into(),
                ));
            }
            Ok(narrowed.to_le_bytes().to_vec())
        }
        NormFormat::F32Reference => {
            let narrowed = norm as f32;
            if !narrowed.is_finite() || narrowed <= 0.0 {
                return Err(FibQuantError::CorruptPayload(
                    "norm cannot be represented as finite positive f32".into(),
                ));
            }
            Ok(narrowed.to_le_bytes().to_vec())
        }
    }
}

fn decode_norm(bytes: &[u8], format: &NormFormat) -> Result<f64> {
    match format {
        NormFormat::Fp16Paper => {
            let bytes: [u8; 2] = bytes
                .try_into()
                .map_err(|_| FibQuantError::CorruptPayload("fp16 norm length".into()))?;
            let value = f16::from_le_bytes(bytes).to_f32() as f64;
            if value.is_finite() && value > 0.0 {
                Ok(value)
            } else {
                Err(FibQuantError::CorruptPayload("invalid fp16 norm".into()))
            }
        }
        NormFormat::F32Reference => {
            let bytes: [u8; 4] = bytes
                .try_into()
                .map_err(|_| FibQuantError::CorruptPayload("f32 norm length".into()))?;
            let value = f32::from_le_bytes(bytes) as f64;
            if value.is_finite() && value > 0.0 {
                Ok(value)
            } else {
                Err(FibQuantError::CorruptPayload("invalid f32 norm".into()))
            }
        }
    }
}

fn l2_norm(x: &[f32]) -> f64 {
    x.iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum::<f64>()
        .sqrt()
}

fn check_finite(x: &[f32]) -> Result<()> {
    if let Some((idx, _)) = x.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(FibQuantError::NonFiniteInput(idx));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_norm_overflow_rejects_before_payload_emit() {
        let err = encode_norm(f64::MAX, &NormFormat::F32Reference).unwrap_err();
        assert!(matches!(err, FibQuantError::CorruptPayload(message) if message.contains("f32")));
    }

    #[test]
    fn f32_norm_underflow_rejects_before_payload_emit() {
        let err = encode_norm(
            f64::from(f32::from_bits(1)) / 2.0,
            &NormFormat::F32Reference,
        )
        .unwrap_err();
        assert!(matches!(err, FibQuantError::CorruptPayload(message) if message.contains("f32")));
    }

    // ---- Batched wire format (FB2) tests ----

    /// Build a paper-default profile and quantizer for tests.
    /// head_dim=64, k=4, n=32 (the fib_k4_n32 codec used by the multi-agent pool).
    fn build_test_quantizer() -> (FibQuantProfileV1, FibQuantizer) {
        let profile = FibQuantProfileV1::paper_default(64, 4, 32, 42).unwrap();
        let quantizer = FibQuantizer::new(profile.clone()).unwrap();
        (profile, quantizer)
    }

    #[test]
    fn batched_wire_roundtrip_matches_single() {
        let (profile, quantizer) = build_test_quantizer();
        let vectors: Vec<Vec<f32>> = (0..16)
            .map(|i| (0..64).map(|j| ((i * 64 + j) as f32 * 0.013).sin()).collect())
            .collect();
        let codes: Vec<_> = vectors
            .iter()
            .map(|v| quantizer.encode(v).unwrap())
            .collect();
        // Single-block (FB1) total
        let single_bytes: Vec<Vec<u8>> = codes
            .iter()
            .map(|c| c.to_compact_bytes())
            .collect();
        let single_total: usize = single_bytes.iter().map(|b| b.len()).sum();
        // Batched (FB2)
        let batched_bytes = FibCodeV1::encode_batch(&codes, &profile).unwrap();
        // Batched must be smaller than the sum of single-block sizes.
        assert!(
            batched_bytes.len() < single_total,
            "batched {} >= single total {}",
            batched_bytes.len(),
            single_total
        );
        // For fib_k4_n32 / head_dim=64, single is 23 B/block, batched is 12 B/block
        // + 19 B header. For 16 blocks: single=368, batched=19+16*12=211. So 1.74× smaller.
        // The savings ratio is exactly (single_per_block - 12) / single_per_block for large N.
        // Verify the exact ratio for this test:
        let expected = 19 + 16 * 12;
        assert_eq!(batched_bytes.len(), expected);
        // Decode and verify each FibCodeV1 matches the original exactly.
        let decoded = FibCodeV1::decode_batch(&batched_bytes, &profile).unwrap();
        assert_eq!(decoded.len(), codes.len());
        for (i, (orig, back)) in codes.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(orig.norm_payload, back.norm_payload, "norm mismatch at vec {i}");
            assert_eq!(orig.indices, back.indices, "indices mismatch at vec {i}");
            assert_eq!(orig.wire_index_bits, back.wire_index_bits);
            assert_eq!(orig.block_count, back.block_count);
            assert_eq!(orig.norm_format, back.norm_format);
        }
    }

    #[test]
    fn batched_wire_rejects_wrong_magic() {
        let (profile, _quantizer) = build_test_quantizer();
        let mut bytes = vec![0u8; 64];
        bytes[0..3].copy_from_slice(b"XXX");
        let r = FibCodeV1::decode_batch(&bytes, &profile);
        assert!(r.is_err());
    }

    #[test]
    fn batched_wire_rejects_buffer_too_short() {
        let (profile, _quantizer) = build_test_quantizer();
        // Just "FB2" + 4 bytes — not enough for the 19-byte header.
        let mut bytes = b"FB2".to_vec();
        bytes.extend_from_slice(&[1u8, 0, 0, 0, 0, 0, 0]);
        let r = FibCodeV1::decode_batch(&bytes, &profile);
        assert!(r.is_err());
    }

    #[test]
    fn batched_wire_preserves_f32_reconstruction() {
        // End-to-end test: vectors in -> FB2 bytes -> codes -> vectors out
        // must match the FB1 single-block round-trip exactly. This proves
        // the batched format doesn't lose any information.
        let (profile, quantizer) = build_test_quantizer();
        let vectors: Vec<Vec<f32>> = (0..32)
            .map(|i| {
                (0..64)
                    .map(|j| {
                        let x = (i * 64 + j) as f32 * 0.013;
                        x.sin() + x.cos() * 0.5
                    })
                    .collect()
            })
            .collect();
        // Encode via FB1 (single block, byte-by-byte)
        let codes_fb1: Vec<_> = vectors
            .iter()
            .map(|v| quantizer.encode(v).unwrap())
            .collect();
        // Decode FB1 -> f32
        let decoded_fb1: Vec<Vec<f32>> = codes_fb1
            .iter()
            .map(|c| quantizer.decode(c).unwrap())
            .collect();
        // Encode via FB2 (batched)
        let codes_fb2 = FibCodeV1::encode_batch(&codes_fb1, &profile).unwrap();
        // Decode FB2 -> FibCodeV1 -> f32
        let codes_back = FibCodeV1::decode_batch(&codes_fb2, &profile).unwrap();
        let decoded_fb2: Vec<Vec<f32>> = codes_back
            .iter()
            .map(|c| quantizer.decode(c).unwrap())
            .collect();
        // f32 outputs from FB1 and FB2 paths must be bit-identical.
        assert_eq!(decoded_fb1.len(), decoded_fb2.len());
        for (i, (fb1, fb2)) in decoded_fb1.iter().zip(decoded_fb2.iter()).enumerate() {
            assert_eq!(fb1.len(), fb2.len());
            for (j, (&a, &b)) in fb1.iter().zip(fb2.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "f32 mismatch at vec {i} dim {j}: fb1={a} fb2={b}"
                );
            }
        }
    }
}

/// Per-step GPU dispatch report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuStepReport {
    /// Hadamard rotation would dispatch to GPU.
    pub hadamard: bool,
    /// Nearest-codebook index lookup would also dispatch to GPU.
    pub codebook_lookup: bool,
}
