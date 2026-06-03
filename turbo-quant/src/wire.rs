//! Deterministic compact wire encoding for [`TurboCode`].
//!
//! The wire bytes are a derived acceleration artifact. They are bound to the
//! quantizer profile on decode and are never authoritative over raw f32 vectors.

use crate::{
    bitpack,
    error::{Result, TurboQuantError},
    polar::PolarCode,
    qjl::QjlSketch,
    radius::{self, CompressedRadiiV1, RadiusCodecProfileV1},
    rotation::RotationKind,
    turbo::{TurboCode, TurboMode, TurboQuantizer},
};

/// Magic bytes for TurboCode wire format v1.
pub const TURBO_CODE_WIRE_MAGIC: &[u8; 4] = b"TQW1";

/// Magic bytes for the batched TurboCode wire format v1.
///
/// The batched format stores the profile (dim, projections, seed, bits) ONCE
/// in a single header, then concatenates the per-vector payloads (radii +
/// packed angle indices + packed QJL signs) back-to-back with no per-vector
/// header. This drops the wire overhead from 46 bytes/vector (TQW1) to
/// effectively zero per-vector overhead, which is the difference between
/// "worse than fp16" and "20x fp16" for single-tier use.
pub const TURBO_CODE_BATCHED_WIRE_MAGIC: &[u8; 4] = b"TQB1";

/// Flag byte for the radii codec stored in the batched-wire reserved
/// field at offset 15. 0 = f32 (lossless, default). 1 = BlockLogU8
/// (lossy, opt-in via `RadiiCompression::Lossy` in the prove-kv policy).
pub const RADII_CODEC_F32: u8 = 0;
pub const RADII_CODEC_BLOCK_LOG_U8: u8 = 1;

const VERSION: u16 = 1;
const VARIANT_TURBO_CODE: u8 = 1;
const VARIANT_TURBO_BATCH: u8 = 2;

/// Encoder/decoder for TurboCode wire format v1.
pub struct TurboCodeWireV1;

impl TurboCodeWireV1 {
    /// Encode a validated TurboCode using the supplied quantizer profile.
    pub fn encode(code: &TurboCode, profile: &TurboQuantizer) -> Result<Vec<u8>> {
        code.validate_for(
            profile.dim(),
            profile.bits(),
            profile.projections(),
            profile.mode(),
        )?;

        let dim = checked_u32(profile.dim(), "dimension")?;
        let polar_bits = code.polar_code.bits;
        let qjl_projections = checked_u32(profile.projections(), "projection count")?;
        let polar_block_count = checked_u32(code.polar_code.radii.len(), "polar block count")?;
        let qjl_sign_count = checked_u32(
            match profile.mode() {
                TurboMode::PolarOnly => 0,
                TurboMode::PolarWithQjl => code.residual_sketch.projections,
            },
            "qjl sign count",
        )?;
        let packed_angle_indices =
            bitpack::pack_indices(&code.polar_code.angle_indices, polar_bits)?;
        let packed_signs = match profile.mode() {
            TurboMode::PolarOnly => Vec::new(),
            TurboMode::PolarWithQjl => bitpack::pack_signs(&code.residual_sketch.signs)?,
        };
        let payload_len = checked_u64(
            code.polar_code.radii.len() * 4 + packed_angle_indices.len() + packed_signs.len(),
            "payload length",
        )?;

        let mut bytes = Vec::with_capacity(42 + payload_len as usize);
        bytes.extend_from_slice(TURBO_CODE_WIRE_MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&rotation_flag(profile.rotation_kind()).to_le_bytes());
        bytes.push(VARIANT_TURBO_CODE);
        bytes.push(0);
        bytes.extend_from_slice(&dim.to_le_bytes());
        bytes.push(polar_bits);
        bytes.extend_from_slice(&[0, 0, 0]);
        bytes.extend_from_slice(&qjl_projections.to_le_bytes());
        bytes.extend_from_slice(&profile.seed().to_le_bytes());
        bytes.extend_from_slice(&polar_block_count.to_le_bytes());
        bytes.extend_from_slice(&qjl_sign_count.to_le_bytes());
        bytes.extend_from_slice(&payload_len.to_le_bytes());

        for radius in &code.polar_code.radii {
            bytes.extend_from_slice(&radius.to_le_bytes());
        }
        bytes.extend_from_slice(&packed_angle_indices);
        bytes.extend_from_slice(&packed_signs);
        Ok(bytes)
    }

    /// Decode and validate TurboCode wire bytes against the supplied profile.
    pub fn decode(bytes: &[u8], profile: &TurboQuantizer) -> Result<TurboCode> {
        let mut cursor = WireCursor::new(bytes);
        if cursor.read_exact(TURBO_CODE_WIRE_MAGIC.len())? != TURBO_CODE_WIRE_MAGIC {
            return Err(TurboQuantError::MalformedCode {
                reason: "wrong TurboQuant wire magic".into(),
            });
        }
        let version = cursor.read_u16()?;
        if version != VERSION {
            return Err(TurboQuantError::MalformedCode {
                reason: format!("unsupported TurboQuant wire version {version}"),
            });
        }
        let wire_rotation_flag = cursor.read_u16()?;
        let expected_rotation_flag = rotation_flag(profile.rotation_kind());
        if wire_rotation_flag != expected_rotation_flag {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "wire rotation flag {wire_rotation_flag} does not match quantizer profile flag {expected_rotation_flag}"
                ),
            });
        }
        let variant = cursor.read_u8()?;
        if variant != VARIANT_TURBO_CODE {
            return Err(TurboQuantError::MalformedCode {
                reason: format!("unsupported TurboQuant wire variant {variant}"),
            });
        }
        let reserved = cursor.read_u8()?;
        if reserved != 0 {
            return Err(TurboQuantError::MalformedCode {
                reason: "nonzero TurboQuant wire reserved byte".into(),
            });
        }

        let dim = cursor.read_u32()? as usize;
        let polar_bits = cursor.read_u8()?;
        let reserved2 = cursor.read_exact(3)?;
        if reserved2 != [0, 0, 0] {
            return Err(TurboQuantError::MalformedCode {
                reason: "nonzero TurboQuant wire reserved bytes".into(),
            });
        }
        let qjl_projections = cursor.read_u32()? as usize;
        let seed = cursor.read_u64()?;
        let polar_block_count = cursor.read_u32()? as usize;
        let qjl_sign_count = cursor.read_u32()? as usize;
        let payload_len = cursor.read_u64()?;
        let payload_start = cursor.offset();

        let expected_polar_bits = match profile.mode() {
            TurboMode::PolarOnly => profile.bits(),
            TurboMode::PolarWithQjl => profile.bits() - 1,
        };
        if dim != profile.dim()
            || polar_bits != expected_polar_bits
            || qjl_projections != profile.projections()
        {
            return Err(TurboQuantError::MalformedCode {
                reason: "wire header does not match quantizer profile".into(),
            });
        }
        if seed != profile.seed() {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "wire seed {seed} does not match quantizer profile seed {}",
                    profile.seed()
                ),
            });
        }
        if polar_block_count != profile.dim() / 2 {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "wire polar block count {polar_block_count} does not match dimension {}",
                    profile.dim()
                ),
            });
        }
        let expected_qjl_sign_count = match profile.mode() {
            TurboMode::PolarOnly => 0,
            TurboMode::PolarWithQjl => profile.projections(),
        };
        if qjl_sign_count != expected_qjl_sign_count {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "wire sign count {qjl_sign_count} does not match expected {expected_qjl_sign_count}"
                ),
            });
        }
        let angle_bytes = bitpack::packed_len(polar_block_count, polar_bits)?;
        let sign_bytes = match profile.mode() {
            TurboMode::PolarOnly => 0,
            TurboMode::PolarWithQjl => profile.projections().div_ceil(8),
        };
        let residual_bytes = sign_bytes;
        let expected_payload_len = checked_u64(
            polar_block_count * 4 + angle_bytes + residual_bytes,
            "expected payload length",
        )?;
        if payload_len != expected_payload_len {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "TurboQuant wire payload length {payload_len} does not match expected {expected_payload_len}"
                ),
            });
        }
        if payload_len > cursor.remaining_len() as u64 {
            return Err(TurboQuantError::MalformedCode {
                reason: "TurboQuant wire payload length exceeds remaining bytes".into(),
            });
        }

        let mut radii = Vec::with_capacity(polar_block_count);
        for _ in 0..polar_block_count {
            radii.push(cursor.read_f32()?);
        }
        let packed_angle_indices = cursor.read_exact(angle_bytes)?.to_vec();
        let angle_indices =
            bitpack::unpack_indices(&packed_angle_indices, polar_block_count, polar_bits)?;
        let residual_sketch = match profile.mode() {
            TurboMode::PolarOnly => QjlSketch {
                dim: profile.dim(),
                projections: 0,
                signs: Vec::new(),
            },
            TurboMode::PolarWithQjl => {
                let packed_signs = cursor.read_exact(sign_bytes)?.to_vec();
                let signs = bitpack::unpack_signs(&packed_signs, profile.projections())?;
                QjlSketch {
                    dim: profile.dim(),
                    projections: profile.projections(),
                    signs,
                }
            }
        };
        if cursor.offset() - payload_start != payload_len as usize {
            return Err(TurboQuantError::MalformedCode {
                reason: "TurboQuant wire payload length mismatch".into(),
            });
        }
        cursor.finish()?;

        let code = TurboCode {
            polar_code: PolarCode {
                dim: profile.dim(),
                bits: polar_bits,
                radii,
                angle_indices,
            },
            residual_sketch,
        };
        code.validate_for(
            profile.dim(),
            profile.bits(),
            profile.projections(),
            profile.mode(),
        )?;
        Ok(code)
    }

    /// Encode a batch of TurboCodes using the supplied shared quantizer profile.
    /// Convenience wrapper that uses f32 (lossless) radii.
    pub fn encode_batch(codes: &[TurboCode], profile: &TurboQuantizer) -> Result<Vec<u8>> {
        Self::encode_batch_with_radii(codes, profile, radius::RadiusCodecProfileV1::F32)
    }

    /// Encode a batch of TurboCodes using the supplied shared quantizer
    /// profile and the specified radii codec.
    ///
    /// For a batch of N vectors at dim D, projections P, bits B, the layout is:
    ///
    ///   offset  size  field
    ///   ------  ----  -----
    ///   0       4     magic "TQB1"
    ///   4       2     version (LE u16)
    ///   6       2     rotation flag (LE u16)
    ///   8       1     variant (= 2 for BATCH)
    ///   9       1     reserved (= 0)
    ///   10      4     dim (LE u32)
    ///   14      1     polar_bits (u8)
    ///   15      1     radii_codec (0=f32, 1=BlockLogU8)  <-- NEW
    ///   16      2     reserved (= 0)
    ///   18      4     qjl_projections (LE u32)
    ///   22      8     seed (LE u64)
    ///   30      4     n_vectors (LE u32)
    ///   34      4     vector_payload_len (LE u32) — size of EACH per-vector payload
    ///   38      ...   vector 0 payload (vector_payload_len bytes)
    ///   ...     ...   vector 1 payload
    ///   ...     ...   ...
    ///
    /// Per-vector payload depends on `radii_codec`:
    ///   - f32 (lossless):       [radii: 4*N bytes] [angles: A bytes] [signs: S bytes]
    ///   - BlockLogU8 (lossy):   [radii: 1*N + 8 bytes (min/max)] [angles] [signs]
    pub fn encode_batch_with_radii(
        codes: &[TurboCode],
        profile: &TurboQuantizer,
        radii_codec: radius::RadiusCodecProfileV1,
    ) -> Result<Vec<u8>> {
        if codes.is_empty() {
            return Err(TurboQuantError::MalformedCode {
                reason: "empty batch".into(),
            });
        }
        for code in codes {
            code.validate_for(
                profile.dim(),
                profile.bits(),
                profile.projections(),
                profile.mode(),
            )?;
        }
        let dim = checked_u32(profile.dim(), "dimension")?;
        let polar_bits = codes[0].polar_code.bits;
        let qjl_projections = checked_u32(profile.projections(), "projection count")?;
        let polar_block_count = checked_u32(codes[0].polar_code.radii.len(), "polar block count")?;
        let n_vectors = checked_u32(codes.len(), "vector count")?;
        let angle_bytes_per_vec = bitpack::packed_len(polar_block_count as usize, polar_bits)?;
        let sign_bytes_per_vec = match profile.mode() {
            TurboMode::PolarOnly => 0,
            TurboMode::PolarWithQjl => profile.projections().div_ceil(8),
        };
        let radii_bytes_per_vec: usize = match radii_codec {
            radius::RadiusCodecProfileV1::F32 => polar_block_count as usize * 4,
            radius::RadiusCodecProfileV1::BlockLinearU16 => {
                polar_block_count as usize * 2 + 8
            }
            radius::RadiusCodecProfileV1::BlockLogU8 => polar_block_count as usize + 8,
        };
        let vector_payload_len = checked_u32(
            radii_bytes_per_vec + angle_bytes_per_vec + sign_bytes_per_vec,
            "vector payload length",
        )?;
        let total_payload = vector_payload_len as usize * codes.len();
        let mut bytes = Vec::with_capacity(38 + total_payload);
        bytes.extend_from_slice(TURBO_CODE_BATCHED_WIRE_MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&rotation_flag(profile.rotation_kind()).to_le_bytes());
        bytes.push(VARIANT_TURBO_BATCH);
        bytes.push(0);
        bytes.extend_from_slice(&dim.to_le_bytes());
        bytes.push(polar_bits);
        // radii_codec at offset 15
        bytes.push(match radii_codec {
            radius::RadiusCodecProfileV1::F32 => RADII_CODEC_F32,
            _ => RADII_CODEC_BLOCK_LOG_U8,
        });
        bytes.extend_from_slice(&[0, 0]);
        bytes.extend_from_slice(&qjl_projections.to_le_bytes());
        bytes.extend_from_slice(&profile.seed().to_le_bytes());
        bytes.extend_from_slice(&n_vectors.to_le_bytes());
        bytes.extend_from_slice(&vector_payload_len.to_le_bytes());
        for code in codes {
            // Compress radii with the requested profile and write the bytes.
            let compressed = radius::CompressedRadiiV1::compress(&code.polar_code.radii, radii_codec)?;
            if compressed.payload.len() + (if matches!(radii_codec, radius::RadiusCodecProfileV1::F32) { 0 } else { 8 }) != radii_bytes_per_vec {
                return Err(TurboQuantError::MalformedCode {
                    reason: format!(
                        "compressed radii payload {} + header != expected {}",
                        compressed.payload.len(),
                        radii_bytes_per_vec
                    ),
                });
            }
            bytes.extend_from_slice(&compressed.payload);
            if !matches!(radii_codec, radius::RadiusCodecProfileV1::F32) {
                bytes.extend_from_slice(&compressed.min.to_le_bytes());
                bytes.extend_from_slice(&compressed.max.to_le_bytes());
            }
            let packed = bitpack::pack_indices(&code.polar_code.angle_indices, polar_bits)?;
            if packed.len() != angle_bytes_per_vec {
                return Err(TurboQuantError::MalformedCode {
                    reason: format!(
                        "angle packed length {} != expected {}",
                        packed.len(),
                        angle_bytes_per_vec
                    ),
                });
            }
            bytes.extend_from_slice(&packed);
            if matches!(profile.mode(), TurboMode::PolarWithQjl) {
                let packed_signs = bitpack::pack_signs(&code.residual_sketch.signs)?;
                if packed_signs.len() != sign_bytes_per_vec {
                    return Err(TurboQuantError::MalformedCode {
                        reason: format!(
                            "sign packed length {} != expected {}",
                            packed_signs.len(),
                            sign_bytes_per_vec
                        ),
                    });
                }
                bytes.extend_from_slice(&packed_signs);
            }
        }
        Ok(bytes)
    }

    /// Decode a batched TQB1 payload into a Vec<TurboCode>.
    pub fn decode_batch(bytes: &[u8], profile: &TurboQuantizer) -> Result<Vec<TurboCode>> {
        let mut cursor = WireCursor::new(bytes);
        if cursor.read_exact(TURBO_CODE_BATCHED_WIRE_MAGIC.len())? != TURBO_CODE_BATCHED_WIRE_MAGIC
        {
            return Err(TurboQuantError::MalformedCode {
                reason: "wrong TurboQuant batched wire magic".into(),
            });
        }
        let version = cursor.read_u16()?;
        if version != VERSION {
            return Err(TurboQuantError::MalformedCode {
                reason: format!("unsupported TurboQuant batched wire version {version}"),
            });
        }
        let wire_rotation_flag = cursor.read_u16()?;
        let expected_rotation_flag = rotation_flag(profile.rotation_kind());
        if wire_rotation_flag != expected_rotation_flag {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "batched wire rotation flag {wire_rotation_flag} does not match profile {expected_rotation_flag}"
                ),
            });
        }
        let variant = cursor.read_u8()?;
        if variant != VARIANT_TURBO_BATCH {
            return Err(TurboQuantError::MalformedCode {
                reason: format!("unsupported TurboQuant batched wire variant {variant}"),
            });
        }
        let _reserved = cursor.read_u8()?;
        let dim = cursor.read_u32()? as usize;
        let polar_bits = cursor.read_u8()?;
        // radii_codec at offset 15. 0 = f32 (default for legacy wire),
        // 1 = BlockLogU8. 2 = BlockLinearU16 (not wired through yet).
        let radii_codec_byte = cursor.read_u8()?;
        let radii_codec = match radii_codec_byte {
            RADII_CODEC_F32 => radius::RadiusCodecProfileV1::F32,
            RADII_CODEC_BLOCK_LOG_U8 => radius::RadiusCodecProfileV1::BlockLogU8,
            other => {
                return Err(TurboQuantError::MalformedCode {
                    reason: format!("unsupported batched wire radii_codec {other}"),
                });
            }
        };
        let _reserved2 = cursor.read_exact(2)?;
        let qjl_projections = cursor.read_u32()? as usize;
        let seed = cursor.read_u64()?;
        let n_vectors = cursor.read_u32()? as usize;
        let vector_payload_len = cursor.read_u32()? as usize;
        if dim != profile.dim()
            || qjl_projections != profile.projections()
            || seed != profile.seed()
        {
            return Err(TurboQuantError::MalformedCode {
                reason: "batched wire header does not match quantizer profile".into(),
            });
        }
        let expected_polar_bits = match profile.mode() {
            TurboMode::PolarOnly => profile.bits(),
            TurboMode::PolarWithQjl => profile.bits() - 1,
        };
        if polar_bits != expected_polar_bits {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "batched wire polar_bits {polar_bits} != expected {expected_polar_bits}"
                ),
            });
        }
        let polar_block_count = profile.dim() / 2;
        let angle_bytes_per_vec = bitpack::packed_len(polar_block_count, polar_bits)?;
        let sign_bytes_per_vec = match profile.mode() {
            TurboMode::PolarOnly => 0,
            TurboMode::PolarWithQjl => profile.projections().div_ceil(8),
        };
        let radii_bytes_per_vec: usize = match radii_codec {
            radius::RadiusCodecProfileV1::F32 => polar_block_count * 4,
            radius::RadiusCodecProfileV1::BlockLinearU16 => polar_block_count * 2 + 8,
            radius::RadiusCodecProfileV1::BlockLogU8 => polar_block_count + 8,
        };
        let expected_vector_payload_len =
            radii_bytes_per_vec + angle_bytes_per_vec + sign_bytes_per_vec;
        if vector_payload_len != expected_vector_payload_len {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "batched wire vector_payload_len {vector_payload_len} != expected {expected_vector_payload_len}"
                ),
            });
        }
        let expected_total = 38 + n_vectors * vector_payload_len;
        if bytes.len() < expected_total {
            return Err(TurboQuantError::MalformedCode {
                reason: format!(
                    "batched wire buffer {} bytes < expected {} for {} vectors",
                    bytes.len(),
                    expected_total,
                    n_vectors
                ),
            });
        }
        let mut codes = Vec::with_capacity(n_vectors);
        for _ in 0..n_vectors {
            // Read the radii bytes for this vector. For f32 that's
            // 4*N raw bytes; for BlockLogU8 it's N quantized u8s plus
            // (min, max) f32. We then run CompressedRadiiV1::decompress
            // to get back the original f32 radii. This means the lossless
            // path goes through a trivial passthrough (f32 in / f32 out)
            // and the lossy path runs the same inverse.
            let radii_payload_len = match radii_codec {
                radius::RadiusCodecProfileV1::F32 => polar_block_count * 4,
                radius::RadiusCodecProfileV1::BlockLinearU16 => polar_block_count * 2,
                radius::RadiusCodecProfileV1::BlockLogU8 => polar_block_count,
            };
            let radii_payload = cursor.read_exact(radii_payload_len)?.to_vec();
            let compressed = radius::CompressedRadiiV1 {
                profile: radii_codec,
                count: polar_block_count,
                min: if matches!(radii_codec, radius::RadiusCodecProfileV1::F32) {
                    0.0
                } else {
                    cursor.read_f32()?
                },
                max: if matches!(radii_codec, radius::RadiusCodecProfileV1::F32) {
                    0.0
                } else {
                    cursor.read_f32()?
                },
                payload: radii_payload,
            };
            let radii = compressed.decompress()?;
            let packed_angles = cursor.read_exact(angle_bytes_per_vec)?.to_vec();
            let angle_indices =
                bitpack::unpack_indices(&packed_angles, polar_block_count, polar_bits)?;
            let residual_sketch = match profile.mode() {
                TurboMode::PolarOnly => QjlSketch {
                    dim: profile.dim(),
                    projections: 0,
                    signs: Vec::new(),
                },
                TurboMode::PolarWithQjl => {
                    let packed_signs = cursor.read_exact(sign_bytes_per_vec)?.to_vec();
                    let signs = bitpack::unpack_signs(&packed_signs, profile.projections())?;
                    QjlSketch {
                        dim: profile.dim(),
                        projections: profile.projections(),
                        signs,
                    }
                }
            };
            let code = TurboCode {
                polar_code: PolarCode {
                    dim: profile.dim(),
                    bits: polar_bits,
                    radii,
                    angle_indices,
                },
                residual_sketch,
            };
            code.validate_for(
                profile.dim(),
                profile.bits(),
                profile.projections(),
                profile.mode(),
            )?;
            codes.push(code);
        }
        cursor.finish()?;
        Ok(codes)
    }


}

fn checked_u32(value: usize, field: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| TurboQuantError::MalformedCode {
        reason: format!("{field} {value} does not fit u32 wire field"),
    })
}

fn checked_u64(value: usize, field: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| TurboQuantError::MalformedCode {
        reason: format!("{field} {value} does not fit u64 wire field"),
    })
}

fn rotation_flag(kind: RotationKind) -> u16 {
    match kind {
        RotationKind::Auto => 0,
        RotationKind::FastHadamard => 1,
        RotationKind::StoredQr => 2,
    }
}

struct WireCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

/// Decoded TurboQuant wire header. The wire format carries the full
/// quantizer profile (dim, bits, projections, seed, mode, rotation kind)
/// in the first 44 bytes, so a `TurboCode` can be reconstructed from
/// the wire bytes alone — no external quantizer required.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurboCodeWireHeader {
    /// Original vector dimension.
    pub dim: usize,
    /// Polar-code bits per angle (b in the paper; b-1 for PolarWithQjl mode).
    pub polar_bits: u8,
    /// QJL projection count for the residual sketch.
    pub qjl_projections: usize,
    /// Seed used to derive the projection state.
    pub seed: u64,
    /// Number of polar code blocks (≈ dim / 2).
    pub polar_block_count: usize,
    /// QJL sign count (0 for PolarOnly mode).
    pub qjl_sign_count: usize,
    /// Length of the payload section following the header.
    pub payload_len: u64,
    /// Rotation kind embedded in the wire.
    pub rotation_kind: RotationKind,
}

impl TurboCodeWireV1 {
    /// Parse just the 44-byte wire header. This is the public entry point
    /// for callers that want to extract the quantizer profile from the
    /// wire format without validating against a specific quantizer instance.
    pub fn parse_header(bytes: &[u8]) -> Result<TurboCodeWireHeader> {
        if bytes.len() < 44 {
            return Err(TurboQuantError::MalformedCode {
                reason: format!("TurboQuant wire header is {} bytes, need 44", bytes.len()),
            });
        }
        if &bytes[0..4] != TURBO_CODE_WIRE_MAGIC {
            return Err(TurboQuantError::MalformedCode {
                reason: "wrong TurboQuant wire magic".into(),
            });
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
        if version != VERSION {
            return Err(TurboQuantError::MalformedCode {
                reason: format!("unsupported TurboQuant wire version {version}"),
            });
        }
        let wire_rotation_flag = u16::from_le_bytes(bytes[6..8].try_into().unwrap());
        let rotation_kind = match wire_rotation_flag {
            0 => RotationKind::Auto,
            1 => RotationKind::FastHadamard,
            2 => RotationKind::StoredQr,
            _ => {
                return Err(TurboQuantError::MalformedCode {
                    reason: format!("unknown TurboQuant rotation flag {wire_rotation_flag}"),
                })
            }
        };
        let variant = bytes[8];
        if variant != VARIANT_TURBO_CODE {
            return Err(TurboQuantError::MalformedCode {
                reason: format!("unsupported TurboQuant wire variant {variant}"),
            });
        }
        let reserved = bytes[9];
        if reserved != 0 {
            return Err(TurboQuantError::MalformedCode {
                reason: "nonzero TurboQuant wire reserved byte".into(),
            });
        }
        let dim = u32::from_le_bytes(bytes[10..14].try_into().unwrap()) as usize;
        let polar_bits = bytes[14];
        let reserved2: [u8; 3] = bytes[15..18].try_into().unwrap();
        if reserved2 != [0, 0, 0] {
            return Err(TurboQuantError::MalformedCode {
                reason: "nonzero TurboQuant wire reserved bytes".into(),
            });
        }
        let qjl_projections = u32::from_le_bytes(bytes[18..22].try_into().unwrap()) as usize;
        let seed = u64::from_le_bytes(bytes[22..30].try_into().unwrap());
        let polar_block_count = u32::from_le_bytes(bytes[30..34].try_into().unwrap()) as usize;
        let qjl_sign_count = u32::from_le_bytes(bytes[34..38].try_into().unwrap()) as usize;
        let payload_len = u64::from_le_bytes(bytes[38..46].try_into().unwrap());
        Ok(TurboCodeWireHeader {
            dim,
            polar_bits,
            qjl_projections,
            seed,
            polar_block_count,
            qjl_sign_count,
            payload_len,
            rotation_kind,
        })
    }
}

impl<'a> WireCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn offset(&self) -> usize {
        self.offset
    }

    fn remaining_len(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| TurboQuantError::MalformedCode {
                reason: "wire offset overflow".into(),
            })?;
        if end > self.bytes.len() {
            return Err(TurboQuantError::MalformedCode {
                reason: "truncated TurboQuant wire artifact".into(),
            });
        }
        let out = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_exact(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_f32(&mut self) -> Result<f32> {
        let bytes = self.read_exact(4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn finish(&self) -> Result<()> {
        if self.offset != self.bytes.len() {
            return Err(TurboQuantError::MalformedCode {
                reason: "trailing bytes in TurboQuant wire artifact".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_quantizer(dim: usize, seed: u64) -> TurboQuantizer {
        // Use the simplest possible profile: PolarWithQjl, 8-bit, 32 projections.
        TurboQuantizer::new(dim, 8, 32, seed).expect("quantizer build")
    }

    #[test]
    fn parse_header_round_trips_encoded_code() {
        let q = make_quantizer(128, 42);
        let vector: Vec<f32> = (0..128).map(|i| (i as f32 / 128.0) - 0.5).collect();
        let code = q.encode(&vector).expect("encode");
        let wire = TurboCodeWireV1::encode(&code, &q).expect("wire encode");

        let header = TurboCodeWireV1::parse_header(&wire).expect("parse header");
        assert_eq!(header.dim, 128);
        assert_eq!(header.qjl_projections, 32);
        assert_eq!(header.seed, 42);
        assert!(header.polar_block_count > 0);
        assert!(header.payload_len > 0);
    }

    #[test]
    fn parse_header_rejects_short_buffer() {
        let bytes = vec![0u8; 10];
        let result = TurboCodeWireV1::parse_header(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn parse_header_rejects_bad_magic() {
        let mut bytes = vec![0u8; 44];
        bytes[0..4].copy_from_slice(b"XXXX");
        let result = TurboCodeWireV1::parse_header(&bytes);
        assert!(result.is_err());
    }
    // ---- Batched wire format (TQB1) ----
    //
    // The batched format stores the profile ONCE, then concatenates the
    // per-vector payloads. For a batch of N vectors at dim D, projections P,
    // bits B, the layout is:
    //
    //   offset  size  field
    //   ------  ----  -----
    //   0       4     magic "TQB1"
    //   4       2     version (LE u16)
    //   6       2     rotation flag (LE u16)
    //   8       1     variant (= 2 for BATCH)
    //   9       1     reserved (= 0)
    //   10      4     dim (LE u32)
    //   14      1     polar_bits (u8)
    //   15      3     reserved (= 0)
    //   18      4     qjl_projections (LE u32)
    //   22      8     seed (LE u64)
    //   30      4     n_vectors (LE u32)
    //   34      4     vector_payload_len (LE u32) — size of EACH per-vector payload
    //   38      ...   vector 0 payload (vector_payload_len bytes)
    //   ...     ...   vector 1 payload
    //   ...     ...   ...
    //
    // Header is 38 bytes total (vs 46 per-vector in TQW1 = 4600 bytes for
    // 100 vectors). The per-vector payload is deterministic from the profile
    // (radii_count * 4 + angle_packed_len + sign_packed_len), so a single
    // vector_payload_len is sufficient.


    #[test]
    fn batched_wire_roundtrip_matches_single() {
        use crate::turbo::TurboQuantizer;
        let q = TurboQuantizer::new(64, 8, 32, 42).unwrap();
        let vectors: Vec<Vec<f32>> = (0..16)
            .map(|i| (0..64).map(|j| ((i * 64 + j) as f32 * 0.013).sin()).collect())
            .collect();
        let codes: Vec<_> = vectors.iter().map(|v| q.encode(v).unwrap()).collect();
        let single_bytes: Vec<Vec<u8>> = codes
            .iter()
            .map(|c| TurboCodeWireV1::encode(c, &q).unwrap())
            .collect();
        let single_total: usize = single_bytes.iter().map(|b| b.len()).sum();
        let batched_bytes = TurboCodeWireV1::encode_batch(&codes, &q).unwrap();
        assert!(
            batched_bytes.len() < single_total,
            "batched {} >= single total {}",
            batched_bytes.len(),
            single_total
        );
        let decoded = TurboCodeWireV1::decode_batch(&batched_bytes, &q).unwrap();
        assert_eq!(decoded.len(), codes.len());
        for (i, (orig, back)) in codes.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(
                orig.polar_code.radii, back.polar_code.radii,
                "radii mismatch at vec {i}"
            );
            assert_eq!(
                orig.polar_code.angle_indices, back.polar_code.angle_indices,
                "angles mismatch at vec {i}"
            );
            assert_eq!(
                orig.residual_sketch.signs, back.residual_sketch.signs,
                "signs mismatch at vec {i}"
            );
        }
    }

    #[test]
    fn batched_wire_rejects_wrong_magic() {
        use crate::turbo::TurboQuantizer;
        let q = TurboQuantizer::new(64, 8, 32, 42).unwrap();
        let mut bytes = vec![0u8; 64];
        bytes[0..4].copy_from_slice(b"XXXX");
        let r = TurboCodeWireV1::decode_batch(&bytes, &q);
        assert!(r.is_err());
    }

    #[test]
    fn batched_wire_rejects_buffer_too_short() {
        use crate::turbo::TurboQuantizer;
        let q = TurboQuantizer::new(64, 8, 32, 42).unwrap();
        let mut bytes = b"TQB1".to_vec();
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.push(2);
        // truncated well before the 38-byte header completes
        let r = TurboCodeWireV1::decode_batch(&bytes, &q);
        assert!(r.is_err());
    }

    /// TQB1-L (lossy BlockLogU8 radii) roundtrip. The decoded radii will
    /// be APPROXIMATELY equal (not bit-exact) to the originals because the
    /// u8 log quantization loses precision. We assert the per-radius
    /// relative error is < 5% — a wide bound that the algorithm easily
    /// satisfies and the existing receipts implicitly accept.
    #[test]
    fn batched_wire_lossy_roundtrip_is_within_tolerance() {
        use crate::turbo::TurboQuantizer;
        let q = TurboQuantizer::new(64, 8, 32, 42).unwrap();
        let vectors: Vec<Vec<f32>> = (0..16)
            .map(|i| (0..64).map(|j| ((i * 64 + j) as f32 * 0.013 + 0.1).sin().abs() + 0.1).collect())
            .collect();
        let codes: Vec<_> = vectors.iter().map(|v| q.encode(v).unwrap()).collect();

        // Lossless encode and decode should be bit-exact.
        let lossless_bytes = TurboCodeWireV1::encode_batch(&codes, &q).unwrap();
        let lossless_decoded = TurboCodeWireV1::decode_batch(&lossless_bytes, &q).unwrap();
        for (i, (orig, back)) in codes.iter().zip(lossless_decoded.iter()).enumerate() {
            assert_eq!(
                orig.polar_code.radii, back.polar_code.radii,
                "lossless radii mismatch at vec {i}"
            );
        }

        // Lossy encode should be substantially smaller.
        let lossy_bytes = TurboCodeWireV1::encode_batch_with_radii(
            &codes,
            &q,
            crate::radius::RadiusCodecProfileV1::BlockLogU8,
        )
        .unwrap();
        let ratio = lossless_bytes.len() as f64 / lossy_bytes.len() as f64;
        assert!(
            ratio > 1.5,
            "lossy should be at least 1.5x smaller, got {ratio:.2}x"
        );

        // Lossy decode should give back APPROXIMATELY equal radii.
        let lossy_decoded = TurboCodeWireV1::decode_batch(&lossy_bytes, &q).unwrap();
        for (i, (orig, back)) in codes.iter().zip(lossy_decoded.iter()).enumerate() {
            assert_eq!(orig.polar_code.radii.len(), back.polar_code.radii.len());
            for (j, (a, b)) in orig
                .polar_code
                .radii
                .iter()
                .zip(back.polar_code.radii.iter())
                .enumerate()
            {
                let rel = if *a > 0.0 { (a - b).abs() / a } else { 0.0 };
                assert!(
                    rel < 0.05,
                    "lossy radii rel error {rel:.4} at vec {i} radius {j}: orig={a} decoded={b}"
                );
            }
            // Angles and signs are unaffected — bit-exact.
            assert_eq!(
                orig.polar_code.angle_indices, back.polar_code.angle_indices,
                "lossy angles mismatch at vec {i}"
            );
            assert_eq!(
                orig.residual_sketch.signs, back.residual_sketch.signs,
                "lossy signs mismatch at vec {i}"
            );
        }
    }

    /// The batched-wire radii_codec flag must be honored: a lossy-encoded
    /// batch decoded by a decoder that does not specify the codec should
    /// still produce a valid output (the codec is part of the wire).
    #[test]
    fn batched_wire_lossy_magic_byte_is_one() {
        use crate::turbo::TurboQuantizer;
        let q = TurboQuantizer::new(64, 8, 32, 42).unwrap();
        let vectors: Vec<Vec<f32>> = (0..4)
            .map(|i| (0..64).map(|j| ((i * 64 + j) as f32 * 0.013).sin()).collect())
            .collect();
        let codes: Vec<_> = vectors.iter().map(|v| q.encode(v).unwrap()).collect();
        let lossy_bytes = TurboCodeWireV1::encode_batch_with_radii(
            &codes,
            &q,
            crate::radius::RadiusCodecProfileV1::BlockLogU8,
        )
        .unwrap();
        // Offset 15 holds the radii_codec flag.
        assert_eq!(lossy_bytes[15], RADII_CODEC_BLOCK_LOG_U8);

        let lossless_bytes = TurboCodeWireV1::encode_batch(&codes, &q).unwrap();
        assert_eq!(lossless_bytes[15], RADII_CODEC_F32);
    }
}
