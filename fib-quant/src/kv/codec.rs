use serde::{Deserialize, Serialize};

use crate::{FibQuantError, FibQuantizer, Result};

use super::{
    block::{KvBlockEncodingV1, KvEncodedBlockV1},
    layout::KvCacheLayoutV1,
    page::KvEncodedPageV1,
    profile::{KvAxisPolicyV1, KvCompressionProfileV1, KvFallbackModeV1},
    receipt::{
        kv_tensor_digest, now_unix_seconds, KvCompressionReceiptV1, KvDecodeReceiptV1,
        KvOperationKindV1, KV_RECEIPT_SCHEMA,
    },
    shape::KvTensorShapeV1,
};

/// Encoded tensor artifact with pages and compression receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvEncodedTensorV1 {
    /// Logical shape.
    pub shape: KvTensorShapeV1,
    /// Physical layout.
    pub layout: KvCacheLayoutV1,
    /// Compression profile.
    pub profile: KvCompressionProfileV1,
    /// Encoded pages.
    pub pages: Vec<KvEncodedPageV1>,
    /// Compression receipt.
    pub receipt: KvCompressionReceiptV1,
}

/// Decoded tensor and receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvDecodedTensorV1 {
    /// Canonical contiguous f32 values.
    pub values: Vec<f32>,
    /// Decode receipt.
    pub receipt: KvDecodeReceiptV1,
}

/// Encode a canonical contiguous f32 KV tensor.
pub fn encode_kv_tensor(
    shape: KvTensorShapeV1,
    layout: KvCacheLayoutV1,
    profile: KvCompressionProfileV1,
    values: &[f32],
) -> Result<KvEncodedTensorV1> {
    shape.validate()?;
    layout.validate_for_shape(&shape)?;
    profile.validate_for_shape(&shape)?;
    if values.len() != shape.element_count()? {
        return Err(FibQuantError::CorruptPayload(format!(
            "kv input has {} values, expected {}",
            values.len(),
            shape.element_count()?
        )));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(FibQuantError::CorruptPayload(
            "kv input contains non-finite value".into(),
        ));
    }

    let quantizer = build_quantizer(&profile)?;
    let source_digest = kv_tensor_digest(values)?;
    let profile_digest = profile.digest(&shape)?;
    let mut pages = Vec::new();
    let mut compressed_blocks = 0u32;
    let mut raw_fallback_blocks = 0u32;
    let mut fallback_reasons = Vec::new();
    let page_count = profile.page_geometry.page_count(&shape)?;

    for page_id in 0..page_count {
        let token_start = page_id * profile.page_geometry.tokens_per_page;
        let token_end = (token_start + profile.page_geometry.tokens_per_page).min(shape.tokens);
        let token_count = token_end - token_start;
        let mut blocks = Vec::new();
        for batch in 0..shape.batch {
            for layer in 0..shape.layers {
                for head in 0..shape.kv_heads {
                    for token in token_start..token_end {
                        let block_id = blocks.len() as u32;
                        let vector = vector_slice(values, &shape, batch, layer, head, token)?;
                        let protected = profile
                            .protected_policy
                            .is_protected(&shape, layer, head, token);
                        let block = if protected {
                            raw_block(
                                block_id,
                                batch,
                                layer,
                                head,
                                token,
                                vector,
                                profile.page_geometry.encoded_block_bytes,
                                "protected_region",
                            )
                        } else {
                            encode_vector_block(
                                &quantizer, &profile, block_id, batch, layer, head, token, vector,
                            )?
                        };
                        if block.raw_fallback {
                            raw_fallback_blocks += 1;
                            if !fallback_reasons.contains(&block.reason) {
                                fallback_reasons.push(block.reason.clone());
                            }
                        } else {
                            compressed_blocks += 1;
                        }
                        blocks.push(block);
                    }
                }
            }
        }
        pages.push(KvEncodedPageV1::new(
            page_id,
            token_start,
            token_count,
            source_digest.clone(),
            profile_digest.clone(),
            &shape,
            profile.page_geometry.clone(),
            blocks,
        )?);
    }

    let page_digests = pages.iter().map(|page| page.page_digest.clone()).collect();
    let receipt = KvCompressionReceiptV1 {
        schema_version: KV_RECEIPT_SCHEMA.into(),
        operation_kind: KvOperationKindV1::Compress,
        source_digest,
        profile_digest,
        shape_digest: shape.digest()?,
        page_digests,
        codebook_digest: profile.codebook_digest.clone(),
        rotation_digest: profile.rotation_digest.clone(),
        encoded_pages: pages.len() as u32,
        compressed_blocks,
        raw_fallback_blocks,
        fallback_reasons,
        recorded_unix_seconds: now_unix_seconds(),
    };
    Ok(KvEncodedTensorV1 {
        shape,
        layout,
        profile,
        pages,
        receipt,
    })
}

/// Decode encoded pages into canonical contiguous f32 values.
pub fn decode_kv_pages(encoded: &KvEncodedTensorV1) -> Result<KvDecodedTensorV1> {
    encoded.shape.validate()?;
    encoded.layout.validate_for_shape(&encoded.shape)?;
    encoded.profile.validate_for_shape(&encoded.shape)?;
    encoded.receipt.validate()?;
    let profile_digest = encoded.profile.digest(&encoded.shape)?;
    if encoded.receipt.profile_digest != profile_digest {
        return Err(FibQuantError::ProfileDigestMismatch {
            expected: profile_digest,
            actual: encoded.receipt.profile_digest.clone(),
        });
    }
    let quantizer = build_quantizer(&encoded.profile)?;
    let mut values = vec![0.0; encoded.shape.element_count()?];
    let mut page_digests = Vec::with_capacity(encoded.pages.len());
    let mut raw_fallback_blocks = 0u32;
    for page in &encoded.pages {
        page.validate(&encoded.shape)?;
        if page.profile_digest != encoded.receipt.profile_digest {
            return Err(FibQuantError::ProfileDigestMismatch {
                expected: encoded.receipt.profile_digest.clone(),
                actual: page.profile_digest.clone(),
            });
        }
        page_digests.push(page.page_digest.clone());
        for block in &page.encoded_blocks {
            if block.batch >= encoded.shape.batch
                || block.layer >= encoded.shape.layers
                || block.kv_head >= encoded.shape.kv_heads
                || block.token >= encoded.shape.tokens
            {
                return Err(FibQuantError::CorruptPayload(
                    "kv block index outside shape".into(),
                ));
            }
            let decoded = match &block.encoding {
                KvBlockEncodingV1::RawF32 { values } => {
                    raw_fallback_blocks += 1;
                    values.clone()
                }
                KvBlockEncodingV1::FibQuant { code } => quantizer.decode(code)?,
            };
            if decoded.len() != encoded.shape.head_dim as usize {
                return Err(FibQuantError::CorruptPayload(
                    "decoded kv vector head_dim mismatch".into(),
                ));
            }
            let out = vector_slice_mut(
                &mut values,
                &encoded.shape,
                block.batch,
                block.layer,
                block.kv_head,
                block.token,
            )?;
            out.copy_from_slice(&decoded);
        }
    }
    let decoded_digest = kv_tensor_digest(&values)?;
    Ok(KvDecodedTensorV1 {
        values,
        receipt: KvDecodeReceiptV1 {
            schema_version: KV_RECEIPT_SCHEMA.into(),
            operation_kind: KvOperationKindV1::Decode,
            decoded_digest,
            profile_digest: encoded.receipt.profile_digest.clone(),
            shape_digest: encoded.shape.digest()?,
            page_digests,
            codebook_digest: encoded.profile.codebook_digest.clone(),
            rotation_digest: encoded.profile.rotation_digest.clone(),
            decoded_pages: encoded.pages.len() as u32,
            raw_fallback_blocks,
            recorded_unix_seconds: now_unix_seconds(),
        },
    })
}

fn build_quantizer(profile: &KvCompressionProfileV1) -> Result<FibQuantizer> {
    let quantizer = FibQuantizer::new(profile.fib_profile.clone())?;
    if quantizer.codebook().codebook_digest != profile.codebook_digest {
        return Err(FibQuantError::CodebookDigestMismatch {
            expected: quantizer.codebook().codebook_digest.clone(),
            actual: profile.codebook_digest.clone(),
        });
    }
    Ok(quantizer)
}

#[allow(clippy::too_many_arguments)]
fn encode_vector_block(
    quantizer: &FibQuantizer,
    profile: &KvCompressionProfileV1,
    block_id: u32,
    batch: u32,
    layer: u32,
    head: u32,
    token: u32,
    vector: &[f32],
) -> Result<KvEncodedBlockV1> {
    match profile.axis_policy {
        KvAxisPolicyV1::Raw => Ok(raw_block(
            block_id,
            batch,
            layer,
            head,
            token,
            vector,
            profile.page_geometry.encoded_block_bytes,
            "raw_axis_policy",
        )),
        KvAxisPolicyV1::PerToken => match quantizer.encode(vector) {
            Ok(code) => Ok(KvEncodedBlockV1::fib_quant(
                block_id,
                batch,
                layer,
                head,
                token,
                code,
                profile.page_geometry.encoded_block_bytes,
                "fib_quant_per_token",
            )),
            Err(err) if profile.fallback_policy.mode == KvFallbackModeV1::KeepRaw => Ok(raw_block(
                block_id,
                batch,
                layer,
                head,
                token,
                vector,
                profile.page_geometry.encoded_block_bytes,
                format!("encode_fallback:{err}"),
            )),
            Err(err) => Err(err),
        },
        KvAxisPolicyV1::PerChannel | KvAxisPolicyV1::RoleAwareKiviStyle => {
            if profile.fallback_policy.mode == KvFallbackModeV1::KeepRaw {
                Ok(raw_block(
                    block_id,
                    batch,
                    layer,
                    head,
                    token,
                    vector,
                    profile.page_geometry.encoded_block_bytes,
                    "unsupported_axis_raw_fallback",
                ))
            } else {
                Err(FibQuantError::DependencyUnsupported(
                    "CPU reference codec supports per-token FibQuant compression only".into(),
                ))
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn raw_block(
    block_id: u32,
    batch: u32,
    layer: u32,
    head: u32,
    token: u32,
    vector: &[f32],
    fixed_size_bytes: u32,
    reason: impl Into<String>,
) -> KvEncodedBlockV1 {
    KvEncodedBlockV1::raw(
        block_id,
        batch,
        layer,
        head,
        token,
        vector.to_vec(),
        fixed_size_bytes,
        reason,
    )
}

fn vector_offset(
    shape: &KvTensorShapeV1,
    batch: u32,
    layer: u32,
    head: u32,
    token: u32,
) -> Result<usize> {
    if batch >= shape.batch
        || layer >= shape.layers
        || head >= shape.kv_heads
        || token >= shape.tokens
    {
        return Err(FibQuantError::CorruptPayload(
            "kv vector index outside shape".into(),
        ));
    }
    let vectors_before = (((batch as usize * shape.layers as usize + layer as usize)
        * shape.kv_heads as usize
        + head as usize)
        * shape.tokens as usize)
        + token as usize;
    vectors_before
        .checked_mul(shape.head_dim as usize)
        .ok_or_else(|| FibQuantError::ResourceLimitExceeded("kv vector offset overflow".into()))
}

fn vector_slice<'a>(
    values: &'a [f32],
    shape: &KvTensorShapeV1,
    batch: u32,
    layer: u32,
    head: u32,
    token: u32,
) -> Result<&'a [f32]> {
    let start = vector_offset(shape, batch, layer, head, token)?;
    let end = start + shape.head_dim as usize;
    values
        .get(start..end)
        .ok_or_else(|| FibQuantError::CorruptPayload("kv vector slice out of bounds".into()))
}

fn vector_slice_mut<'a>(
    values: &'a mut [f32],
    shape: &KvTensorShapeV1,
    batch: u32,
    layer: u32,
    head: u32,
    token: u32,
) -> Result<&'a mut [f32]> {
    let start = vector_offset(shape, batch, layer, head, token)?;
    let end = start + shape.head_dim as usize;
    values
        .get_mut(start..end)
        .ok_or_else(|| FibQuantError::CorruptPayload("kv vector slice out of bounds".into()))
}
