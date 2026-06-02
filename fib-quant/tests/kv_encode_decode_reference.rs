#![cfg(feature = "kv")]

use fib_quant::kv::{
    decode_kv_pages, encode_kv_tensor, KvAttentionKind, KvAxisPolicyV1, KvCacheLayoutV1,
    KvCompressionProfileV1, KvDType, KvPageGeometryV1, KvRole, KvRopeState, KvTensorShapeV1,
};
use fib_quant::{FibQuantProfileV1, FibQuantizer};

fn fixture(
    axis: KvAxisPolicyV1,
) -> (
    KvTensorShapeV1,
    KvCacheLayoutV1,
    KvCompressionProfileV1,
    Vec<f32>,
) {
    let shape = KvTensorShapeV1::new(
        KvRole::Value,
        KvAttentionKind::Mha,
        1,
        1,
        1,
        1,
        4,
        8,
        KvDType::F32,
        KvRopeState::NotApplicable,
    );
    let layout = KvCacheLayoutV1::canonical(&shape).unwrap();
    let mut fib_profile = FibQuantProfileV1::paper_default(8, 2, 8, 13).unwrap();
    fib_profile.training_samples = 64;
    fib_profile.lloyd_restarts = 1;
    fib_profile.lloyd_iterations = 1;
    let quantizer = FibQuantizer::new(fib_profile.clone()).unwrap();
    let profile = KvCompressionProfileV1::from_parts(
        "encode-decode",
        &shape,
        fib_profile,
        quantizer.codebook().codebook_digest.clone(),
        axis,
        KvPageGeometryV1::new(2, 8, 64),
    )
    .unwrap();
    let values = (0..shape.element_count().unwrap())
        .map(|idx| ((idx as f32 + 1.0) * 0.07).sin() + 0.25)
        .collect();
    (shape, layout, profile, values)
}

#[test]
fn per_token_encode_decode_preserves_shape_and_receipts() {
    let (shape, layout, profile, values) = fixture(KvAxisPolicyV1::PerToken);
    let encoded = encode_kv_tensor(shape.clone(), layout, profile, &values).unwrap();
    assert_eq!(encoded.shape, shape);
    assert_eq!(encoded.receipt.encoded_pages, 2);
    assert!(encoded.receipt.compressed_blocks > 0);
    assert_eq!(encoded.receipt.page_digests.len(), encoded.pages.len());

    let decoded = decode_kv_pages(&encoded).unwrap();
    assert_eq!(decoded.values.len(), values.len());
    assert_eq!(decoded.receipt.decoded_pages, encoded.pages.len() as u32);
}

#[test]
fn unsupported_axis_uses_raw_fallback() {
    let (shape, layout, profile, values) = fixture(KvAxisPolicyV1::PerChannel);
    let encoded = encode_kv_tensor(shape, layout, profile, &values).unwrap();
    assert_eq!(encoded.receipt.compressed_blocks, 0);
    assert_eq!(encoded.receipt.raw_fallback_blocks, 4);
    let decoded = decode_kv_pages(&encoded).unwrap();
    assert_eq!(decoded.values, values);
}
