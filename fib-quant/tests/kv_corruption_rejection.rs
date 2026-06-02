#![cfg(feature = "kv")]

use fib_quant::kv::{
    decode_kv_pages, encode_kv_tensor, KvAttentionKind, KvAxisPolicyV1, KvCacheLayoutV1,
    KvCompressionProfileV1, KvDType, KvPageGeometryV1, KvRole, KvRopeState, KvTensorShapeV1,
};
use fib_quant::{FibQuantProfileV1, FibQuantizer};

#[test]
fn page_digest_tamper_rejects_decode() {
    let shape = KvTensorShapeV1::new(
        KvRole::Value,
        KvAttentionKind::Mha,
        1,
        1,
        1,
        1,
        2,
        8,
        KvDType::F32,
        KvRopeState::NotApplicable,
    );
    let layout = KvCacheLayoutV1::canonical(&shape).unwrap();
    let mut fib_profile = FibQuantProfileV1::paper_default(8, 2, 8, 17).unwrap();
    fib_profile.training_samples = 64;
    fib_profile.lloyd_restarts = 1;
    fib_profile.lloyd_iterations = 1;
    let quantizer = FibQuantizer::new(fib_profile.clone()).unwrap();
    let profile = KvCompressionProfileV1::from_parts(
        "corruption",
        &shape,
        fib_profile,
        quantizer.codebook().codebook_digest.clone(),
        KvAxisPolicyV1::PerToken,
        KvPageGeometryV1::new(2, 8, 64),
    )
    .unwrap();
    let values: Vec<f32> = (0..shape.element_count().unwrap())
        .map(|idx| idx as f32 * 0.05 + 0.5)
        .collect();
    let mut encoded = encode_kv_tensor(shape, layout, profile, &values).unwrap();
    encoded.pages[0].page_digest.push_str("-tampered");
    assert!(decode_kv_pages(&encoded).is_err());
}
