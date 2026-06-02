#![cfg(feature = "kv")]

use fib_quant::kv::{
    KvAttentionKind, KvAxisPolicyV1, KvCacheLayoutV1, KvCompressionProfileV1, KvDType,
    KvPageGeometryV1, KvRole, KvRopeState, KvTensorShapeV1,
};
use fib_quant::{FibQuantProfileV1, FibQuantizer};

fn shape(role: KvRole) -> KvTensorShapeV1 {
    KvTensorShapeV1::new(
        role,
        KvAttentionKind::Mha,
        1,
        1,
        1,
        1,
        4,
        8,
        KvDType::F32,
        match role {
            KvRole::Key => KvRopeState::PostRope,
            _ => KvRopeState::NotApplicable,
        },
    )
}

#[test]
fn shape_layout_and_profile_serde_roundtrip() {
    let shape = shape(KvRole::Key);
    shape.validate().unwrap();
    let shape_digest = shape.digest().unwrap();
    let decoded_shape: KvTensorShapeV1 =
        serde_json::from_str(&serde_json::to_string(&shape).unwrap()).unwrap();
    assert_eq!(decoded_shape.digest().unwrap(), shape_digest);

    let layout = KvCacheLayoutV1::canonical(&shape).unwrap();
    layout.validate_for_shape(&shape).unwrap();
    let decoded_layout: KvCacheLayoutV1 =
        serde_json::from_str(&serde_json::to_string(&layout).unwrap()).unwrap();
    assert_eq!(
        decoded_layout.digest(&shape).unwrap(),
        layout.digest(&shape).unwrap()
    );

    let mut fib_profile = FibQuantProfileV1::paper_default(8, 2, 8, 11).unwrap();
    fib_profile.training_samples = 64;
    fib_profile.lloyd_restarts = 1;
    fib_profile.lloyd_iterations = 1;
    let quantizer = FibQuantizer::new(fib_profile.clone()).unwrap();
    let profile = KvCompressionProfileV1::from_parts(
        "shape-contract",
        &shape,
        fib_profile,
        quantizer.codebook().codebook_digest.clone(),
        KvAxisPolicyV1::PerToken,
        KvPageGeometryV1::new(2, 8, 64),
    )
    .unwrap();
    let profile_digest = profile.digest(&shape).unwrap();
    let decoded_profile: KvCompressionProfileV1 =
        serde_json::from_str(&serde_json::to_string(&profile).unwrap()).unwrap();
    assert_eq!(decoded_profile.digest(&shape).unwrap(), profile_digest);
}

#[test]
fn invalid_shapes_reject() {
    let mut bad = shape(KvRole::Key);
    bad.rope_state = KvRopeState::NotApplicable;
    assert!(bad.validate().is_err());

    let mut bad = shape(KvRole::Value);
    bad.rope_state = KvRopeState::PostRope;
    assert!(bad.validate().is_err());

    let mut bad = shape(KvRole::Key);
    bad.query_heads = 2;
    assert!(bad.validate().is_err());
}
