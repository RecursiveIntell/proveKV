#![cfg(feature = "kv")]

use fib_quant::kv::{
    calibrate_kv_tensor, compare_attention_fixture, KvAttentionKind, KvDType, KvRole, KvRopeState,
    KvTensorShapeV1,
};

#[test]
fn synthetic_attention_quality_is_stable() {
    let head_dim = 4usize;
    let query = vec![0.2, -0.1, 0.3, 0.4];
    let keys = vec![0.1, 0.2, 0.3, 0.4, -0.2, 0.1, 0.0, 0.5, 0.4, 0.3, 0.2, 0.1];
    let decoded_keys: Vec<f32> = keys.iter().map(|value| value * 0.98).collect();
    let values = vec![
        0.3, 0.1, -0.1, 0.2, 0.0, 0.4, 0.2, -0.2, 0.5, -0.1, 0.1, 0.3,
    ];
    let decoded_values: Vec<f32> = values.iter().map(|value| value * 1.01).collect();
    let report = compare_attention_fixture(
        &query,
        &keys,
        &decoded_keys,
        &values,
        &decoded_values,
        head_dim,
        2,
    )
    .unwrap();
    report.validate().unwrap();
    assert!(report.key_logit_mse.unwrap() >= 0.0);
    assert!(report.topk_attention_agreement.unwrap() > 0.0);
}

#[test]
fn calibration_recommends_role_axes() {
    let key_shape = KvTensorShapeV1::new(
        KvRole::Key,
        KvAttentionKind::Mha,
        1,
        1,
        1,
        1,
        3,
        4,
        KvDType::F32,
        KvRopeState::PostRope,
    );
    let values = vec![0.1; key_shape.element_count().unwrap()];
    let summary = calibrate_kv_tensor(&key_shape, &values, 0.3).unwrap();
    summary.validate().unwrap();
    assert_eq!(summary.role, KvRole::Key);
    assert!(summary.calibration_digest.starts_with("blake3:"));
}
