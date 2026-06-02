#![cfg(feature = "kv")]

use fib_quant::kv::{
    decide_kv_compression, KvAttentionKind, KvAxisPolicyV1, KvCompressionPolicyV1,
    KvCompressionStrategyV1, KvDType, KvDecisionActionV1, KvProtectedPolicyV1, KvRole, KvRopeState,
    KvTensorShapeV1,
};

fn shape(role: KvRole) -> KvTensorShapeV1 {
    KvTensorShapeV1::new(
        role,
        KvAttentionKind::Gqa,
        1,
        2,
        2,
        4,
        8,
        16,
        KvDType::F32,
        match role {
            KvRole::Key => KvRopeState::PreRope,
            _ => KvRopeState::NotApplicable,
        },
    )
}

#[test]
fn role_aware_policy_selects_different_axes() {
    let policy = KvCompressionPolicyV1::role_aware_baseline();
    let key = decide_kv_compression(&policy, &shape(KvRole::Key), 0, 0, 1, true).unwrap();
    let value = decide_kv_compression(&policy, &shape(KvRole::Value), 0, 0, 1, true).unwrap();
    assert_eq!(key.action, KvDecisionActionV1::Compress);
    assert_eq!(value.action, KvDecisionActionV1::Compress);
    assert_eq!(key.axis_policy, KvAxisPolicyV1::PerChannel);
    assert_eq!(value.axis_policy, KvAxisPolicyV1::PerToken);
}

#[test]
fn protected_and_calibration_missing_paths_keep_raw() {
    let mut policy = KvCompressionPolicyV1 {
        strategy: KvCompressionStrategyV1::FibQuantPerToken,
        protected_policy: KvProtectedPolicyV1 {
            first_tokens_raw: 1,
            last_tokens_raw: 0,
            raw_layers: vec![1],
            raw_heads: Vec::new(),
        },
        require_calibration: true,
        allow_raw_fallback: true,
    };
    let shape = shape(KvRole::Value);
    let protected = decide_kv_compression(&policy, &shape, 0, 0, 0, false).unwrap();
    assert_eq!(protected.action, KvDecisionActionV1::KeepRaw);

    policy.protected_policy.first_tokens_raw = 0;
    let missing_calibration = decide_kv_compression(&policy, &shape, 0, 0, 2, false).unwrap();
    assert_eq!(missing_calibration.action, KvDecisionActionV1::KeepRaw);
}
