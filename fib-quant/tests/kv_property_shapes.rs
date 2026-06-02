#![cfg(feature = "kv")]

use fib_quant::kv::{
    KvAttentionKind, KvCacheLayoutV1, KvDType, KvPageGeometryV1, KvRole, KvRopeState,
    KvTensorShapeV1,
};
use proptest::prelude::*;

proptest! {
    #[test]
    fn bounded_valid_shapes_validate(
        batch in 1u32..3,
        layers in 1u32..3,
        heads in 1u32..4,
        tokens in 1u32..8,
        head_dim_mult in 1u32..5,
    ) {
        let head_dim = head_dim_mult * 2;
        let shape = KvTensorShapeV1::new(
            KvRole::Key,
            KvAttentionKind::Mha,
            batch,
            layers,
            heads,
            heads,
            tokens,
            head_dim,
            KvDType::F32,
            KvRopeState::PostRope,
        );
        prop_assert!(shape.validate().is_ok());
        prop_assert!(shape.validate_block_dim(2).is_ok());
        let layout = KvCacheLayoutV1::canonical(&shape).unwrap();
        prop_assert!(layout.validate_for_shape(&shape).is_ok());
        let geometry = KvPageGeometryV1::new(tokens.min(2), head_dim, 64);
        prop_assert!(geometry.validate_for_shape(&shape).is_ok());
    }
}
