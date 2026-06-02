use fib_quant::{
    beta_d_k, DirectionMethod, FibQuantError, FibQuantProfileV1, NormFormat, RadiusMethod,
    SourceMode,
};

#[test]
fn profile_digest_is_stable_and_math_sensitive() {
    let base = FibQuantProfileV1::paper_default(8, 2, 8, 11).unwrap();
    let same = FibQuantProfileV1::paper_default(8, 2, 8, 11).unwrap();
    let mut changed = base.clone();
    changed.codebook_seed = changed.codebook_seed.wrapping_add(1);

    assert_eq!(base.digest().unwrap(), same.digest().unwrap());
    assert_ne!(base.digest().unwrap(), changed.digest().unwrap());
    assert!(base.digest().unwrap().starts_with("blake3:"));
}

#[test]
fn invalid_profile_parts_reject() {
    assert!(matches!(
        FibQuantProfileV1::paper_default(0, 2, 8, 0),
        Err(FibQuantError::ZeroDimension)
    ));
    assert!(matches!(
        FibQuantProfileV1::paper_default(8, 0, 8, 0),
        Err(FibQuantError::InvalidBlockDim { .. })
    ));
    assert!(matches!(
        FibQuantProfileV1::paper_default(10, 4, 8, 0),
        Err(FibQuantError::DimensionNotDivisible { .. })
    ));
    assert!(matches!(
        FibQuantProfileV1::paper_default(8, 2, 1, 0),
        Err(FibQuantError::InvalidCodebookSize(1))
    ));
}

#[test]
fn paper_and_wire_rates_are_distinct_when_needed() {
    let profile = FibQuantProfileV1::paper_default(8, 2, 3, 1).unwrap();
    assert_eq!(profile.wire_index_bits, 2);
    assert!((profile.paper_rate_bits_per_coord - (3.0f64).log2() / 2.0).abs() < 1e-12);
    assert_eq!(profile.wire_bits_per_coord, 1.0);
}

#[test]
fn beta_d_k_underflow_edge_stays_finite() {
    let beta = beta_d_k(4, 3).unwrap();
    assert!(beta.is_finite());
    assert!(beta > 0.0);
    assert!((beta - 0.7).abs() < 1.0e-12);
}

#[test]
fn profile_rate_tampering_rejects() {
    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 11).unwrap();
    profile.paper_rate_bits_per_coord += 0.01;
    assert!(matches!(
        profile.validate(),
        Err(FibQuantError::CorruptPayload(_))
    ));

    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 11).unwrap();
    profile.wire_bits_per_coord = f64::NAN;
    assert!(matches!(
        profile.validate(),
        Err(FibQuantError::CorruptPayload(_))
    ));
}

#[test]
fn profile_method_and_schema_tampering_rejects() {
    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 11).unwrap();
    profile.schema_version = "wrong".into();
    assert!(matches!(
        profile.validate(),
        Err(FibQuantError::CorruptPayload(_))
    ));

    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 11).unwrap();
    profile.radius_method = RadiusMethod::BetaQuantile;
    assert!(matches!(
        profile.validate(),
        Err(FibQuantError::CorruptPayload(_))
    ));

    let mut profile = FibQuantProfileV1::paper_default(9, 3, 8, 11).unwrap();
    profile.direction_method = DirectionMethod::RobertsKronecker;
    assert!(matches!(
        profile.validate(),
        Err(FibQuantError::CorruptPayload(_))
    ));
}

#[test]
fn profile_source_norm_and_lloyd_tampering_rejects() {
    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 11).unwrap();
    profile.norm_format = NormFormat::F32Reference;
    assert!(matches!(
        profile.validate(),
        Err(FibQuantError::CorruptPayload(_))
    ));

    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 11).unwrap();
    profile.source_mode = SourceMode::ReferenceGaussianProjection;
    assert!(matches!(
        profile.validate(),
        Err(FibQuantError::CorruptPayload(_))
    ));

    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 11).unwrap();
    profile.rotation_algorithm_version = "wrong".into();
    assert!(matches!(
        profile.validate(),
        Err(FibQuantError::CorruptPayload(_))
    ));

    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 11).unwrap();
    profile.lloyd_iterations = 0;
    assert!(matches!(
        profile.validate(),
        Err(FibQuantError::CorruptPayload(_))
    ));

    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 11).unwrap();
    profile.training_samples = profile.codebook_size - 1;
    assert!(matches!(
        profile.validate(),
        Err(FibQuantError::CorruptPayload(_))
    ));
}
