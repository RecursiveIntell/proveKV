use fib_quant::{FibQuantError, FibQuantProfileV1, FibQuantizer};
use proptest::prelude::*;

fn small_profile(seed: u64) -> fib_quant::Result<FibQuantProfileV1> {
    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, seed)?;
    profile.training_samples = 64;
    profile.lloyd_restarts = 1;
    profile.lloyd_iterations = 1;
    Ok(profile)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn finite_encode_decode_never_self_invalid(seed in 1u64..10_000, values in prop::collection::vec(-10.0f32..10.0, 8)) {
        prop_assume!(values.iter().any(|value| *value != 0.0));
        let quantizer = FibQuantizer::new(small_profile(seed)?)?;
        let code = quantizer.encode(&values)?;
        let decoded = quantizer.decode(&code)?;
        prop_assert_eq!(decoded.len(), values.len());
        prop_assert!(decoded.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn profile_tamper_rejects_decode() {
    let quantizer = FibQuantizer::new(small_profile(101).unwrap()).unwrap();
    let mut code = quantizer.encode(&[0.25; 8]).unwrap();
    code.profile_digest = "blake3:tampered".into();
    assert!(matches!(
        quantizer.decode(&code),
        Err(FibQuantError::ProfileDigestMismatch { .. })
    ));
}

#[test]
fn codebook_tamper_rejects_decode() {
    let quantizer = FibQuantizer::new(small_profile(102).unwrap()).unwrap();
    let mut code = quantizer.encode(&[0.25; 8]).unwrap();
    code.codebook_digest = "blake3:tampered".into();
    assert!(matches!(
        quantizer.decode(&code),
        Err(FibQuantError::CodebookDigestMismatch { .. })
    ));
}

#[test]
fn rotation_tamper_rejects_decode() {
    let quantizer = FibQuantizer::new(small_profile(103).unwrap()).unwrap();
    let mut code = quantizer.encode(&[0.25; 8]).unwrap();
    code.rotation_digest = "blake3:tampered".into();
    assert!(matches!(
        quantizer.decode(&code),
        Err(FibQuantError::RotationDigestMismatch { .. })
    ));
}

#[test]
fn payload_corruption_rejects_decode() {
    let quantizer = FibQuantizer::new(small_profile(104).unwrap()).unwrap();
    let mut code = quantizer.encode(&[0.25; 8]).unwrap();
    code.indices.push(0);
    assert!(matches!(
        quantizer.decode(&code),
        Err(FibQuantError::CorruptPayload(_))
    ));
}
