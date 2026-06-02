use fib_quant::{FibQuantError, FibQuantProfileV1, FibQuantizer};

fn quantizer() -> FibQuantizer {
    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 61).unwrap();
    profile.training_samples = 128;
    profile.lloyd_restarts = 1;
    profile.lloyd_iterations = 2;
    FibQuantizer::new(profile).unwrap()
}

#[test]
fn profile_and_codebook_digest_corruption_rejects() {
    let quantizer = quantizer();
    let input = vec![0.25; 8];
    let mut code = quantizer.encode(&input).unwrap();
    code.profile_digest = "blake3:bad".into();
    assert!(matches!(
        quantizer.decode(&code),
        Err(FibQuantError::ProfileDigestMismatch { .. })
    ));

    let mut code = quantizer.encode(&input).unwrap();
    code.codebook_digest = "blake3:bad".into();
    assert!(matches!(
        quantizer.decode(&code),
        Err(FibQuantError::CodebookDigestMismatch { .. })
    ));

    let mut code = quantizer.encode(&input).unwrap();
    code.rotation_digest = "blake3:bad".into();
    assert!(matches!(
        quantizer.decode(&code),
        Err(FibQuantError::RotationDigestMismatch { .. })
    ));
}

#[test]
fn corrupt_payload_and_non_finite_input_reject() {
    let quantizer = quantizer();
    assert!(matches!(
        quantizer.encode(&[f32::NAN; 8]),
        Err(FibQuantError::NonFiniteInput(0))
    ));
    assert!(matches!(
        quantizer.encode(&[0.0; 8]),
        Err(FibQuantError::ZeroNorm)
    ));
    let input = vec![0.25; 8];
    let mut code = quantizer.encode(&input).unwrap();
    code.indices.push(0);
    assert!(matches!(
        quantizer.decode(&code),
        Err(FibQuantError::CorruptPayload(_))
    ));
}

#[test]
fn invalid_code_schema_rejects() {
    let quantizer = quantizer();
    let input = vec![0.25; 8];
    let mut code = quantizer.encode(&input).unwrap();
    code.schema_version = "fib_code_v0".into();

    assert!(matches!(
        quantizer.decode(&code),
        Err(FibQuantError::CorruptPayload(_))
    ));
}
