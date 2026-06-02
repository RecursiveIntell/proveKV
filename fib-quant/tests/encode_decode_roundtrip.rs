use fib_quant::{FibQuantProfileV1, FibQuantizer};

#[test]
fn encode_decode_roundtrip_is_finite_and_fixed_rate() {
    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 51).unwrap();
    profile.training_samples = 128;
    profile.lloyd_restarts = 2;
    profile.lloyd_iterations = 3;
    let quantizer = FibQuantizer::new(profile).unwrap();
    let input = vec![0.25, -0.5, 0.75, 1.0, -1.25, 0.5, 0.125, -0.875];
    let code = quantizer.encode(&input).unwrap();
    assert_eq!(code.block_count, 4);
    assert_eq!(code.indices.len(), 2);
    let decoded = quantizer.decode(&code).unwrap();
    assert_eq!(decoded.len(), input.len());
    assert!(decoded.iter().all(|value| value.is_finite()));
    assert!(quantizer.reconstruction_mse(&input).unwrap().is_finite());
    assert!(quantizer.cosine_similarity(&input).unwrap().is_finite());
}

#[test]
fn encode_with_receipt_records_digest_chain() {
    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 52).unwrap();
    profile.training_samples = 128;
    profile.lloyd_restarts = 1;
    profile.lloyd_iterations = 2;
    let quantizer = FibQuantizer::new(profile).unwrap();
    let input = vec![1.0, 0.5, -0.25, 0.125, -1.5, 0.75, 0.25, -0.5];
    let (_code, receipt) = quantizer.encode_with_receipt(&input).unwrap();
    assert!(receipt.profile_digest.starts_with("blake3:"));
    assert!(receipt.codebook_digest.starts_with("blake3:"));
    assert!(receipt.source_vector_digest.starts_with("blake3:"));
    assert!(receipt.encoded_digest.starts_with("blake3:"));
    assert_eq!(receipt.code_schema_version, "fib_code_v1");
    assert_eq!(receipt.profile_schema_version, "fib_quant_profile_v1");
    assert_eq!(receipt.codebook_schema_version, "fib_codebook_v1");
    assert!(receipt.mse.unwrap().is_finite());
    assert!(receipt.cosine_similarity.unwrap().is_finite());
}

#[test]
fn receipt_source_digest_changes_with_input() {
    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 53).unwrap();
    profile.training_samples = 128;
    profile.lloyd_restarts = 1;
    profile.lloyd_iterations = 2;
    let quantizer = FibQuantizer::new(profile).unwrap();

    let input_a = vec![1.0, 0.5, -0.25, 0.125, -1.5, 0.75, 0.25, -0.5];
    let input_b = vec![1.0, 0.5, -0.25, 0.125, -1.5, 0.75, 0.25, -0.25];
    let (_, receipt_a) = quantizer.encode_with_receipt(&input_a).unwrap();
    let (_, receipt_b) = quantizer.encode_with_receipt(&input_b).unwrap();

    assert_ne!(
        receipt_a.source_vector_digest,
        receipt_b.source_vector_digest
    );
}
