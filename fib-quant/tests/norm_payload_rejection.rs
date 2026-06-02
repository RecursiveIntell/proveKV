use fib_quant::{FibQuantError, FibQuantProfileV1, FibQuantizer, NormFormat};

fn tiny_profile(norm_format: NormFormat) -> FibQuantProfileV1 {
    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 71).unwrap();
    profile.training_samples = 64;
    profile.lloyd_restarts = 1;
    profile.lloyd_iterations = 1;
    profile.norm_format = norm_format;
    profile
}

#[test]
fn fp16_norm_underflow_rejects_before_artifact_emit() {
    let quantizer = FibQuantizer::new(tiny_profile(NormFormat::Fp16Paper)).unwrap();
    let input = vec![f32::from_bits(1); 8];
    let err = quantizer.encode(&input).unwrap_err();
    assert!(matches!(err, FibQuantError::CorruptPayload(message) if message.contains("fp16")));
}

#[test]
fn fp16_norm_overflow_rejects_before_artifact_emit() {
    let quantizer = FibQuantizer::new(tiny_profile(NormFormat::Fp16Paper)).unwrap();
    let input = vec![f32::MAX; 8];
    let err = quantizer.encode(&input).unwrap_err();
    assert!(matches!(err, FibQuantError::CorruptPayload(message) if message.contains("fp16")));
}
