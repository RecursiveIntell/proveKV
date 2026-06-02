use fib_quant::{FibQuantError, FibQuantProfileV1, MAX_CODEBOOK_SIZE, MAX_TRAINING_SAMPLES};

#[test]
fn d_equal_k_is_rejected_for_alpha() {
    assert!(matches!(
        FibQuantProfileV1::paper_default(8, 8, 8, 1),
        Err(FibQuantError::InvalidBlockDim {
            ambient_dim: 8,
            block_dim: 8
        })
    ));
}

#[test]
fn oversized_rotation_matrix_rejects_before_allocation() {
    assert!(matches!(
        FibQuantProfileV1::paper_default(8194, 2, 8, 1),
        Err(FibQuantError::ResourceLimitExceeded(message)) if message.contains("rotation matrix")
    ));
}

#[test]
fn oversized_codebook_values_reject() {
    assert!(matches!(
        FibQuantProfileV1::paper_default(256, 128, MAX_CODEBOOK_SIZE, 1),
        Err(FibQuantError::ResourceLimitExceeded(message)) if message.contains("codebook values")
    ));
}

#[test]
fn excessive_training_samples_reject() {
    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 1).unwrap();
    profile.training_samples = MAX_TRAINING_SAMPLES + 1;
    assert!(matches!(
        profile.validate(),
        Err(FibQuantError::CorruptPayload(message)) if message.contains("training_samples")
    ));
}
