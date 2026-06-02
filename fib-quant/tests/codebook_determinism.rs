use fib_quant::{build_initial_codebook, FibCodebookV1, FibQuantProfileV1};

fn small_profile() -> FibQuantProfileV1 {
    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 31).unwrap();
    profile.training_samples = 128;
    profile.lloyd_restarts = 2;
    profile.lloyd_iterations = 3;
    profile
}

#[test]
fn initial_and_refined_codebooks_are_deterministic() {
    let profile = small_profile();
    let init_a = build_initial_codebook(&profile).unwrap();
    let init_b = build_initial_codebook(&profile).unwrap();
    assert_eq!(init_a, init_b);
    assert_eq!(init_a.len(), 16);
    for codeword in init_a.chunks_exact(2) {
        let r2 = codeword.iter().map(|value| value * value).sum::<f64>();
        assert!(r2 <= 1.0 + 1e-12);
    }

    let codebook_a = FibCodebookV1::build(profile.clone()).unwrap();
    let codebook_b = FibCodebookV1::build(profile).unwrap();
    assert_eq!(codebook_a.codebook_digest, codebook_b.codebook_digest);
    assert_eq!(codebook_a.rotation_digest, codebook_b.rotation_digest);
    assert!(codebook_a.rotation_digest.starts_with("blake3:"));
    assert_eq!(codebook_a.codewords, codebook_b.codewords);
}
