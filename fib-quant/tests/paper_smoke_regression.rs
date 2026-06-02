use fib_quant::{FibQuantProfileV1, FibQuantizer};

#[test]
fn paper_matrix_small_cases_build_and_roundtrip() {
    let cases = [
        (8, 2, 8, 11),
        (12, 3, 12, 12),
        (16, 4, 16, 13),
        (32, 8, 32, 14),
    ];
    for (d, k, n, seed) in cases {
        let mut profile = FibQuantProfileV1::paper_default(d, k, n, seed).unwrap();
        profile.training_samples = if d <= 16 { 160 } else { 224 };
        profile.lloyd_restarts = 2;
        profile.lloyd_iterations = 3;
        let quantizer = FibQuantizer::new(profile).unwrap();
        let input: Vec<f32> = (0..d)
            .map(|idx| ((idx as f32 + 1.0) * 0.137).sin())
            .collect();
        let code = quantizer.encode(&input).unwrap();
        let decoded = quantizer.decode(&code).unwrap();
        assert_eq!(decoded.len(), d);
        assert!(decoded.iter().all(|value| value.is_finite()));
        assert!(quantizer.codebook().training_mse <= quantizer.codebook().init_mse + 1e-12);
    }
}
