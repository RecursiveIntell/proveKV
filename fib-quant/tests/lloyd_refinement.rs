use fib_quant::{FibCodebookV1, FibQuantProfileV1};

#[test]
fn lloyd_refinement_does_not_worsen_initial_mse() {
    let mut profile = FibQuantProfileV1::paper_default(12, 3, 12, 41).unwrap();
    profile.training_samples = 180;
    profile.lloyd_restarts = 2;
    profile.lloyd_iterations = 4;
    let codebook = FibCodebookV1::build(profile).unwrap();
    assert!(codebook.training_mse <= codebook.init_mse + 1e-12);
    assert!(codebook.refinement_report.best_mse <= codebook.refinement_report.init_mse + 1e-12);
}
