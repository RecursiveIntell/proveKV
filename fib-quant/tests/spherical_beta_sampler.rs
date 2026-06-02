use fib_quant::{sample_reference_projection, sample_spherical_beta};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn spherical_beta_moments_match_paper_law() {
    let d = 16;
    let k = 4;
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let mut mean_r2 = 0.0;
    let mut coord_var = 0.0;
    let samples = 8_000;
    for _ in 0..samples {
        let sample = sample_spherical_beta(d, k, &mut rng).unwrap();
        let r2: f64 = sample.iter().map(|value| value * value).sum();
        assert!(r2 <= 1.0 + 1e-12);
        mean_r2 += r2;
        coord_var += sample[0] * sample[0];
    }
    mean_r2 /= samples as f64;
    coord_var /= samples as f64;
    assert!((mean_r2 - k as f64 / d as f64).abs() < 0.02);
    assert!((coord_var - 1.0 / d as f64).abs() < 0.01);
}

#[test]
fn gaussian_projection_agrees_on_coarse_moment() {
    let d = 12;
    let k = 3;
    let mut rng = ChaCha8Rng::seed_from_u64(12);
    let samples = 4_000;
    let mut mean_r2 = 0.0;
    for _ in 0..samples {
        let sample = sample_reference_projection(d, k, &mut rng).unwrap();
        mean_r2 += sample.iter().map(|value| value * value).sum::<f64>();
    }
    mean_r2 /= samples as f64;
    assert!((mean_r2 - k as f64 / d as f64).abs() < 0.025);
}
