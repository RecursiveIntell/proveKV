use fib_quant::{radius_quantile, radius_quantile_k2_closed_form};

#[test]
fn k2_closed_form_matches_general_path_and_is_monotone() {
    let d = 16;
    let n_total = 64;
    let mut previous = 0.0;
    for n in 1..=n_total {
        let q = (n as f64 - 0.5) / n_total as f64;
        let closed = radius_quantile_k2_closed_form(d, q).unwrap();
        let general_entry = radius_quantile(d, 2, n, n_total).unwrap();
        assert!((closed - general_entry).abs() < 1e-14);
        assert!((0.0..=1.0).contains(&closed));
        assert!(closed > previous);
        previous = closed;
    }
}
