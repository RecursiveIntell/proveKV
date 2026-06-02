use fib_quant::{fibonacci_sphere_3d, fibonacci_spiral_2d, roberts_kronecker};

#[test]
fn direction_generators_are_unit_and_deterministic() {
    let a = fibonacci_spiral_2d(16).unwrap();
    let b = fibonacci_spiral_2d(16).unwrap();
    assert_eq!(a, b);
    for value in a {
        let norm = (value[0] * value[0] + value[1] * value[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    for value in fibonacci_sphere_3d(16).unwrap() {
        let norm = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
    }

    let rk = roberts_kronecker(5, 16).unwrap();
    assert_eq!(rk, roberts_kronecker(5, 16).unwrap());
    for value in rk {
        let norm = value.iter().map(|entry| entry * entry).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
        assert!(value.iter().all(|entry| entry.is_finite()));
    }
}
