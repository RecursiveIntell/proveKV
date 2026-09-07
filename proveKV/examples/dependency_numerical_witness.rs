//! Deterministic numerical witness for the paste-to-pastey dependency repair.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rotations = Vec::new();
    for dim in [4, 8, 32] {
        for seed in [0, 42, 999] {
            rotations.push(serde_json::json!({
                "dim": dim, "seed": seed,
                "fib": fib_quant::rotation::StoredRotation::new(dim, seed)?,
                "turbo": turbo_quant::rotation::StoredRotation::new(dim, seed)?,
            }));
        }
    }
    let mut beta = Vec::new();
    for q in [0.0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0] {
        for (a, b) in [(0.5, 2.0), (2.0, 10.0), (16.0, 48.0)] {
            beta.push(serde_json::json!([
                q,
                a,
                b,
                fib_quant::beta_inv::beta_inv(q, a, b)?
            ]));
        }
    }
    let directions = fib_quant::directions::roberts_kronecker(8, 32)?;
    println!(
        "{}",
        serde_json::json!({"rotations":rotations,"beta":beta,"directions":directions})
    );
    Ok(())
}
