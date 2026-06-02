//! `encode_batch_microbench` — isolate fib-quant encode_batch from pool build.
//!
//! Times just the fib-quant `encode_batch` call (no codebook construction,
//! no digest math, no pool manifest). Compares:
//!   - CPU-only: encode_batch runs the per-vector StoredRotation + nearest_index
//!   - GPU Hadamard: the Hadamard step dispatches to gpu-backend; codebook stays CPU
//!   - GPU full: both Hadamard and codebook_lookup dispatch to GPU
//!
//! Each call takes a fresh batch, so the only thing being measured is the
//! encode pipeline itself.
//!
//! Usage:
//!   cargo run --release --example encode_batch_microbench
//!   cargo run --release --example encode_batch_microbench --features gpu,gpu-backend/precompiled-ptx
//!   cargo run --release --example encode_batch_microbench --features gpu_codebook_lookup,gpu-backend/precompiled-ptx

use std::time::Instant;

use fib_quant::{FibQuantProfileV1, FibQuantizer};
use rand::Rng;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};

fn make_quantizer(d: usize, k: usize, n_codewords: usize, seed: u64) -> FibQuantizer {
    let profile = FibQuantProfileV1::paper_default(d, k, n_codewords, seed).unwrap();
    // paper_default is fine; the same codebook will be built on every run
    FibQuantizer::new(profile).unwrap()
}

fn make_inputs(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

fn run_one(d: usize, k: usize, n_codewords: usize, n: usize, label: &str) {
    // Build quantizer once outside the timed region.
    let q = make_quantizer(d, k, n_codewords, 42);
    let inputs = make_inputs(n, d, 0xDEAD);
    let refs: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();

    // Warm up.
    let _ = q.encode_batch(&refs).unwrap();

    let start = Instant::now();
    let _codes = q.encode_batch(&refs).unwrap();
    let wall = start.elapsed();

    let per_vec_us = wall.as_micros() as f64 / n as f64;
    let report = q.gpu_steps_for(n, d);
    let hadamard_str = if report.hadamard { "gpu" } else { "cpu" };
    let codebook_str = if report.codebook_lookup { "gpu" } else { "cpu" };

    println!(
        "  {label:24} n={n:>4} d={d:>4} k={k} N={n_codewords:>2}  wall={w:>5} ms  \
         per_vec={pv:>5.1} us  vec/s={vs:>9.0}  hadamard={h:>3}  codebook={cb:>3}",
        label = label,
        n = n,
        d = d,
        k = k,
        n_codewords = n_codewords,
        w = wall.as_millis(),
        pv = per_vec_us,
        vs = n as f64 / wall.as_secs_f64(),
        h = hadamard_str,
        cb = codebook_str,
    );
}

fn main() {
    println!("fib-quant encode_batch microbench");
    println!("compile-time: gpu feature = {}", cfg!(feature = "gpu"));
    println!();

    // Paper-default for proveKV pool: k=4, N=32, dim 64/128/2560
    // Note: d=2560 is included but each run takes ~30s on CPU; the bench
    // is meant for A/B comparison, so all three configs pay the same cost.
    println!("=== paper_default(k=4, N=32) ===");
    for (d, label) in &[(64usize, "tiny"), (128, "small"), (768, "nomic")] {
        for n in &[4usize, 20, 80] {
            run_one(*d, 4, 32, *n, label);
        }
        println!();
    }
    // qwen3-dim 2560: just n=4 to have a data point without 30s per run
    #[allow(clippy::single_element_loop)]
    for n in &[4usize] {
        run_one(2560, 4, 32, *n, "qwen3");
    }
    println!();

    println!("Notes:");
    println!("  - gpu_probe=full means both Hadamard and codebook_lookup go to GPU");
    println!("  - gpu_probe=device-only means device exists but batch too small for codebook path");
    println!("  - gpu_probe=cpu means no GPU feature compiled in");
    println!("  - per_vec is microseconds per encoded vector");
    println!("  - vec/s is throughput (vectors per second)");
}
