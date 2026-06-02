//! Microbenchmark for the codebook_lookup kernel.
//!
//! Runs the GPU codebook lookup on a representative fib-quant workload and
//! compares wall time to the CPU fallback. The point is to isolate the
//! kernel from the rest of the pool build so we can see if it actually
//! wins.
//!
//! Run with:
//!   cargo run --release --example codebook_lookup_microbench --features gpu,gpu-backend/precompiled-ptx

use std::time::Instant;

use gpu_backend::codebook_lookup_batch;
use rand::Rng;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};

fn make_inputs(n: usize, d: usize, n_codewords: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let input: Vec<f32> = (0..n * d).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let codebook: Vec<f32> = (0..n_codewords * k)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    (input, codebook)
}

fn run_one(n: usize, d: usize, k: usize, label: &str) {
    let n_codewords = 32; // paper_default for fib_k4_n32
    let (input, codebook) = make_inputs(n, d, n_codewords, k);

    // Warm up: build quantizer and run once outside the timed region.
    let _ = codebook_lookup_batch(&input, &codebook, n, d, k).unwrap();

    // Timed GPU run (which may fall back to CPU).
    let start = Instant::now();
    let indices = codebook_lookup_batch(&input, &codebook, n, d, k).unwrap();
    let wall = start.elapsed();

    let total_blocks = n * (d / k);
    let ops = total_blocks * n_codewords * k;
    let blocks_per_sec = total_blocks as f64 / wall.as_secs_f64();
    let ops_per_sec = ops as f64 / wall.as_secs_f64();

    println!(
        "  {label:32} n={n:>3} d={d:>4}  blocks={tb:>6}  wall={w:>5} ms  \
         blocks/s={bps:>10.0}  ops/s={ops:>11.0}  out_len={ol}",
        label = label,
        n = n,
        d = d,
        tb = total_blocks,
        w = wall.as_millis(),
        bps = blocks_per_sec,
        ops = ops_per_sec,
        ol = indices.len(),
    );
}

fn main() {
    println!("codebook_lookup_batch microbenchmark");
    println!("compile-time: gpu feature = {}", cfg!(feature = "gpu"));
    println!(
        "device available: {}",
        gpu_backend::GpuContext::is_available()
    );
    println!();

    println!("=== qwen3 2560-dim, k=4, N=32 ===");
    for n in &[4usize, 20, 80, 200, 800] {
        run_one(*n, 2560, 4, "qwen3-2560");
    }
    println!();

    println!("=== nomic 768-dim, k=4, N=32 ===");
    for n in &[4usize, 20, 80, 200, 800] {
        run_one(*n, 768, 4, "nomic-768");
    }
    println!();

    println!("Notes:");
    println!("  - 'blocks' = n * (d / k) = total codebook lookups");
    println!("  - 'ops' = blocks * N * k = total squared-distance ops");
    println!("  - Falls back to CPU when n < 16 or d < 64");
}
