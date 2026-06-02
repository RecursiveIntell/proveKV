//! GPU-accelerated benchmark using the gpu-backend crate directly.
//! Compares CPU vs GPU hadamard + lloyd-max throughput for 768-dim and 2560-dim vectors.

use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       GPU Acceleration Benchmark — gpu-backend              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let configs = vec![
        ("nomic-embed-text", 768, 200),
        ("qwen3-embedding", 2560, 50),
    ];

    for (name, dim, num_docs) in &configs {
        println!(
            "{:-^70}",
            format!(" {} — {} dim, {} docs ", name, dim, num_docs)
        );

        // Generate realistic vectors
        let vectors = generate_vectors(*dim, *num_docs);
        let k: usize = 4;
        let n_levels: usize = 32;
        let seed: u64 = 42;

        // ── GPU status ──
        #[cfg(feature = "gpu")]
        {
            let gpu_available = gpu_backend::GpuContext::is_available();
            println!("  GPU available: {}", gpu_available);
            if !gpu_available {
                println!("  ⚠ PTX not loaded — falling back to CPU");
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            println!("  ⚠ GPU feature not enabled — using CPU");
        }

        // ── Hadamard benchmark ──
        println!("\n  ── Hadamard WHT ──");
        let mut hadamard_data = vectors.clone();
        let start = Instant::now();
        gpu_backend::hadamard_batch(&mut hadamard_data, *num_docs, *dim, seed)
            .expect("hadamard failed");
        let elapsed = start.elapsed();
        println!("  {} vectors × {} dim: {:?}", num_docs, dim, elapsed);
        println!(
            "  throughput: {:.0} vectors/sec",
            *num_docs as f64 / elapsed.as_secs_f64()
        );

        // ── Lloyd-Max encode benchmark ──
        println!("\n  ── Lloyd-Max Encode (k={}, N={}) ──", k, n_levels);
        let start = Instant::now();
        let (indices, norms) =
            gpu_backend::lloyd_max_batch(&hadamard_data, *num_docs, *dim, k, n_levels, seed)
                .expect("lloyd_max encode failed");
        let elapsed = start.elapsed();
        println!("  {} vectors encoded: {:?}", num_docs, elapsed);
        println!(
            "  indices: {} bytes, norms: {} f32s",
            indices.len(),
            norms.len()
        );
        println!(
            "  throughput: {:.0} vectors/sec",
            *num_docs as f64 / elapsed.as_secs_f64()
        );
        println!(
            "  per-vector: {:.1}ms",
            elapsed.as_secs_f64() * 1000.0 / *num_docs as f64
        );

        // ── Lloyd-Max decode benchmark ──
        println!("\n  ── Lloyd-Max Decode ──");
        let start = Instant::now();
        let decoded = gpu_backend::lloyd_max_decode_batch(
            &indices, &norms, *num_docs, *dim, k, n_levels, seed,
        )
        .expect("lloyd_max decode failed");
        let elapsed = start.elapsed();
        println!("  {} vectors decoded: {:?}", num_docs, elapsed);

        // ── Fidelity check ──
        let mut total_cos = 0.0f64;
        let mut total_mse = 0.0f64;
        for i in 0..*num_docs {
            let orig = &vectors[i * dim..(i + 1) * dim];
            let dec = &decoded[i * dim..(i + 1) * dim];
            let dot: f64 = orig
                .iter()
                .zip(dec.iter())
                .map(|(a, b)| (*a as f64) * (*b as f64))
                .sum();
            let mag_o: f64 = orig.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
            let mag_d: f64 = dec.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
            total_cos += if mag_o > 0.0 && mag_d > 0.0 {
                dot / (mag_o * mag_d)
            } else {
                0.0
            };
            total_mse += orig
                .iter()
                .zip(dec.iter())
                .map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2))
                .sum::<f64>()
                / *dim as f64;
        }
        println!("  avg cosine fidelity: {:.6}", total_cos / *num_docs as f64);
        println!("  avg MSE: {:.6}", total_mse / *num_docs as f64);

        // ── Compression ratio ──
        let raw_bytes = num_docs * dim * 4;
        let indices_bytes = indices.len();
        let norms_bytes = norms.len() * 4;
        let comp_bytes = indices_bytes + norms_bytes;
        let ratio = raw_bytes as f64 / comp_bytes as f64;
        println!("\n  ── Compression ──");
        println!("  raw: {} KB", raw_bytes / 1024);
        println!("  compressed: {} KB ({:.1}×)", comp_bytes / 1024, ratio);
    }

    println!("\n✅ GPU benchmark complete.");
}

fn generate_vectors(dim: usize, num_docs: usize) -> Vec<f32> {
    use std::num::Wrapping;
    let mut state = Wrapping(42u64);
    let std_dev = 1.0 / (dim as f64).sqrt() as f32;
    let mut data = Vec::with_capacity(num_docs * dim);
    for _ in 0..num_docs {
        let mut vec = Vec::with_capacity(dim);
        let mut norm_sq = 0.0f64;
        for _ in 0..dim {
            state = state * Wrapping(6364136223846793005) + Wrapping(1442695040888963407);
            let u = (state.0 as f64) / (u64::MAX as f64);
            let v = ((-2.0 * (1.0 - u).ln()).sqrt() * std_dev as f64) as f32;
            norm_sq += (v as f64).powi(2);
            vec.push(v);
        }
        // Normalize to unit length (matching real embedding vectors)
        let norm = norm_sq.sqrt() as f32;
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        data.extend(vec);
    }
    data
}
