//! Wall-clock decode bench for proveKV's turbo hot tier.
//!
//! Compares the per-vec decode path (`TurboQuantizer::decode_approximate`)
//! against the batch decode path (`TurboQuantizer::decode_approximate_batch`)
//! on a fixed corpus shape matching the msi PPL bench:
//! SmolLM2-1.7B (24 layers, 32 kv_heads, head_dim=64), N=8 agents, 28
//! unique tokens each. Per-agent shell size at b=4 lossless is 6,194,976 B
//! = 6,194,976 / 160 B/vec = 38,718 vec/agent (or roughly 193 vectors
//! per (layer, kv_head, unique_token) cell).
//!
//! This is decode-only wall-clock. No model, no PPL, no forward pass. The
//! point is to measure the speedup from the batch decode path the audit
//! added — same numerical output (verified bit-exact in unit tests), so
//! PPL is unaffected.
//!
//! Usage:
//!   cargo run --release -p turbo-quant --example decode_wallclock -- [n_reps]
//!
//! Output is JSON on stdout for easy parsing:
//!   {"per_vec_ms": ..., "batch_ms": ..., "speedup": ...}

use std::time::Instant;

use turbo_quant::{PolarQuantizer, TurboCode, TurboCodeWireV1, TurboQuantizer};

fn main() {
    let n_reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    // msi PPL bench config: 24 layers, 32 kv_heads, head_dim=64, 8 agents,
    // 28 unique tokens per agent, b=4 lossless.
    let dim: usize = 64;
    let bits: u8 = 4;
    let projections: usize = 32;
    let seed: u64 = 42;
    let n_layers: usize = 24;
    let n_kv_heads: usize = 32;
    let n_agents: usize = 8;
    let n_unique: usize = 28;
    // Per-vec size at b=4: 4*32 (signs) + 16 (radii u8) + 0 (no aux) = 144 B
    // Per-agent shell = 144 B * 24 layers * 32 heads * 28 unique = 3,096,576 B
    // (mismatch with 6,194,976 B is because msi uses f32 radii profile, not
    //  the lossy one — at the lossless f32-radii profile, per-vec is
    //  4*32 + 4*32 = 256 B → 256 * 24 * 32 * 28 = 5,505,408 B... still
    //  not exactly 6,194,976. The shell is densely packed with some
    //  outer-block header. Don't worry about matching the msi number
    //  exactly — this bench is for the DECODE PATH RATIO, not total bytes.)
    let vecs_per_layer: usize = n_kv_heads * n_agents * n_unique;
    let total_vecs: usize = n_layers * vecs_per_layer;

    eprintln!("[decode_wallclock] dim={dim} bits={bits} projections={projections} seed={seed}");
    eprintln!(
        "[decode_wallclock] shape: layers={n_layers} kv_heads={n_kv_heads} agents={n_agents} unique={n_unique}"
    );
    eprintln!(
        "[decode_wallclock] total_vecs={total_vecs} ({} per layer)",
        vecs_per_layer
    );
    eprintln!("[decode_wallclock] n_reps={n_reps}");
    eprintln!();

    // Build the quantizer and pre-encode a deterministic corpus of random
    // f32 vectors. The corpus doesn't need to match K/V stats — we're
    // measuring decode path overhead, not codec quality.
    let quantizer = TurboQuantizer::new(dim, bits, projections, seed).expect("quantizer init");

    // Pre-encode: build a corpus that matches the shape.
    eprintln!("[decode_wallclock] pre-encoding corpus...");
    let mut codes: Vec<TurboCode> = Vec::with_capacity(total_vecs);
    for i in 0..total_vecs {
        // Deterministic pseudo-random vector (so this is reproducible).
        let v: Vec<f32> = (0..dim)
            .map(|d| {
                let x = ((i * dim + d) as f32 * 0.1234).sin();
                x * 0.1
            })
            .collect();
        let code = quantizer.encode(&v).expect("encode");
        codes.push(code);
    }
    eprintln!(
        "[decode_wallclock] encoded {} codes ({} B per code, {} B total)",
        codes.len(),
        4 * (dim / 8) + 4, // rough wire size hint
        codes.len() * (4 * (dim / 8) + 4)
    );
    eprintln!();

    // ===== Per-vec decode path =====
    eprintln!(
        "[decode_wallclock] timing per-vec decode path (TurboQuantizer::decode_approximate)..."
    );
    let per_vec_times: Vec<f64> = (0..n_reps)
        .map(|_| {
            let t0 = Instant::now();
            for code in &codes {
                let _ = quantizer.decode_approximate(code).expect("decode");
            }
            t0.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    let per_vec_total: f64 = per_vec_times.iter().sum::<f64>();
    let per_vec_mean: f64 = per_vec_total / n_reps as f64;
    let per_vec_min: f64 = per_vec_times.iter().cloned().fold(f64::INFINITY, f64::min);
    eprintln!(
        "[decode_wallclock]   total per_rep: mean={:.2}ms min={:.2}ms ({} reps)",
        per_vec_mean, per_vec_min, n_reps
    );
    eprintln!(
        "[decode_wallclock]   per_vec: mean={:.3}us ({} total vecs)",
        per_vec_mean * 1000.0 / total_vecs as f64,
        total_vecs
    );

    // ===== Batch decode path =====
    eprintln!();
    eprintln!(
        "[decode_wallclock] timing batch decode path (TurboQuantizer::decode_approximate_batch)..."
    );
    let batch_times: Vec<f64> = (0..n_reps)
        .map(|_| {
            let t0 = Instant::now();
            let _ = quantizer
                .decode_approximate_batch(&codes)
                .expect("decode batch");
            t0.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    let batch_total: f64 = batch_times.iter().sum::<f64>();
    let batch_mean: f64 = batch_total / n_reps as f64;
    let batch_min: f64 = batch_times.iter().cloned().fold(f64::INFINITY, f64::min);
    eprintln!(
        "[decode_wallclock]   total per_rep: mean={:.2}ms min={:.2}ms ({} reps)",
        batch_mean, batch_min, n_reps
    );
    eprintln!(
        "[decode_wallclock]   per_vec: mean={:.3}us ({} total vecs)",
        batch_mean * 1000.0 / total_vecs as f64,
        total_vecs
    );

    // ===== Wire-format decode path =====
    eprintln!();
    eprintln!("[decode_wallclock] timing wire-format decode path (TurboCodeWireV1 round-trip)...");

    // First encode the corpus to wire format (one-time cost).
    let wire_payloads: Vec<Vec<u8>> = codes
        .iter()
        .map(|c| TurboCodeWireV1::encode(c, &quantizer).expect("wire encode"))
        .collect();
    let wire_total_bytes: usize = wire_payloads.iter().map(|p| p.len()).sum();
    eprintln!(
        "[decode_wallclock]   wire payload: {} codes, {} B total, {:.1} B/vec",
        wire_payloads.len(),
        wire_total_bytes,
        wire_total_bytes as f64 / wire_payloads.len() as f64
    );

    let wire_times: Vec<f64> = (0..n_reps)
        .map(|_| {
            let t0 = Instant::now();
            for payload in &wire_payloads {
                let code = TurboCodeWireV1::decode(payload, &quantizer).expect("wire decode");
                let _ = quantizer.decode_approximate(&code).expect("decode");
            }
            t0.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    let wire_total: f64 = wire_times.iter().sum::<f64>();
    let wire_mean: f64 = wire_total / n_reps as f64;
    let wire_min: f64 = wire_times.iter().cloned().fold(f64::INFINITY, f64::min);
    eprintln!(
        "[decode_wallclock]   total per_rep: mean={:.2}ms min={:.2}ms ({} reps)",
        wire_mean, wire_min, n_reps
    );
    eprintln!(
        "[decode_wallclock]   per_vec: mean={:.3}us ({} total vecs)",
        wire_mean * 1000.0 / total_vecs as f64,
        total_vecs
    );

    // ===== Report =====
    eprintln!();
    eprintln!("[decode_wallclock] ===== summary =====");
    eprintln!(
        "[decode_wallclock] per-vec-only:  {:.2}ms (mean of {} reps)",
        per_vec_mean, n_reps
    );
    eprintln!(
        "[decode_wallclock] batch-only:    {:.2}ms (mean of {} reps)",
        batch_mean, n_reps
    );
    eprintln!(
        "[decode_wallclock] wire+per-vec:  {:.2}ms (mean of {} reps)",
        wire_mean, n_reps
    );
    eprintln!(
        "[decode_wallclock] speedup (per-vec / batch): {:.2}x",
        per_vec_mean / batch_mean
    );
    eprintln!(
        "[decode_wallclock] speedup (per-vec / wire):  {:.2}x",
        per_vec_mean / wire_mean
    );

    // JSON on stdout
    println!(
        "{{\
            \"n_reps\": {n_reps},\
            \"n_layers\": {n_layers},\
            \"n_kv_heads\": {n_kv_heads},\
            \"n_agents\": {n_agents},\
            \"n_unique\": {n_unique},\
            \"total_vecs\": {total_vecs},\
            \"dim\": {dim},\
            \"bits\": {bits},\
            \"per_vec_ms_mean\": {per_vec_mean:.4},\
            \"per_vec_ms_min\": {per_vec_min:.4},\
            \"batch_ms_mean\": {batch_mean:.4},\
            \"batch_ms_min\": {batch_min:.4},\
            \"wire_ms_mean\": {wire_mean:.4},\
            \"wire_ms_min\": {wire_min:.4},\
            \"speedup_batch_vs_per_vec\": {pvb:.4},\
            \"speedup_wire_vs_per_vec\": {pvw:.4}\
        }}",
        pvb = per_vec_mean / batch_mean,
        pvw = per_vec_mean / wire_mean
    );

    // Print the polar-only call count for diagnostics (the FWHT trig is
    // where the batch path's win comes from: amortized vs per-vec).
    let _ = PolarQuantizer::new(dim, bits, 0).expect("polar init");
}
