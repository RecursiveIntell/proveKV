//! Parity test: `decode_batch_fast` must produce byte-identical output to
//! per-call `decode` (within f32 epsilon, since both end with `as f32`).
//!
//! The fast path is used by proveKV roundtrip CLI for the PPL validation
//! to make 1.5M-block decompresses tractable. Any silent divergence here
//! would invalidate the PPL delta measurement.

use fib_quant::{FibQuantProfileV1, FibQuantizer};

#[test]
fn decode_batch_fast_matches_per_call_decode() {
    let mut profile = FibQuantProfileV1::paper_default(64, 4, 32, 42).unwrap();
    profile.training_samples = 256;
    profile.lloyd_restarts = 2;
    profile.lloyd_iterations = 3;
    let d = profile.ambient_dim as usize;
    let quantizer = FibQuantizer::new(profile.clone()).unwrap();

    // Build a small batch of vectors, encode each.
    let n = 8;
    let mut codes = Vec::new();
    for i in 0..n {
        // Deterministic but non-trivial inputs.
        let x: Vec<f32> = (0..d)
            .map(|j| ((i * 17 + j * 3 + 1) as f32 * 0.1).sin())
            .collect();
        codes.push(quantizer.encode(&x).unwrap());
    }    // Reference: per-call decode.
    let reference: Vec<Vec<f32>> = codes.iter().map(|c| quantizer.decode(c).unwrap()).collect();

    // Fast path: single call.
    let fast = quantizer.decode_batch_fast(&codes).unwrap();

    assert_eq!(fast.len(), reference.len());
    for (i, (r, f)) in reference.iter().zip(fast.iter()).enumerate() {
        assert_eq!(r.len(), f.len(), "vec {i} length mismatch");
        // The fast path is allowed to differ by a small f32 epsilon because
        // it computes the rotation in f32 directly (vs the reference which
        // uses f64 then casts back to f32). We allow 1e-4 absolute per coord,
        // which is well below the codebook quantization noise.
        for (j, (rv, fv)) in r.iter().zip(f.iter()).enumerate() {
            let diff = (rv - fv).abs();
            assert!(
                diff < 1e-4,
                "vec {i} coord {j} differs by {diff}: ref={rv} fast={fv}"
            );
        }
    }
}

#[test]
fn decode_batch_fast_handles_empty_batch() {
    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 100).unwrap();
    profile.training_samples = 64;
    profile.lloyd_restarts = 1;
    profile.lloyd_iterations = 1;
    let quantizer = FibQuantizer::new(profile).unwrap();
    let out = quantizer.decode_batch_fast(&[]).unwrap();
    assert!(out.is_empty());
}
