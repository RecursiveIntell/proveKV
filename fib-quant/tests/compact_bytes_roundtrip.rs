//! Roundtrip test: encode a FibCodeV1, serialize to compact bytes,
//! deserialize, decode back to a vector, verify the round-trip is exact
//! (modulo f32 epsilon).
//!
//! This is the contract for the wire format change. The compact format
//! is what makes fib-quant actually compress real LLM K/V caches
//! (the JSON-serialized FibCodeV1 was 474 bytes/block for 12 bytes of
//! actual data — the JSON overhead was the entire "compression" story).

use fib_quant::{FibCodeV1, FibQuantProfileV1, FibQuantizer};

#[test]
fn compact_bytes_roundtrip_is_lossless() {
    let mut profile = FibQuantProfileV1::paper_default(64, 4, 32, 42).unwrap();
    profile.training_samples = 256;
    profile.lloyd_restarts = 2;
    profile.lloyd_iterations = 3;
    let quantizer = FibQuantizer::new(profile.clone()).unwrap();

    let v: Vec<f32> = (0..64)
        .map(|i| ((i as f32) * 0.137).sin() * 1.5)
        .collect();
    let original = quantizer.encode(&v).unwrap();
    let original_decoded = quantizer.decode(&original).unwrap();

    // Round trip through compact bytes
    let compact = original.to_compact_bytes();
    let restored = FibQuantizer::new(profile.clone())
        .unwrap()
        .decode(&FibCodeV1::from_compact_bytes(&compact, &profile).unwrap())
        .unwrap();

    // Allow 1e-4 absolute per coord (well below codebook quantization noise).
    for (i, (a, b)) in original_decoded.iter().zip(restored.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(diff < 1e-4, "coord {i} differs by {diff}");
    }

    // Verify compact size is dramatically smaller than JSON.
    let json_size = serde_json::to_string(&original).unwrap().len();
    assert!(
        compact.len() < json_size / 10,
        "compact should be at least 10x smaller than JSON (got {} vs {} = {:.1}x)",
        compact.len(),
        json_size,
        json_size as f64 / compact.len() as f64,
    );
    println!(
        "fib_k4_n32: JSON={} bytes, compact={} bytes, ratio={:.1}x",
        json_size,
        compact.len(),
        json_size as f64 / compact.len() as f64,
    );
}

#[test]
fn compact_bytes_rejects_bad_magic() {
    let bytes = vec![b'X', b'B', b'1', 1, 5, 0, 0, 0, 16, 0, 0];
    let r = FibCodeV1::from_compact_bytes(&bytes, &FibQuantProfileV1::paper_default(64, 4, 32, 42).unwrap());
    assert!(r.is_err(), "should reject bad magic");
}

#[test]
fn compact_bytes_rejects_truncated() {
    let bytes = vec![b'F', b'B', b'1', 1]; // only 4 bytes
    let r = FibCodeV1::from_compact_bytes(&bytes, &FibQuantProfileV1::paper_default(64, 4, 32, 42).unwrap());
    assert!(r.is_err(), "should reject truncated header");
}

#[test]
fn compact_bytes_rejects_wrong_indices_len() {
    // valid header, norm_len=2, then 3 bytes (but block_count=16, wire=5
    // means we need 10 bytes of indices — give only 1 to trigger the
    // length mismatch).
    let mut bytes = vec![b'F', b'B', b'1', 1, 5];
    bytes.extend_from_slice(&16u32.to_le_bytes()); // block_count=16
    bytes.extend_from_slice(&2u16.to_le_bytes()); // norm_len=2
    bytes.extend_from_slice(&[0u8; 2]); // norm payload
    bytes.push(0); // only 1 index byte, need 10
    let r = FibCodeV1::from_compact_bytes(&bytes, &FibQuantProfileV1::paper_default(64, 4, 32, 42).unwrap());
    assert!(r.is_err(), "should reject mismatched indices length");
}
