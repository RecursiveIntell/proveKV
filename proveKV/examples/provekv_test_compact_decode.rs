use fib_quant::{FibCodeV1, FibQuantProfileV1, FibQuantizer};

fn main() {
    let profile = FibQuantProfileV1::paper_default(64, 4, 32, 42).unwrap();
    let q = FibQuantizer::new(profile.clone()).unwrap();
    let n = 10000usize;
    let codes: Vec<FibCodeV1> = (0..n)
        .map(|i| {
            let v: Vec<f32> = (0..64)
                .map(|j| ((i * 64 + j) as f32 * 0.137).sin() * 1.5)
                .collect();
            q.encode(&v).unwrap()
        })
        .collect();
    let compact: Vec<u8> = codes.iter().flat_map(|c| c.to_compact_bytes()).collect();
    let compact_per = compact.len() / n;
    println!(
        "compact bytes: {} total, {} per block, codes JSON would be ~472 per block",
        compact.len(),
        compact_per
    );

    // Time from_compact_bytes in a loop
    let refs: Vec<Vec<u8>> = codes.iter().map(|c| c.to_compact_bytes()).collect();
    let byte_slices: Vec<&[u8]> = refs.iter().map(|v| v.as_slice()).collect();
    let t = std::time::Instant::now();
    let mut decoded_codes = Vec::with_capacity(n);
    for s in &byte_slices {
        decoded_codes.push(FibCodeV1::from_compact_bytes(s, &profile).unwrap());
    }
    let dt = t.elapsed();
    println!("from_compact_bytes x{} in {:?}", n, dt);

    // Time decode_batch_fast
    let t = std::time::Instant::now();
    let _decoded = q.decode_batch_fast(&decoded_codes).unwrap();
    let dt = t.elapsed();
    println!("decode_batch_fast x{} in {:?}", n, dt);
}
