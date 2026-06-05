//! Measure turbo-only on the shared pool corpus (no fib at all).
//! Quick test: build the shared pool with TQB1 instead of FB2, see the ratio.

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use provekv::codec::TurboQuantAdapter;
use provekv::policy::TurboConfig;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct InputJson {
    shape: ShapeJson,
    shared_tokens: Vec<TokenJson>,
}

#[derive(Debug, Deserialize)]
struct ShapeJson {
    num_layers: u32,
    num_kv_heads: u32,
    head_dim: usize,
}

#[derive(Debug, Deserialize, Clone)]
struct TokenJson {
    #[serde(rename = "id")]
    _id: String,
    vector: Vec<f32>,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: {} <input.json>", args[0]);
        std::process::exit(1);
    }
    let input: InputJson = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let head_dim = input.shape.head_dim;
    let num_layers = input.shape.num_layers as usize;
    let num_kv_heads = input.shape.num_kv_heads as usize;

    // Each token's vector is [layer0_head0_k..., layer0_head0_v..., layer0_head1_k..., ...].
    // The per-vector size for one (layer, head) is head_dim * 2.
    let per_layer_per_head_len = head_dim * 2;
    let adapter = TurboQuantAdapter::new(head_dim, 8, 32).unwrap();
    let seed = 42;

    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    for (layer_idx, _) in (0..num_layers).enumerate() {
        for head_idx in 0..num_kv_heads {
            for token in &input.shared_tokens {
                let offset = (layer_idx * num_kv_heads + head_idx) * per_layer_per_head_len;
                all_vectors.push(token.vector[offset..offset + head_dim].to_vec());
            }
        }
    }

    let n_vectors = all_vectors.len();
    let t = Instant::now();
    let refs: Vec<&[f32]> = all_vectors.iter().map(|v| v.as_slice()).collect();
    let payload = adapter.encode_batch_compact(&refs, seed).expect("encode");
    let elapsed = t.elapsed().as_millis();

    let raw_bytes = n_vectors * head_dim * 4;
    let ratio = raw_bytes as f64 / payload.len() as f64;
    println!(
        "TURBO-ONLY shared pool: n_vectors={} payload={}B raw={}B ratio={:.2}x ({}ms)",
        n_vectors,
        payload.len(),
        raw_bytes,
        ratio,
        elapsed
    );
    let cfg = TurboConfig::default_8bit();
    let _ = cfg;
    // Force inclusion for cfg.
    let _ = PathBuf::from("/tmp");
}
