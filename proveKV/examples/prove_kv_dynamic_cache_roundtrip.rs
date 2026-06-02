//! `prove_kv_dynamic_cache_roundtrip` — the HuggingFace integration
//! validation tool.
//!
//! Reads a JSON file with a flat corpus of K/V tensors, builds a
//! `SharedKVPool`, dumps:
//! 1. The compressed pool bytes (as a JSON manifest)
//! 2. The decompressed K/V tensors (one JSON per layer, in the
//!    HuggingFace-friendly layout: `keys[head][token][head_dim]` flattened
//!    to `keys[head][token * head_dim + j]`).
//!
//! This is the round-trip path the Python validation script uses to
//! measure PPL delta between baseline and proveKV-compressed KV caches.
//!
//! Usage:
//!   prove_kv_dynamic_cache_roundtrip <input.json> <output_dir>
//!
//! Input JSON shape (see `scripts/build_prove_kv_corpus.py` for the
//! Python builder):
//!   {
//!     "shape": {
//!       "attention_type": "MHA" | "GQA" | "MQA",
//!       "num_layers": 24,
//!       "num_heads": 32,
//!       "num_kv_heads": 32,
//!       "head_dim": 64,
//!       "hidden_size": 2048
//!     },
//!     "tokens": [
//!       {"id": "tok_0", "vector": [f32; num_layers * num_kv_heads * head_dim * 2]},
//!       ...
//!     ]
//!   }
//!
//! Output:
//!   <output_dir>/manifest.json     - PoolManifest + PoolBuildReceipt
//!   <output_dir>/layer_<i>.json    - DecompressedLayer for each layer

use std::env;
use std::fs;
use std::path::PathBuf;

use prove_kv::policy::CompressionPolicy;
use prove_kv::shape::{AttentionType, KvTensorShape};
use prove_kv::SharedKVPool;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct InputJson {
    shape: ShapeJson,
    tokens: Vec<TokenJson>,
    seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ShapeJson {
    attention_type: String,
    num_layers: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: usize,
    hidden_size: usize,
}

#[derive(Debug, Deserialize)]
struct TokenJson {
    id: String,
    vector: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct OutputManifest {
    pool_id: String,
    num_shared_tokens: u32,
    num_layers: u32,
    num_kv_heads: u32,
    head_dim: usize,
    shared_codec: String,
    compression_ratio: f64,
    pool_size_bytes: u64,
    total_compressed_bytes: u64,
    backend: String,
    fib_build_ms: u64,
    build_seed: u64,
    built_at_unix: i64,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <input.json> <output_dir>", args[0]);
        std::process::exit(1);
    }
    let input_path = PathBuf::from(&args[1]);
    let output_dir = PathBuf::from(&args[2]);
    fs::create_dir_all(&output_dir).expect("create output dir");

    let input_bytes = fs::read(&input_path).expect("read input");
    let input: InputJson =
        serde_json::from_slice(&input_bytes).expect("parse input json");

    let attn_type = match input.shape.attention_type.as_str() {
        "MHA" => AttentionType::MHA,
        "GQA" => AttentionType::GQA,
        "MQA" => AttentionType::MQA,
        other => panic!("unknown attention_type: {other}"),
    };
    let shape = KvTensorShape {
        attention_type: attn_type,
        num_layers: input.shape.num_layers,
        num_heads: input.shape.num_heads,
        num_kv_heads: input.shape.num_kv_heads,
        head_dim: input.shape.head_dim,
        hidden_size: input.shape.hidden_size,
    };
    let corpus: Vec<(String, Vec<f32>)> = input
        .tokens
        .into_iter()
        .map(|t| (t.id, t.vector))
        .collect();
    let seed = input.seed.unwrap_or(42);

    let policy = CompressionPolicy::default_two_tier();
    eprintln!(
        "[proveKV] building pool: shape={:?} num_tokens={} num_layers={} num_kv_heads={} head_dim={}",
        attn_type,
        corpus.len(),
        shape.num_layers,
        shape.num_kv_heads,
        shape.head_dim
    );
    let (pool, receipt) = SharedKVPool::build(&corpus, &shape, seed).expect("build pool");
    eprintln!(
        "[proveKV] build ok: pool_id={} backend={} codec={:?} compression_ratio={:.2}x size={} bytes",
        &pool.manifest.pool_id[..12],
        receipt.backend,
        pool.manifest.shared_codec,
        pool.manifest.compression_ratio,
        pool.manifest.pool_size_bytes
    );

    // Write the manifest
    let output_manifest = OutputManifest {
        pool_id: pool.manifest.pool_id.clone(),
        num_shared_tokens: pool.manifest.num_shared_tokens,
        num_layers: pool.manifest.num_layers,
        num_kv_heads: pool.manifest.shape.num_kv_heads,
        head_dim: pool.manifest.shape.head_dim,
        shared_codec: format!("{:?}", pool.manifest.shared_codec),
        compression_ratio: pool.manifest.compression_ratio,
        pool_size_bytes: pool.manifest.pool_size_bytes,
        total_compressed_bytes: receipt.pool_size_bytes,
        backend: receipt.backend.clone(),
        fib_build_ms: receipt.fib_build_ms,
        build_seed: pool.manifest.build_seed,
        built_at_unix: pool.manifest.built_at_unix,
    };
    fs::write(
        output_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&output_manifest).expect("serialize manifest"),
    )
    .expect("write manifest");

    // Write the PoolBuildReceipt as a separate JSON for receipts verification
    let receipt_json = serde_json::to_string_pretty(&receipt).expect("serialize receipt");
    fs::write(output_dir.join("receipt.json"), receipt_json).expect("write receipt");

    // Decompress and write each layer
    for layer_idx in 0..pool.manifest.num_layers as usize {
        let decompressed = pool
            .decompress_layer(layer_idx)
            .expect("decompress layer");
        let json = serde_json::to_vec(&decompressed).expect("serialize layer");
        fs::write(
            output_dir.join(format!("layer_{layer_idx:03}.json")),
            json,
        )
        .expect("write layer");
    }
    eprintln!(
        "[proveKV] wrote {} layer files to {}",
        pool.manifest.num_layers,
        output_dir.display()
    );
}
