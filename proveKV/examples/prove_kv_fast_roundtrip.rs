//! `prove_kv_fast_roundtrip` — fast compress+decompress for the PPL validation.
//!
//! Same input format as `prove_kv_dynamic_cache_roundtrip`. Output is a single
//! binary file containing the manifest (as JSON, length-prefixed) followed by
//! raw f32 LE data for each layer's K and V tensors.
//!
//! Uses `FibQuantizer::decode_batch_fast` which:
//! - Allocates a single output Vec<f32> per code (no per-index alloc)
//! - Skips f32→f64 roundtrip on the codeword gather
//! - Operates the inverse rotation in f32 (no per-call matrix conversion)
//!
//! Usage: prove_kv_fast_roundtrip <input.json> <output.bin>

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use prove_kv::codec::FibQuantAdapter;
use prove_kv::policy::{CompressionPolicy, FibConfig};
use prove_kv::shape::{AttentionType, KvTensorShape};
use prove_kv::SharedKVPool;

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
        eprintln!("usage: {} <input.json> <output.bin>", args[0]);
        std::process::exit(1);
    }
    let input_path = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);

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
    let corpus: Vec<(String, Vec<f32>)> =
        input.tokens.into_iter().map(|t| (t.id, t.vector)).collect();
    let seed = input.seed.unwrap_or(42);

    let policy = CompressionPolicy::default_two_tier();
    let fib_cfg: FibConfig = policy.fib_config.clone();
    eprintln!(
        "[fast] building pool: shape={:?} num_tokens={} num_layers={} num_kv_heads={} head_dim={}",
        attn_type,
        corpus.len(),
        shape.num_layers,
        shape.num_kv_heads,
        shape.head_dim
    );
    let t_build = Instant::now();
    let (pool, receipt) = SharedKVPool::build(&corpus, &shape, seed).expect("build pool");
    let build_ms = t_build.elapsed().as_millis() as u64;
    eprintln!(
        "[fast] build ok in {build_ms}ms: pool_id={} backend={} codec={:?} ratio={:.2}x size={} bytes",
        &pool.manifest.pool_id[..12],
        receipt.backend,
        pool.manifest.shared_codec,
        pool.manifest.compression_ratio,
        pool.manifest.pool_size_bytes
    );

    let manifest = OutputManifest {
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
        fib_build_ms: build_ms,
        build_seed: pool.manifest.build_seed,
        built_at_unix: pool.manifest.built_at_unix,
    };
    let manifest_json = serde_json::to_vec(&manifest).expect("serialize manifest");

    // Decode all layers in parallel using the fast path.
    eprintln!("[fast] decompressing all layers (fast path)...");
    let t_dec = Instant::now();
    let num_layers = pool.manifest.num_layers as usize;
    let num_tokens = corpus.len();
    let num_kv_heads = pool.manifest.shape.num_kv_heads as usize;
    let head_dim = pool.manifest.shape.head_dim;

    // Build a single adapter that knows the right profile for the entire pool.
    // All blocks share the same codec/profile (fib_k4_n32 by default), so one
    // adapter is byte-identical to per-block ones.
    let adapter = FibQuantAdapter::new(
        head_dim,
        fib_cfg.k,
        fib_cfg.n,
        fib_cfg.training_samples,
        fib_cfg.lloyd_restarts,
        fib_cfg.lloyd_iterations,
    )
    .expect("create adapter");
    // We need a separate quantizer per rayon thread (each thread takes one
    // layer). Build them per-thread inside the par_iter.
    let layer_data: Vec<Vec<u8>> = (0..num_layers)
        .into_par_iter()
        .map(|layer_idx| {
            let quantizer = adapter
                .build_quantizer(seed)
                .expect("create quantizer");
            let layer = &pool.layers[layer_idx];
            // Deserialize all K and V codes for this layer. The pool
            // stores compact binary bytes (per the new wire format
            // introduced for fib_k4_n32), with a JSON fallback for
            // pools written by older proveKV versions.
            let decode_one = |b: &prove_kv::codec::CompressedBlock| -> fib_quant::FibCodeV1 {
                let bytes = &b.encoded_payload;
                if bytes.len() >= 3 && bytes[0..3] == fib_quant::COMPACT_MAGIC {
                    fib_quant::FibCodeV1::from_compact_bytes(bytes, quantizer.profile())
                        .expect("decode K/V FibCodeV1 compact")
                } else {
                    serde_json::from_slice(bytes).expect("decode K/V FibCodeV1 json fallback")
                }
            };
            let k_codes: Vec<fib_quant::FibCodeV1> =
                layer.key_blocks.iter().map(decode_one).collect();
            let v_codes: Vec<fib_quant::FibCodeV1> =
                layer.value_blocks.iter().map(decode_one).collect();

            // Fast batch decode. Each result has one Vec<f32> per code,
            // each of length head_dim.
            let k_decoded = quantizer
                .decode_batch_fast(&k_codes)
                .expect("decode K batch");
            let v_decoded = quantizer
                .decode_batch_fast(&v_codes)
                .expect("decode V batch");

            // Flatten in decode order: k_decoded[i] is the k-block for the
            // i-th encoded block, which is laid out as
            // [tok_0_head_0, tok_0_head_1, ..., tok_T-1_head_H-1].
            // We need to transpose to per-head layout for the proveKV
            // DecompressedLayer contract.
            let mut k_flat: Vec<f32> = Vec::with_capacity(num_tokens * num_kv_heads * head_dim);
            for chunk in &k_decoded {
                k_flat.extend_from_slice(chunk);
            }
            let mut v_flat: Vec<f32> = Vec::with_capacity(num_tokens * num_kv_heads * head_dim);
            for chunk in &v_decoded {
                v_flat.extend_from_slice(chunk);
            }
            let mut keys_per_head: Vec<Vec<f32>> =
                vec![Vec::with_capacity(num_tokens * head_dim); num_kv_heads];
            let mut vals_per_head: Vec<Vec<f32>> =
                vec![Vec::with_capacity(num_tokens * head_dim); num_kv_heads];
            for t in 0..num_tokens {
                for h in 0..num_kv_heads {
                    let base = t * num_kv_heads * head_dim + h * head_dim;
                    keys_per_head[h].extend_from_slice(&k_flat[base..base + head_dim]);
                    vals_per_head[h].extend_from_slice(&v_flat[base..base + head_dim]);
                }
            }
            #[derive(Serialize)]
            struct LayerOut {
                layer_index: u32,
                num_tokens: usize,
                num_heads: usize,
                head_dim: usize,
                keys: Vec<Vec<f32>>,
                values: Vec<Vec<f32>>,
            }
            let out = LayerOut {
                layer_index: layer_idx as u32,
                num_tokens,
                num_heads: num_kv_heads,
                head_dim,
                keys: keys_per_head,
                values: vals_per_head,
            };
            serde_json::to_vec(&out).expect("serialize layer")
        })
        .collect();
    let dec_ms = t_dec.elapsed().as_millis() as u64;
    eprintln!(
        "[fast] decompressed {} layers in {}ms (parallel + fast path)",
        num_layers, dec_ms
    );

    let mut f = fs::File::create(&output_path).expect("create output");
    f.write_all(&(manifest_json.len() as u64).to_le_bytes())
        .expect("write manifest len");
    f.write_all(&manifest_json).expect("write manifest");
    for layer in &layer_data {
        f.write_all(&(layer.len() as u64).to_le_bytes())
            .expect("write layer len");
        f.write_all(layer).expect("write layer");
    }
    drop(f);
    let total_bytes =
        8 + manifest_json.len() + layer_data.iter().map(|l| 8 + l.len()).sum::<usize>();
    eprintln!(
        "[fast] wrote {} layers to {} ({} MB total)",
        num_layers,
        output_path.display(),
        total_bytes as f64 / 1e6
    );
}
