//! `prove_kv_gpu_bench` — end-to-end proveKV pool build benchmark.
//!
//! Runs the SharedKVPool build path on a deterministic synthetic corpus
//! across two model shapes (nomic-embed 768-dim and qwen3-embedding 2560-dim)
//! and three corpus sizes (4, 20, 80 documents). Reports:
//!
//!   - wall: total pool build time (includes codebook + encode + digest math)
//!   - receipt_ms: fib_build_ms from the receipt
//!   - encode_only: time spent in encode_batch (codebook build excluded)
//!   - gpu_dispatch: per-call probe — would this batch actually use GPU?
//!   - backend: what the receipt said happened
//!   - ratio, size: pool characteristics
//!
//! Usage:
//!   cargo run --release --example prove_kv_gpu_bench
//!   cargo run --release --example prove_kv_gpu_bench --features gpu

use std::time::Instant;

use prove_kv::codec::create_codec;
use prove_kv::pool::SharedKVPool;
use prove_kv::policy::CODEC_FIB_K4_N32;
use prove_kv::shape::{AttentionType, KvTensorShape};
use rand::Rng;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};

#[derive(Debug, Clone, Copy)]
struct ModelShape {
    name: &'static str,
    num_layers: u32,
    num_kv_heads: u32,
    head_dim: usize,
}

const NOMIC: ModelShape = ModelShape {
    name: "nomic-embed-text (768-dim)",
    num_layers: 12,
    num_kv_heads: 12,
    head_dim: 64,
};

const QWEN3: ModelShape = ModelShape {
    name: "qwen3-embedding (2560-dim)",
    num_layers: 28,
    num_kv_heads: 4,
    head_dim: 128,
};

fn make_shape(m: ModelShape) -> KvTensorShape {
    KvTensorShape {
        attention_type: AttentionType::GQA,
        num_layers: m.num_layers,
        num_heads: m.num_kv_heads * 4, // pretend GQA with 4x q heads
        num_kv_heads: m.num_kv_heads,
        head_dim: m.head_dim,
        hidden_size: m.num_kv_heads as usize * 4 * m.head_dim,
    }
}

fn make_corpus(m: ModelShape, n_tokens: usize) -> Vec<(String, Vec<f32>)> {
    let mut rng = ChaCha8Rng::seed_from_u64(0xDEAD_BEEF);
    let vec_len = m.num_layers as usize * m.num_kv_heads as usize * m.head_dim * 2;
    (0..n_tokens)
        .map(|i| {
            let v: Vec<f32> = (0..vec_len).map(|_| rng.gen_range(-1.0..1.0)).collect();
            (format!("doc_{i}"), v)
        })
        .collect()
}

/// Time just the encode_batch portion, using a codec that already had its
/// codebook built (so the Lloyd-Max / codebook training cost is excluded).
fn time_encode_only(
    model: ModelShape,
    n_tokens: usize,
    corpus: &[(String, Vec<f32>)],
) -> u128 {
    // First, do one full build to build the codebook, then immediately
    // re-use the resulting quantizer for the timed encode-only run.
    let shape = make_shape(model);
    let _ = SharedKVPool::build(corpus, &shape, 42).unwrap();

    // Now time only the encode_batch calls. This is what the GPU would
    // actually accelerate.
    let policy = prove_kv::policy::CompressionPolicy::default_two_tier();
    let codec = create_codec(
        CODEC_FIB_K4_N32,
        model.head_dim,
        Some(&policy.fib_config),
        None,
    )
    .unwrap();

    let head_dim = model.head_dim;
    let num_kv_heads = model.num_kv_heads as usize;
    let num_layers = model.num_layers as usize;

    let start = Instant::now();
    for layer_idx in 0..num_layers {
        let mut key_inputs: Vec<Vec<f32>> = Vec::with_capacity(n_tokens * num_kv_heads);
        let mut value_inputs: Vec<Vec<f32>> = Vec::with_capacity(n_tokens * num_kv_heads);
        for (_token_id, vec) in corpus.iter() {
            for head_idx in 0..num_kv_heads {
                let base_offset =
                    layer_idx * num_kv_heads * head_dim * 2 + head_idx * head_dim * 2;
                let key_end = base_offset + head_dim;
                let value_end = key_end + head_dim;
                key_inputs.push(vec[base_offset..key_end].to_vec());
                value_inputs.push(vec[key_end..value_end].to_vec());
            }
        }
        let key_refs: Vec<&[f32]> = key_inputs.iter().map(|v| v.as_slice()).collect();
        let value_refs: Vec<&[f32]> = value_inputs.iter().map(|v| v.as_slice()).collect();
        let _ = codec.encode_batch(&key_refs, 42).unwrap();
        let _ = codec.encode_batch(&value_refs, 42).unwrap();
    }
    start.elapsed().as_millis()
}

fn run_one(model: ModelShape, n_tokens: usize) {
    let shape = make_shape(model);
    let corpus = make_corpus(model, n_tokens);

    // warm the codec + codebook build once outside the timed region
    let _ = SharedKVPool::build(&corpus[..1.min(corpus.len())], &shape, 42).unwrap();

    // Time the encode-only portion (codebook build excluded).
    let encode_only_ms = time_encode_only(model, n_tokens, &corpus);

    // Now time the full build.
    let start = Instant::now();
    let (pool, receipt) = SharedKVPool::build(&corpus, &shape, 42).unwrap();
    let wall = start.elapsed();

    let batch_n = n_tokens * model.num_kv_heads as usize;
    let gpu_dispatch_would = if batch_n >= 16 && model.head_dim >= 64 && cfg!(feature = "gpu") {
        "yes"
    } else {
        "no"
    };

    println!(
        "  {model:32} n={n_tokens:>3}  wall={wall_ms:>6} ms  encode_only={enc:>5} ms  \
         codebook={cb:>5} ms  batch={bn:>3}  gpu_dispatch={gd:>3}  backend={bk:>3}  ratio={ratio:.2}x  size={kb} KB",
        model = format!("{} {}", model.name, ""),
        n_tokens = n_tokens,
        wall_ms = wall.as_millis(),
        enc = encode_only_ms,
        cb = wall.as_millis() as i64 - encode_only_ms as i64,
        bn = batch_n,
        gd = gpu_dispatch_would,
        bk = receipt.backend,
        ratio = receipt.compression_ratio,
        kb = receipt.pool_size_bytes / 1024,
    );

    assert_eq!(pool.manifest.num_shared_tokens, n_tokens as u32);
    // Receipt backend must agree with the per-call probe.
    let expected = if gpu_dispatch_would == "yes" { "gpu" } else { "cpu" };
    assert_eq!(
        receipt.backend, expected,
        "receipt backend drift: probe said {expected}, receipt said {}",
        receipt.backend
    );
}

fn main() {
    println!("proveKV pool-build benchmark");
    println!("compile-time: gpu feature = {}", cfg!(feature = "gpu"));
    println!();

    for model in &[NOMIC, QWEN3] {
        println!("=== {} ===", model.name);
        for n in &[4usize, 20, 80] {
            run_one(*model, *n);
        }
        println!();
    }

    println!("Notes:");
    println!("  - 'encode_only' excludes codebook build (Lloyd-Max training).");
    println!("  - 'codebook' = wall - encode_only, the one-time cost per quantizer.");
    println!("  - GPU only accelerates 'encode_only' (Hadamard rotation).");
    println!("  - GPU threshold is n>=16, dim>=64. n=4 qwen3 batch=16 is exactly at the edge.");
    println!("  - 'backend' must agree with 'gpu_dispatch' — a drift here is a receipt bug.");
}
