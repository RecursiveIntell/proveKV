//! `prove_kv_turbo_only_bench` — measure the system-level ratio if turbo
//! alone replaces fib for the shared pool (and the per-agent shells).
//!
//! This is a head-to-head with the two-tier default: same corpus, same
//! shells, but the shared pool is built with TQB1 (turbo-quant) instead
//! of FB2 (fib-quant). The fib tier is dropped entirely.
//!
//! IMPORTANT: this is a LOSSY variant — turbo drops the QJL residual, so
//! the shared pool is no longer bit-exact. The output is a system-level
//! size comparison only.
//!
//! Usage: prove_kv_turbo_only_bench <input.json> <output_dir> [--lossy]

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use prove_kv::policy::{
    CompressionPolicy, RadiiCompression, TurboConfig, CODEC_TURBO_8BIT_BATCHED,
    CODEC_TURBO_8BIT_BATCHED_LOSSY,
};
use prove_kv::shape::{AttentionType, KvTensorShape};
use prove_kv::SharedKVPool;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct InputJson {
    shape: ShapeJson,
    shared_tokens: Vec<TokenJson>,
    agents: Vec<AgentJson>,
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

#[derive(Debug, Deserialize, Clone)]
struct TokenJson {
    id: String,
    vector: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct AgentJson {
    id: String,
    tokens: Vec<TokenJson>,
}

#[derive(Debug, Serialize)]
struct BenchState {
    tier: String,
    lossy: bool,
    n_agents: u32,
    shared_pool_bytes: u64,
    total_shell_bytes: u64,
    total_with_sharing_bytes: u64,
    naive_total_bytes: u64,
    memory_reduction_factor: f64,
}

fn write_kv_binary(path: &PathBuf, manifest: &serde_json::Value, layers: &[(Vec<f32>, Vec<f32>)]) {
    let mut f = fs::File::create(path).expect("create kv bin");
    let manifest_bytes = serde_json::to_vec(manifest).expect("serialize manifest");
    f.write_all(&(manifest_bytes.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&manifest_bytes).unwrap();
    for (k, v) in layers {
        f.write_all(&(k.len() as u32).to_le_bytes()).unwrap();
        for x in k {
            f.write_all(&x.to_le_bytes()).unwrap();
        }
        f.write_all(&(v.len() as u32).to_le_bytes()).unwrap();
        for x in v {
            f.write_all(&x.to_le_bytes()).unwrap();
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <input.json> <output_dir> [--lossy]", args[0]);
        std::process::exit(1);
    }
    let lossy = args.iter().any(|a| a == "--lossy");
    let input_path = PathBuf::from(&args[1]);
    let output_dir = PathBuf::from(&args[2]);
    fs::create_dir_all(&output_dir).expect("create output dir");

    let input_bytes = fs::read(&input_path).expect("read input");
    let input: InputJson = serde_json::from_slice(&input_bytes).expect("parse input json");
    let seed = input.seed.unwrap_or(42);

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

    // Build policy: shared = TQB1 (turbo), shell = TQB1 (turbo).
    // This drops fib entirely. Both tiers are turbo-quant.
    let mut policy = CompressionPolicy::default_two_tier();
    policy.shared_codec = if lossy {
        CODEC_TURBO_8BIT_BATCHED_LOSSY.to_string()
    } else {
        CODEC_TURBO_8BIT_BATCHED.to_string()
    };
    policy.shell_codec = if lossy {
        CODEC_TURBO_8BIT_BATCHED_LOSSY.to_string()
    } else {
        CODEC_TURBO_8BIT_BATCHED.to_string()
    };
    if lossy {
        policy.turbo_config = TurboConfig {
            bits: 8,
            projections: 32,
            radii_compression: RadiiCompression::Lossy,
        };
    }

    eprintln!(
        "[turbo-only] building pool: shared_tokens={} n_agents={} lossy={}",
        input.shared_tokens.len(),
        input.agents.len(),
        lossy
    );
    let t0 = Instant::now();
    let shared_corpus: Vec<(String, Vec<f32>)> = input
        .shared_tokens
        .iter()
        .map(|t| (t.id.clone(), t.vector.clone()))
        .collect();
    let (pool, _receipt) =
        SharedKVPool::build_with_policy(&shared_corpus, &shape, seed, policy.clone())
            .expect("build pool");
    eprintln!(
        "[turbo-only] pool built in {}ms: {} bytes, ratio={:.2}x codec={}",
        t0.elapsed().as_millis(),
        pool.manifest.pool_size_bytes,
        pool.manifest.compression_ratio,
        pool.manifest.shared_codec
    );

    // Compute total shell bytes
    let mut total_shell_bytes: u64 = 0;
    for agent in &input.agents {
        let agent_corpus: Vec<(String, Vec<f32>)> = agent
            .tokens
            .iter()
            .map(|t| (t.id.clone(), t.vector.clone()))
            .collect();
        let (shell, _) = pool
            .materialize_shell(&agent.id, &agent_corpus, seed)
            .expect("materialize shell");
        let shell_bytes: u64 = shell
            .unique_layers
            .iter()
            .map(|l| {
                l.key_blocks
                    .iter()
                    .map(|b| b.compressed_bytes as u64)
                    .sum::<u64>()
                    + l.value_blocks
                        .iter()
                        .map(|b| b.compressed_bytes as u64)
                        .sum::<u64>()
            })
            .sum();
        total_shell_bytes += shell_bytes;
    }

    let n_agents = input.agents.len() as u32;
    let num_layers = shape.num_layers as u64;
    let num_kv_heads = shape.num_kv_heads as u64;
    let head_dim = shape.head_dim as u64;
    let naive_total_bytes: u64 = (n_agents as u64 + 1)
        * (input.shared_tokens.len() as u64
            * num_layers
            * num_kv_heads
            * head_dim
            * 2
            * 4);
    let total_with_sharing_bytes = pool.manifest.pool_size_bytes + total_shell_bytes;
    let state = BenchState {
        tier: "turbo_only".to_string(),
        lossy,
        n_agents,
        shared_pool_bytes: pool.manifest.pool_size_bytes,
        total_shell_bytes,
        total_with_sharing_bytes,
        naive_total_bytes,
        memory_reduction_factor: naive_total_bytes as f64 / total_with_sharing_bytes as f64,
    };
    fs::write(
        output_dir.join("state.json"),
        serde_json::to_vec_pretty(&state).unwrap(),
    )
    .expect("write state");
    eprintln!(
        "[turbo-only] DONE n_agents={} shared={}B total_shell={}B system={}B naive={}B ratio={:.2}x",
        n_agents,
        state.shared_pool_bytes,
        state.total_shell_bytes,
        state.total_with_sharing_bytes,
        state.naive_total_bytes,
        state.memory_reduction_factor
    );
}
