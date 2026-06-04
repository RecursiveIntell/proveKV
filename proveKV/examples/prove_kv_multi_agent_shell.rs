//! `prove_kv_multi_agent_shell` — produce the artifact set the Python PPL
//! bench (`ppl_multi_agent.py`) needs:
//!   - shared_pool_receipt.json
//!   - shared_kv.bin           (decompressed shared K/V, f32 LE)
//!   - agents_receipt.json
//!   - agent_<i>_kv.bin        (decompressed shell K/V, f32 LE, per agent)
//!
//! Input JSON shape:
//!   {
//!     "shape": { attention_type, num_layers, num_heads, num_kv_heads, head_dim, hidden_size },
//!     "shared_tokens":  [ {id, vector}, ... ],
//!     "agents":         [ {id, tokens: [{id, vector}, ...]}, ... ],
//!     "seed":           <int>  (optional, default 42)
//!   }
//!
//! Usage: prove_kv_multi_agent_shell <input.json> <output_dir> [--lossy]
//!
//! On success, writes a state.json summarizing the run (sizes, ratios, codec ids).

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use provekv::policy::{CompressionPolicy, RadiiCompression, CODEC_FIB_K4_N32_BATCHED};
use provekv::shape::{AttentionType, KvTensorShape};
use provekv::SharedKVPool;
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
struct SharedPoolReceipt {
    pool_id: String,
    num_shared_tokens: u32,
    num_layers: u32,
    num_kv_heads: u32,
    head_dim: usize,
    shared_codec: String,
    compression_ratio: f64,
    pool_size_bytes: u64,
    build_seed: u64,
    built_at_unix: i64,
}

#[derive(Debug, Serialize)]
struct AgentsReceipt {
    n_agents: u32,
    per_agent: Vec<PerAgentReceipt>,
    total_shell_bytes: u64,
    avg_shell_bytes: f64,
}

#[derive(Debug, Serialize, Clone)]
struct PerAgentReceipt {
    agent_id: String,
    num_unique_tokens: u32,
    shell_bytes: u64,
    shell_codec: String,
}

#[derive(Debug, Serialize)]
struct BenchState {
    model: String,
    n_agents: u32,
    shared_pool_bytes: u64,
    total_shell_bytes: u64,
    total_with_sharing_bytes: u64,
    naive_total_bytes: u64,
    memory_reduction_factor: f64,
    shell_codec: String,
    shared_codec: String,
    radii_compression: String,
    lossy: bool,
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
        eprintln!(
            "usage: {} <input.json> <output_dir> [--lossy] [--bits N]",
            args[0]
        );
        std::process::exit(1);
    }
    let lossy = args.iter().any(|a| a == "--lossy");
    // Parse --bits N. Default 4 (PPL-validated 40.53x lossless / 76.55x lossy).
    let bits: u8 = match args.iter().position(|a| a == "--bits") {
        Some(i) if i + 1 < args.len() => args[i + 1]
            .parse()
            .expect("--bits value must be a u8 in [2, 16]"),
        _ => 4,
    };
    if !(2..=16).contains(&bits) {
        eprintln!("--bits must be in [2, 16], got {bits}");
        std::process::exit(1);
    }
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
    let num_layers = shape.num_layers as usize;
    let num_kv_heads = shape.num_kv_heads as usize;
    let head_dim = shape.head_dim;

    // Build the compression policy.
    let mut policy = CompressionPolicy::default_two_tier();
    // Override the shell tier's bits if --bits was provided. The fib tier
    // is unaffected (it doesn't use turbo). The default_two_tier shell codec
    // is `turbo_<bits>bit_batched` (or _lossy variant).
    policy.turbo_config.bits = bits;
    if lossy {
        policy.turbo_config.radii_compression = RadiiCompression::Lossy;
    }

    // Build the pool from the shared tokens.
    eprintln!(
        "[multi-agent] building pool: shared_tokens={} n_agents={} lossy={}",
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
    let (pool, _receipt) = SharedKVPool::build_with_policy(&shared_corpus, &shape, seed, policy.clone())
        .expect("build pool");
    eprintln!(
        "[multi-agent] pool built in {}ms: {} bytes, ratio={:.2}x codec={}",
        t0.elapsed().as_millis(),
        pool.manifest.pool_size_bytes,
        pool.manifest.compression_ratio,
        pool.manifest.shared_codec
    );

    // Decode all pool layers and write shared_kv.bin.
    eprintln!("[multi-agent] decoding shared pool layers...");
    let t1 = Instant::now();
    let shared_layers = pool.decompress_all_layers_with_seed(seed).expect("decompress shared");
    eprintln!(
        "[multi-agent] shared decoded in {}ms ({} layers)",
        t1.elapsed().as_millis(),
        shared_layers.len()
    );
    let shared_manifest = serde_json::json!({
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "num_tokens": shared_corpus.len(),
        "pool_id": pool.manifest.pool_id,
        "pool_size_bytes": pool.manifest.pool_size_bytes,
    });
    write_kv_binary(&output_dir.join("shared_kv.bin"), &shared_manifest, &shared_layers);

    let shared_pool_receipt = SharedPoolReceipt {
        pool_id: pool.manifest.pool_id.clone(),
        num_shared_tokens: pool.manifest.num_shared_tokens,
        num_layers: pool.manifest.num_layers,
        num_kv_heads: shape.num_kv_heads,
        head_dim: shape.head_dim,
        shared_codec: pool.manifest.shared_codec.clone(),
        compression_ratio: pool.manifest.compression_ratio,
        pool_size_bytes: pool.manifest.pool_size_bytes,
        build_seed: pool.manifest.build_seed,
        built_at_unix: pool.manifest.built_at_unix,
    };
    fs::write(
        output_dir.join("shared_pool_receipt.json"),
        serde_json::to_vec_pretty(&shared_pool_receipt).unwrap(),
    )
    .expect("write shared pool receipt");

    // Build shells for each agent, decode them, write per-agent kv bins.
    // The shell codec id reflects the policy's bit rate and radii
    // compression. Per-agent receipts read this from the block, but the
    // top-level `state.shell_codec` field is derived here.
    let shell_codec_str = if lossy {
        format!("turbo_{bits}bit_batched_lossy")
    } else {
        format!("turbo_{bits}bit_batched")
    };

    let mut per_agent = Vec::with_capacity(input.agents.len());
    let mut total_shell_bytes: u64 = 0;
    for (i, agent) in input.agents.iter().enumerate() {
        let agent_corpus: Vec<(String, Vec<f32>)> = agent
            .tokens
            .iter()
            .map(|t| (t.id.clone(), t.vector.clone()))
            .collect();
        eprintln!(
            "[multi-agent] agent {i} {}: {} unique tokens",
            agent.id,
            agent_corpus.len()
        );
        let (shell, _shell_receipt) = pool
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
        let agent_codec = shell
            .unique_layers
            .first()
            .and_then(|l| l.key_blocks.first())
            .map(|b| b.codec.clone())
            .unwrap_or_default();
        per_agent.push(PerAgentReceipt {
            agent_id: agent.id.clone(),
            num_unique_tokens: agent_corpus.len() as u32,
            shell_bytes,
            shell_codec: agent_codec,
        });

        // Decode shell layers and write agent_<i>_kv.bin.
        let agent_layers = shell
            .decompress_all_layers_with_seed(seed)
            .expect("decompress shell");
        let agent_manifest = serde_json::json!({
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "num_tokens": agent_corpus.len(),
            "agent_id": agent.id,
            "shell_bytes": shell_bytes,
        });
        write_kv_binary(
            &output_dir.join(format!("agent_{i}_kv.bin")),
            &agent_manifest,
            &agent_layers,
        );
    }

    let agents_receipt = AgentsReceipt {
        n_agents: input.agents.len() as u32,
        per_agent: per_agent.clone(),
        total_shell_bytes,
        avg_shell_bytes: total_shell_bytes as f64 / input.agents.len() as f64,
    };
    fs::write(
        output_dir.join("agents_receipt.json"),
        serde_json::to_vec_pretty(&agents_receipt).unwrap(),
    )
    .expect("write agents receipt");

    // Compute the bench state.
    let n_agents = input.agents.len() as u32;
    let naive_total_bytes: u64 = (n_agents as u64 + 1)
        * (input.shared_tokens.len() as u64
            * num_layers as u64
            * num_kv_heads as u64
            * head_dim as u64
            * 2
            * 4);
    let total_with_sharing_bytes = pool.manifest.pool_size_bytes + total_shell_bytes;
    let state = BenchState {
        model: format!(
            "Qwen2.5-0.5B ({} layers, {} kv heads, head_dim {})",
            num_layers, num_kv_heads, head_dim
        ),
        n_agents,
        shared_pool_bytes: pool.manifest.pool_size_bytes,
        total_shell_bytes,
        total_with_sharing_bytes,
        naive_total_bytes,
        memory_reduction_factor: naive_total_bytes as f64 / total_with_sharing_bytes as f64,
        shell_codec: shell_codec_str.to_string(),
        shared_codec: CODEC_FIB_K4_N32_BATCHED.to_string(),
        radii_compression: if lossy { "Lossy".to_string() } else { "Lossless".to_string() },
        lossy,
    };
    fs::write(
        output_dir.join("state.json"),
        serde_json::to_vec_pretty(&state).unwrap(),
    )
    .expect("write state");
    eprintln!(
        "[multi-agent] DONE n_agents={} shared={}B total_shell={}B system={}B naive={}B ratio={:.2}x",
        n_agents,
        state.shared_pool_bytes,
        state.total_shell_bytes,
        state.total_with_sharing_bytes,
        state.naive_total_bytes,
        state.memory_reduction_factor
    );
}
