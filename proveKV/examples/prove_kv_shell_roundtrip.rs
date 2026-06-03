//! `prove_kv_shell_roundtrip` — shell-tier PPL bench driver.
//!
//! Given a corpus of per-token K/V vectors (the "agent's unique prefix"
//! extracted from a real forward pass on an LLM):
//!
//! 1. Build a SharedKVPool from the corpus
//! 2. Materialize an AgentShell for the same tokens (1:1 with the pool —
//!    every token is "unique" to this agent in this test)
//! 3. Decompress the shell back to f32 K/V
//! 4. Write the decompressed K/V to a binary file in the same format
//!    `ppl_validate_shell.py` expects
//!
//! Supports `--lossy` to use the TQB1-L (BlockLogU8 radii) wire format
//! for the shell. Default is lossless TQB1.
//!
//! Output binary format (per layer):
//!   u32 K_len (in floats) | K data as f32 | u32 V_len | V data as f32
//!
//! Usage:
//!   prove_kv_shell_roundtrip <input.json> <output.bin> [--lossy]
//!
//! The input.json is the same shape `build_prove_kv_corpus.py` produces:
//!   { "shape": {...}, "tokens": [{"id": "...", "vector": [f32, ...]}, ...], "seed": 42 }

use provekv::codec::FibQuantAdapter;
use provekv::manifest::ShellManifest;
use provekv::policy::{CompressionPolicy, FibConfig, RadiiCompression, TurboConfig};
use provekv::shape::{AttentionType, KvTensorShape};
use provekv::{AgentShell, SharedKVPool};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Deserialize)]
struct InputShape {
    num_layers: u32,
    num_kv_heads: u32,
    head_dim: u32,
    attention_type: String,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Deserialize)]
struct InputToken {
    id: String,
    vector: Vec<f32>,
}

#[derive(Deserialize)]
struct InputJson {
    shape: InputShape,
    tokens: Vec<InputToken>,
}

#[derive(Serialize)]
struct ShellBenchReceipt {
    pool_id: String,
    pool_size_bytes: u64,
    pool_compression_ratio: f64,
    shell_digest: String,
    shell_size_bytes: u64,
    shell_codec: String,
    num_tokens: u32,
    num_layers: u32,
    num_kv_heads: u32,
    head_dim: u32,
    build_ms: u64,
    roundtrip_ms: u64,
    lossy: bool,
    output_bytes: u64,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!("usage: {} <input.json> <output.bin> [--lossy]", args[0]);
        std::process::exit(1);
    }
    let lossy = args.iter().any(|a| a == "--lossy");
    let input_path = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);

    let input_bytes = fs::read(&input_path).expect("read input");
    let input: InputJson = serde_json::from_slice(&input_bytes).expect("parse input json");

    let attn_type = match input.shape.attention_type.as_str() {
        "MHA" => AttentionType::MHA,
        "MQA" => AttentionType::MQA,
        "GQA" => AttentionType::GQA,
        other => panic!("unknown attention_type: {other}"),
    };
    let shape = KvTensorShape {
        attention_type: attn_type,
        num_layers: input.shape.num_layers,
        num_heads: input.shape.num_kv_heads, // assume MHA: heads == kv_heads
        num_kv_heads: input.shape.num_kv_heads,
        head_dim: input.shape.head_dim as usize,
        hidden_size: (input.shape.num_kv_heads * input.shape.head_dim) as usize, // not used for MHA
    };
    let seed = input.shape.seed.unwrap_or(42);

    // Build a lossless-OR-lossy policy. For the SHELL tier only — but
    // the shared pool's codec is controlled by the fib_config which
    // we leave at the default (lossless fib_k4_n32). The lossless
    // flag only affects the turbo tier (shell).
    let mut policy = CompressionPolicy::default_two_tier();
    if lossy {
        policy.turbo_config = TurboConfig {
            bits: 8,
            projections: 32,
            radii_compression: RadiiCompression::Lossy,
        };
        eprintln!("[shell] lossy mode: TQB1-L (BlockLogU8 radii)");
    } else {
        eprintln!("[shell] lossless mode: TQB1 (f32 radii)");
    }
    let fib_cfg: FibConfig = policy.fib_config.clone();

    // Build the corpus as (id, vector) pairs.
    let corpus: Vec<(String, Vec<f32>)> = input
        .tokens
        .into_iter()
        .map(|t| (t.id, t.vector))
        .collect();
    let num_tokens = corpus.len();
    eprintln!(
        "[shell] building pool: num_tokens={} num_layers={} num_kv_heads={} head_dim={}",
        num_tokens, shape.num_layers, shape.num_kv_heads, shape.head_dim
    );
    let t_build = Instant::now();
    let (pool, pool_receipt) =
        SharedKVPool::build_with_policy(&corpus, &shape, seed, policy.clone())
            .expect("build pool");
    let build_ms = t_build.elapsed().as_millis() as u64;
    eprintln!(
        "[shell] pool build ok in {build_ms}ms: pool_id={} ratio={:.2}x size={} bytes",
        pool.manifest.pool_id,
        pool_receipt.compression_ratio,
        pool_receipt.pool_size_bytes
    );

    // Materialize a shell that covers ALL tokens (1:1 with the pool).
    // This is the worst case for shell size — every token is "unique".
    // For a real multi-agent bench, this is what a single agent with
    // no overlap looks like.
    let agent_id = "shell_ppl_agent";
    let agent_tokens: Vec<(String, Vec<f32>)> = corpus.clone();
    eprintln!("[shell] materializing shell for {num_tokens} unique tokens");
    let t_shell = Instant::now();
    let (shell, shell_receipt) =
        provekv::shell::materialize_shell(&pool, agent_id, &agent_tokens, seed)
            .expect("materialize shell");
    let shell_ms = t_shell.elapsed().as_millis() as u64;
    let shell_size = shell_receipt.shell_size_bytes;
    eprintln!(
        "[shell] shell materialization ok in {shell_ms}ms: shell_digest={} size={} bytes",
        shell_receipt.shell_digest, shell_size
    );

    // Decompress the shell back to f32 K/V.
    eprintln!("[shell] decompressing shell to f32 K/V");
    let t_decode = Instant::now();
    let layers = shell
        .decompress_all_layers_with_seed(seed)
        .expect("decompress shell");
    let decode_ms = t_decode.elapsed().as_millis() as u64;
    eprintln!(
        "[shell] decompressed {} layers in {}ms",
        layers.len(),
        decode_ms
    );

    // Write the binary file. Format: per layer, u32 K_len | K f32 | u32 V_len | V f32.
    // Layout per layer: K_flat and V_flat are [num_kv_heads, num_tokens, head_dim] per
    // the existing prove_kv/multi_agent_shell convention. PPL bench reads these directly.
    let mut out_file = fs::File::create(&output_path).expect("create output");
    let mut total_bytes: u64 = 0;
    for (layer_idx, (k_flat, v_flat)) in layers.iter().enumerate() {
        let k_len = k_flat.len() as u32;
        let v_len = v_flat.len() as u32;
        out_file
            .write_all(&k_len.to_le_bytes())
            .expect("write k_len");
        for v in k_flat {
            out_file
                .write_all(&v.to_le_bytes())
                .expect("write k data");
        }
        out_file
            .write_all(&v_len.to_le_bytes())
            .expect("write v_len");
        for v in v_flat {
            out_file
                .write_all(&v.to_le_bytes())
                .expect("write v data");
        }
        total_bytes += 8 + (k_len as u64) * 4 + (v_len as u64) * 4;
        if layer_idx == 0 || layer_idx == layers.len() - 1 {
            eprintln!(
                "[shell] layer {}: K_len={} V_len={}",
                layer_idx, k_len, v_len
            );
        }
    }
    out_file.flush().expect("flush");
    eprintln!(
        "[shell] wrote {} bytes to {}",
        total_bytes,
        output_path.display()
    );

    // Write the receipt next to the binary.
    let receipt_path = output_path.with_extension("receipt.json");
    let codec = if lossy {
        "turbo_8bit_batched_lossy"
    } else {
        "turbo_8bit_batched"
    };
    let receipt = ShellBenchReceipt {
        pool_id: pool.manifest.pool_id.clone(),
        pool_size_bytes: pool_receipt.pool_size_bytes,
        pool_compression_ratio: pool_receipt.compression_ratio,
        shell_digest: shell_receipt.shell_digest.clone(),
        shell_size_bytes: shell_size,
        shell_codec: codec.to_string(),
        num_tokens: num_tokens as u32,
        num_layers: shape.num_layers,
        num_kv_heads: shape.num_kv_heads,
        head_dim: shape.head_dim as u32,
        build_ms,
        roundtrip_ms: decode_ms,
        lossy,
        output_bytes: total_bytes,
    };
    fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt).unwrap())
        .expect("write receipt");
    eprintln!("[shell] receipt: {}", receipt_path.display());
}
