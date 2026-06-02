# fib-quant

The cold-tier vector codec. ~50× compression. 100% recall on
the canonical benchmark corpus.

> Implementation of FibQuant-style radial-angular vector
> quantization for KV-cache compression, based on the public
> FibQuant reference (Lee & Kim 2026). The implementation is
> independent and **not** the original paper code — see the
> Attribution section below.

`fib-quant` decomposes a vector into spherical blocks,
quantizes each block against a Fibonacci-optimized codebook,
and stores only the codebook indices. The result: a 768-dim
f32 vector (3,072 bytes) becomes ~860 bytes in JSON, or
~64 bytes with binary packing. And it still finds the right
document at rank 1 in 100% of the canonical test queries.

This is the **cold-tier codec** in the proveKV pool. It
handles shared context that's large, stable, and accessed by
many agents:

```
┌──────────────────────────────────┐
│    SHARED POOL — fib-quant       │  ← you are here
│    System prompts, few-shot      │
│    examples, shared docs         │
│    50× compression, cos 0.863    │
└──────────┬──────────┬────────────┘
           │          │
      ┌────▼───┐ ┌───▼────┐
      │ Agent0 │ │ Agent1 │  ...  ← turbo-quant hot tier
      └────────┘ └────────┘
```

## What's in the box

- **Codecs** (`src/codec.rs`, 842 lines) — `FibQuantizer`
  with `encode`, `decode`, `encode_batch`, `decode_batch`.
  The `encode_batch` is the Rayon-parallel path that wins
  the proveKV pool build.
- **Codebook** (`src/codebook.rs`, 204 lines) —
  `LloydRefinement` of a Fibonacci-sampled seed codebook.
  Parity-verified against the reference.
- **Rotation** (`src/rotation.rs`, 189 lines) — fast
  Walsh-Hadamard rotation, with a CPU fallback and an
  optional CUDA dispatch via `gpu-backend`.
- **Spherical beta** (`src/spherical_beta.rs`, 139 lines) —
  spherical-Beta direction samplers for codebook seeding.
- **KV-cache codec** (`src/kv/`, ~2,500 lines) — a separate
  `KvCacheCodec` impl that operates on `KvTensorShape` rather
  than raw `Vec<f32>`. Includes attention-quality metrics,
  shape contracts, and policy-role-aware dispatch.
- **Profiles** (`src/profile.rs`, 408 lines) — typed
  `FibProfile` with `paper_default` (k=4, N=32), `compact`
  (k=4, N=32, binary-packed), and `kv` (KV-cache-tuned).
- **Receipts** (`src/receipt.rs`, `src/kv/receipt.rs`) —
  typed `FibEncodeReceipt` and `KvEncodeReceipt` capturing
  every parameter of the encode pipeline for audit.

## Quick Start

```rust
use fib_quant::{FibQuantizer, FibProfile};
use quant_codec_core::CodecProfile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a quantizer with the default paper profile.
    let profile = FibProfile::paper_default(32);  // 32-dim vectors
    let quantizer = FibQuantizer::new(profile.clone());

    // Encode a single vector.
    let vector: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
    let block = quantizer.encode(&vector)?;
    let reconstructed = quantizer.decode(&block)?;

    // The cosine similarity should be high.
    let cos = quantizer.cosine_similarity(&vector, &reconstructed);
    assert!(cos > 0.85);

    // Or encode a batch in parallel.
    let corpus: Vec<Vec<f32>> = /* ... */;
    let (blocks, receipt) = quantizer.encode_batch(&corpus)?;
    println!("Batch receipt: {:?}", receipt);
    Ok(())
}
```

Run it: `cargo run --release --example encode_decode`.

## Benchmarks — measured

### Compression ratios (768-dim nomic-embed-v1.5)

| Format | Bytes per vector | Ratio |
|---|---|---|
| Raw f32 | 3,072 | 1.0× |
| fib-quant JSON (default profile) | 860 | 3.6× |
| fib-quant binary-packed (`PackedFibCode`) | ~64 | ~48× |
| fib-quant KV-cache JSON | 1,200 | 2.6× |
| fib-quant KV-cache binary | ~80 | ~38× |

The "theoretical 50×" is the binary-packed compact form.
The "JSON 3.6×" is what you get with the default wire
format — the JSON envelope is 12× bigger than the actual
codebook indices. **If you're optimizing for storage, use
the binary wire format.**

### Retrieval quality (P26 measurement, semantic-memory harness)

8 queries, 200 docs, 768-dim, k=10, oversample=4:

| Route | Recall@1 | Recall@10 | nDCG@10 | Mean rank drift |
|---|---|---|---|---|
| exact_scan (no compression) | 1.000 | 1.000 | 1.000 | — |
| **fib-quant only** | **1.000** | **1.000** | **1.000** | **0.33** |
| turbo-quant only | 1.000 | 1.000 | 1.000 | 0.03 |
| proveKV (two-tier) | 1.000 | 1.000 | 1.000 | 0.25 |

**Cosine fidelity: 0.863** (single vector), **0.9996** (after
turbo-quant rerank in proveKV).

### Encode_batch throughput — "Do All" perf pass 2026-06-01

The `encode_batch` loop is the dominant cost in the proveKV
pool build. After the June 1 perf pass (AVX2+FMA SIMD +
Rayon):

| Workload | Old (f64 ref) | + SIMD | + Rayon (parallel) | + full stack |
|---|---|---|---|---|
| qwen3 2560 n=80 | 13763ms | — | 1250ms | **346ms** (40×) |
| nomic 768 n=80 | 4552ms | 94ms | 407ms | **133ms** (34×) |
| qwen3 2560 n=4 | 1449ms | 418ms | 893ms | **256ms** (5.7×) |

Numbers from `proveKV/benchmarks/DO_ALL_PERF_PASS_2026-06-01.md`.

### GPU path — measured

| Shape | n | CPU | Hadamard-GPU | Full-GPU | Best |
|---|---|---|---|---|---|
| d=64 | 80 | 14ms | **13ms (-7%)** | 14ms | Hadamard |
| d=128 | 80 | 57ms | **54ms (-5%)** | 56ms | Hadamard |
| d=768 | 80 | 2143ms | **2103ms (-2%)** | 2133ms | Hadamard |
| d=2560 | 4 | 1571ms | 1564ms (0%) | 1554ms (0%) | tie |

**Honest takeaway:** fib-quant's `encode_batch` is 2-7%
faster on a real GPU (msi i7-6700HQ + GTX 1070) with the
Hadamard path engaged. The codebook_lookup kernel exists and
is parity-verified, but the per-call H2D/D2H overhead
currently negates its win. A device-side pipeline is the
next step.

### Test coverage

- **23 integration test files** in `tests/`:
  - `bitpack_indices`, `codebook_determinism`,
    `compact_bytes_roundtrip`, `corruption_rejection`,
    `decode_batch_fast_parity`, `direction_generators`,
    `encode_decode_roundtrip`, `kv_attention_quality`,
    `kv_corruption_rejection`, `kv_encode_decode_reference`,
    `kv_policy_role_aware`, `kv_property_shapes`,
    `kv_shape_contracts`, `lloyd_refinement`,
    `norm_payload_rejection`, `paper_k2_radius_closed_form`,
    `paper_smoke_regression`, `profile_digest`,
    `profile_resource_bounds`, `property_bitpack`,
    `property_codec`, `rotation_identity`,
    `spherical_beta_sampler`.
- **4 examples**: `build_codebook`, `encode_batch_microbench`,
  `encode_decode`, `test_compact_decode`.
- **4 benches** (criterion): `codebook_build`, `encode_decode`,
  `kv_attention_ref`, `kv_encode_decode`.
- `cargo test` clean, `cargo clippy --all-targets -- -D warnings` clean.

## MSRV

Rust 1.75 (2021 edition). `#![forbid(unsafe_code)]` at the
crate level.

## Dependencies

- `serde` (with `derive`).
- `serde_json`.
- `blake3`.
- `rand` + `rand_chacha` (dev).
- `gpu-backend` (optional) — for the GPU Hadamard dispatch.
- `rayon` (optional, behind the `parallel` feature) — for
  parallel batch encoding.
- `proptest` (dev).
- `criterion` (dev).
- `nalgebra` (with `serde-serialize`).

## License

Apache-2.0. See `LICENSE-APACHE` for the full text.

## Changelog

See `CHANGELOG.md` for the release history.

## Attribution

This crate is an independent Rust implementation of the
FibQuant compression technique described in the public
literature (Lee & Kim, 2026). It is **not** the original
authors' reference code, and does not claim affiliation with
the FibQuant paper authors. The mathematical approach
(radial-angular block decomposition with Fibonacci-sampled
codebook seeding) follows the published specification.

The `kv-cache` codec profile, the parallel encode pipeline,
the GPU dispatch path, the receipt infrastructure, and the
test suite are original to this implementation.

## Where it's used

`fib-quant` is the cold-tier codec for:

- `proveKV` — the shared pool is fib-quant compressed.
  Every shared system prompt, every shared few-shot example,
  every shared doc goes through `encode_batch`.
- `semantic-memory` — when `AdmissibilityClass::Standard` is
  selected by `quant-governor`, semantic-memory can route to
  the fib-quant sidecar for candidate generation.
- `scr-runtime-compression` — the `fib` feature is the
  fib-quant adapter for the runtime.

Any system that needs a **high-ratio, medium-fidelity** vector
codec — search-only recall, cold tier, ~50× storage savings
— can adopt `fib-quant` directly.
