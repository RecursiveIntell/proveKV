# fib-quant

> **Historical claim boundary:** the ~50× compression and 100% recall
> statements retained below are historical unqualified wording, not
> current-revision certification. The current profile records an explicit
> index bit rate, and the wire stores a norm payload plus packed indices.
> Byte counts and quality require a matching profile and source-bound receipt.

An experimental cold-tier vector codec. The size and retrieval numbers below
are historical, profile- and fixture-specific reports, not guarantees for the
current source or a new workload.

> Implementation of FibQuant-style radial-angular vector
> quantization for KV-cache compression, based on the public
> FibQuant reference (Lee & Kim 2026). The implementation is
> independent and **not** the original paper code — see the
> Attribution section below.

`fib-quant` decomposes a vector into spherical blocks,
quantizes each block against a Fibonacci-optimized codebook,
and stores only the codebook indices. For one previously reported 768-dim profile, a 3,072-byte f32 vector
became about 860 bytes in JSON or 64 bytes in a compact binary form.
The reported rank-1 result came from eight fixture queries; neither
size nor retrieval quality is certified for a new profile or corpus.

This is the **cold-tier codec** in the proveKV pool. It
handles shared context that's large, stable, and accessed by
many agents:

```
┌──────────────────────────────────┐
│    SHARED POOL — fib-quant       │  ← you are here
│    System prompts, few-shot      │
│    examples, shared docs         │
│    historical profile report      │
└──────────┬──────────┬────────────┘
           │          │
      ┌────▼───┐ ┌───▼────┐
      │ Agent0 │ │ Agent1 │  ...  ← turbo-quant hot tier
      └────────┘ └────────┘
```

## What's in the box

- **Vector codec** (`src/codec.rs`) — `FibQuantizer` provides
  `encode`, `decode`, `encode_with_receipt`, `encode_batch`,
  and `decode_batch`. Batch encoding accepts borrowed vector slices
  and returns encoded codes.
- **Codebook** (`src/codebook.rs`, `src/lloyd.rs`) —
  Fibonacci-style direction seeding and Lloyd refinement.
- **Rotation** (`src/rotation.rs`) — a seeded stored orthogonal
  rotation. The feature-gated batch path calls `gpu-backend`;
  its public Hadamard operation currently selects the CPU reference.
- **Spherical beta** (`src/spherical_beta.rs`) —
  spherical-Beta direction samplers for codebook seeding.
- **KV-cache helpers** (`src/kv/`, `kv` feature, default-off) —
  experimental CPU reference encode/decode functions, explicit
  `KvTensorShapeV1` contracts, policy decisions, and attention-quality
  reports. This module uses its own typed KV contracts; it is not an
  implementation of `quant_codec_core::KvCacheCodec`.
- **Profiles** (`src/profile.rs`) — `FibQuantProfileV1`, constructed
  with `paper_default(ambient_dim, block_dim, codebook_size, seed)`.
  Profile construction and quantizer construction are fallible.
- **Receipts** (`src/receipt.rs`, `src/kv/receipt.rs`) —
  `FibQuantCompressionReceiptV1`, `KvCompressionReceiptV1`,
  `KvDecodeReceiptV1`, and `KvEvalReceiptV1`.

## Quick Start

The following uses the checked-in `examples/encode_decode.rs` API.
It is a small encode/decode example, not a quality benchmark.

```rust
use fib_quant::{FibQuantProfileV1, FibQuantizer};

fn main() -> fib_quant::Result<()> {
    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, 42)?;
    profile.training_samples = 128;
    profile.lloyd_restarts = 1;
    profile.lloyd_iterations = 2;

    let quantizer = FibQuantizer::new(profile)?;
    let input = vec![1.0, 0.5, -0.25, 0.125, -1.5, 0.75, 0.25, -0.5];
    let (code, receipt) = quantizer.encode_with_receipt(&input)?;
    let decoded = quantizer.decode(&code)?;

    println!("encoded_digest={}", receipt.encoded_digest);
    println!("source_vector_digest={}", receipt.source_vector_digest);
    println!("decoded_len={}", decoded.len());
    Ok(())
}
```

From the workspace root, run:

```sh
cargo run --release -p fib-quant --example encode_decode
```

For a batch, pass `&[&[f32]]` to `encode_batch`; it returns
`Vec<FibCodeV1>`. The `parallel` feature is enabled by default.
The `gpu` feature enables CUDA readiness through `gpu-backend`;
`gpu_codebook_lookup` additionally enables the optional codebook
dispatch in this crate. Actual CUDA lookup still depends on the
backend's readiness and narrow shape contract. See
[`gpu-backend`](../gpu-backend/README.md) for the current dispatch boundary.

## Historical benchmark reports

The figures below were previously reported for specific profiles and fixtures.
They have not been reproduced for this HEAD in this README review. The cited
June performance source file, `proveKV/benchmarks/DO_ALL_PERF_PASS_2026-06-01.md`,
is not present in the current repository tree, so those throughput rows need
an independently accessible receipt before use as current performance claims.

### Compression ratios (768-dim nomic-embed-v1.5)

| Format | Bytes per vector | Ratio |
|---|---|---|
| Raw f32 | 3,072 | 1.0× |
| fib-quant JSON (default profile) | 860 | 3.6× |
| fib-quant binary-packed (`PackedFibCode`) | ~64 | ~48× |
| fib-quant KV-cache JSON | 1,200 | 2.6× |
| fib-quant KV-cache binary | ~80 | ~38× |

The historical ~48–50× figure describes the binary-packed form in the
reported profile, while the reported JSON representation was about 3.6×
smaller than raw f32. These are profile- and format-dependent byte ratios;
measure the complete representation selected by your application.

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

### Historical GPU-path report

The table below is retained as a previously reported measurement.
It does not certify the current dispatch path: the public backend
Hadamard operation now selects the CPU reference, and only the
narrow codebook-lookup contract can dispatch to CUDA. Re-establish
source-bound hardware evidence before making a current GPU speedup claim.


| Shape | n | CPU | Hadamard-GPU | Full-GPU | Best |
|---|---|---|---|---|---|
| d=64 | 80 | 14ms | **13ms (-7%)** | 14ms | Hadamard |
| d=128 | 80 | 57ms | **54ms (-5%)** | 56ms | Hadamard |
| d=768 | 80 | 2143ms | **2103ms (-2%)** | 2133ms | Hadamard |
| d=2560 | 4 | 1571ms | 1564ms (0%) | 1554ms (0%) | tie |

Those historical rows are not a current Hadamard-GPU capability claim.
The current public Hadamard call runs on CPU; see
[`gpu-backend`](../gpu-backend/README.md) for the active CUDA lookup
contract and typed readiness checks.

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
- To check the current source, run `cargo test -p fib-quant` and
  `cargo clippy -p fib-quant --all-targets -- -D warnings` from the
  workspace root. These checks were not executed in this README review.

## MSRV

Rust 1.75 (2021 edition). `#![forbid(unsafe_code)]` at the
crate level.

## Dependencies

- `serde` (with `derive`).
- `serde_json`.
- `blake3`.
- `rand` + `rand_chacha` (dev).
- `gpu-backend` — CPU/SIMD primitives, with optional CUDA codebook lookup.
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

A consumer evaluating this experimental vector codec should measure
storage and retrieval quality on its own profile and workload before
adopting it; the historical ~50× figure is not a general saving.
