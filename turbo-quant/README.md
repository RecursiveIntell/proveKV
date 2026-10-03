# turbo-quant

Experimental vector compression sidecars for embedding search.

`turbo-quant` implements three compression sidecars (`PolarQuant`,
`TurboQuant`, and QJL sketches) and the surrounding infrastructure
for bit-packed wire formats, approximate candidate generation,
KV-cache shadow mode, and codec evaluation. Exact reranking uses
caller-owned raw vectors; the candidate index does not own that step.

**Status:** experimental / research substrate. See the
"Scope and limits" section below for what this crate is and is not
safe to claim.

## What's in the box

- **PolarQuant** — angular quantization of vectors onto a uniform
  angle grid. Asymmetric: scoring in compressed space only
  (no decode). Source code: `src/polar.rs`.
- **TurboQuant** — adds a QJL residual sketch on top of PolarQuant
  to recover accuracy. Symmetric: the residual sketch lets you
  approximate the inner product, and exact rerank uses the raw
  vector. Source: `src/turbo.rs`.
- **QJL sketches** — randomized sign-based Johnson-Lindenstrauss
  projections for cheap approximate inner product estimation.
  Source: `src/qjl.rs`.
- **Bit-packed wire formats** — `PackedPolarCode`,
  `PackedQjlSketch`, `PackedTurboCode` with a fixed
  `storage_layout: "polar_radii_f32_angles_bitpacked_qjl_signs_bitpacked"`.
  Source: `src/packed.rs`, `src/wire.rs`.
- **KV-cache shadow mode** — `KvRuntimeConfig`, `KvShadowToken`.
  Lets you score a compressed KV cache against a raw baseline and
  emit a `KvShadowReceipt`. Experimental, not for production.
  Source: `src/kv.rs`.
- **Codec profiles** — typed `CodecProfileV1` that captures the
  codec kind, dim, bits, projections, rotation, and a
  `profile_digest` (FNV-1a 64-bit) for receipt comparison.
  Source: `src/profile.rs`.
- **Codec benchmark example** — `examples/bench_embeddings.rs`
  evaluates a generated corpus and emits a `BenchmarkReceiptV1`.
  The historical external semantic-memory harness discussed below
  is not the same executable.

## Quick Start

`TurboSidecarIndex` owns compressed candidate generation. Keep the raw
vectors separately and rerank the returned candidates in the caller.
Candidate retrieval remains approximate, so reranking cannot recover a
raw top result that was absent from the candidate set.

```rust
use turbo_quant::{SearchOptions, TurboQuantizer, TurboSidecarIndex};

fn main() -> turbo_quant::Result<()> {
    let dim = 32;
    let corpus = vec![vec![0.1_f32; dim], vec![0.2_f32; dim]];
    let query = vec![0.15_f32; dim];
    let quantizer = TurboQuantizer::new(dim, 8, 16, 42)?;
    let mut index = TurboSidecarIndex::new(quantizer);

    for (id, vector) in corpus.iter().enumerate() {
        index.add(id, vector, None)?;
    }

    let (candidates, receipt) = index.search(
        &query,
        SearchOptions { top_k: 1, oversample: 4 },
    )?;

    // Rerank candidates against caller-owned raw vectors.
    let mut reranked: Vec<(usize, f32)> = candidates
        .iter()
        .map(|candidate| {
            let score = corpus[candidate.id]
                .iter()
                .zip(&query)
                .map(|(x, y)| x * y)
                .sum();
            (candidate.id, score)
        })
        .collect();
    reranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    reranked.truncate(1);
    println!("reranked={reranked:?}, receipt={receipt:?}");
    Ok(())
}
```

The crate exports `CodecProfileV1`, not a `CodecProfile` builder.
`TurboSidecarIndex::new` takes a quantizer; `add` encodes each vector;
`search` returns candidates and a `SearchReceiptV1` with
`approximate_only = true` and `exact_rerank_required = true`.
There is no built-in `TurboSidecarCode` type or `exact_rerank` method.

The checked-in examples include `bench_embeddings.rs`,
`kv_shadow.rs`, `profile_receipt.rs`, and `compat_0_1_smoke.rs`.
From the workspace root, run the compatibility example with:

```sh
cargo run --release -p turbo-quant --example compat_0_1_smoke
```

## Benchmarks — measured

These are real numbers from the P26 release-evidence
(`docs/release-evidence/0.2.0/semantic_memory_harness_receipt.json`).
They are run on a synthetic deterministic corpus
(`synthetic-deterministic-unit-vectors-v1`, 128 vectors × 32 dim,
12 queries, k=10, oversample=4, TurboQuant 8-bit, 16 QJL
projections, fast-Hadamard rotation, seed=42):

| Metric | Value | Notes |
|---|---|---|
| **Recall@1 (after exact rerank)** | 0.917 | 11/12 queries had the raw top-1 in the reranked top-1 |
| **Recall@5 (after exact rerank)** | 0.983 | 11.8/12 queries |
| **Recall@10 (after exact rerank)** | 0.992 | 11.9/12 queries |
| **Exact-rerank recovery rate** | 0.917 | top-1 in compressed candidates → top-1 in raw |
| **Top-k overlap (compressed top-10 vs raw top-10)** | 0.567 | Exploratory metric — see note |
| **Rank drift (mean)** | 0.083 | Position changes in top-10 |
| **Rank drift (p95)** | 1.0 | |
| **Rank drift (max)** | 1 | |
| **Score error (mean)** | 0.479 | Absolute score difference, compressed vs raw |
| **Score error (p95)** | 1.085 | |
| **Score error (max)** | 1.203 | |

**Storage (128 vectors × 32 dim):**

| Layout | Bytes | Ratio to raw |
|---|---|---|
| Raw f32 (reference) | 16,384 | 1.0× |
| f16 baseline | 8,192 | 0.5× |
| TurboQuant sidecar (no fallback) | 10,240 | 0.625× |
| TurboQuant sidecar + exact fallback | 26,624 | 1.625× |

The sidecar is **0.625× the raw size** when the raw fallback is
*not* kept. With the fallback, total storage is 1.625× raw — this
is the production configuration where you keep the compressed
sidecar *and* the raw vector for the rerank step. The
"exact-rerank recovery rate" is the gate: at 0.917 (11/12 queries
exact top-1) the harness passed.

**P31 retrieval-benchmark numbers (semantic-memory, May 2026):**

Run on a different harness with 1,000 vectors × 384 dim, 50
queries, k=10, candidate multiplier 20:

| Metric | Value |
|---|---|
| Candidate scoring p50 | 138.0 ms |
| Candidate scoring p95 | 148.1 ms |
| Exact rerank p95 | 0.087 ms |
| Mean abs score error | 0.0024 |
| P95 abs score error | 0.0061 |
| NDCG@10 | 1.0 |
| Mean rank drift | 0.0 |
| Recall@10 | 1.0 |

**P32 retrieval-benchmark numbers (semantic-memory, May 2026, smoke):**

Same harness, 1,000 × 384, 50 queries, with the optimized
candidate-then-exact flow:

| Metric | Value |
|---|---|
| Candidate scoring p50 | 109.1 ms |
| Candidate scoring p95 | 111.3 ms |
| Exact rerank p95 | 0.046 ms |
| Mean abs score error | 0.0024 |
| P95 abs score error | 0.0061 |
| NDCG@10 | 1.0 |
| Recall@10 | 1.0 |
| Fallback rate | 0.0 |

Both P31 and P32 receipts classify as `green`. The full receipts
are at
`semantic-memory/docs/codex-runs/archive/.../turboquant-*-benchmark-summary.json`.

From the workspace root, `cargo run --release -p turbo-quant --example bench_embeddings`
runs the current synthetic codec benchmark. It does not reproduce the
external P31/P32 semantic-memory harness reports above.

## Scope and limits

This crate is **experimental**. The following claims are explicitly
**forbidden** in documentation, rustdoc, README, and release notes
unless scoped to a specific external paper claim or local receipt
evidence:

- "zero accuracy loss"
- "zero overhead"
- "production KV cache runtime"
- "drop-in replacement"
- "better than semantic-memory"
- "proven deployment quality"
- "no dataset-specific calibration needed"

What's allowed:

- "experimental codec substrate"
- "derived sidecar (not canonical vectors)"
- "approximate scoring; exact fallback or rerank required"
- "workload-specific benchmark receipts required"
- "semantic-memory reference harness validates retrieval drift locally"

The historical P26 release-claim law referenced `turbo-quant/AGENTS.md`.
That file is not present in this repository snapshot; the scope limits above
remain explicit documentation boundaries, not evidence of a current release gate.

## What's verified

- `cargo test --all-targets --all-features --locked` passes
  (29 tests, 11.3s in CI).
- `cargo check --all-targets --all-features --locked` clean.
- `cargo fmt --all -- --check` clean.
- `python3 scripts/assert_p26_invariants.py .` passes — all
  required P26 release artifacts are present and content-addressed.
- `cargo package` succeeds.
- The `SemanticMemoryHarnessSummaryV1` is emitted and SHA-256
  recorded in `docs/release-evidence/0.2.0/release_receipt.json`.

## Test coverage

- 18 integration test files in `tests/`:
  - `api_compat.rs`, `bitpack.rs`, `determinism.rs`,
    `encoded_size.rs`, `inner_product.rs`, `invalid_inputs.rs`,
    `kv_policy.rs`, `malformed_artifacts.rs`, `packed_index.rs`,
    `profile_receipt.rs`, `query_workspace.rs`, `readiness.rs`,
    `rotation_policy.rs`, `serialization.rs`, `wire_format.rs`,
    `workspace.rs` — plus 2 more.
- 4 examples: `bench_embeddings`, `compat_0_1_smoke`,
  `kv_shadow`, `profile_receipt`.
- 1 criterion bench: `benches/turbo_quant_search.rs`.

## MSRV

Rust 1.75 (2021 edition). Stable features only.

## Dependencies

- `serde`, `nalgebra` (with `serde-serialize`).
- `bitvec` (transitive, for `BitPack`).
- Workspace `Cargo.toml` pin.

Zero platform-specific code, zero FFI, zero unsafe (`unsafe_code`
is denied at the workspace level).

## License

MIT OR Apache-2.0 (dual-licensed). See `LICENSE-MIT` and
`LICENSE-APACHE` for the full texts.

## Changelog

See `CHANGELOG.md` for the release history. The v0.2.0 release
notes are in `RELEASE_NOTES.md` and the receipts in
`docs/release-evidence/0.2.0/`.

## Where it's used

`turbo-quant` is the experimental vector compression sidecar for:

- [`semantic-memory`](https://github.com/RecursiveIntell/semantic-memory) — every projection
  import with `AdmissibilityClass::Standard` or below can route
  through the sidecar (gated by `quant-governor`).
- [`scr-runtime-compression`](https://github.com/RecursiveIntell/Libraries/tree/main/scr-runtime-compression) —
  the cross-runtime compression scheduler can use
  `TurboSidecarCode` for batched candidate generation.
- The KV-cache shadow mode is used by the
  `tools/semantic_memory_harness/` to validate the sidecar
  against a raw-vector reference.

Adopting `turbo-quant` directly is appropriate for systems that
need a vector compression sidecar with a documented, receipted
benchmark harness, and that can run their own workload-specific
benchmarks to confirm the sidecar is appropriate.
