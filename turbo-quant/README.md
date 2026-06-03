# turbo-quant

Experimental vector compression sidecars for embedding search.

`turbo-quant` implements three compression sidecars (`PolarQuant`,
`TurboQuant`, and QJL sketches) and the surrounding infrastructure
needed to use them in a real retrieval system: bit-packed wire
formats, candidate generation, exact rerank, KV-cache shadow mode,
and a complete benchmark harness that validates quality against a
raw-vector reference.

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
- **Benchmark harness** — `tools/semantic_memory_harness/`
  validates the sidecar against `semantic_memory::search::cosine_similarity`
  as the raw-vector reference, and emits a
  `SemanticMemoryHarnessSummaryV1` receipt.

## Quick Start

```rust
use turbo_quant::{CodecProfile, TurboSidecarCode, TurboSidecarIndex};
use nalgebra::DVector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a codec profile.
    let profile = CodecProfile::turbo_quant_8bit(32)
        .with_projections(16)
        .with_seed(42);

    // Encode a corpus.
    let corpus: Vec<DVector<f32>> = /* your vectors */;
    let code = TurboSidecarCode::encode(&profile, &corpus)?;
    let index = TurboSidecarIndex::build(&profile, code)?;

    // Search — get candidates in compressed space, then rerank on raw.
    let query = DVector::from_vec(/* query vector */);
    let candidates = index.candidates(&profile, &query, 40, 10)?;  // oversample × top_k
    let reranked = index.exact_rerank(&candidates, &corpus, &query, 10)?;

    Ok(())
}
```

The `examples/` directory has runnable versions of this and three
other flows: `bench_embeddings.rs`, `kv_shadow.rs`,
`profile_receipt.rs`, and `compat_0_1_smoke.rs` (the P26
release-gate smoke test).

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

To reproduce: `cd turbo-quant && cargo run --release --example bench_embeddings`.

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

The full release-claim law is at
`turbo-quant/AGENTS.md` (P26 patch).

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

- [`semantic-memory`](../semantic-memory) — every projection
  import with `AdmissibilityClass::Standard` or below can route
  through the sidecar (gated by `quant-governor`).
- [`scr-runtime-compression`](../scr-runtime-compression) —
  the cross-runtime compression scheduler can use
  `TurboSidecarCode` for batched candidate generation.
- The KV-cache shadow mode is used by the
  `tools/semantic_memory_harness/` to validate the sidecar
  against a raw-vector reference.

Adopting `turbo-quant` directly is appropriate for systems that
need a vector compression sidecar with a documented, receipted
benchmark harness, and that can run their own workload-specific
benchmarks to confirm the sidecar is appropriate.
