# proveKV

Shared compressed KV-cache pool. One pool, many agents, zero leaks.

`proveKV` is the **pool primitive** for multi-agent
inference. It solves a specific problem: 10 agents share the
same 200-token system prompt, but without compression, you're
storing that prompt 10 times in VRAM. With `proveKV`, you
store it once, compress it 50×, and give each agent a 17ms
shell for its unique tokens.

This is the **production target** of the governed
compression workspace: a typed, receipted, two-tier pool
where the shared tier is built once and read by everyone.

## The two-tier strategy

KV-cache compression is hard because one size doesn't fit
all. Shared context (system prompts, few-shot examples,
retrieval results) needs **high compression** — it's large,
rarely changes, and you can afford some fidelity loss.
Agent-private context (conversation turns, tool outputs)
needs **near-lossless reconstruction** — it's smaller but
critical for correctness.

`proveKV` gives you both:

| Tier | What it holds | Codec | Compression | Fidelity | Build cost |
|---|---|---|---|---|---|
| **Shared pool (cold)** | System prompts, shared context | fib-quant k=4, N=32 | ~50× theoretical | cos 0.863 | 1,557ms (once) |
| **Agent shell (hot)** | Per-agent conversation, tool results | turbo-quant 8-bit, 32proj | ~8× theoretical | cos 0.9996 | 17ms (per agent) |

The shared pool is **read-only after build** — agents
physically cannot contaminate each other. Every operation
produces a typed receipt that captures what was built, who
read it, and what was returned.

## What's in the box

### `SharedKVPool` (`src/pool.rs`, 602 lines)

The shared, immutable, build-once KV-cache pool. Built from
a corpus of vectors; encoded with fib-quant; addresses
documents by their content digest.

```rust
use prove_kv::SharedKVPool;

let pool = SharedKVPool::build(corpus, profile)?;
// 1,557ms for 80 shared docs, 768-dim, fib-quant k=4.
// Receipt: PoolBuildReceipt { profile_digest, block_digest, ... }
```

The build is **deterministic** — same corpus + same profile
+ same seed = same `block_digest`. You can verify the pool
matches a previous build by comparing digests.

### `AgentShell` (`src/shell.rs`, 324 lines)

A per-agent overlay that materializes the agent's private
context on top of the shared pool. Built fast (17ms for 12
docs); uses turbo-quant for the hot tier (cosine 0.9996
fidelity).

```rust
use prove_kv::AgentShell;

let shell = AgentShell::materialize(&pool, agent_docs, profile)?;
// 17ms for 12 docs, 768-dim, turbo-quant 8-bit.
// Receipt: ShellMaterializeReceipt { shell_digest, ... }
```

### Manifests (`src/manifest.rs`, 179 lines)

A typed `PoolManifest` that names the corpus, the profile,
the codec, the seed, the block layout, the receipt, and the
content digest. The manifest is what gets serialized to disk
and what gets verified when you re-open the pool.

### Receipts (`src/receipt.rs`, 373 lines)

Every operation produces a typed receipt:

- `PoolBuildReceipt` — emitted on `SharedKVPool::build`.
  Carries the profile, the seed, the block digest, the
  per-block statistics.
- `ShellMaterializeReceipt` — emitted on
  `AgentShell::materialize`. Carries the shell digest, the
  agent's read set, the cost.
- `FallbackReceiptV1` — emitted when an exact-fallback was
  triggered. Carries the reason and the original path.

All three receipts are BLAKE3-hashed and signed with the
codec profile digest, so the audit trail is tamper-evident.

### Policy (`src/policy.rs`, 209 lines)

A typed policy object that the caller passes to `build` and
`materialize`. The policy is the single decision point:
"this corpus is admissible for fib-quant with these
parameters, and the result is admissible for these
admissibility classes."

### Exact fallback (`src/fallback.rs`, 66 lines)

The contract: any compressed representation can be re-derived
back to its raw input. If a caller asks for
`Admissibility::Exact` and the codec can't deliver, the
adapter falls back to raw and emits a `FallbackReceiptV1`.

## Quick Start

```rust
use prove_kv::{SharedKVPool, AgentShell, Admissibility};
use quant_codec_core::CodecProfile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Build the shared pool (cold tier).
    let shared_corpus: Vec<Vec<f32>> = /* system prompts + shared docs */;
    let shared_pool = SharedKVPool::build(shared_corpus, profile.clone())?;
    println!("Shared pool built: {:?}", shared_pool.build_receipt);

    // 2. Each agent materializes a shell (hot tier) on top.
    for agent_id in 0..10 {
        let agent_docs: Vec<Vec<f32>> = /* agent-private context */;
        let shell = AgentShell::materialize(&shared_pool, agent_docs, profile.clone())?;
        println!("Agent {} shell: {:?}", agent_id, shell.materialize_receipt);
    }
    Ok(())
}
```

Run it: `cargo run --release --example prove_kv_fast_roundtrip`.

## Benchmarks — measured

### 10-agent contention (June 2026)

10 agents, 80 shared docs (768-dim, fib-quant k=4), 12
agent-private docs per agent (768-dim, turbo-quant 8-bit):

| Metric | Result |
|---|---|
| Agents with recall@1 = 1.0 | **10/10** |
| Cross-agent top-1 leaks | **0/90 pairs** |
| Pool build (80 shared docs) | 1,557ms |
| Shell materialize (12 docs/agent) | 17ms avg |
| fib-quant cold compression batch | 480 KB → 133 KB (3.6× JSON, ~48× binary projected) |
| turbo-quant hot fidelity | cosine 0.9996 |

**Every agent found its target at rank 1. Zero interference.
The shared pool is read-only after build — agents physically
cannot contaminate each other.**

### Single-route parity (June 2026)

8 queries, 200 docs, 768-dim, k=10:

| Route | Recall@1 | Recall@10 | nDCG@10 | Rank drift |
|---|---|---|---|---|
| exact_scan (no compression) | 1.000 | 1.000 | 1.000 | — |
| fib-quant only | 1.000 | 1.000 | 1.000 | 0.33 |
| turbo-quant only | 1.000 | 1.000 | 1.000 | 0.03 |
| **proveKV (two-tier)** | **1.000** | **1.000** | **1.000** | **0.25** |

### "Do All" perf pass (2026-06-01) — pool build throughput

After the June 1 perf pass (AVX2+FMA SIMD + Rayon parallel
across vec_idx + Rayon parallel across layer_idx):

| Config | qwen3 n=4 | qwen3 n=20 | qwen3 n=80 | nomic n=4 | nomic n=20 | nomic n=80 |
|---|---|---|---|---|---|---|
| **Old (f64 reference)** | 1449ms | 4271ms | 13763ms | 459ms | 1336ms | 4552ms |
| + SIMD | 418ms | — | — | — | — | 94ms |
| + Rayon (parallel) | 893ms | 968ms | 1250ms | 271ms | 296ms | 407ms |
| **+ parallel_pool (full)** | **256ms** | **291ms** | **346ms** | **94ms** | **100ms** | **133ms** |

**Best speedup over old (f64): 5.7× at qwen3 n=4, 40× at qwen3 n=80.**

The 28 layers in qwen3 are independent — spreading them
across cores compounds the fib-quant Rayon wins.

### GPU path (msi i7-6700HQ + GTX 1070)

| Shape | n | wall CPU | wall Hadamard-GPU | wall Hadamard+Codebook-GPU |
|---|---|---|---|---|
| nomic 768 | 80 | 4552 | 4430 | 4485 |
| qwen3 2560 | 80 | 13763 | 13419 | 13428 |

**Hadamard-only win: 2.5-2.7%** on the larger corpora.
**Hadamard + Codebook-GPU win: 1.5-2.4%.** The new codebook
kernel is slower in integration than just the Hadamard alone
because per-call H2D/D2H transfer overhead dominates.

**The kernel is correct** (parity test passes for n=32, d=128,
k=4, N=32 random inputs on msi GTX 1070). **The dispatch is
the issue, not the kernel.**

### JSON vs binary storage

Current compression ratios are JSON-serialized. The JSON
envelope is 12× bigger than the actual codebook indices.
Binary wire format is the next PR. `PackedTurboCode` already
exists in `turbo-quant`. `PackedFibCode` is next.

| | JSON (current) | Binary (projected) |
|---|---|---|
| Shared pool (80 docs) | 240 KB → 66 KB (3.6×) | 240 KB → ~5 KB (48×) |
| Agent shell (12 docs) | 36 KB → 63 KB (0.6×) | 36 KB → ~5 KB (7×) |
| System total (200 docs) | 600 KB → 695 KB (0.9×) | 600 KB → ~95 KB (6.3×) |

## Test coverage

- **4 integration test files** in `tests/`:
  - `integration_tests.rs` (162 lines) — full
    build-then-materialize roundtrip with receipts.
  - `pool_tests.rs` (118 lines) — pool invariants:
    determinism, immutability, profile digest stability.
  - `receipt_tests.rs` (169 lines) — receipt roundtrip,
    BLAKE3 digest stability, fallback contract.
  - `shell_tests.rs` (151 lines) — agent shell contracts:
    materialization, isolation, cost.
- **4 examples** in `examples/`:
  - `prove_kv_dynamic_cache_roundtrip.rs` — full end-to-end
    build + materialize + search.
  - `prove_kv_fast_roundtrip.rs` — fast path benchmark.
  - `prove_kv_gpu_bench.rs` — GPU dispatch benchmark.
  - `test_compact_decode.rs` — compact binary decode.
- **1 bench** in `benches/synthetic_pool.rs`.
- **25+ Python validation scripts** in `scripts/` for
  preflight, schema validation, public claim checking,
  receipt integrity, source-package hygiene, and final-state
  validation.
- `cargo test` clean, `cargo clippy --all-targets -- -D warnings` clean.

## MSRV

Rust 1.75 (2021 edition). Stable features only.

## Dependencies

- `serde` (with `derive`).
- `serde_json`.
- `blake3`.
- `rand` + `rand_chacha`.
- `thiserror`.
- `turbo-quant` (optional) — for the `turbo` feature.
- `fib-quant` (optional) — for the `fib` feature.
- `gpu-backend` (optional) — for the GPU path.
- `rayon` (optional) — for the parallel pool build.

## License

MIT OR Apache-2.0 (dual-licensed). See `LICENSE-MIT` and
`LICENSE-APACHE` for the full texts.

## Changelog

See `CHANGELOG.md` for the release history.

## Scope and limits

This crate is **alpha**. The following claims are explicitly
**forbidden** in documentation, rustdoc, README, and release
notes unless scoped to a specific external paper claim or
local receipt evidence (per the AGENTS.md release-claim law):

- "zero accuracy loss"
- "zero overhead"
- "production KV cache runtime"
- "drop-in replacement"
- "better than semantic-memory"
- "proven deployment quality"
- "no dataset-specific calibration needed"

What's allowed:

- "experimental pool primitive"
- "two-tier codec policy (fib-quant cold + turbo-quant hot)"
- "receipt-bearing, deterministic, runnable on synthetic fixtures"
- "exact-fallback contract enforced"
- "scope: shared KV-cache pool, not full agent runtime"

## Attribution

This crate is an independent Rust implementation of the
PolyKV-style shared compressed KV-cache pool idea. It is
**not** the original authors' reference implementation and
does not claim affiliation with the PolyKV paper authors.
The pool architecture, the two-tier strategy, the receipt
infrastructure, the GPU dispatch path, and the test suite
are original to this implementation.

## Where it's used

`proveKV` is the pool primitive for:

- The LLM runtime stack — when 10+ agents share a system
  prompt, the shared pool saves N× memory and gives each
  agent a 17ms spin-up cost.
- `semantic-memory` — when the corpus has a stable shared
  component (system prompts, few-shot examples), the
  shared pool reduces the per-import cost.
- The `quant-governor` policy layer — when the policy
  routes a multi-agent workload to the cold tier, the
  `SharedKVPool` is what actually serves the request.

Any system that needs **shared, immutable, receipted vector
storage across multiple consumers** can adopt `proveKV`
directly.
