# KV Source Inventory

Recorded on 2026-05-16 before KV implementation work.

## Workspace Shape

- Intended crate root: `/home/sikmindz/Coding/Libraries/fib-quant`.
- Git toplevel reported by `git rev-parse --show-toplevel`: `/home/sikmindz/Coding/Libraries`.
- `fib-quant/Cargo.toml` is not tracked by the parent Git index at Phase 0 inventory time.
- Parent worktree is dirty with many unrelated changes outside `fib-quant`; KV work must remain scoped to `fib-quant`.
- Cargo build artifacts are written to the parent shared target directory: `/home/sikmindz/Coding/Libraries/target`.

This differs from a standalone tracked crate assumption. The source basis remains usable for local crate work, but release/publish claims must account for the parent worktree shape and dirty state.

## Current Crate Surface

- Package: `fib-quant`
- Version: `0.1.0-alpha.1`
- Rust edition: `2021`
- Rust version: `1.75`
- Current package description: experimental FibQuant radial-angular vector quantization core.
- Current public posture in `src/lib.rs`: explicitly not a production KV-cache compressor, not a benchmark reproduction package, and not integrated with a parent workspace memory crate.
- Existing `Cargo.toml` has no feature declarations yet.
- Existing `Cargo.toml` package include list contains `docs/compression/**` but not `docs/kv/**`.
- Existing dependencies include `serde`, `serde_json`, `blake3`, `half`, `nalgebra`, `rand`, `rand_chacha`, `rand_distr`, `statrs`, and `thiserror`.
- Existing dev dependencies include `criterion` and `proptest`.

## Required Phase 0 Files Inspected

- `Cargo.toml`
- `src/profile.rs`
- `src/codebook.rs`
- `src/codec.rs`
- `src/rotation.rs`
- `src/receipt.rs`
- `src/bitpack.rs`
- `src/lib.rs`
- `tests/*`
- `benches/*`
- `docs/compression/*`
- `examples/*`

## Existing Math Core Summary

- `FibQuantProfileV1` validates paper-profile parameters, resource bounds, rate fields, method choices, norm format, source mode, and rotation algorithm identity.
- `FibCodebookV1` builds and validates deterministic codebooks, profile digests, rotation digests, codebook digests, shape lengths, and finite codewords.
- `FibQuantizer` provides vector-level encode/decode and `encode_with_receipt`.
- `FibCodeV1` is a fixed-rate vector artifact containing profile/codebook/rotation digests, norm payload, index width, block count, and packed indices.
- `StoredRotation` provides deterministic QR/Gaussian rotation with digest identity.
- `FibQuantCompressionReceiptV1` records vector compression metadata and optional reconstruction metrics.
- `bitpack` packs and unpacks fixed-width indices and rejects nonzero padding bits.

No source file inspected contained an existing `src/kv` module or KV-cache contract types at Phase 0.

## Existing Tests, Benches, Docs

Current tests cover:

- bitpack packing and padding rejection;
- codebook determinism and tamper rejection;
- codec encode/decode roundtrip and corruption rejection;
- direction generators;
- Lloyd refinement;
- norm payload rejection;
- paper smoke regressions;
- profile digest and resource bounds;
- rotation identity;
- spherical-Beta sampler behavior;
- property tests for bitpack and codec.

Current benches:

- `benches/encode_decode.rs`
- `benches/codebook_build.rs`

Current committed docs are under `docs/compression/` and describe math conformance, nonclaims, publication readiness, benchmark plans, and release decisions for the vector math crate.

## Package List

Phase 0 ran:

```text
cargo package --list --allow-dirty
```

Receipt path:

```text
target/kv-production-receipts/phase0/cargo-package-list.txt
```

Observed package list includes source files, tests, benches, examples, `docs/compression/**`, `Cargo.lock`, `.cargo_vcs_info.json`, and `Cargo.toml.orig`. It does not include `z.py`, root workbench prompt files, `.codex`, or future `docs/kv` files.

## Phase 0 Gate Receipts

All commands below completed with exit code 0:

```text
python3 scripts/kv_static_preflight.py
cargo package --list --allow-dirty
cargo fmt --all --check
cargo check --all-features
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --no-deps --all-features
```

Receipt files:

```text
target/kv-production-receipts/phase0/kv_static_preflight.txt
target/kv-production-receipts/phase0/cargo-package-list.txt
target/kv-production-receipts/phase0/cargo-fmt-check.txt
target/kv-production-receipts/phase0/cargo-check-all-features.txt
target/kv-production-receipts/phase0/cargo-test-all-features.txt
target/kv-production-receipts/phase0/cargo-clippy-all-targets.txt
target/kv-production-receipts/phase0/cargo-doc-all-features.txt
```

Static preflight warnings at Phase 0:

- `src/kv` does not exist yet.
- root workbench artifacts remain: `01_CODEX_MASTER_PROMPT.md`, `OPERATOR_PASTE_FIRST.md`, `overlays`, `.agents`, `.codex`.
- `z.py` exists in the repo context; package list currently excludes it.

## Assumptions For Implementation

- Preserve the existing vector math core unless KV correctness requires a narrowly scoped addition.
- Add KV functionality behind a non-default `kv` feature.
- Keep KV compression default-off.
- Keep raw/uncompressed fallback available at page/block granularity.
- Keep `semantic-memory` out of scope for this run.
- Do not claim production readiness or paper benchmark reproduction without local receipts.
