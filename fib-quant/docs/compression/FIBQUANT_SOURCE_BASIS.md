# FibQuant Source Basis

Date: 2026-05-16

This crate is being hardened as an experimental, paper-faithful Rust research implementation of:

Namyoon Lee and Yongjune Kim, "FibQuant: Universal Vector Quantization for Random-Access KV-Cache Compression", arXiv:2605.11478v1, May 12 2026.

## Release Posture

The supported public claim is limited to the core radial-angular vector quantization math and fixed-rate artifact path implemented in this crate. This crate is not a production KV-cache compressor, is not default-enabled in any surrounding system, and does not claim local reproduction of the paper's benchmark numbers.

## Local Source Inventory

- Package root: `/home/sikmindz/Coding/Libraries/fib-quant`.
- Parent Cargo workspace root reported by `cargo metadata`: `/home/sikmindz/Coding/Libraries`.
- Workspace default member reported by metadata: `fib-quant`.
- Rust sources: `src/*.rs`.
- Integration tests: `tests/*.rs`.
- Existing run-context docs: `docs/codex-runs/*`.
- Root archive and sidecar artifacts exist and must be excluded from the published crate.
- Root `z.py` exists and must be excluded from the published crate.

## Preflight Receipt

`python3 scripts/publish_preflight.py` was run before code changes. It reported 20 findings:

- `WORKSPACE_INHERITANCE` in `Cargo.toml`.
- Missing Cargo metadata: `readme`, `repository`, `documentation`.
- Unbounded package surface.
- Missing release files: `LICENSE`, `CITATION.cff`, `CHANGELOG.md`, `RELEASE_CHECKLIST.md`.
- Missing compression docs: source basis, math conformance, benchmark plan, publication non-claims.
- `z.py` present in the crate root.
- `BETA_DK_USIZE_UNDERFLOW` in `src/spherical_beta.rs`.
- Weak profile rate enforcement in `src/profile.rs`.
- Missing receipt source digest in `src/receipt.rs`.
- Fail-open encoded digest in `src/codec.rs`.
- `encoded_digest()` not returning `Result`.
- Thin README public-positioning language.

## Non-Goals

- No integration with `semantic-memory`.
- No mutation of `turbo-quant`.
- No default-on compression path.
- No FEUT/SCR variant work.
- No benchmark reproduction claim without local benchmark receipts.
