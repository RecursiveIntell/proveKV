# FibQuant Publish Readiness Report

Date: 2026-05-16

## Summary

`fib-quant` has been hardened for a cautious `0.1.0-alpha.1` experimental research release. The crate is positioned as a paper-faithful core math implementation, not as a production KV-cache compressor and not as a locally performance-validated reproduction of the FibQuant paper.

## Changed Files

- `Cargo.toml`
- `LICENSE`
- `CITATION.cff`
- `CHANGELOG.md`
- `RELEASE_CHECKLIST.md`
- `README.md`
- `docs/compression/FIBQUANT_SOURCE_BASIS.md`
- `docs/compression/FIBQUANT_PUBLISH_HARDENING_AUDIT.md`
- `docs/compression/FIBQUANT_MATH_CONFORMANCE.md`
- `docs/compression/FIBQUANT_BENCHMARK_PLAN.md`
- `docs/compression/FIBQUANT_PUBLICATION_NONCLAIMS.md`
- `docs/compression/FIBQUANT_PUBLISH_READINESS_REPORT.md`
- `examples/build_codebook.rs`
- `examples/encode_decode.rs`
- `scripts/publish_final_assert.py`
- `src/bitpack.rs`
- `src/codebook.rs`
- `src/codec.rs`
- `src/directions.rs`
- `src/lloyd.rs`
- `src/profile.rs`
- `src/receipt.rs`
- `src/rotation.rs`
- `src/spherical_beta.rs`
- `tests/corruption_rejection.rs`
- `tests/encode_decode_roundtrip.rs`
- `tests/profile_digest.rs`

## Validation Results

- `python3 scripts/publish_preflight.py`: PASS after the public README rewrite. It may still warn locally that `z.py` exists in the crate root; package include rules and `.gitignore` prevent shipping it.
- `cargo fmt --all --check`: PASS.
- `cargo test`: PASS, 22 integration tests plus doc-test harness.
- `cargo clippy --all-targets --all-features -- -D warnings`: PASS.
- `cargo test --examples`: PASS.
- `cargo doc --no-deps`: PASS.
- `cargo package --list`: FAIL in this untracked dirty overlay. Exact stderr: `error: 1 files in the working directory contain changes that were not yet committed into git: Cargo.toml ... to proceed despite this and include the uncommitted changes, pass the --allow-dirty flag`.
- `cargo package --list --allow-dirty`: PASS; excludes `z.py`, `docs/codex-runs/**`, root archives, zip files, and generated sidecars.
- `cargo package --allow-dirty`: PASS.
- `cargo publish --dry-run --allow-dirty`: PASS; Cargo packaged, verified, reached upload, and aborted only because this was a dry run.
- `python3 scripts/publish_final_assert.py`: PASS.

## Package and Repository Surface

The package surface is allowlisted in `Cargo.toml`: source, tests, benches, examples, release docs, and selected `docs/compression/**` and `docs/kv/**` documentation. It does not contain `z.py`, `docs/codex-runs/**`, root `.zip` archives, `.manifest.json`, `.findings.json`, `.excluded.json`, or `.codex-archive.json` sidecars.

The public GitHub surface should include release source, tests, examples, scripts, docs, `.github/workflows/**`, `CONTRIBUTING.md`, and `SECURITY.md`. Local Codex/operator bundle files are ignored by `.gitignore`.

## Fixed Blockers

- Removed workspace dependency and lint inheritance from `Cargo.toml`.
- Added standalone Cargo metadata and an allowlisted package surface.
- Added release, citation, conformance, benchmark-plan, non-claims, and readiness docs.
- Fixed `beta_d_k()` to avoid unsigned underflow and reject non-positive/non-finite beta shapes.
- Enforced profile schema, rate, method, norm, source, Lloyd, and training sample invariants.
- Made `radius_method` and `direction_method` authoritative for codebook initialization dispatch.
- Added source vector digest and schema/norm metadata to compression receipts.
- Changed encoded digest generation to return `Result<String>` and fail closed.
- Added decode rejection for invalid `FibCodeV1.schema_version`.
- Added regression coverage for underflow, tampering, schema rejection, and receipt source digest changes.

## Remaining Deviations

- No fused attention-kernel decompression.
- No production KV-cache integration.
- No integration with `semantic-memory`.
- No mutation or replacement of `turbo-quant`.
- No local reproduction of the paper's GPT-2, TinyLlama, throughput, memory, or perplexity benchmark results.
- Current public profile validation accepts the paper-path source mode and fp16 norm side header only.

## Publish Recommendation

Use `0.1.0-alpha.1` for any first public release. Do not perform an actual `cargo publish` from this dirty/untracked overlay; publish only from a clean VCS checkout or after intentionally committing the crate state.
