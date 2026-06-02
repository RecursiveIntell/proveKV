# KV Final Auditor Handoff

## Changed Files

- `Cargo.toml`
- `src/lib.rs`
- `src/kv/*`
- `tests/kv_*`
- `benches/kv_encode_decode.rs`
- `benches/kv_attention_ref.rs`
- `docs/kv/*`
- `docs/codex-runs/fibquant-final-hardening/archive-inputs/01_CODEX_MASTER_PROMPT.md`
- `docs/codex-runs/fibquant-final-hardening/archive-inputs/09_PUBLISH_POSITIONING.md`

## Commands Run

```text
python3 scripts/kv_static_preflight.py
python3 scripts/check_no_kv_product_claims.py
cargo fmt --all --check
cargo check --all-features
cargo test --all-features
cargo test --examples --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo doc --no-deps --all-features
cargo bench --no-run --all-features
cargo package --list --allow-dirty
cargo package --allow-dirty
cargo publish --dry-run --allow-dirty
```

## Tests Passed

- Existing math-core tests passed under `cargo test --all-features`.
- New KV tests passed:
  - `kv_shape_contracts`
  - `kv_policy_role_aware`
  - `kv_encode_decode_reference`
  - `kv_attention_quality`
  - `kv_corruption_rejection`
  - `kv_property_shapes`

## Tests Failed

None remaining.

## Tests Skipped And Why

- No model-level quality tests were run because no model integration is in scope for this run.
- No GPU tests were run because no GPU kernels are implemented.
- No external backend adapter tests were run because no adapter is implemented.

## Current Maturity Stage

M2 synthetic/reference stage.

## Exact Production Claims Allowed

- Experimental default-off KV contracts and CPU reference encode/decode exist.
- Receipts are emitted for material encode/decode operations in the CPU reference path.
- Synthetic quality and calibration helpers exist.

## Forbidden Claims Still Forbidden

- Ready for production serving.
- Model-level quality validated.
- Paper benchmark reproduction.
- vLLM, FlashInfer, TensorRT-LLM, HQQ, Quanto, KIVI, or KVQuant replacement.

## Known Deviations From Production KV-Cache Readiness

- No model-captured calibration.
- No backend adapter.
- No fused kernel.
- No serving benchmark.
- Per-channel policy falls back to raw in the CPU reference codec.

## Benchmark Receipt Paths

```text
target/kv-production-receipts/final/cargo-bench-no-run-all-features.txt
```

Bench targets compile, but no benchmark timing run was used for a performance claim.

## Package/Publish Receipt Paths

```text
target/kv-production-receipts/final/cargo-package-list-allow-dirty.txt
target/kv-production-receipts/final/cargo-package-allow-dirty.txt
target/kv-production-receipts/final/cargo-publish-dry-run-allow-dirty.txt
```

## Next Run Recommendation

Implement a fixture-captured calibration harness and a true per-channel CPU reference codec for key tensors before attempting any backend adapter work.
