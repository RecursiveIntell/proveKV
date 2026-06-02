# KV Production Readiness Report

Recorded on 2026-05-16.

## Current Maturity

Current maturity: M2 synthetic/reference stage.

Implemented:

- non-default `kv` feature;
- KV shape, layout, page geometry, profile, block, page, and receipt contracts;
- role-aware policy decisions for raw, per-token, per-channel, KIVI-style, and experimental role-aware strategies;
- CPU reference per-token FibQuant encode/decode over canonical contiguous f32 tensors;
- fixed token pages with page digests;
- raw fallback for protected regions, unsupported axes, and per-vector encode fallback;
- synthetic attention-logit, softmax, top-k, and value aggregation quality metrics;
- fixture-driven calibration summaries;
- KV tests, property tests, and benchmark build targets;
- runtime/backend plan docs without GPU kernel implementation.

Not implemented:

- model-captured KV calibration;
- Hugging Face runtime adapter;
- vLLM or FlashInfer backend integration;
- fused GPU kernels;
- model-level quality evaluation;
- named-hardware serving benchmarks.

## Claims Allowed

Allowed:

- `fib-quant` now has an experimental, default-off KV feature with typed contracts and a CPU reference path.
- The CPU reference path can encode/decode canonical f32 KV tensors using per-token FibQuant where supported.
- Unsupported per-channel policy paths keep raw fallback blocks in the CPU reference codec.
- Synthetic attention quality and calibration helpers are available for local experiments.

Forbidden:

- any claim that this is ready for production serving;
- any claim that model-level quality has been validated;
- any claim that paper benchmark results were reproduced locally;
- any claim that this replaces vLLM, FlashInfer, TensorRT-LLM, HQQ, Quanto, KIVI, or KVQuant.

## Validation Receipts

Receipt directory:

```text
target/kv-production-receipts/final/
```

Passing receipts captured:

```text
kv-static-preflight.txt
no-kv-product-claims.txt
cargo-fmt-check.txt
cargo-check-all-features.txt
cargo-test-all-features.txt
cargo-test-examples-all-features.txt
cargo-clippy-all-targets-all-features.txt
cargo-doc-all-features.txt
cargo-bench-no-run-all-features.txt
cargo-package-list-allow-dirty.txt
cargo-package-allow-dirty.txt
cargo-publish-dry-run-allow-dirty.txt
```

`cargo publish --dry-run --allow-dirty` reached the dry-run upload abort after package verification.

## Source Shape Notes

Phase 0 found that `fib-quant` is inside a larger dirty parent Git worktree at `/home/sikmindz/Coding/Libraries`. Work in this run was scoped to `fib-quant`. This workspace shape should be cleaned or isolated before any actual release action.

## Release Decision

Do not make a production serving claim. The code is suitable for continued experimental KV-contract and CPU-reference work under the default-off `kv` feature.
