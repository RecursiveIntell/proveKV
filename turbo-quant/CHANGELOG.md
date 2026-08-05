# Changelog

## Unreleased

- Added sparse-prefill planning and comparison receipts for A-shape,
  vertical-slash, and block-sparse KV attention probes.
- Added `sparse_prefill_probe` example to compare exact retained KV shadows and
  compressed-key attention scores before fused-kernel work.
- Added a hybrid anchor/recent/block sparse-prefill pattern and multi-trace
  benchmark receipt with kernel-readiness gates.
- Added a HuggingFace attention trace dumper in the proveKV scripts tree for
  generating real `SparsePrefillTraceV1` inputs without modifying locked PPL
  validation receipts.
- Added an adaptive softmax-mass sparse-prefill pattern and saved SmolLM2
  real-trace receipts showing it improves fixed policies but still does not pass
  the strict kernel gate.

## 0.2.0

- Preserved the `0.1.0` public struct literal shapes for legacy code.
- Added packed sidecar payload types without replacing legacy logical structs.
- Added deterministic wire encoding and strict decode validation for TurboCode.
- Added codec profiles, compression receipts, benchmark receipts, and sidecar
  search receipts.
- Added explicit QJL source-norm provenance APIs and removed hidden process-global
  norm dependence from legacy QJL scoring.
- Added KV shadow-mode runtime configuration and exact-shadow comparison helpers.
- Added semantic-memory reference harness support for local retrieval drift
  validation with exact rerank.
- Reworked public docs around experimental sidecar semantics and caller-owned
  exact-vector authority.
