# HF Transformers Adapter Plan

The `kv` feature currently exposes Rust contracts and a CPU reference codec. A future Hugging Face experiment should adapt those artifacts to a `QuantizedCache`-like flow without changing the default behavior of this crate.

## Adapter Steps

1. Capture synthetic and model-generated K/V tensors in canonical f32 form.
2. Convert model cache metadata into `KvTensorShapeV1`.
3. Build per-role `KvCompressionProfileV1` profiles.
4. Encode pages with `encode_kv_tensor`.
5. Decode with `decode_kv_pages` for CPU reference quality checks.
6. Compare against raw cache behavior with attention-logit and value aggregation metrics.
7. Save receipts under `target/kv-production-receipts/`.

## Required Receipts

- shape/profile/page digests;
- compression and decode receipts;
- attention quality report;
- model/task identifier;
- hardware and software versions;
- raw memory footprint and encoded artifact size.

## Non-Goals For This Run

- No Python package is built here.
- No Hugging Face runtime adapter is committed here.
- No model-level quality claim is made from synthetic fixtures.
