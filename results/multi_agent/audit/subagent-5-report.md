# proveKV claim evidence coverage audit

Source: `/home/sikmindz/proveKV/CLAIMS.json` (schema 1.0.0).

## Summary

- **Total claims audited: 12**
  - `claims`: 6
  - `wire_lossless_certs`: 2
  - `hybrid_state`: 4
- **Claim classes present:**
  - `PPL_validated`: 5
  - `size_only`: 1
  - `wire_lossless`: 2
  - `test_gate`: 4
- **Claims with a declared receipt/reference:** 10/12
- **Claims with at least one receipt path that exists on disk:** 9/12
  - The `p0b_rust_gate` receipt is the command string `cargo test -p provekv`, not a filesystem receipt, so it is declared but not a path.
  - The two `wire_lossless` certificates have no receipt field/path.

## Claim-by-claim coverage

| Claim ID | Class | Declared receipt(s) | Evidence status |
|---|---|---|---|
| `smollm2_wikitext2_n8_lossless_default` | `PPL_validated` | 2 JSON paths | Present; both paths exist |
| `smollm2_wikitext2_n8_lossy_default` | `PPL_validated` | 2 JSON paths | Present; both paths exist |
| `smollm2_wikitext2_n8_lossless_legacy_b8` | `PPL_validated` | 1 JSON path | Present; path exists |
| `smollm2_wikitext2_n8_lossy_legacy_b8` | `PPL_validated` | 1 JSON path | Present; path exists |
| `pool_only_smollm2` | `PPL_validated` | 1 JSON path | Present; path exists |
| `qwen0_5b_synthetic_n8` | `size_only` | 2 JSON paths | Present; both paths exist |
| `fib_fb2_batched` | `wire_lossless` | None | **Missing receipt**; only test name is recorded |
| `turbo_tqb1_batched` | `wire_lossless` | None | **Missing receipt**; only test name is recorded |
| `p0b_rust_gate` | `test_gate` | `cargo test -p provekv` | Command reference only; no durable receipt path |
| `p0c_capture_replay_gate` | `test_gate` | `/tmp/provekv-capture-test/20260805T041422Z/receipt.json` | Present; path exists |
| `p0c_msi_gate` | `test_gate` | `results/bench/hybrid_state/qwen25/capture/20260805T042634Z/receipt.json` | Present; path exists |
| `p0c_gpu_gate` | `test_gate` | `results/bench/hybrid_state/qwen25/capture-gpu/20260805T050409Z/receipt.json` | Present; path exists |

## Findings

The main evidence gap is the pair of wire-format claims (`fib_fb2_batched` and `turbo_tqb1_batched`): they identify tests but provide no receipt artifact. The Rust gate also lacks a durable receipt and records only the test command. All six entries in the main `claims` section have declared JSON receipts, and those paths currently exist.
