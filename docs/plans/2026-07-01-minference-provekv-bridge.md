# MInference/proveKV Bridge Plan

Date: 2026-07-01

## Decision

The sparse-attention/KV-cache track is relevant to this repository. The highest
ROI next step is not a CUDA kernel first; it is a receipt-backed bridge between
MInference-style sparse prefill patterns and proveKV/turbo-quant KV shadow
experiments.

## Claim Boundary

This bridge may claim:

- deterministic sparse-prefill token/block plans for score-vector probes
- top-k retention and softmax-mass coverage against full attention logits
- compatibility with existing KV shadow-mode experiments
- score-read savings estimates before kernel work

This bridge must not claim:

- fused-kernel speedup
- decode wall-clock speedup
- deployed attention quality
- PPL neutrality
- framework KV-cache byte reduction

Those claims require separate runtime, kernel, and PPL receipts.

## Build Order

1. Add additive V1 sparse-prefill plan and comparison receipt types.
2. Support A-shape, vertical-slash, and block-sparse pattern probes.
3. Compare sparse plans against exact retained KV shadow logits.
4. Compare sparse plans against compressed-key attention scoring.
5. Use the receipt metrics to decide which pattern deserves CUDA work.
6. Build a fused sparse/decode kernel only after top-k recall and softmax-mass
   coverage survive realistic traces.

## Current Highest-ROI Additions

The first follow-up implementation adds:

- `HybridAnchorRecentBlocks`, because pure block-sparse can miss the recent
  softmax mass that dominates the attention output.
- softmax-mass block scoring, because absolute logit energy is not the right
  proxy for sparse attention quality.
- a multi-trace benchmark receipt with explicit kernel-readiness gates.
- a small sparse-budget sweep in the trace benchmark example, so near-miss
  configs are visible without relaxing the gate.
- an adaptive mass selector that keeps full top-k and adds highest-mass tokens
  until either the target mass or read-savings cap is reached.

## Current Real-Trace Finding

The saved SmolLM2 receipt at
`results/sparse_prefill/smollm2-1.7b/bench_adaptive_n128_l4_h4_q4.json`
does not justify CUDA work yet. `AdaptiveMass` improves the fixed policies, but
under the strict gate it still fails worst-case mass coverage:

- best strict 50% score-read-savings pass rate: `0.65625`
- best strict 50% score-read-savings minimum softmax mass: `0.6225`
- diagnostic 25% score-read-savings minimum softmax mass: `0.8548`

The next useful work is not a fused kernel. It is trace stratification and a
different sparse primitive for the low-mass outliers, or a measured decision to
lower the mass gate for specific layers/heads.

## First Gate

Run:

```bash
cargo run -p turbo-quant --example sparse_prefill_probe
cargo run -p turbo-quant --example sparse_prefill_trace_bench
cargo test -p turbo-quant sparse_prefill
```

The useful first-pass signal is:

- high `top_k_recall`
- high `softmax_mass_coverage`
- meaningful `estimated_score_reads_saved_ratio`
- warnings that make omitted-top-k cases visible

For real traces, write a JSON array of `SparsePrefillTraceV1` objects and pass
the path to:

```bash
cargo run -p turbo-quant --example sparse_prefill_trace_bench -- traces.json
```

The HF trace dumper writes that exact format:

```bash
python proveKV/scripts/dump_sparse_prefill_traces.py \
  --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --device cuda \
  --n-tokens 256 \
  --max-layers 8 \
  --max-heads 8 \
  --query-count 8 \
  --output /tmp/sparse_prefill_traces.json

cargo run -p turbo-quant --example sparse_prefill_trace_bench -- \
  /tmp/sparse_prefill_traces.json
```

The dumper stores `log(attention_probability + eps)`, not raw pre-softmax
logits. This is intentional: the sparse-prefill gate softmaxes the stored
scores, reconstructing the real model attention distribution closely enough for
token/block selection decisions.

## Integration Point

The first implementation lives in `turbo-quant` because that crate already owns
KV shadow mode, approximate key scoring, sidecar receipts, and benchmark
harnesses. `proveKV` can consume these receipts later when the pool/shell system
needs a runtime sparse-prefill policy.
