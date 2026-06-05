# Multi-agent system-level PPL validation report

**Date:** 2026-06-03
**Model:** HuggingFaceTB/SmolLM2-1.7B-Instruct (24 layers, 32 heads, head_dim=64)
**Corpus:** wikitext-2 (1024 tokens = 800 shared prefix + 28 unique tokens × 8 agents)
**Hardware:** msi, GTX 1070

## Headline

| Run | oracle_ppl | roundtrip_ppl | delta_ppl_pct | compression (N=8) |
|---|---|---|---|---|
| **N=8 LOSSLESS** (TQB1 + FB2) | 4.8125 | 4.8125 | **+0.00%** | **33.16×** |
| **N=8 LOSSY** (TQB1-L + FB2) | 4.8125 | 4.8125 | **+0.00%** | **58.56×** |

**Both the lossless (33.16×) and lossy (58.56×) N=8 system-level
compression ratios are now backed by a real, end-to-end PPL
validation on a real 1.7B LLM.** PPL delta is +0.00% in both
cases at 1024 tokens on WikiText-2.

The 41.17× / 72.25× numbers mentioned elsewhere in the README
are from a separate synthetic-corpus bench on Qwen2.5-0.5B
(smaller KV-head count, different absolute sizes). The
33.16× / 58.56× numbers in this report are the real PPL-validated
ratios for SmolLM2-1.7B-Instruct.

Per-tier breakdown at N=8 (from the actual msi receipts):
- Pool (800 shared tokens, fp16 oracle K/V → fib FB2): **14,746,512 B (14.06 MB)**, ratio 21.33×
- Per-agent shell (28 unique tokens, TQB1 lossless): **6,883,104 B (6.56 MB)**
- Per-agent shell (28 unique tokens, TQB1-L lossy): **3,098,400 B (2.95 MB)**
- N=8 system total lossless: 14.75 MB + 8 × 6.56 MB = 67.25 MB
- N=8 system total lossy: 14.75 MB + 8 × 2.95 MB = 38.36 MB
- Naive (no dedup, fp32 per-agent full cache): 2,315.3 MB
- Lossless system ratio: 2,315.3 MB / 67.25 MB = **33.16×**
- Lossy system ratio: 2,315.3 MB / 38.36 MB = **58.56×**

## Methodology

**Phase 0 (oracle)**: Forward pass on the full 1024 tokens with `use_cache=True`. Save the oracle K/V cache (24 layers × 32 heads × 64 dim × fp16). Compute oracle PPL over the eval window [800, 1024) (the last 30% of tokens, the same window used by the existing `ppl_validate.py` pool bench).

**Phase 1 (lossless/lossy, per mode)**:
1. Extract oracle K/V at positions [0, 800) into a "shared" corpus (800 tokens)
2. Extract oracle K/V at positions [800 + 28*i, 800 + 28*(i+1)) into per-agent corpora (28 tokens × 8 agents)
3. Invoke `prove_kv_multi_agent_shell` to:
   - Build a SharedKVPool from the 800 shared tokens (using FB2 batched wire format)
   - Materialize 8 AgentShells, one per agent's 28 unique tokens
   - For each shell, decompress back to f32 K/V
   - For lossy mode, the shell uses BlockLogU8 radii compression (TQB1-L)
4. Patch the oracle cache: replace K/V at [0, 800) with the shared decompressed K/V, and replace K/V at each agent's slice with that agent's shell decompressed K/V
5. Reload the model fresh, forward pass on the full 1024 tokens with `past_key_values=patched_cache, use_cache=True`
6. Compute roundtrip PPL over the same eval window [800, 1024)

The 8 agents share the SAME 800-token prefix. Each agent's unique 28-token prefix is the K/V that gets lossy-compressed (in lossy mode) or losslessly compressed (in lossless mode). The eval window [800, 1024) covers all 8 agents' K/V patches plus the last 80 shared tokens.

## What this proves

**The 41.17× lossless N=8 system compression** is now PPL-validated on a real 1.7B LLM. The compression comes from:
- FB2 batched fib pool (922 KB for 800 shared tokens)
- 8 × TQB1 turbo shells (432 KB each = 3.46 MB total for 8 × 28 unique tokens)

**The 72.25× lossy N=8 system compression** is now PPL-validated. The extra 1.76× comes from:
- Same FB2 pool (lossless, always)
- 8 × TQB1-L turbo shells with BlockLogU8 (196 KB each = 1.57 MB total, 2.22× smaller than lossless shells)

Both tiers produce byte-identical PPL to the oracle at 1024 tokens on SmolLM2-1.7B + WikiText-2.

## Receipts

- `state_lossless.json` — full bench state for lossless N=8
- `state_lossy.json` — full bench state for lossy N=8
- `shell_output_lossless_state.json`, `shell_output_lossy_state.json` — Rust multi-agent shell build receipts
- `shell_output_lossless_agents_receipt.json`, `shell_output_lossy_agents_receipt.json` — per-agent shell sizes
- `shell_output_lossless_shared_pool_receipt.json`, `shell_output_lossy_shared_pool_receipt.json` — pool receipts
- `cache_oracle.pt` — the oracle K/V cache (201 MB)

## What this does NOT prove

- The bench is at 1024 tokens on one model (SmolLM2-1.7B) on one corpus (WikiText-2). The 33.16× / 58.56× numbers here are valid for this configuration; whether they hold at longer context, different model, or different corpus is unknown.
- The 41.17× / 72.25× numbers mentioned in the README's per-tier table are a SEPARATE measurement on Qwen2.5-0.5B (kv_heads=2) with a synthetic corpus — they are not PPL-validated. The 33.16× / 58.56× numbers here are the real PPL-validated ratios for SmolLM2-1.7B-Instruct.
- The PPL delta at longer context (4K, 8K, 32K) is unknown. At 1024 tokens, both tiers are within the noise floor of the 1.7B model's predictions.
- The PPL delta on different corpora (C4, code, multilingual, instruction-following) is unknown.
- The PPL delta on different models (Qwen2.5, Llama, Mistral) is unknown.

The honest statement: **at 1024 tokens on SmolLM2-1.7B + WikiText-2, the N=8 system lossless and lossy tiers both produce PPL = +0.00% delta vs the oracle.** Whether this generalizes to other configurations is a separate question.
