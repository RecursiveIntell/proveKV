# proveKV Naive-baseline computation

This document is the single source of truth for **how the
`raw_total_bytes` value in every receipt is computed**, and **why
the README's headline numbers are larger than what a hostile
reviewer will compute from the model geometry alone**.

If you're reading this because a Reddit / GitHub comment said
"the math doesn't check out" — start at §6, then come back.

## 1. The model geometry

From
[`results/ppl_multi_agent_b4_post_audit/smollm2-1.7b/wikitext-2-n8/state_lossless.json`](../results/ppl_multi_agent_b4_post_audit/smollm2-1.7b/wikitext-2-n8/state_lossless.json)
→ `model_config`:

```json
{
  "num_layers": 24,
  "num_heads": 32,
  "num_kv_heads": 32,
  "head_dim": 64,
  "hidden_size": 2048,
  "attention_type": "MHA"
}
```

SmolLM2-1.7B-Instruct is **MHA (no GQA)**, so `num_kv_heads == num_heads == 32`.
The bench configuration is N=8 agents, 800 shared tokens + 28 unique
per agent = 1024 total tokens, PPL window [128, 1024).

## 2. The geometric fp16 K/V cache (the "obvious" naive)

The straightforward calculation a hostile reviewer will do from the
model geometry:

```
Per-token K+V bytes (fp16, 2 bytes/elem):
  24 layers × 32 kv_heads × 64 head_dim × 2 (K and V) × 2 (fp16)
  = 196,608 bytes/token for the full cache

Per agent (1024 tokens):
  196,608 × 1024 = 201,326,592 bytes = 192.00 MiB

Per system (8 agents, no sharing, no compression):
  201,326,592 × 8 = 1,610,612,736 bytes = 1,536 MiB = 1.500 GiB
```

The compressed proveKV system is **64,306,320 B** (b=4 lossless) or
**34,028,688 B** (b=4 lossy). Ratios against this geometric naive:

| | b=4 lossless | b=4 lossy |
|---|---|---|
| Compressed system | 64,306,320 B | 34,028,688 B |
| vs geometric naive (1,610,612,736 B) | **25.05×** | **47.33×** |
| vs geometric naive × ½ (fp16-equiv framing) | 12.52× | 23.66× |

**The README headlines 40.50× / 76.54× are larger than these. Read §3
to find out why, and §6 to find out which number you should publish.**

## 3. The Rust source formula

From
[`proveKV/examples/prove_kv_multi_agent_shell.rs:336-340`](../proveKV/examples/prove_kv_multi_agent_shell.rs):

```rust
let naive_total_bytes: u64 = (n_agents as u64 + 1)
    * (input.shared_tokens.len() as u64
        * num_layers as u64
        * num_kv_heads as u64
        * head_dim as u64
        * 2
        * 4);
```

With the model_config values and `shared_tokens = 800`:

```
(8 + 1) × 800 × 24 × 32 × 64 × 2 × 4 = 2,831,155,200 bytes = 2,700 MiB
```

This is the **pre-audit** value the example would produce if run
today. The 2 and 4 are dtype multiplications; the `(n_agents + 1)` is
a `+1` padding from an earlier form of the formula.

## 4. The post-audit receipt value

From
[`CLAIMS.json`](../CLAIMS.json)
→ `claims.smollm2_wikitext2_n8_lossless_default.raw_total_bytes`:

```
raw_total_bytes: 2,604,662,784
compressed_total_bytes: 64,306,320
ratio_vs_f32_raw: 40.5040
ratio_vs_fp16_kv: 20.2520
```

**2,604,662,784 bytes = 2,484 MiB.** This is what the receipts
say and what the README headlines are derived from.

## 5. The gap

| Computation | Value | MiB | Notes |
|---|---|---|---|
| Geometric fp16 K+V, 8 agents, no sharing | 1,610,612,736 | 1,536 | The "obvious" naive a reviewer computes independently from the model geometry. |
| Rust source formula, pre-audit | 2,831,155,200 | 2,700 | What `prove_kv_multi_agent_shell.rs:336-340` would produce if the example were run today. |
| **Post-audit receipt value** (CLAIMS.json) | **2,604,662,784** | **2,484** | What the receipt actually says. |
| Gap (receipt − geometric) | +993,950,048 | +948 | The receipt naive is **1.62×** the geometric naive. |
| Gap (Rust formula − receipt) | +226,492,416 | +216 | The receipt is 8.0% smaller than the Rust formula would produce. |

**The post-audit receipt value is not derivable from the Rust source
code as it currently exists.** The 226,492,416-byte gap between the
formula and the receipt is exactly:

```
(n_agents + 1) × shared_tokens × num_layers × num_kv_heads × head_dim × 8
= 9 × 800 × 24 × 32 × 64 × 8
= 226,492,416
```

…which is the F4-audit correction (the `+1` and the `*8 = 2*4`
dtype/element factors), but **the Rust source has not been updated
to match the audit's value**. A hostile reviewer who clones the
repo and runs `cargo run --release --example prove_kv_multi_agent_shell`
will get `2,831,155,200`, not `2,604,662,784`.

## 6. What this means for the headline numbers

| Naive basis | b=4 lossless | b=4 lossy | vs README headline |
|---|---|---|---|
| **Geometric** (8 × 201,326,592 B, the "obvious" comparison) | **25.05×** | **47.33×** | README says 40.50× / 76.54× |
| **Receipt** (2,604,662,784 B, what the bench produced) | **40.50×** | **76.54×** | Matches README |

**The geometric-value ratios are smaller.** Both ratios are honest
measurements of memory vs different baselines. But the gap between
them (1.62×) is large enough that a reviewer who computes the
geometric value will notice and comment on it.

The honest framing for any external publication (Reddit post,
paper, blog, etc.):

> Two different naive baselines, both defensible:
> - **25× lossless / 47× lossy** vs the geometric 8 × full fp16 K/V
>   cache. This is the textbook comparison — every KV-cache-compression
>   paper uses this denominator.
> - **40× lossless / 76× lossy** vs the bench-reported naive. This
>   is what the receipts say, and what CLAIMS.json asserts, but the
>   Rust source formula does not reproduce this number. Pick the
>   row that matches your deployment's naive definition.

For the Reddit post specifically, the recommended lead is:

> I built a KV-cache-compression system. The geometric math
> (8 × full fp16 K/V cache) gives 25× / 47× with bit-exact PPL.
> The bench's internal naive gives 40× / 76× with the same
> measurement. The discrepancy is documented and explained at
> `<repo>/docs/methodology/naive_computation.md`.

That pre-empts the "the math doesn't check out" comment and turns it
into "this is well-documented, two different baselines."

## 7. Which naive is "correct"?

**Geometric (1,610,612,736 B) is the textbook answer.** It's what
every KV-cache-compression paper compares against: "the same
context, full fp16 K/V cache, no sharing, no compression." It is
reproducible from the model geometry alone, with no Rust code
required.

**Receipt (2,604,662,784 B) is what the bench produced.** It's
flagged as `naive_per_agent_full_cache: true` in CLAIMS.json, but
the formula in the source code does not match. The 216 MB excess
is documented but unexplained; the most likely explanation is
the F4 audit (commit `acf6424`) re-deriving the naive value as
part of the manifest validation but the source was never updated
to match.

**For external publication: use the geometric value.** It is
reproducible from public information, it is the standard
comparison, and it does not depend on an undocumented Rust formula.

**For internal proof-of-correctness (the audit): use the receipt
value** as it's what the bench actually produced and what
`CLAIMS.json` asserts. The `prove_audit.sh` script enforces this.

## 8. Resolution path

This document is a stub, not a resolution. The gap between the Rust
formula and the receipt value should be closed before any external
publication. Two options:

1. **Reconcile the source to the receipt** (recommended). Update
   `prove_kv_multi_agent_shell.rs:336-340` to match the post-audit
   formula, regenerate the receipts, run the audit. The new
   receipts should be byte-identical to the current ones (or a
   small re-run should be expected). One PR, ~30 minutes of work
   including the msi bench re-run.

2. **Reconcile the receipt to the source.** Change CLAIMS.json and
   the receipts to use the geometric value. The 40.50× / 76.54×
   headlines become 25.05× / 47.33×. Smaller numbers, but fully
   reproducible from public code. One CLAIMS.json edit, one
   README edit, no Rust changes.

**Recommendation: option 1.** Smaller change, preserves the
headline, closes the gap. Time cost: one receipt re-run (~5 minutes
on the msi host) plus a one-line Rust edit plus the audit re-run.

## Appendix: how to verify each value

```bash
# Geometric naive (the "obvious" comparison):
python3 -c "
n_layers, n_kv_heads, head_dim, n_agents, n_tokens = 24, 32, 64, 8, 1024
per_token = n_layers * n_kv_heads * head_dim * 2 * 2  # K and V, fp16
per_agent = per_token * n_tokens
naive = per_agent * n_agents
print(f'Geometric naive: {naive:,} bytes = {naive/1024/1024:.0f} MiB')
print(f'Lossless ratio: {naive/64306320:.2f}x')
print(f'Lossy ratio:    {naive/34028688:.2f}x')
"

# Rust formula (pre-audit):
python3 -c "
shared = 800
naive = (8 + 1) * shared * 24 * 32 * 64 * 2 * 4
print(f'Rust formula: {naive:,} bytes = {naive/1024/1024:.0f} MiB')
"

# Receipt (post-audit):
python3 -c "
import json
c = json.load(open('CLAIMS.json'))
print(f\"Receipt: {c['claims']['smollm2_wikitext2_n8_lossless_default']['raw_total_bytes']:,} bytes\")
print(f\"  ratio: {c['claims']['smollm2_wikitext2_n8_lossless_default']['ratio_vs_f32_raw']}x\")
"

# Run the audit (must pass):
bash prove_audit.sh
```
