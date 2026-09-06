# proveKV Naive-baseline computation

This document defines the denominators behind the public proveKV ratios.
It exists because different honest baselines produce different ratios.

## 1. Model geometry

The checked-in post-audit SmolLM2-1.7B receipts use:

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

Size estimand: N=8 independent contexts, each containing the same 800-token
prefix plus one 28-token private tail. PPL estimand: one separate aggregate
1024-token fixture containing the prefix plus eight contiguous 28-token slices.
The current state receipts score exact target indices [800, 1024) and bind the
model revision, token digest, source snapshot, and cache identity.
The cache-aligned shared-prefix continuation result is oracle PPL 7.203125
versus roundtrip PPL 24.234375 (+236.44%) for both current b=4 runs. The
forward loads reconstructed positions [0, 799) and recomputes token 799 onward,
so shell K/V is not consumed by this score. It rejects PPL neutrality; only the
separately derived size ratios below are publication-eligible.

## 2. Geometric fp16 K/V cache denominator

This is the reviewer-obvious framework-cache calculation:

```text
per-token K+V bytes = 24 layers × 32 kv_heads × 64 head_dim × 2 (K,V) × 2 bytes fp16
                    = 196,608 bytes/token
per-agent cache     = 196,608 × (800 + 28) = 162,791,424 bytes
N=8 naive cache     = 162,791,424 × 8 = 1,302,331,392 bytes
```

Compressed proveKV bytes:

| Config | Compressed bytes | Ratio vs geometric fp16 naive |
|---|---:|---:|
| b=4 lossless | 64,306,320 | 20.25× |
| b=4 lossy | 34,028,688 | 38.27× |

Use this denominator when comparing against a conventional fp16
framework K/V cache.

## 3. CLAIMS.json f32-raw denominator

`CLAIMS.json` is the repo's canonical claim ledger. The current default
size results use:

```text
raw_total_bytes        = 2,604,662,784
lossless compressed   = 64,306,320
lossy compressed      = 34,028,688
lossless f32 ratio    = 40.50×
lossy f32 ratio       = 76.54×
lossless fp16-equiv   = 20.25×
lossy fp16-equiv      = 38.27×
```

The ratios are byte-derived:

```text
2,604,662,784 / 64,306,320 = 40.50×
2,604,662,784 / 34,028,688 = 76.54×
```

## 4. Which number to publish?

Publish both when space permits:

| Baseline | Lossless | Lossy | Use when |
|---|---:|---:|---|
| Independent-context f32 bytes | 40.50× | 76.54× | Comparing compressed bytes with eight independent 828-token f32 contexts |
| Independent-context fp16-equivalent | 20.25× | 38.27× | Translating the same geometry to 2-byte elements |

Always name the baseline. Keep the size result separate from the aggregate
fixture PPL result; the latter does not establish eight independent-prompt PPL
measurements and currently fails the PPL-neutrality gate.

## 5. Verification

```bash
python3 - <<'PY'
import json
c=json.load(open('CLAIMS.json'))['claims']
for name in ['smollm2_wikitext2_n8_lossless_default','smollm2_wikitext2_n8_lossy_default']:
    x=c[name]
    print(name, x['raw_total_bytes']/x['compressed_total_bytes'], x['ratio_vs_f32_raw'])
PY
bash prove_audit.sh
```
