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

Bench shape: N=8 agents, 800 shared tokens, 28 unique tokens per
agent, 1024 tokens total. The checked-in state receipts evaluate PPL on
window [800, 1024), and oracle PPL equals roundtrip PPL on that window.

## 2. Geometric fp16 K/V cache denominator

This is the reviewer-obvious framework-cache calculation:

```text
per-token K+V bytes = 24 layers × 32 kv_heads × 64 head_dim × 2 (K,V) × 2 bytes fp16
                    = 196,608 bytes/token
per-agent cache     = 196,608 × 1024 = 201,326,592 bytes
N=8 naive cache     = 201,326,592 × 8 = 1,610,612,736 bytes
```

Compressed proveKV bytes:

| Config | Compressed bytes | Ratio vs geometric fp16 naive |
|---|---:|---:|
| b=4 lossless | 64,306,320 | 25.05× |
| b=4 lossy | 34,028,688 | 47.33× |

Use this denominator when comparing against a conventional fp16
framework K/V cache.

## 3. CLAIMS.json f32-raw denominator

`CLAIMS.json` is the repo's canonical claim ledger. The current default
PPL-validated claims use:

```text
raw_total_bytes        = 2,315,255,808
lossless compressed   = 64,306,320
lossy compressed      = 34,028,688
lossless f32 ratio    = 36.00×
lossy f32 ratio       = 68.04×
lossless fp16-equiv   = 18.00×
lossy fp16-equiv      = 34.02×
```

The ratios are byte-derived:

```text
2,315,255,808 / 64,306,320 = 36.00×
2,315,255,808 / 34,028,688 = 68.04×
```

## 4. Which number to publish?

Publish both when space permits:

| Baseline | Lossless | Lossy | Use when |
|---|---:|---:|---|
| CLAIMS f32-raw bytes | 36.00× | 68.04× | Referring to proveKV's checked-in claim ledger and audit gate |
| CLAIMS fp16-equivalent | 18.00× | 34.02× | Translating the same bytes to a 2-byte/element baseline |
| Geometric fp16 cache | 25.05× | 47.33× | Comparing with a conventional framework fp16 K/V cache |

Do not mix denominators in a single sentence. Always name the baseline.

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
