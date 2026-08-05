# Backend Capability Matrix

Probed and admitted capabilities per backend. Empty cells = not yet probed.

| Capability | Qwen2.5-0.5B (CPU) | Qwen3.5-2B | llama.cpp | vLLM |
|---|---|---|---|---|
| **Full-attention K/V capture** | ✅ (24 layers, 48 pages) | ❌ (blocked) | | |
| **Convolution state** | N/A (full-attn only) | N/A | | |
| **Recurrent state** | N/A | N/A | | |
| **DynamicCache reconstruction** | ✅ | ❌ | | |
| **Offline load (HF_HUB_OFFLINE=1)** | ✅ | N/A (no weights) | | |
| **trust_remote_code=false** | ✅ | — | | |
| **Raw page roundtrip (BLAKE3)** | ✅ | ❌ | | |
| **Immutable page views** | ✅ (Rust store) | | | |
| **O(1)-metadata fork** | ✅ (Rust store) | | | |
| **Lease authority** | ✅ (Rust store) | | | |
| **Mark-and-sweep GC** | ✅ (Rust store) | | | |
| **CPU-only execution** | ✅ | ✅ (when available) | | |
| **GPU execution** | Not targeted | Not targeted | | |
| **Batch > 1** | Not targeted | Not targeted | | |
| **Beam search** | Not targeted | Not targeted | | |

## Fallback policy

| Condition | Fallback |
|---|---|
| Unsupported model | Recompute from scratch |
| Unknown component kind | Reject (conservative) or raw fallback (permissive) |
| Unadmitted lossy profile | Raw exact |
| State store unavailable | Typed `RecomputeOrUnsupported` |
| Corrupted page | Quarantine page; state invalid |
| Expired/revoked lease | Reject without existence leak |

## Admission gates

A new backend/model pair is admitted only after:
1. Capability probe with explicit source revision
2. Capture → persist → reopen → raw roundtrip
3. Baseline agreement (N ≥ 3 forward passes)
4. Frozen tolerance profile
5. Held-out suffix replay

No `unknown` or `source-only` capability is marked supported.
