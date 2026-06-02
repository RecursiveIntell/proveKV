# KV Runtime ABI

This document defines the current CPU reference artifact boundary for the
`kv` feature. It is an interface contract for later adapters, not a fused
serving backend.

## Artifact Inputs

- `KvTensorShapeV1`: role, attention kind, batch, layers, KV heads, query heads, tokens, head dimension, dtype, and RoPE state.
- `KvCacheLayoutV1`: canonical contiguous `[batch][layer][kv_head][token][head_dim]` layout for the CPU reference path.
- `KvCompressionProfileV1`: shape digest, FibQuant profile digest, codebook digest, rotation digest, role policy, axis policy, page geometry, protected policy, fallback policy, quality budget, and calibration digest.
- Canonical f32 tensor values for the CPU reference encode path.

## Encoded Page ABI

`KvEncodedPageV1` binds:

- page id;
- token start and token count;
- source tensor digest;
- profile digest;
- shape digest;
- page geometry;
- encoded blocks;
- raw fallback block count;
- page digest.

The current block unit is one `[head_dim]` vector. `encoded_block_bytes` is a fixed reservation for random access. The CPU reference implementation validates the page digest before decoding.

## Decode Modes

- `RawF32`: raw fallback or protected region.
- `FibQuant`: existing `FibCodeV1` vector artifact.

Future fused attention kernels must use the same shape/profile/page digest chain. They must reject stale profiles, mismatched roles, invalid page geometry, and corrupted pages before returning approximate values.

## Current Limits

- CPU reference supports per-token FibQuant compression.
- Per-channel policy decisions are represented, but the CPU codec keeps those blocks raw until a validated channel codec exists.
- No GPU kernel ABI is implemented in this crate.
