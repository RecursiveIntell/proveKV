# vLLM and FlashInfer Adapter Plan

The current crate provides page contracts that can inform later vLLM or FlashInfer experiments. It does not replace an attention backend.

## Required Adapter Mapping

- Page size maps to `KvPageGeometryV1.tokens_per_page`.
- Block size maps to one logical vector today.
- Role maps to `KvRole`.
- RoPE state maps to `KvRopeState` and must be explicit for keys.
- Layout maps to `KvCacheLayoutV1`.
- Decode mode maps to raw fallback, eager decode, or future fused attention.
- Capability flags must declare supported roles, axis policies, dtypes, and page geometry.

## Validation Before Backend Use

- Validate shape and layout.
- Validate profile digest, codebook digest, and rotation digest.
- Validate each page digest.
- Reject profile/shape/role mismatches.
- Preserve raw fallback pages and blocks.
- Emit receipts for encode, decode, and quality evaluation.

## Deferred Work

- Backend-specific memory allocator integration.
- PagedAttention kernel integration.
- Heterogeneous page scheduling.
- Model-captured quality and latency comparisons.
