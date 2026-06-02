# CUDA Kernel Roadmap

No CUDA kernel is implemented in this run. Future kernels should be built only after the CPU reference path and adapter receipts identify a profile worth optimizing.

## Kernel Requirements

- Fixed page lookup by page id and block id.
- Role-aware decode path.
- Raw fallback block support.
- Shape/profile/page digest validation at the host boundary.
- Eager decode and fused attention modes declared separately.
- No silent fallback from approximate output to exact-output labels.

## Candidate Kernel Stages

1. Eager-decode page blocks to f16/bf16/f32 workspace.
2. Fused key logit path for supported key profiles.
3. Fused value aggregation path for supported value profiles.
4. Mixed raw/compressed page scheduling.
5. Backend-specific autotuning with receipts.

## Required Evidence Before Claims

- named GPU hardware;
- model and context length;
- latency, throughput, and memory receipts;
- quality receipts against raw cache;
- comparison to a practical baseline;
- corruption and fallback tests.
