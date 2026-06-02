# FibQuant Math Conformance

Date: 2026-05-16

Primary source: Namyoon Lee and Yongjune Kim, "FibQuant: Universal Vector Quantization for Random-Access KV-Cache Compression", arXiv:2605.11478v1.

## Implemented

- Vector normalization before block quantization.
- Deterministic stored rotation for the ambient vector path.
- Spherical-Beta block source sampler and Gaussian projection reference sampler.
- Bennett-Gersho radial companding shape `beta_{d,k}` with fail-closed finite positive validation.
- k=2 closed-form radial quantile path.
- k>=3 Beta-quantile radial path.
- k=2 Fibonacci spiral directions.
- k=3 Fibonacci sphere directions.
- k>=4 Roberts-Kronecker directions.
- Deterministic Lloyd-Max refinement with multiple restarts, finite-MSE validation, and non-worsening fallback to the initialized codebook.
- Fixed-rate index packing using `ceil(log2(N))` wire width.

## Enforced Profile Law

`FibQuantProfileV1::validate()` rejects:

- wrong profile schema marker;
- non-finite or tampered paper rate;
- non-finite or tampered wire rate;
- radius/direction methods incompatible with `k`;
- non-paper source mode for the current release posture;
- non-fp16 norm format for the current paper path;
- zero or excessive Lloyd restarts/iterations;
- training sample counts below codebook size or above the configured hard bound.
- degenerate `d == k` profiles; unit-sphere whole-vector mode is deferred for alpha.

The profile method fields are authoritative for codebook construction. Radius and direction dispatch go through `radius_method` and `direction_method`; unsupported combinations reject.

## Receipt and Wire Conformance

- `FibCodeV1` uses schema marker `fib_code_v1`.
- `FibCodebookV1` uses schema marker `fib_codebook_v1`.
- `StoredRotation` uses schema marker `fib_rotation_v1`.
- `LloydReportV1` uses schema marker `lloyd_report_v1`.
- `FibQuantCompressionReceiptV1` uses schema marker `fib_quant_compression_receipt_v1`.
- Rotation generation is identified as `qr-gaussian-chacha8-sign-corrected-v1`: draw a deterministic Gaussian `d x d` matrix from `ChaCha8Rng`, run QR, and flip each `Q` column whose corresponding `R[j,j]` is negative.
- Decode rejects mismatched code schema, profile digest, codebook digest, rotation digest, dimensions, block count, wire width, or norm format.
- Receipts include source vector, profile, codebook, rotation, and encoded payload digests.
- Alpha reproducibility is local to this implementation and dependency set. The rotation digest records the exact generated matrix so replay/tamper checks do not rely on a seed-only identity claim.

## Not Implemented

- Fused attention-kernel decode path.
- End-to-end transformer KV-cache integration.
- Paper benchmark reproduction harness.
- Production fallback codec policy.
- Hardware-specific optimization claims.
