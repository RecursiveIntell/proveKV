# Fib-Quant encode_batch GPU microbench — 2026-06-01 (revised, with real GPU)

## What we tested

The `fib_quant::FibQuantizer::encode_batch` call in isolation — no pool
manifest, no digest math, no codebook construction. Just the per-batch
encode pipeline. Compared three configs on msi i7-6700HQ + GTX 1070:

- **CPU**: no `gpu` feature compiled in. Pure CPU encode.
- **Hadamard-GPU**: `--features gpu,gpu-backend/precompiled-ptx`. The
  Hadamard rotation dispatches to the new gpu-backend kernel. The
  codebook lookup (nearest_index loop) stays on CPU.
- **Full-GPU**: adds `--features gpu_codebook_lookup`. Both Hadamard
  and codebook_lookup dispatch to GPU.

## Numbers (msi, k=4, N=32, paper_default profile)

| Shape | n | CPU | Hadamard-GPU | Full-GPU | Best |
|---|---|---|---|---|---|
| d=64   | 80  | 14ms   | **13ms (-7%)** | 14ms | Hadamard |
| d=128  | 80  | 57ms   | **54ms (-5%)** | 56ms | Hadamard |
| d=768  | 80  | 2143ms | **2103ms (-2%)** | 2133ms | Hadamard |
| d=2560 | 4   | 1571ms | 1564ms (0%) | 1554ms (0%) | tie |

For d=64 n=20, d=128 n=20, d=768 n=20 the win is also 1-7%.
For n=4 (below the 16-vector batch threshold), the GPU path doesn't
engage, so all three configs are identical CPU.

## What the new codebook kernel reveals

1. **The GPU is not the bottleneck for fib-quant's encode_batch.** The
   nearest_index loop (32 codeword lookups × 640 blocks for d=2560)
   is the dominant cost. The Hadamard rotation is one step in a
   multi-step pipeline; saving it saves ~2-3% of the total time.

2. **Per-call H2D/D2H overhead is the GPU's enemy.** The
   `codebook_lookup` kernel itself runs in microseconds. The dispatch
   path through `gpu_backend` pays H2D + D2H + synchronize per call,
   which costs more than the kernel. The Hadamard-GPU path has the
   same overhead but wins on a bigger per-call saving.

3. **The kernel is correct.** The parity test in gpu-backend (random
   inputs, byte-identical to CPU) is the receipt that the codebook
   kernel produces the right answer. The kernel is good; the dispatch
   is the issue.

4. **The real "win path" is keeping data on GPU across calls** — a
   device-side pipeline. Then the rotated data from the Hadamard stays
   on device for the codebook lookup, eliminating one H2D+D2H round
   trip per call. Estimated 4-6 hours of careful cudarc work.

## Honest takeaway for fib-quant's GPU story

- The new `codebook_lookup` kernel is **infrastructure, not a win** in
  the current dispatch path. It's parity-verified and ready, but the
  per-call H2D/D2H overhead negates any savings.
- The Hadamard-only GPU path is a **modest 2-7% win** on the encode
  pipeline, matching what we saw in the proveKV pool build.
- A 10× win on d=2560 encode_batch would require something other than
  a faster Hadamard: parallelizing the nearest_index loop, a
  device-side pipeline, or batching many encode_batch calls together.

## Public-safe phrasing

"fib-quant's encode_batch is 2-7% faster on a real GPU (msi i7-6700HQ
+ GTX 1070) with the Hadamard path engaged. The codebook_lookup
kernel exists and is parity-verified, but the per-call H2D/D2H
overhead currently negates its win. A device-side pipeline is the
next step."

Do NOT claim "fib-quant is X× faster on GPU." The actual win is 2-7%
on the encode pipeline, depending on (n, d).

## Reproduce

```bash
cd fib-quant

# CPU baseline
cargo run --release --example encode_batch_microbench

# Hadamard-GPU
cargo run --release --example encode_batch_microbench \
  --features gpu,gpu-backend/precompiled-ptx

# Full GPU (Hadamard + codebook_lookup)
cargo run --release --example encode_batch_microbench \
  --features gpu,gpu_codebook_lookup,gpu-backend/precompiled-ptx
```

Note: `combined.ptx` must be present in `gpu-backend/kernels/` or
gpu dispatch falls back to CPU. See proveKV/benchmarks/GPU_BENCH_RESULTS_2026-06-01.md
for the build instructions.
