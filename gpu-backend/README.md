# gpu-backend

Shared CUDA GPU backend for `fib-quant` and `turbo-quant` vector
quantization.

`gpu-backend` provides the GPU-side primitives that the codec
crates dispatch to: a **Hadamard rotation kernel** (in-place
fast Walsh-Hadamard transform) and a **codebook lookup kernel**
(nearest-codeword search for block-quantized vectors). Both
kernels are **parity-verified** — they produce byte-identical
results to the CPU reference on random inputs, which is the
audit handle that GPU dispatch doesn't silently change the
result.

**Status:** alpha. The kernels are correct. The dispatch path
through `cudarc` pays per-call H2D/D2H overhead that, for the
workloads in the current benchmark suite, is more expensive
than the kernel runtime. The kernels are exposed as
**infrastructure** that a future device-side pipeline can
use to keep rotated data resident on device between calls.

## What's in the box

- **`simd_nearest_codeword`** — AVX2+FMA SIMD implementation
  of the k=4, N=32 codebook lookup loop. For 4-element blocks,
  two codewords fit in one `__m256` and are evaluated with
  FMA + horizontal add. Runtime feature detection via
  `is_x86_feature_detected!` — falls back to a scalar f32 loop
  on platforms without AVX2+FMA.
- **Hadamard kernel** — In-place fast Walsh-Hadamard transform
  on d-dim vectors. CPU fallback when `gpu` feature is not
  enabled; CUDA kernel when it is.
- **Codebook lookup kernel** — Nearest-codeword search for
  block-quantized vectors. CUDA kernel when `gpu` feature is
  enabled; CPU fallback otherwise.
- **Parity tests** — Random inputs (16 seeds, 32 dims × 32
  codewords, 4 blocks) verified byte-identical against the
  CPU reference. The audit handle that the kernel is correct.

## Quick Start

```rust
use gpu_backend::simd_nearest_codeword;

fn main() {
    // AVX2+FMA SIMD path — no GPU needed, just x86 with AVX2+FMA.
    let codebook: Vec<f32> = /* 32 codewords × 4 dims */;
    let input: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let (index, score) = simd_nearest_codeword(&input, &codebook, 4, 32);
    println!("Nearest codeword: index={}, score={}", index, score);
}
```

Run it: `cargo run --release --example simd_nearest_demo`.

## Features

| Feature | Default | What it enables |
|---|---|---|
| `gpu` | off | CUDA dispatch via `cudarc` |
| `precompiled-ptx` | off | Loads the precompiled `combined.ptx` at runtime; required for real GPU dispatch |
| `default = []` | yes | Pure CPU, no CUDA dependency compiled in |

Both `gpu` and `precompiled-ptx` are required for real GPU
dispatch. Without `--features precompiled-ptx`, all GPU
operations fall back to CPU.

## Benchmarks — measured

The `gpu-backend` kernels were measured on **msi i7-6700HQ + GTX 1070**
(matched to the fib-quant and turbo-quant bench environments).

### `simd_nearest_codeword` (AVX2+FMA CPU path)

For k=4, N=32 codebook lookups:

| Workload | Scalar f32 | AVX2+FMA SIMD | Speedup |
|---|---|---|---|
| 16 random seeds, 32 dims × 32 codewords | 8.0ms | 1.2ms | **6.7×** |

The SIMD path is the dominant cost saver in the fib-quant
encode_batch loop. **Parity test: byte-identical to scalar f32
on all 16 random seeds.**

### `codebook_lookup` kernel (CUDA)

| Workload | CPU fallback | GPU kernel | Ratio |
|---|---|---|---|
| qwen3 n=80 d=2560 k=4 | 8ms (6.27M blocks/s) | 14ms (3.42M blocks/s) | GPU 0.5× CPU |
| nomic n=80 d=768 k=4 | 2ms (6.36M blocks/s) | 4ms (3.44M blocks/s) | GPU 0.5× CPU |

**The GPU is 1.8× slower per call than the tight CPU loop** for
these batch sizes. Root cause: every call to
`gpu_backend::codebook_lookup_batch` pays H2D + D2H transfer
overhead. The rotated input is `n * d * 4` bytes uploaded, the
indices are `n * (d/k) * 4` bytes downloaded, plus
`synchronize()` between.

For n=80, d=2560: 800KB H2D + 100KB D2H per call. PCIe 2.0 x16
practical throughput is ~4GB/s, so the transfers alone are
~225μs. The kernel runtime is microseconds. **Transfer overhead
dominates.**

**The kernel is correct** (parity test passes for n=32, d=128,
k=4, N=32 random inputs on msi GTX 1070). **The dispatch is the
issue, not the kernel.**

### Hadamard kernel (CUDA)

| Workload | CPU | GPU Hadamard-only |
|---|---|---|
| nomic 768 n=80 | 4552ms wall | 4430ms wall (-2.7%) |
| qwen3 2560 n=80 | 13763ms wall | 13419ms wall (-2.5%) |

**Hadamard-only GPU win: 2.5-2.7%** on the larger corpora.
The win is real but small — the dominant cost in
fib-quant's encode_batch is the codebook lookup, not the
Hadamard.

## What would actually win

A **device-side pipeline** that keeps the rotated data on GPU
between the Hadamard and the codebook lookup:

1. H2D input (once per pool build)
2. GPU Hadamard (in-place on device)
3. GPU codebook lookup (no H2H roundtrip)
4. D2H indices (just the small result array)

This requires restructuring `gpu_backend` to expose a
`GpuPipeline` handle that holds the device buffer across calls.
The current design allocates and frees per-call, which is
correct but defeats the purpose of GPU compute for this
workload.

Estimated effort: 4-6 hours of careful `cudarc` work, plus a
parity test that proves device-side indices match the CPU
reference.

## Test coverage

- **16 parity tests** in `tests/`:
  - SIMD vs scalar f32, 16 random seeds, byte-identical.
  - Hadamard vs reference, 8 random seeds, byte-identical.
  - Codebook lookup vs reference, 8 random seeds, byte-identical
    (only on machines with `gpu` feature + CUDA runtime).
- **3 examples**: `simd_nearest_demo`, `codebook_lookup_microbench`,
  `hadamard_microbench`.
- `cargo test` clean, `cargo clippy --all-targets -- -D warnings` clean.

## MSRV

Rust 1.75 (2021 edition). Stable features only (the CUDA
dispatch uses `cudarc` which is also stable).

## Dependencies

- `cudarc` (optional, behind the `gpu` feature) — CUDA driver bindings.
- `blake3` — for parity-test digests.
- `serde` (with `derive`).
- `thiserror` — for error types.
- `rand`, `rand_chacha` (dev) — for random test inputs.

The `cudarc` dep is **optional** — building with no features
produces a pure-CPU crate with no CUDA runtime, no PTX loading,
no GPU driver.

## License

MIT. See `LICENSE-MIT` for the full text.

## Changelog

See `CHANGELOG.md` for the release history.

## Where it's used

`gpu-backend` is the GPU-side primitive for:

- `fib-quant` — dispatches Hadamard rotation to GPU when
  `--features gpu` is enabled, and codebook lookup when
  `--features gpu_codebook_lookup` is enabled.
- `turbo-quant` — historically had a `gpu` feature that was
  removed in v0.2.0 because the dispatch overhead negated
  the kernel speedup; the kernels live here for future use.
- `proveKV` — gates the GPU codebook lookup path on this
  crate's `gpu` feature.

Any system that needs a parity-verified GPU Hadamard or
codebook lookup can adopt `gpu-backend` directly.
