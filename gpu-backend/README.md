# gpu-backend

CPU/SIMD primitives and a feature-gated CUDA codebook lookup backend
for vector quantization.

**Status:** alpha. The public backend currently activates one CUDA
operation: codebook lookup for `k = 4` and exactly 32 codewords.
It compiles the checked-in `kernels/codebook_lookup.cu` through NVRTC
at runtime. Hadamard, Lloyd-Max encode/decode, and bitpacking public
operations currently select their CPU reference paths.

CUDA readiness is reported by `GpuContext::availability()`.
`Ready` means the driver and NVRTC loaded, the canonical source
compiled, its module loaded, and the required kernel symbol resolved.
It is a readiness signal, not a hardware-parity or performance result.

## What's in the box

- **`nearest_codeword_f32`** — returns a nearest-codeword index.
  On x86/x86_64, the `k = 4` path uses AVX2+FMA when those
  features are available at runtime; other cases use scalar f32.
- **`codebook_lookup_batch`** — returns row-major `u32` indices.
  CUDA is eligible only for finite inputs and a finite codebook,
  `k = 4`, exactly 32 codewords, `n >= 16`, `d >= 64`,
  supported index ranges, and a ready CUDA context. Other valid
  shapes or unavailable CUDA use the CPU reference.
- **CPU reference operations** — `hadamard_batch`,
  `lloyd_max_batch`, `lloyd_max_decode_batch`, and `bitpack`.
- **Readiness and conditional parity tests** — source includes
  disabled-feature, unavailable-runtime, input-validation, and
  CUDA/CPU index-parity checks. The CUDA parity test skips when
  CUDA is unavailable; a passing CPU-only test run is not proof
  that CUDA executed.

## Quick Start

```rust
use gpu_backend::nearest_codeword_f32;

fn main() {
    // One 4-element sample and two 4-element codewords.
    let codebook = vec![0.0_f32, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let index = nearest_codeword_f32(&input, &codebook, 4);
    println!("Nearest codeword: index={index}");
}
```

This CPU/SIMD API needs no CUDA feature. It returns an index, not
an `(index, score)` tuple.

From the workspace root, the following commands exercise the package
tests. The CUDA-feature command still requires inspection of readiness
and test output to distinguish an executed CUDA test from a skip.

```sh
cargo test -p gpu-backend
cargo test -p gpu-backend --features gpu -- --nocapture
```

## Features

| Feature | Default | What it enables |
|---|---|---|
| `gpu` | off | CUDA driver/NVRTC dynamic loading and the narrow codebook-lookup dispatch |
| `precompiled-ptx` | off | Retained feature declaration; not required by the active NVRTC-compiled codebook-lookup path |
| `default = []` | yes | CPU/SIMD primitives without compiling the CUDA dependency |

The active CUDA lookup requires `gpu`, an available CUDA driver,
NVRTC, and the documented shape contract. It does not load
`combined.ptx` for that path. Use the typed availability result
rather than assuming that compiling a feature proves GPU execution.

## Historical benchmark reports

The following numbers are retained as previously reported measurements.
They do not certify the current source revision or its dispatch path.
In particular, the historical Hadamard-GPU rows do not describe the
current public `hadamard_batch` operation, which now runs on CPU.
Re-establish source-bound hardware receipts before using these rows
as current performance or parity claims.

## Benchmarks — measured

The `gpu-backend` kernels were measured on **msi i7-6700HQ + GTX 1070**
(matched to the fib-quant and turbo-quant bench environments).

### `nearest_codeword_f32` (historical AVX2+FMA CPU report)

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

The historical report described parity for n=32, d=128, k=4,
N=32 on msi GTX 1070. Current CUDA parity must be established
with evidence that the active kernel actually executed; the
conditional test alone can skip on unavailable hardware.

### Hadamard kernel (CUDA)

| Workload | CPU | GPU Hadamard-only |
|---|---|---|
| nomic 768 n=80 | 4552ms wall | 4430ms wall (-2.7%) |
| qwen3 2560 n=80 | 13763ms wall | 13419ms wall (-2.5%) |

These historical Hadamard timings are not a current CUDA capability
claim. The public Hadamard operation now selects the CPU reference.

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

The source includes CUDA readiness diagnostics, invalid-input checks,
and a conditional codebook index-parity test. That parity fixture
uses n=32, d=128, k=4, and 32 codewords, and skips when CUDA
is unavailable.

Run the commands in Quick Start against the revision you intend to use.
A passing run without an executed CUDA fixture establishes no hardware
parity result, and this README makes no current test-count or timing claim.

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

- `fib-quant` — its `gpu` feature enables backend readiness;
  `gpu_codebook_lookup` additionally enables its optional lookup
  call. The backend's public Hadamard call currently stays on CPU.
- `turbo-quant` — historically had a `gpu` feature that was
  removed in v0.2.0 because the dispatch overhead negated
  the kernel speedup; the kernels live here for future use.
- `proveKV` — reaches backend primitives through its fib codec
  dependency. Its own `gpu` feature is an empty cfg-probe feature;
  it does not enable `gpu-backend/gpu`.

Systems can adopt the CPU/SIMD primitives directly or evaluate the
narrow CUDA lookup path with workload-specific hardware receipts.
