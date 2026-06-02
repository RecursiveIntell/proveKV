# quant-codec-core

The smallest stable interface layer shared by the governed
compression workspace (`proveKV`, `fib-quant`, `turbo-quant`,
`scr-runtime-compression`, `quant-eval`).

`quant-codec-core` is **boring, deterministic, dependency-light, and
free of runtime authority**. It defines:

- Typed ID and digest primitives (`CodecId`, `CodecProfileDigest`,
  `ArtifactDigest`, `ModelFingerprint`, `TokenizerFingerprint`).
- A shape model for KV-cache tensors (`KvTensorShape`,
  `KvRole`, `KvLayout`, `DType`, `LayerId`, `HeadId`, `TokenSpan`).
- A trait surface that concrete codec implementations must satisfy
  (`CodecProfile`, `VectorCodec`, `KvCacheCodec`).
- An `EvalReport` type for codec quality comparison.

This crate is the **type contract** for the codec workspace. It does
**not** own the codec math, the GPU dispatch, the runtime, the
manifests, the receipts, or the policy. It owns types and traits,
nothing more.

## Why a separate crate?

A codec is a complex thing. It has a profile, a shape, a digest,
a name, a version, an encoding, a decoding, and an evaluation. If
each of the codecs in the workspace (`fib-quant`, `turbo-quant`,
`polar`, `qjl`, future `biquant`, etc.) re-invents these, you get
duplicated IDs, duplicate digests with subtly different canonical
forms, and shape types that don't compose.

`quant-codec-core` is the **one place** in the workspace where the
ID is a `CodecId` and the digest is a `CodecProfileDigest` and
they mean the same thing everywhere.

## Quick Start

```rust
use quant_codec_core::{
    CodecId, CodecProfileDigest, KvRole, KvTensorShape, KvLayout, DType,
    CodecProfile,
};

fn main() {
    // Define a shape for a model.
    let shape = KvTensorShape {
        layers: 32,
        key_heads: 8,
        value_heads: 8,
        seq_len: 2048,
        head_dim: 128,
        layout: KvLayout::LayersHeadsTokensDim,
        dtype: DType::F16,
    };

    // Validate invariants.
    assert!(shape.layers > 0);
    assert_eq!(shape.key_heads, shape.value_heads, "MHA");

    // Build a codec ID.
    let id = CodecId::new("fib-quant").expect("non-empty");
    assert_eq!(id.as_str(), "fib-quant");
}
```

## What's in the box

### IDs and digests

| Type | Purpose |
|---|---|
| `CodecId(String)` | Identifier for a concrete codec (e.g. `"fib-quant"`, `"turbo-quant"`, `"qjl"`). |
| `CodecProfileDigest([u8; 32])` | 256-bit deterministic digest of a codec's profile. Used to identify "the same profile" across versions. |
| `ArtifactDigest([u8; 32])` | 256-bit deterministic digest of an encoded artifact. |
| `ModelFingerprint(String)` | Identifier for the model that produced the source data (e.g. `"qwen3-7b"`, `"nomic-embed-v1.5"`). |
| `TokenizerFingerprint(String)` | Identifier for the tokenizer. |

All five validate empty IDs as errors and provide `Display`, `Debug`,
`Clone`, `Eq`, `Hash`, `Serialize`, `Deserialize`. Digest inputs are
canonicalized via the workspace's `blake3` chain so different
serialization orderings produce the same digest.

### Shape model

The shape types describe a KV-cache tensor — the dominant data
shape in LLM inference.

```rust
pub enum KvRole { Key, Value }
pub enum DType { F32, F16, BF16, I8, U8, PackedBits }
pub enum KvLayout {
    LayersHeadsTokensDim,
    LayersTokensHeadsDim,
    RuntimeSpecific(String),
}

pub struct LayerId(pub u32);
pub struct HeadId(pub u32);
pub struct TokenSpan { pub start: u64, pub end: u64 }

pub struct KvTensorShape {
    pub layers: u32,
    pub key_heads: u32,
    pub value_heads: u32,
    pub seq_len: u64,
    pub head_dim: u32,
    pub layout: KvLayout,
    pub dtype: DType,
}
```

Validation rules:

- `layers > 0`
- `head_dim > 0`
- `seq_len > 0`
- Token spans are half-open `[start, end)` and non-empty
- GQA/MQA: `key_heads != value_heads` represents grouped/multi-query
  attention; `key_heads == value_heads` is multi-head attention (MHA)

### Trait surface

Concrete codec implementations must implement these traits:

```rust
pub trait CodecProfile {
    fn codec_id(&self) -> CodecId;
    fn codec_version(&self) -> &str;
    fn profile_digest(&self) -> CodecProfileDigest;
    fn fixed_rate_bits(&self) -> Option<u16>;
    fn block_dim(&self) -> Option<u16>;
    fn is_lossy(&self) -> bool;
}

pub trait VectorCodec {
    type EncodedBlock;
    type Error;

    fn encode_block(&self, input: &[f32]) -> Result<Self::EncodedBlock, Self::Error>;
    fn decode_block(&self, block: &Self::EncodedBlock, out: &mut [f32]) -> Result<(), Self::Error>;
}

pub trait KvCacheCodec: VectorCodec {
    type EncodedCache;

    fn encode_kv_cache(
        &self,
        tensors: &[f32],
        shape: KvTensorShape,
    ) -> Result<Self::EncodedCache, Self::Error>;

    fn decode_slice(
        &self,
        cache: &Self::EncodedCache,
        request: KvSliceRequest,
        out: &mut [f32],
    ) -> Result<(), Self::Error>;
}
```

`CodecProfile::is_lossy()` is the critical safety predicate. Codecs
that declare `is_lossy = true` are **not allowed** to be the
fallback path for an exact-fallback contract — the caller must
keep the raw data separately.

### Eval types

```rust
pub struct EvalReport {
    pub mse: Option<f64>,
    pub cosine_similarity: Option<f64>,
    pub max_abs_error: Option<f64>,
    pub bytes_exact: u64,
    pub bytes_encoded: u64,
    pub passed: bool,
    pub notes: Vec<String>,
}
```

The eval report is a **data type**, not a runner. The actual
benchmark harness lives in `quant-eval`. `quant-codec-core` provides
the report struct so all codecs can produce a result of the same
shape.

## Source-of-truth ownership

`quant-codec-core` is authoritative for:

- ID and digest shapes and validation
- The `KvTensorShape` model
- The `CodecProfile` / `VectorCodec` / `KvCacheCodec` trait surface
- The `EvalReport` struct

`quant-codec-core` is **forbidden** from owning:

- Codec math (lives in `fib-quant`, `turbo-quant`)
- GPU dispatch (lives in `gpu-backend`)
- Runtime manifests / receipts (lives in `proveKV`)
- Benchmark runners (lives in `quant-eval`)
- Adaptive routing (lives in `quant-governor`)

This is enforced by the `AGENTS.md` boundary contract in the
parent repo.

## Test coverage

- 12 unit tests in `src/` covering:
  - Shape validation (positive and negative)
  - Token span validation
  - Digest stability across runs
  - Serde round-trip for all public types
  - Trait mock compile test
- `cargo test` clean, `cargo clippy --all-targets -- -D warnings` clean.

## MSRV

Rust 1.75 (2021 edition). Stable features only.

## Dependencies

- `serde` (optional, behind the `serde` feature)
- `thiserror`
- `blake3` (for digest computation)

Zero platform-specific code, zero FFI, zero async, zero ML
dependencies. Builds in <1s.

## License

MIT OR Apache-2.0 (dual-licensed). See `LICENSE-MIT` and
`LICENSE-APACHE` for the full texts.

## Changelog

See `CHANGELOG.md` for the release history.

## Where it's used

`quant-codec-core` is a foundational dependency of:

- `proveKV` (the shared KV-cache pool primitive)
- `fib-quant` (the radial-angular vector codec)
- `turbo-quant` (the experimental vector compression sidecar)
- `scr-runtime-compression` (the runtime integration adapter)
- `quant-eval` (the benchmark suite)

Any system that needs a typed codec contract — the
shape of the data, the digest of the profile, the trait the
codec implements — can adopt `quant-codec-core` directly.
