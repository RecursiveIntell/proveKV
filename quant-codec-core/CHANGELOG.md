# Changelog

All notable changes to `quant-codec-core` are documented here.

## [Unreleased]

## [0.1.0-alpha.1] — 2026-06-02

First crates.io release.

### Added

- `CodecId(String)` — typed identifier for codecs. Validates non-empty.
- `CodecProfileDigest([u8; 32])` — 256-bit profile digest.
- `ArtifactDigest([u8; 32])` — 256-bit artifact digest.
- `ModelFingerprint(String)` — typed model identifier.
- `TokenizerFingerprint(String)` — typed tokenizer identifier.
- `KvRole` — `Key` or `Value` discriminator.
- `DType` — `F32`, `F16`, `BF16`, `I8`, `U8`, `PackedBits`.
- `KvLayout` — `LayersHeadsTokensDim`, `LayersTokensHeadsDim`,
  or a runtime-specific string.
- `LayerId(u32)`, `HeadId(u32)`, `TokenSpan { start, end }` — addressing
  primitives.
- `KvTensorShape` — the canonical KV-cache shape struct.
- `CodecProfile` trait — `codec_id`, `codec_version`,
  `profile_digest`, `fixed_rate_bits`, `block_dim`, `is_lossy`.
- `VectorCodec` trait — `encode_block`, `decode_block`.
- `KvCacheCodec` trait — `encode_kv_cache`, `decode_slice`.
- `EvalReport` — quality metrics: mse, cosine_similarity,
  max_abs_error, bytes_exact, bytes_encoded, passed, notes.

### Lints

- `#![forbid(unsafe_code)]`.
- `cargo clippy --all-targets -- -D warnings` clean.

### Test coverage

- 12 unit tests in `src/`.
- All shape validation invariants tested.
- Digest stability tested across runs.
- Serde round-trip for all public types.
- Trait mock compile test.

[Unreleased]: https://github.com/RecursiveIntell/Libraries/tree/main/proveKV/crates/quant-codec-core/compare/quant-codec-core-v0.1.0-alpha.1...HEAD
[0.1.0-alpha.1]: https://github.com/RecursiveIntell/Libraries/tree/main/proveKV/crates/quant-codec-core/releases/tag/quant-codec-core-v0.1.0-alpha.1
