# KV Unresolved Risks

## Codec Scope

- CPU reference compression currently supports per-token FibQuant only.
- Per-channel and KIVI-style policy selections are represented but keep raw fallback blocks in the CPU codec.
- The block unit is one `[head_dim]` vector; grouped vectors and channel-block encodings remain future work.

## Quality Scope

- Quality metrics are synthetic and fixture-driven.
- No model-captured KV cache has been evaluated.
- No perplexity, long-context retrieval, or model-specific layer/head sensitivity receipts exist.

## Runtime Scope

- No Hugging Face, vLLM, FlashInfer, TensorRT-LLM, or CUDA integration is implemented.
- No serving scheduler, allocator, or heterogeneous page backend exists.
- No named-hardware latency or throughput receipts exist.

## Release Scope

- The crate lives inside a larger dirty Git worktree in this environment.
- `cargo publish --dry-run --allow-dirty` passed, but no actual publish action was taken.
- Future release work should run from a clean, intentionally tracked repository state.

## Security And Correctness Scope

- Serialized artifacts are validated for schema, shape, profile, page digest, codebook digest, and rotation digest in the CPU path.
- Backend adapters must preserve those checks rather than trusting serialized page data.
- Approximate decoded values must remain labeled as approximate in downstream systems.
