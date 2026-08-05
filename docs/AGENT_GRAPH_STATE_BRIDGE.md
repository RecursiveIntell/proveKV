# ADR: Agent Graph Model-Invocation Seam

**Status:** Proposed (not implemented)
**Date:** 2026-08-05
**Git commit:** 9295db0

## Context

P0-C proves that Qwen2.5-0.5B state can be captured, persisted, reopened, and
replayed. To integrate this with Agent Graph orchestrated LLM workflows, we need
a generic invocation seam that carries a non-model-visible lease reference to a
local model backend without putting lease material in prompts, graph state,
provider bodies, or logs.

## Current state (2026-08-05)

- `LlmNode` in `agent-graph-mcp/src/nodes.rs` renders JSON into a prompt and
  directly builds `llm_pipeline::LlmCall`/`ExecCtx`.
- `Tool` and `External` graph node types are not executable.
- The default HTTP/OpenRouter provider path must remain compatible.
- No proveKV semantics exist in graph state, prompts, or node output.

## Decision

Add a generic `ModelInvocationExecutor` trait (or invocation-context attachment)
keyed by `(run_id, node_id, attempt_id)`. It carries a non-model-visible lease
reference to a local backend. The existing default path is unchanged.

### Required properties

1. **Lease never enters model-visible data.** Not in rendered prompt, graph
   state, node output, terminal receipt, provider request body, or logs.
2. **Default path preserved.** When no executor is attached, the existing
   HTTP/OpenRouter path operates identically.
3. **Attachment/cleanup/retry lineage.** A fake executor test proves that
   attachment, cleanup, and retry work without any model call.
4. **Security boundary.** Local Unix socket or in-process backend, peer
   credentials, `0700/0600`, short TTL, least rights, revocation, quotas,
   no network listener.

### Interface sketch

```rust
/// Attached to a graph run before node execution.
trait ModelInvocationExecutor {
    /// Called before LlmNode::execute. Returns a backend handle or None
    /// to fall through to the default provider path.
    fn acquire(
        &self,
        run_id: &str,
        node_id: &str,
        attempt_id: u32,
    ) -> Option<BackendHandle>;

    /// Called after node completion (success or failure).
    fn release(&self, handle: BackendHandle);
}

struct BackendHandle {
    lease_digest: String,  // safe for logs
    state_id: String,      // content-addressed, not a bearer secret
    // ... internal backend connection
}
```

### RED gates (must fail)

- Lease appears in rendered prompt
- Expired/cross-principal/sibling/write-widened lease
- Cancelled attempt leaks lease
- Retry reuses mutated failed state
- Model/provider sees lease material
- State store unavailable → typed recompute/unsupported

### GREEN gates

- Fake executor test: attachment/cleanup/retry lineage without model call
- Default graph tests and provider path remain green
- N=2,4,8 local fixture proves controller-side fork/retry/release
- Exact Qwen replay through attached executor

## Rollback

Feature/config flag off. Default: independent calls through existing provider
path. No bridge means no proveKV/graph integration.

## Evidence

- Architecture decision record (this document)
- Source citations from `agent-graph-mcp/src/nodes.rs`, `compiler.rs`,
  `run_manager.rs`, `spec.rs`
- `llm-pipeline/src/exec_ctx.rs`, `llm_call.rs`, backend contracts
- RED/GREEN test plan above

## Claim

A generic invocation seam exists. This does not claim proveKV/graph integration
exists — only that the seam is designed and tested in isolation.
