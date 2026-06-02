# proveKV — System Name, Naming Doctrine, and Public-Facing Brand

> **Status:** internal branding + public claim record. Versioned with the
> workspace. Authoritative on scope of what is and is not unique to this
> project.

This document records the naming decision for the two-tier compressed
KV-cache pool that previously lived as `poly-kv` in this workspace, and
the boundary between what is **the system** (unique, ours) and what is
**a primitive inside the system** (named for its source, not ours).

## The decision

The two-tier pool is renamed to **`proveKV`**.

The codec primitives inside the pool keep their existing names:

- `fib_k4_n32` — the shared-tier (cold) codec. A clean-room Rust port of
  FibQuant (Lee & Kim, arXiv 2605.11478, May 2026). The codec identity
  belongs to the paper; the wire format, the dispatch, and the pool
  integration around it are ours.
- `turbo_8bit` — the per-agent-shell (hot) codec. Identity belongs to
  the upstream `turbo-quant` crate from `RecursiveIntell/Libraries`.

## What is and is not unique

| Layer | Unique? | Reason |
|---|---|---|
| `fib_k4_n32` codec math | **No** | Port of FibQuant (Lee & Kim 2026). Algorithm and core
mathematics are the paper authors'. |
| `fib_k4_n32` wire format and dispatch | **Yes** | The compact binary envelope
(`FibCodeV1::to_compact_bytes`), the 17 μs decode path, and the
4-byte-block compression are this project's engineering. |
| `turbo_8bit` codec | **No** | Identity belongs to the upstream `turbo-quant`
crate. |
| Two-tier pool architecture (shared cold + per-agent hot) | **Yes** | The split,
the policy, and the build/materialize separation are this project's design. |
| Receipted, content-addressed, build-once pool | **Yes** | Every operation
emits a typed receipt; the pool is hash-stable; the audit trail is the
contract the codec is built against. |
| 11.13× lossless at ΔPPL=+0.00% measurement | **Yes** | This measurement,
on this corpus, on these three models, with these state.json receipts,
is this project's evidence. |
| The combined artifact — a two-tier pool with receipted dispatch, content-
addressed manifest, and a measured 11.13× lossless headline | **Yes** | This is
the artifact `proveKV` names. |

## Why "proveKV"

The name is a system name, not a result name. The properties the name
encodes are durable:

- **Provenance.** Every artifact in the pool is content-addressed.
- **Proof.** Every build, materialize, and fallback emits a receipt.
- **Verification.** Receipts are BLAKE3-hashed against the codec profile
  digest; the audit trail is tamper-evident.
- **KV cache.** The system targets the K/V cache, not a generic vector
  store.

The name does *not* encode:

- A specific compression ratio. Numbers move; doctrines don't.
- A specific bit rate. Codecs can be swapped; the system stays.
- A specific model. The pool is model-agnostic.
- A claim that the underlying codec math is novel. The math is FibQuant;
  the integration is ours.

## The non-claims (for the public README)

The following sentences may not appear in user-facing copy, READMEs,
release notes, or X posts about this project, unless they are tied to
a specific external paper claim or local receipt evidence (per the
`AGENTS.md` release-claim law and the existing `poly-kv` scope rules):

- "novel codec" / "new quantization algorithm"
- "first lossless KV cache compression"
- "we invented fib_k4_n32"
- "production-ready"
- "zero-overhead"
- "better than TurboQuant" (without a head-to-head at matched bit rate)
- "drop-in replacement for vLLM / llama.cpp"

What may appear:

- "A two-tier, receipted, content-addressed KV-cache pool"
- "Built on a clean-room Rust port of FibQuant (Lee & Kim 2026)"
- "11.13× compression with ΔPPL=+0.00% on SmolLM2-1.7B, TinyLlama-1.1B, and
  Qwen2.5-0.5B" (tied to `state.json` receipts)
- "The shared-tier codec is `fib_k4_n32`; the per-agent-shell codec is
  `turbo_8bit`"
- "Receipts, content addressing, and exact-fallback are the runtime
  contract"

## Mapping from old names to new names

| Old name | New name | Notes |
|---|---|---|
| `poly-kv` (the pool) | `proveKV` | The system. The pool is what we're naming. |
| `SharedKVPool` | `proveKV::SharedPool` | Type-level rename. |
| `AgentShell` | `proveKV::Shell` | Type-level rename. |
| `PoolBuildReceipt` | `proveKV::BuildReceipt` | Type-level rename. |
| `ShellMaterializeReceipt` | `proveKV::MaterializeReceipt` | Type-level rename. |
| `fib_k4_n32` | unchanged | The codec identity belongs to the paper. |
| `turbo_8bit` | unchanged | Belongs to the upstream `turbo-quant` crate. |
| `poly-kv/examples/poly_kv_fast_roundtrip.rs` | `proveKV/examples/prove_kv_fast_roundtrip.rs` | |
| `poly-kv/scripts/ppl_validate.py` | `proveKV/scripts/ppl_validate.py` | |

The migration is **rename, not rewrite.** All existing tests, examples,
scripts, and `state.json` files keep working; the type-level renames
are introduced behind re-exports so a downstream consumer using
`poly_kv::SharedKVPool` can switch to `prove_kv::SharedPool` at their
own pace.

## The one-paragraph public framing

> **proveKV** is a two-tier, receipted, content-addressed KV-cache pool.
> The shared tier uses the `fib_k4_n32` codec (a clean-room Rust port of
> the [FibQuant paper, Lee & Kim 2026](https://arxiv.org/abs/2605.11478))
> and is validated at 11.13× compression with ΔPPL=+0.00% on
> SmolLM2-1.7B, TinyLlama-1.1B, and Qwen2.5-0.5B. The hot tier uses
> `turbo_8bit` for per-agent shells. The pool is the system; the codecs
> are the primitives.

That framing is honest about what is measured, what is named, and what
is unique. It survives a hostile auditor.

## The X bio / GitHub org description (one sentence each)

- **X bio:** "proveKV — receipted, two-tier, content-addressed KV cache.
  fib_k4_n32 cold + turbo_8bit hot. 11.13× lossless ΔPPL on real LLMs."
- **GitHub org description:** "RecursiveIntell — provenance-first AI
  infrastructure. proveKV ships a two-tier, receipted KV-cache pool
  built on a clean-room Rust port of FibQuant (Lee & Kim 2026)."

## What is NOT changing

- The codec math. `fib_k4_n32` is still FibQuant.
- The receipts. `PoolBuildReceipt`, `ShellMaterializeReceipt`, and
  `FallbackReceiptV1` keep their semantics, only the type-level path
  changes.
- The methodology. `ppl_validate.py` keeps its locked procedure; the
  committed `state.json` files are unchanged.
- The licenses. fib-quant stays Apache-2.0; poly-kv / proveKV stays
  MIT OR Apache-2.0; the standalone proof repo stays MIT.

## Open work (unchanged)

1. ~~Multi-agent validation~~ — done 2026-06-02.
2. Head-to-head vs TurboQuant at matched bit rate (not at matched
   headline).
3. Cross-corpus with a real public corpus (Salesforce/wikitext-2 with
   a different split, c4, pg19).
4. Longer context on a larger GPU (A100 40-80 GB or H100).
5. Multi-agent on a larger model (SmolLM2-1.7B and TinyLlama-1.1B).

These are independent of the rename. They were on the open-work list
before this document and they stay there.

## Revision history

- 2026-06-02 — initial drafting. Decision: `proveKV` for the system;
  codecs keep their source-paper identities.
