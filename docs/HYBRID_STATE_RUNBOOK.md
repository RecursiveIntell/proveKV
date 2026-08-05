# Hybrid State Runbook

Operations guide for the proveKV hybrid state runtime.

## Capture

Capture KV cache state from a model into binary pages:

```bash
cd python && uv sync --group dev && cd ..
PATH="python/.venv/bin:$PATH" PYTHONPATH="python" \
  HF_HUB_OFFLINE=1 python3 proveKV/scripts/qwen35_state_capture.py \
  --model Qwen/Qwen2.5-0.5B \
  --device cpu --dtype float32 \
  --tokens 64 \
  --output results/bench/hybrid_state/qwen25/capture
```

Output per run:
- `pages/` — JSON-header + raw-float32 binary pages, one per (layer, K/V)
- `manifest.json` — model identity, token IDs, page digest inventory
- `receipt.json` — machine-readable run evidence

## Replay Gate

Verify pages roundtrip through cache reconstruction and suffix decode:

```bash
PATH="python/.venv/bin:$PATH" PYTHONPATH="python" \
  python3 proveKV/scripts/qwen35_replay_gate.py \
  --capture-dir results/bench/hybrid_state/qwen25/capture/<run-id> \
  --device cpu --baselines 5
```

Checks:
1. N independent baseline forward passes — must all agree on next token
2. Pages reloaded from disk, BLAKE3 payload digests verified
3. DynamicCache reconstructed from pages
4. Suffix tokens replayed through reconstructed cache
5. Frozen tolerance profile computed from baseline jitter

## Rust Store Operations

```rust
use provekv::{
    PageStore, StateStore, HybridStateManifestV1,
    LeaseRights, LeaseRight, StateLease,
};

// Open a page store.
let page_store = PageStore::open("/data/pages")?;

// Write a page atomically (temp → fsync → rename → dir-fsync).
let header = build_page_header(/* ... */);
page_store.write_page(&header, &payload)?;

// Read and validate.
let (header, payload) = page_store.read_page(&digest)?;

// Open a state store.
let mut store = StateStore::open("/data/states")?;

// Commit a root state.
let root_id = store.commit_root(manifest)?;

// Fork with O(1) metadata (no page copies).
let child_id = store.fork(&root_id, child_manifest)?;

// Issue a lease.
let lease = StateLease::new(
    principal, scope, &state_id,
    LeaseRights::empty().with(LeaseRight::Inspect),
    Some(3600_000), // 1-hour TTL
    revocation_epoch, nonce,
)?;
store.register_lease(lease)?;

// GC: collect unreleased, unreachable states.
let report = collect(&mut store)?;
```

## Crash Recovery

On startup:
1. Remove stale `.tmp.*.page` files
2. Validate every committed `.page` file (header + payload digest)
3. Corrupted pages are recorded but not deleted — operator decides quarantine

```rust
let report = recover(&page_store)?;
assert!(report.corrupted_pages.is_empty());
```

## Fork Semantics

- Parent and sibling digests never change on fork
- Pages are shared by reference — only metadata is allocated
- Released states cannot be forked
- Branch lineage is walkable forward (children) and backward (parent)

## GC Policy

- Mark: from every non-released root state, walk children → reachable set
- Sweep: released + unreachable + no active leases → collect pages + state
- Released state with active lease is retained until lease revoked
- Child of reachable parent is reachable even if released

## Lease Authority

- Opaque CSPRNG lease IDs (`lease-v1:<blake3-hex>`)
- Rights: `inspect`, `materialize`, `fork`, `append`, `release`
- Authorization checks: principal, namespace, state ID, rights, expiry, revocation epoch
- Lease digests are safe for logs (non-replayable)
- Least-authority: existence check does not leak cross-principal

## Environment Contract

Pinned dependencies (see `python/pyproject.toml`):
- `torch>=2.5`
- `transformers` (git main, for Qwen3.5 support when available)
- `safetensors>=0.4`
- `blake3>=0.4`

Verify: `PATH="python/.venv/bin:$PATH" pytest python/tests/ -v`

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `qwen3_5` not recognized | Qwen3.5 architecture not in transformers | Use Qwen2.5-0.5B; wait for upstream |
| Digest mismatch on replay | Capture/replay use different hash (SHA vs BLAKE3) | Ensure both use BLAKE3 |
| DynamicCache 3-tuple error | Newer transformers uses `(k, v, None)` tuples | Update adapter — already handled |
| Page not found | Payload digest path mismatch | Check `page_path()` uses same digest as `write_page()` |
| `INTEGRITY_KEY_REQUIRED` | Missing data-dir for persistent store | Pass `--data-dir` or accept volatile mode |
