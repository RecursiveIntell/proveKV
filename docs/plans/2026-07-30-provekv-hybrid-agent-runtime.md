# ProveKV Hybrid Agent-State Runtime Implementation Plan

> **For Hermes:** Execute this plan task-by-task. Use Agent Graph MCP for LLM-only orchestration when available, but use the owning runtime or isolated coding agents for file, terminal, model, and benchmark work. Never treat an agent self-report, graph completion, or process exit code as model-quality evidence.

**Goal:** Evolve proveKV from a conventional K/V compression reference into a CPU-first, branchable hybrid attention-state runtime for exact pinned model/runtime tuples, beginning with Qwen3.5 full-attention K/V plus Gated DeltaNet convolution/recurrent state while preserving v1 readers and evidence-safe claims.

**Architecture:** `quant-codec-core` remains canonical for typed IDs, digests, dtypes, axes, and cache-shape contracts. proveKV owns immutable pages, persisted manifests, state lineage, copy-on-write branches, leases, accounting, and receipts. A pinned Transformers adapter maps runtime-specific cache objects to canonical tensor components. Agent Graph and Hermes may carry only non-model-visible execution references; a trusted model-invocation control plane resolves them. Graph JSON, prompts, transcripts, and model output are never the tensor data plane.

**Tech stack:** Rust 2021 workspace (`quant-codec-core`, `proveKV`, existing FibQuant/TurboQuant codecs), serde/serde_json for bounded control manifests, BLAKE3 through the canonical digest owner, fixed-width binary or Safetensors tensor payloads, a CPU Python environment locked under `python/`, pinned Hugging Face Transformers/Qwen3.5 revisions, owner-only local filesystem/Unix-domain IPC, and receipt-backed validation.

**Plan status:** hardened by direct source inspection on 2026-07-30. This artifact remains an untracked plan; it is not implementation evidence.

---

## 1. Final planning verdict

The direction is sound. Only P0-A source/evidence closure and P0-B foundational contract/store implementation may begin; **Qwen3.5 model integration is NO-GO until P0-A and P0-B pass**. The previous revision had four material defects:

1. Most crate/root paths were one directory too deep.
2. It proposed duplicate shape/identity authority inside proveKV even though `quant-codec-core` explicitly owns those contracts.
3. It treated Agent Graph nodes as if they could safely exchange/invoke state handles; current nodes are JSON/text LLM nodes and Tool/External nodes are rejected.
4. It mixed content identity with authority and lacked sufficient crash, concurrency, GC, confidentiality, and resource-bound gates.

This revision closes those planning defects. It does **not** claim the runtime exists.

### Hardening orchestration receipt

- A purpose-built council graph validated but could not be registered because Agent Graph had reached its 64-graph limit.
- Reuse of the existing compression council passed policy preflight.
- Run `run-19fb59e3c73-6` then failed before any LLM call with OpenRouter HTTP 402 insufficient credits.
- Receipt facts: `llm_calls=0`, `step_count=0`, durable terminal failure projection.
- Therefore no council findings or consensus are claimed. Only live-source findings were adopted.

---

## 2. Evidence snapshot and claim boundary

**Planning checkpoint:** 2026-07-30.

### Local repository

```text
repo:   /home/sikmindz/proveKV
branch: main
HEAD:   e1344992566107bb5043bb3c7287248c4488e99a
state:  ahead of origin/main by 2; dirty with unrelated sparse-prefill work
```

Observed unrelated changes include `.gitignore`, `turbo-quant` sparse-prefill files, `proveKV/scripts/dump_sparse_prefill_traces.py`, and `results/sparse_prefill/`. The hybrid-runtime implementation must use an isolated worktree/branch and must not stage, reset, or rewrite those files.

### MSI target

- “The MSI” is the work laptop.
- The user states that it has **no discrete graphics**.
- CPU and system RAM are the target. Do not imply that an NVIDIA runtime merely needs repair.
- Historical MSI proveKV source used for the completed size sweep: commit `6038bb262d9908cad5b1bd8c7597662d50dd7daf` with a clean worktree at observation time.
- Historical MSI audit report: six claims validated, 55 tests passed, zero failed. Re-run before implementation; this is not a timeless baseline.

### Temporary MSI size-only observation

```text
/tmp/provekv-msi-rerun-20260730/qwen2.5/summary.json
sha256 bc224c8164ad554adc66da7adabd69a835495ce96d87221dedf8048a71f8136c
```

| Agents | Radii-preserved 4-bit factor* | Radii-lossy 4-bit factor* | Shared pool bytes | Preserved system bytes | Lossy system bytes | Naive bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 15.966087× | 29.453396× | 944,400 | 3,781,968 | 2,050,128 | 60,383,232 |
| 4 | 26.584503× | 49.001800× | 944,400 | 3,785,616 | 2,053,776 | 100,638,720 |
| 6 | 37.182473× | 68.480881× | 944,400 | 3,789,264 | 2,057,424 | 140,894,208 |
| 8 | 47.760058× | 87.891008× | 944,400 | 3,792,912 | 2,061,072 | 181,149,696 |

`*` Historical output called the first mode “lossless,” but both modes quantize angular values. “Lossless” referred only to radii storage and must not be presented as byte-exact tensor reconstruction.

These are reconstructed Qwen2.5 synthetic-artifact **size observations** in `/tmp`. They are not durable certification and do not prove PPL, logits, generated-output equivalence, Qwen3.5 support, live cache sharing, framework memory reduction, or GPU performance.

### Failed PPL attempt

The prior command loaded Qwen2.5-0.5B weights and failed at `.to("cuda")`:

```text
RuntimeError: No CUDA GPUs are available
PPL_RC=1
```

No oracle forward pass, cache capture, compressed roundtrip, PPL result, or quality receipt was produced. This is a device-selection failure, not a proveKV quality result.

---

## 3. Canonical ownership map

| Contract/surface | Canonical owner | proveKV role | Forbidden duplication |
|---|---|---|---|
| Typed model/tokenizer/codec IDs | `quant-codec-core` | Compose/use | New string-ID family in proveKV |
| Digest and dtype shapes | `quant-codec-core` plus deliberate Tier-1 migration | Use canonical law | Parallel JSON canonicalizer |
| Conventional KV and hybrid component layout | `quant-codec-core` | Persist/layout-reference | Independent second hybrid-shape authority |
| Codec math | FibQuant/TurboQuant | Invoke under policy | Copy codec implementations |
| Runtime manifests/pages/lineage/leases | proveKV | Own | Put in `quant-codec-core` |
| Benchmark runners | current proveKV scripts / future `quant-eval` adoption | Emit witnessed receipts | Treat manifests as benchmark truth |
| Graph state/checkpoints | Agent Graph | Cite state IDs/lease digests | Replace graph state with cache state |
| Model invocation backend | `llm-pipeline::ExecCtx`/backend plus Agent Graph `LlmNode` composition | Provide optional adapter | Put prompt/provider semantics in proveKV |
| Tool retry/idempotency/side effects | `llm-tool-runtime` and orchestrator | Cite receipts | Let cache retry re-run tools |
| Transcript/tool calls/results | Hermes runtime | Rebuild cache from truth | Treat cache as transcript truth |
| Public numerical claims | root `CLAIMS.json` + receipts | Propose only after gates | Hand-edit unreceipted headlines |

### Source inventory checked

- `Cargo.toml` — workspace and path-bound local dependencies.
- `proveKV/src/{lib,shape,pool,shell,manifest,receipt,error}.rs`.
- `proveKV/Cargo.toml`, examples, scripts, and tests.
- `quant-codec-core/src/{ids,digest,dtype,shape}.rs` and README ownership contract.
- Root `README.md`, `REPRODUCE.md`, `CLAIMS.json`, and `docs/STATE_JSON_SCHEMA.md`.
- FibQuant KV inventory/ABI/adapter plans and unresolved-risk documents.
- `docs/INTEGRATION_TIER1_STACK_IDS_BOUNDARY_COMPILER.md`.
- `docs/plans/2026-07-01-minference-provekv-bridge.md`.
- Agent Graph MCP `nodes.rs`, `compiler.rs`, `spec.rs`, and `run_manager.rs`.
- llm-pipeline `exec_ctx.rs`, `llm_call.rs`, backend modules, retry, parsing, and receipts.
- llm-tool-runtime contracts for retry owner, idempotency, approval, and side-effect classes.
- Upstream Transformers Qwen3.5 model/cache source observed at `main` commit `71c6f699ac9b3f8fc42a6a3e9dc59034c349a678`: `modeling_qwen3_5.py` SHA-256 `190f2eb865a23e79b9bf54a42c3e109a85510a1827a55d65aebb26d120e902dd`; `cache_utils.py` SHA-256 `ad16cd7042e0de6fbff546062b394917a7f6fb04e54af0814039189489391147`. These hashes document this audit only; implementation must pin its own exact package/model tuple.

### Adjacent-plan convergence

The MInference/sparse-prefill plan stays under `turbo-quant`: it selects retained attention reads and has separate quality/kernel gates. Hybrid-state persistence is not its owner and does not depend on it for P0. Integration is considered only after both tracks independently pass their quality gates.

---

## 4. Scope lock

### In scope

1. Canonical hybrid full-attention, convolution, and recurrent component contracts.
2. Immutable page storage, O(1) metadata fork, COW overlays, leases, recovery, GC, and accounting.
3. Batch-size-1 CPU float32 Qwen3.5 capture → persist → reopen → fork → append → replay.
4. Exact source/model/tokenizer/template/adapter/execution identity.
5. Negative, corruption, concurrency, restart, and branch-isolation tests.
6. A trusted model-invocation attachment seam for future Agent Graph/Hermes integration.
7. Separate per-component codec admission and evidence-safe claim generation.

### Hard no list

- No GPU work or GPU claims on the MSI.
- No JSON float tensor blobs, pickle, or untrusted `torch.load`.
- No mutable shared tensors across branches.
- No content hash used as an authorization token.
- No lease/capability value in prompts, graph state, model output, logs, receipts, or filenames.
- No network-exposed bridge in v1.
- No cross-principal deduplication or state sharing.
- No unknown backend/layer fallback that silently drops state.
- No batch, beam, sampling, or multi-sequence claim before their mutable execution state is modeled.
- No “lossless K/V” wording for a quantized angular codec.
- No public quality/memory/speed claim from exit code, compressed file size, or model load alone.
- No v1 artifact rewrite in place.
- No direct commits or destructive reset in the current dirty worktree.

---

## 5. Architectural invariants

### 5.1 State is a rebuildable projection

Canonical truth remains messages, native tool calls, matching tool results/call IDs, model/template identity, graph/checkpoint state, execution permits, and side-effect receipts. A proveKV state ID may be cited by a run receipt but never replaces those records.

### 5.2 Three reuse modes remain distinct

1. `prompt_recognition_recompute`: same token prefix recognized; backend recomputes it.
2. `materialized_prefix_reuse`: persisted state is restored into a backend-owned cache.
3. `shared_live_pages`: backend directly consumes shared immutable pages plus branch overlays.

The Qwen3.5 CPU MVP targets mode 2 only. Mode 3 requires a separate backend-specific proof.

### 5.3 Canonical hybrid layout vs runtime manifest

`quant-codec-core` gains a dependency-light `HybridCacheLayoutV1` composed from existing IDs, dtype, axis, and shape primitives. Qwen-specific runtime names map into generic controlled component kinds such as full-attention key/value, convolution state, and recurrent state. Empty or unknown runtime-specific labels fail closed.

proveKV owns `HybridStateManifestV1`, which binds canonical layout, model/execution identity, immutable page references, lineage, policy decisions, and digests. The current `docs/STATE_JSON_SCHEMA.md` remains a benchmark receipt schema; it is not overloaded as the state manifest.

### 5.4 Content identity is not authority

- `StateId`: deterministic content/lineage identity; grants no access.
- `StateLease`: random opaque lease ID plus principal, namespace, run/node/attempt, state ID, allowed operations, issue/expiry, and revocation epoch.
- Logs/receipts may store only a non-replayable lease digest.
- Lease checks occur before state lookup/materialization so existence is not leaked across principals.

### 5.5 Compatibility identity

A reusable state binds at minimum:

- immutable model repository revision;
- config digest;
- weights index and weight-file digests;
- PEFT/LoRA adapter set, order, merge state, and digests;
- tokenizer and chat-template digests;
- exact prefix token IDs and attention-mask digest;
- position IDs, cache position, and RoPE/scaling configuration;
- layer/component inventory, axes, shapes, dtypes, and layout digest;
- Transformers/backend version and source digest;
- batch size and execution mode;
- state schema and codec-profile digests.

Any mismatch returns a typed rejection. There is no “close enough” reuse.

### 5.6 MVP execution boundary

Initial certification is batch size 1, CPU, float32, `eval()` plus `inference_mode()`, greedy deterministic decode, and one pinned Qwen3.5-2B/Transformers tuple. Sampling requires captured per-branch RNG state; beam search requires reorder/beam state. Both remain unsupported until explicit tasks are added.

### 5.7 Control plane vs tensor data plane

JSON carries bounded manifests, statuses, and receipts only. Tensor payloads use canonical contiguous CPU bytes in fixed-width binary pages or Safetensors. Headers declare magic, schema, endianness, dtype, rank, dimensions, axes, byte length, codec/profile, model/layout identity, and payload/header digests before allocation.

Python stages capture files. Rust verifies bounds and digests and commits the manifest last. Unknown formats, object arrays, pickle metadata, oversized dimensions, overflow, trailing bytes, and unreferenced payloads are rejected.

### 5.8 Crash consistency and recovery

- Immutable pages are written to same-filesystem temporary files.
- Validate, `fsync` file, atomic rename, then `fsync` directory.
- Commit the manifest only after every referenced page is durable.
- A derived reachability/refcount index is never deletion authority.
- Startup removes uncommitted temporaries, validates committed manifests, rebuilds reachability, and quarantines corruption.
- GC is mark-and-sweep from committed manifests plus pinned/unexpired leases.

### 5.9 Concurrency

Define and document lock order. Concurrent fork/append/release/lease-expiry/GC must not mutate a parent/sibling, delete reachable pages, resurrect revoked state, or produce divergent IDs for the same canonical content. Run deterministic stress tests and failure injection; use `loom` only if it pays for a specific concurrency invariant.

### 5.10 Confidentiality and least authority

Cache state is confidential prompt-derived data. Default local modes are directory `0700`, files/socket `0600`, Unix-domain only, peer-credential checked, one OS user/principal namespace, and no cross-principal dedupe. Do not claim secure deletion; document filesystem, swap, and page-cache limitations. Network serving, multi-user tenancy, encryption-at-rest, and remote attestation are separate future threat models.

### 5.11 Resource bounds

Predeclare and enforce with checked arithmetic before allocation:

- manifest/header bytes;
- page count and page bytes;
- component/layer count;
- rank and each dimension;
- total tensor/state bytes;
- branch depth;
- live leases and states per principal;
- request concurrency and decode/materialization budget.

Limit violations are typed and do not partially commit state.

### 5.12 Replay acceptance law

1. Run repeated independent baselines first.
2. Freeze and hash an acceptance profile without looking at candidate replay results.
3. Raw component bytes and manifest digests must round-trip exactly.
4. Record logit max-absolute/max-relative error, top-k agreement, exact greedy token IDs, and full generated sequence.
5. Test one-shot suffix, token-by-token decode, process restart, wrong component/layer/position, and divergent branches.
6. A practical float32 ceiling is `max(1e-6, 10 × observed baseline jitter)` absolute and `max(1e-5, 10 × jitter)` relative; freeze the baseline-derived values before candidate replay.
7. Codec admission uses frozen development/calibration data and one held-out evaluation. Failed profiles are quarantined, not tuned on holdout.

---

## 6. Implementation phases and tasks

Every mutating task requires: exact owner/worktree, entry gate, RED failure, minimal GREEN, package-qualified checks, evidence path, migration/rollback, and narrow claim. No phase advances on skipped or degraded required checks.

## Phase P0-A — source, evidence, and environment closure

### Task 0: Create isolated implementation lanes

**Owner/worktrees:**

- `/home/sikmindz/proveKV` for codec/runtime work.
- Separate `/home/sikmindz/Coding/Libraries` worktree only if Agent Graph/llm-pipeline changes are later authorized.

**Entry:**

```bash
cd /home/sikmindz/proveKV
git status --short --branch
git rev-parse HEAD
git diff --check
cargo metadata --no-deps --format-version 1
```

**RED:** implementation starts without recording HEAD, dirty paths, worktree path, Cargo metadata, and governing instructions.

**GREEN:** create a dedicated branch/worktree containing no sparse-prefill modifications; record parent/sibling status and path-dependency revisions.

**Evidence:** `results/implementation/hybrid_state/<run-id>/source.json`, `commands.log`, and checksums.

**Rollback:** remove only the isolated worktree/branch. Never `git reset --hard` the existing checkout.

**Claim:** source boundary identified; no feature claim.

### Task 1: Durable-copy or retire the MSI size observation

**Files:**

- Create immutable run directory under `results/bench/multi_agent/qwen2.5-0.5b/msi-cpu/<run-id>/` only after copying from MSI.
- Update `REPRODUCE.md`/`CLAIMS.json` only if the complete recipe validates.

**Required packet:** source commit/worktree, original binary manifest digests, reconstruction utility digest, exact commands/logs, all eight state files, summary, environment, and checksums.

**RED:** missing source artifact, command, digest, or geometry rejects promotion from `/tmp`.

**GREEN:** independently recompute summary and checksum; rename legacy modes in derived reporting to `radii_preserved_4bit` and `radii_lossy_4bit` while preserving raw historical fields.

**Evidence:** durable packet plus validator output.

**Rollback:** retain it as temporary observed evidence or delete the local copy; do not elevate partial artifacts.

**Claim:** exact size-only result on the named reconstructed corpus, if promoted.

### Task 2: Add CPU-safe capability and dtype probing

**Files:**

- Modify `proveKV/scripts/ppl_multi_agent.py`, `ppl_validate.py`, `ppl_validate_shell.py`.
- Create `proveKV/scripts/runtime_capability.py`.
- Create `proveKV/tests/test_runtime_capability.py`.

**Required CLI:** `--device auto|cpu|cuda` and `--dtype auto|float32|float16|bfloat16`.

**RED:** explicit CUDA on MSI fails before model load with a typed capability result. Invalid CPU float16/bfloat16 combinations fail rather than silently cast.

**GREEN:** `auto` selects CPU on MSI; receipt records requested/resolved device/dtype, allocation probe, torch/Python versions, CPU/thread settings, and capability status.

```bash
python3 -m pytest proveKV/tests/test_runtime_capability.py -q
ssh msi 'python3 ~/proveKV/proveKV/scripts/ppl_multi_agent.py --help'
```

**Evidence:** `results/bench/runtime-capability/<run-id>/`.

**Rollback:** revert script behavior; preserve historical failed CUDA log.

**Claim:** explicit device/dtype selection only.

### Gate P0-A

```bash
cd /home/sikmindz/proveKV
cargo metadata --no-deps --format-version 1
cargo fmt --all --check
cargo test --workspace --all-features
./prove_audit.sh
```

Pass only when the isolated worktree is clean except intended files, existing warnings are classified, the CPU capability test passes, and the MSI evidence is either durable/promoted or explicitly temporary.

---

## Phase P0-B — canonical contracts, authority, persistence, and COW

### Task 3: Extend the canonical hybrid layout contract

**Owner:** `quant-codec-core`; proveKV only composes the result.

**Files:**

- Create `quant-codec-core/src/hybrid.rs`.
- Modify `quant-codec-core/src/lib.rs`, `error.rs`, and only required existing type modules.
- Create `quant-codec-core/tests/hybrid_shape_validation.rs` and serde/digest fixtures.
- Modify `proveKV/Cargo.toml` to use the workspace dependency deliberately.

**Contract:** controlled component/axis/sequence-semantics types plus `HybridCacheLayoutV1`; use existing `ModelFingerprint`, `TokenizerFingerprint`, `ArtifactDigest`, and `DType`.

**RED:** reject zero dimensions, overflow, duplicate layer/component IDs, wrong layer order, missing required components, empty runtime labels, unsupported dtype/layout, and noncanonical axis order.

**GREEN:** conventional GQA and synthetic hybrid fixtures validate; serde and digest fixtures remain stable across map insertion order and process runs.

```bash
cargo test -p quant-codec-core --test hybrid_shape_validation
cargo test -p quant-codec-core
cargo clippy -p quant-codec-core --all-targets -- -D warnings
```

**Evidence:** canonical fixtures and test logs under `target/kv-production-receipts/hybrid/schema/`.

**Migration:** no rewrite of `KvCacheShapeV2` or existing proveKV `KvTensorShape`. Add explicit adapters and deprecation path only after parity tests.

**Rollback:** remove only the additive hybrid module/export.

**Claim:** canonical hybrid layout validation exists; no runtime/model support.

### Task 4: Add proveKV manifest and compatibility identity

**Files:**

- Create `proveKV/src/hybrid_manifest.rs` and `state_id.rs`.
- Modify `proveKV/src/lib.rs`, `manifest.rs`, `receipt.rs`, and `error.rs`.
- Create `proveKV/tests/hybrid_manifest_identity.rs`.

**Artifact:** `HybridStateManifestV1` / `provekv_hybrid_state_manifest_v1`, distinct from benchmark `state.json`.

**Identity fields:** all fields in §5.5, parent state ID, component page digests, policy decisions, creation source, and canonical schema digest.

**RED:** vary each identity field independently; open/fork/materialize/replay must reject the specific mismatch. Reject missing model revision, mutable branch labels as content identity, unordered adapters, and unbounded strings.

**GREEN:** deterministic state ID for identical canonical input and different ID for every semantic mismatch.

```bash
cargo test -p provekv --test hybrid_manifest_identity
cargo test -p provekv
```

**Evidence:** accepted/rejected manifests plus mismatch field receipts.

**Migration:** v1 pool/shell readers stay unchanged. New artifacts use a separate discriminator/directory and have no implicit v1 downgrade.

**Rollback:** feature-gate/remove new manifest exports; preserve all v1 readers.

**Claim:** identity-bound hybrid manifests; not backend cache handles.

### Task 5: Separate state leases from content identity

**Files:**

- Create `proveKV/src/lease.rs`, `principal.rs`, and `limits.rs`.
- Modify `proveKV/Cargo.toml` for an audited CSPRNG dependency such as `getrandom`; do not implement IDs with timestamps or deterministic hashes.
- Create `proveKV/tests/lease_authority.rs` and `resource_limits.rs`.

**Lease:** 256-bit CSPRNG opaque ID, principal, namespace, run/node/attempt, state ID, rights (`inspect`, `materialize`, `fork`, `append`, `release`), issue/expiry, revocation epoch.

**RED:** content hash used as credential; expired/revoked/cross-principal/cross-namespace/sibling/write-widened lease; state-existence probe; lease serialization into model-visible data.

**GREEN:** least-authority check precedes state lookup; logs expose only lease digest; revocation and expiry are atomic. Each principal uses an isolated store root (or a principal-salted storage namespace) so page deduplication cannot cross principals.

**Evidence:** permission matrix and negative audit log with redacted/non-replayable identifiers.

**Rollback:** keep runtime in-process and single-owner with no bridge. Never fall back to state-ID authority.

**Claim:** local least-authority lease contract for tested operations.

### Task 6: Implement bounded binary page persistence and recovery

**Files:**

- Create `proveKV/src/page_format.rs`, `page_store.rs`, `recovery.rs`.
- Create `proveKV/tests/page_format.rs`, `page_corruption.rs`, `crash_recovery.rs`.

**Header:** magic, schema, component kind, axes/shape/dtype/endianness, model/layout digest, position span, codec/profile, payload length/digest, header digest.

**RED/failure injection:** header/payload bit flip, truncation, trailing bytes, overflow dimensions, oversized allocation, wrong endian/dtype, unknown codec, pickle/object payload, temp-file crash, page durable but manifest absent, manifest references missing page, directory-fsync omission simulation.

**GREEN:** bounded parse before allocation; canonical contiguous raw bytes; file fsync → rename → directory fsync; manifest-last commit; restart quarantine/recovery; no partial visible state.

```bash
cargo test -p provekv --test page_format
cargo test -p provekv --test page_corruption
cargo test -p provekv --test crash_recovery
```

**Evidence:** `target/kv-production-receipts/hybrid/pages/<run-id>/`.

**Migration:** historical JSON remains read-only fixture/benchmark data. No automatic rewrite.

**Rollback:** disable new store and use v1 pool/shell; never accept an unverified page.

**Claim:** durable bounded pages with corruption/restart rejection; not model replay.

### Task 7: Implement immutable store, O(1) forks, concurrency, and GC

**Files:**

- Create `proveKV/src/state_store.rs`, `branch.rs`, `gc.rs`.
- Create `proveKV/tests/state_branching.rs`, `concurrent_lifecycle.rs`, `gc_reachability.rs`.

**API rules:** parent/page references immutable; fork creates metadata/overlay only; append returns a new state ID; release/revocation updates derived indexes; no mutable page API.

**RED:** eight concurrent branches mutate unique suffixes while GC, expiry, release, and restart occur. Parent/sibling digests must never change; reachable pages must never disappear; revoked states must not resurrect.

**GREEN:** explicit lock order, transaction boundary, manifest-last commit, reachability rebuild, and deterministic IDs. Measure fork allocations to prove it does not clone prefix payloads.

**Evidence:** per-branch lineage, shared/copied page counts, allocation trace, concurrency seed, and recovery receipt.

**Rollback:** feature-gate store; v1 `SharedKVPool`/`AgentShell` remain available.

**Claim:** O(1)-metadata fork and branch isolation in the Rust reference store.

### Task 8: Add component-specific codec and fallback policy

**Files:**

- Create `proveKV/src/state_policy.rs`.
- Narrowly reuse codec/page functions from `pool.rs`/`shell.rs`.
- Create `proveKV/tests/state_policy.rs`.

**Modes:** `raw_exact`, `radii_preserved_4bit`, `radii_lossy_4bit`, and future named profiles. Neither 4-bit mode may advertise exact tensor reconstruction.

**Policy:** full-attention K/V may use admitted profiles; convolution/recurrent state defaults to raw; unknown components fail; lossy profiles require exact component/model/workload admission receipt.

**RED:** apply a K/V profile to recurrent state, request unadmitted lossy mode, use stale profile/model digest, or claim `lossless=true` for quantized values.

**GREEN:** one explicit decision/fallback per component included in the manifest digest.

**Evidence:** policy fixtures covering every component kind, selected profile, fallback reason, profile digest, and rejection class.

**Rollback:** raw pages for every v1 hybrid component.

**Claim:** explicit component policy only.

### Gate P0-B

```bash
cargo test -p quant-codec-core
cargo test -p provekv --test hybrid_manifest_identity
cargo test -p provekv --test lease_authority
cargo test -p provekv --test page_corruption
cargo test -p provekv --test crash_recovery
cargo test -p provekv --test state_branching
cargo test -p provekv --test concurrent_lifecycle
cargo test -p provekv --test gc_reachability
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Pass only when v1 tests remain green, all negative/failure-injection cases fail closed, and no duplicate canonical contract exists.

---

## Phase P0-C — pinned Qwen3.5 CPU vertical slice

### Task 9: Pin the Python/model execution environment

**Files:**

- Create `python/pyproject.toml` and generated `python/uv.lock`.
- Create `python/provekv_transformers/versions.py` and `python/tests/test_environment_contract.py`.
- Create `results/bench/hybrid_state/environment-schema.json` if a shared receipt schema is needed.

**Entry:** inventory MSI Python, `uv`, torch, Transformers, CPU ISA, RAM/swap, thread settings, cached model revisions, and model-file digests. Do not install or download silently.

**RED:** mutable Transformers `main`, unresolved model revision, missing weight digest, unsupported Python/torch tuple, non-CPU wheel, `trust_remote_code=true`, or absent offline model.

**GREEN:** lock exact packages/hashes and a specific Transformers revision known to expose the inspected Qwen3.5 cache structures. Record all model/tokenizer/template/weight files. `HF_HUB_OFFLINE=1` succeeds.

```bash
uv lock --project python
uv sync --project python --locked
uv run --project python pytest python/tests/test_environment_contract.py -q
```

If `uv` or the pinned model is absent on MSI, return `blocked` and stop. Do not use ad-hoc mismatched `pip`.

**Evidence:** environment lock, model inventory, source digests, and offline-load receipt.

**Rollback:** delete only the isolated virtual environment; retain lock and blocked receipt.

**Claim:** reproducible CPU environment for the exact tuple.

### Task 10: Build the pinned Transformers capture adapter

**Files:**

- Create `python/provekv_transformers/{__init__,qwen35_adapter,fingerprint,tensor_io}.py`.
- Create `python/tests/test_qwen35_adapter.py`.
- Create `proveKV/scripts/qwen35_state_capture.py`.

**Required mapping:** dynamic/sliding full-attention K/V; Gated DeltaNet `conv_states` and `recurrent_states`; initialized flags; `has_previous_state`; `record_past`; position/cache metadata; exact runtime layer inventory.

**Data plane:** Safetensors or canonical binary pages only. No tensor values in JSON and no pickle.

**RED:** incomplete fake cache, unknown cache-layer class, wrong model/layout fingerprint, omitted recurrent/conv state, unexpected `number_of_states`, noncontiguous/unbounded tensor, or unsupported batch.

**GREEN:** offline CPU capture of a short prefix produces verified pages plus `HybridStateManifestV1` and a complete layer/component inventory.

```bash
HF_HUB_OFFLINE=1 uv run --project python python proveKV/scripts/qwen35_state_capture.py \
  --model Qwen/Qwen3.5-2B \
  --revision <immutable-revision> \
  --device cpu --dtype float32 --batch-size 1 --tokens 64 \
  --output results/bench/hybrid_state/qwen35/capture/<run-id>/
```

**Evidence:** source/model/tokenizer/template/weights/adapters, token IDs, component shapes/dtypes, initialized flags, state/layout digests, commands, and checksums.

**Rollback:** adapter remains outside default Rust features. Unknown upstream change disables capture rather than reinterpreting it.

**Claim:** complete capture for one pinned tuple only.

### Task 11: Freeze baseline tolerance and prove raw persist/reopen/replay

**Files:**

- Create `python/provekv_transformers/qwen35_replay.py`.
- Create `python/tests/test_qwen35_replay.py`.
- Create `proveKV/scripts/qwen35_replay_gate.py`.
- Create immutable `python/tests/fixtures/qwen35_acceptance_v1.json` after baseline-only calibration.

**Protocol:** five independent full baselines → freeze/hash tolerance → prefix capture → Rust persist/reopen → cache reconstruction → one-shot suffix and token-by-token decode → process restart → independent full comparison.

**Required:** raw tensor bytes/digests exact; float32 logits within frozen profile; exact greedy token IDs and generated sequence; record top-k agreement. Test wrong position, mask, component, layer order, dtype, source revision, adapter set, and `has_previous_state`.

**RED:** omitted/swapped recurrent state, wrong cache position, wrong mask/template/model, or replay result used to tune acceptance threshold.

**GREEN:** all positive fixtures pass and every mismatch fails before or during replay with a typed classification.

```bash
uv run --project python pytest python/tests/test_qwen35_replay.py -q
HF_HUB_OFFLINE=1 uv run --project python python proveKV/scripts/qwen35_replay_gate.py --device cpu
```

**Evidence:** `results/bench/hybrid_state/qwen35/replay/<run-id>/` with frozen acceptance digest, raw comparisons, logs, and process-restart receipt.

**Rollback:** capture-only mode or recompute fallback; no replay claim.

**Claim:** materialized-prefix replay for the exact pinned tuple.

### Task 12: Prove divergent branch replay and measured accounting

**Files:**

- Create `python/tests/test_qwen35_branching.py`.
- Create `proveKV/scripts/qwen35_branch_gate.py`, `measure_hybrid_runtime.py`.
- Create/modify `proveKV/src/accounting.rs` and receipts.

**Protocol:** one prefix; N=2,4,8 forks; distinct suffix per branch; independent full-run oracle per branch; concurrent and sequential runs; parent inspection after all writes.

**Accounting:** raw/encoded/page-index/manifest bytes, decoded shared/branch bytes, framework cache bytes, process peak RSS/PSS where available, page-cache caveat, prefill/decode/materialize/fork time, thread count, warm/cold run, and repeated-run distribution.

**RED:** branch mutation changes parent/sibling; size-only receipt populates PPL/RSS/latency; page-cache or framework allocations are counted as proveKV compression.

**GREEN:** every branch passes the frozen replay profile and accounting fields are observed or explicitly unavailable—never estimated silently.

**Evidence:** aggregate lineage/comparison/accounting receipt plus raw `/proc`/`time` logs.

**Rollback:** disable branching and use independent full runs.

**Claim:** branch-isolated materialized reuse and CPU measurements for the exact run.

### Gate P0-C

Pass only when all required full-attention, convolution, recurrent, flag, and position components are captured; raw bytes reopen exactly; replay/branch tests pass the frozen profile; wrong identities fail closed; and the full run is reproducible offline after process restart.

No receipt means no Qwen3.5 claim.

---

## Phase P1 — quality admission and governed orchestration integration

### Task 13: Calibrate per-component codecs

**Files:**

- Modify `proveKV/src/state_policy.rs`.
- Create `proveKV/scripts/qwen35_component_calibration.py`.
- Create `python/tests/test_qwen35_component_quality.py`.

**Protocol:** frozen development/calibration/held-out prompt families; raw baseline; component-at-a-time profile sweep; reconstruction metrics; logits/top-k/token/sequence metrics; PPL/task metrics only when a valid harness exists; predeclared denominators/stopping/rollback.

**RED:** deliberately degraded recurrent state must fail and remain quarantined. No retuning on holdout.

**GREEN:** only an admitted component/profile/model/runtime/workload tuple is selectable. Unknown/stale/revoked profiles use raw fallback.

**Evidence:** calibration and held-out receipts with profile/model/layout/source digests.

**Rollback:** revoke profile; raw state remains the baseline.

**Claim:** exact admitted tuple only; never generic losslessness/PPL neutrality.

### Task 14A: Prove the Agent Graph model-invocation seam before building a bridge

**Owner/worktree:** separate `/home/sikmindz/Coding/Libraries` lane.

**Verified current state:** `LlmNode` renders JSON into a prompt and directly builds `llm_pipeline::LlmCall`/`ExecCtx`; Tool/External graph nodes are not executable. Therefore a proveKV MCP tool or prompt-carried lease is invalid.

**Files to inspect/plan precisely:**

- `agent-graph-mcp/src/nodes.rs`, `compiler.rs`, `run_manager.rs`, `spec.rs` and tests.
- `llm-pipeline/src/exec_ctx.rs`, `llm_call.rs`, backend request/response contracts.

**Required design:** a generic trusted `ModelInvocationExecutor` or invocation-context attachment keyed by `(run_id, node_id, attempt_id)`. It carries a non-model-visible lease reference to a local Qwen3.5 backend. It must preserve the existing default HTTP/OpenRouter path and must not add proveKV semantics to graph state.

**RED:** lease appears in rendered prompt, serialized graph state, node output, terminal receipt, provider request body, or logs; current runtime pretends Tool/External nodes can execute it.

**GREEN:** a fake executor test proves attachment/cleanup/retry lineage without any model call; default graph tests and provider path remain green.

**Evidence:** architecture decision record `docs/AGENT_GRAPH_STATE_BRIDGE.md`, source citations, RED/GREEN tests, and rollback flag.

**Rollback:** no bridge. Continue independent model calls.

**Claim:** a generic invocation seam exists; no proveKV/graph integration.

### Task 14B: Add local Qwen3.5 executor and controller-side fork/retry

**Prerequisite:** Task 14A and P0-C pass.

**Design:** graph controller/executor side table maps run/node/attempt to a lease. Fan-out forks before node invocation; retries use an explicit pre-call checkpoint; completion releases attempt leases; joins perform a fresh model call from canonical branch outputs. Graph JSON contains no bearer secret or tensor metadata beyond safe state lineage digests.

**Security:** local Unix socket or in-process backend, peer credentials, `0700/0600`, short TTL, least rights, revocation, quotas, no network listener.

**RED:** expired/cross-principal/sibling/write-widened lease; cancelled attempt leak; retry reuses mutated failed state; model/provider sees lease; state store unavailable.

**GREEN:** N=2,4,8 local fixture proves controller-side fork/retry/release and exact Qwen replay. Unavailable store returns recompute or typed unsupported according to policy.

**Evidence:** per-node lineage and executor audit receipt with lease digests only.

**Rollback:** feature/config off; default independent calls.

**Claim:** tested local graph fan-out through the pinned executor, not general Agent Graph cache support.

### Task 15: Bind checkpoints without taking tool-side-effect authority

**Owners:** Agent Graph owns graph checkpoints/retries; `llm-pipeline` owns model-call parsing/semantic retry; `llm-tool-runtime` owns tool retry/idempotency/side-effect metadata; Hermes owns transcript/tool receipts. proveKV owns none of them.

**Files:** identify the actual Hermes invocation owner before editing. Do not create a guessed `proveKV/integrations/` authority. Expected source surfaces include Agent Graph run manager/checkpoints, llm-pipeline retry/receipts, and llm-tool-runtime contracts/tests.

**Required ordering:** checkpoint → fork model state → model call → parse/validate → authorize/idempotency-check tool → effect receipt → transcript commit. A parser/model retry may fork from a pre-call state; it may not replay a completed non-idempotent tool.

**RED:** parser failure after a side effect repeats the tool; model-state rollback rewinds transcript/effect truth; retry ownership ambiguous; attempt lease survives cancellation indefinitely.

**GREEN:** exact retry owner, model attempt ID, checkpoint ID, state ID, lease digest, tool idempotency key, parser outcome, and effect receipt are linked without duplicate truth.

**Evidence:** cross-repo integration tests plus a joined lineage receipt covering model retry, parser retry, cancellation, idempotent tool retry, and rejected non-idempotent replay.

**Rollback:** disable state reuse; retain canonical retries/tool receipts.

**Claim:** checkpoint-safe projection for the tested local integration path.

### Gate P1

Pass only when per-component admission is held-out green; the generic invocation seam is real; leases never enter model-visible or graph-serialized data; retries preserve parent/sibling and side-effect truth; cancellation/expiry cleanup passes; and the default non-proveKV graph/provider path remains compatible.

---

## Phase P2 — backend probes and public closure

### Task 16: Probe backend capabilities before adapters

**Files:**

- Create `docs/BACKEND_CAPABILITY_MATRIX.md`.
- Create `proveKV/src/backend_capability.rs` only for generic typed capability records.
- Create one source-bound probe per adopted backend.

**Fields:** external-cache acceptance, immutable-page views, fork, full-attention state, recurrent restore, CPU, direct page views, position/mask restoration, batch/beam support, version/source evidence, and fallback mode.

**RED:** an `unknown` or source-only capability marked supported; a mode-2 materialization path reported as mode 3 live pages.

**GREEN:** witnessed capability or `unknown`; unknown selects recompute/materialize/unsupported explicitly.

**Evidence:** source/API locator, immutable version/revision, probe command/output, capability result, and fallback selection per matrix row.

**Rollback:** no adapter when capability is unknown.

**Claim:** exact matrix entry only.

### Task 17: Reconcile schemas, claims, docs, and runbook

**Files:**

- Modify root `CLAIMS.json`, `README.md`, `REPRODUCE.md` only from receipts.
- Keep `docs/STATE_JSON_SCHEMA.md` scoped to benchmark receipts or version it deliberately.
- Create `docs/HYBRID_STATE_MANIFEST.md` and `docs/HYBRID_STATE_RUNBOOK.md`.

**RED:** claim checker rejects unreceipted headline, “lossless” quantized wording, generic Qwen/backends, production, zero-copy, GPU, or PPL claims.

**GREEN:** generated/source-backed wording cites model/runtime/device/workload/run and receipt digests; blocked/degraded runs remain visible.

**Evidence:** claim-ledger diff, receipt-to-claim mapping, auditor rerun log, docs/schema consistency report, and final artifact checksum manifest.

**Rollback:** last verified M2/reference wording; keep experimental receipts quarantined.

**Claim:** only exact source-backed statements in the ledger.

### Gate P2

There is no aggregate “backend ready” or “production ready” gate. Each backend/model/mode qualifies independently. Public closure also requires a clean exact source commit, complete durable artifact packet, full audit, and claim-ledger reconciliation.

---

## 7. Failure-injection matrix

| Failure | Required result | Quarantine/rollback |
|---|---|---|
| Wrong model/tokenizer/template/adapter | Reject before page decode | Recompute |
| Wrong cache position/mask/RoPE | Reject typed mismatch | Recompute |
| Missing conv/recurrent state | Reject incomplete layout | No hybrid claim |
| Header/payload corruption | Reject page; preserve valid state | Quarantine artifact |
| Truncated/trailing/oversized page | Reject before allocation/commit | Quarantine request |
| Crash before page rename | No visible page | Remove temp on recovery |
| Crash after page rename, before manifest | Page unreachable and GC-safe | Recover/collect later |
| Manifest references missing page | Manifest invalid | Quarantine state |
| Concurrent fork/append/release/GC | Parent/siblings/reachable pages stable | Disable COW runtime |
| Expired/revoked/cross-principal lease | Reject without existence leak | Recompute/deny |
| Cancelled graph node | Attempt lease released/expired | Independent retry |
| Parser retry after tool effect | Tool not repeated | Block integration |
| State store unavailable | Typed recompute/unsupported | Default provider path |
| Candidate exceeds frozen tolerance | Fail/quarantine profile | Raw fallback |
| Agent Graph/provider unavailable | Operational failure only | Direct source audit/manual gate |

---

## 8. Artifact conventions

Do not overwrite historical results.

```text
results/implementation/hybrid_state/<run-id>/
results/bench/hybrid_state/
  environment/<run-id>/
  qwen35/capture/<run-id>/
  qwen35/replay/<run-id>/
  qwen35/branch/<run-id>/
  qwen35/accounting/<run-id>/
  qwen35/calibration/<run-id>/

target/kv-production-receipts/hybrid/
  schema/
  manifests/
  pages/
  leases/
  recovery/
  branches/
```

Every durable run packet contains:

```text
state.json                    # run status/metrics, not the state manifest
hybrid_state_manifest.json    # when applicable
commands.log
environment.json
source.json
acceptance.json               # frozen before candidate evaluation
checksums.txt
stdout.log
stderr.log
```

Status is one of `observed`, `verified`, `blocked`, `degraded`, `quarantined`, or `rejected`. A blocked CUDA run cannot be encoded as a CPU/GPU success.

Sensitive payloads and lease values never enter receipts. Receipts contain state/page/source digests and non-replayable lease digests only.

---

## 9. Claims licensed by gates

### Current safe statements

- proveKV has conventional compressed shared-pool and per-agent-shell primitives.
- A temporary reconstructed Qwen2.5 size sweep completed on MSI CPU; exact factors are size-only observations.
- The historical MSI audit reported six validated claims and 55 passing tests at the recorded source.
- A CUDA-requested PPL run failed before inference because no CUDA GPU was available.
- Current proveKV has no certified Qwen3.5 hybrid capture/replay or Agent Graph integration.

### After P0-B

- Canonical hybrid layout, manifests, least-authority leases, durable pages, and branch isolation exist in the Rust reference store.
- No model replay claim.

### After P0-C

- CPU materialized-prefix capture/replay and divergent branch correctness for the exact pinned Qwen3.5/model/tokenizer/template/weights/adapters/Transformers tuple and frozen tolerance.
- No generic Qwen3.5, live shared pages, or production claim.

### After P1

- Exact per-component profile admission for the tested workload.
- Tested local graph fan-out/retry through the pinned invocation executor, if Task 14B and Task 15 pass.

### Not safe without separate evidence

- Generic Qwen3.5 support.
- Bit-exact reconstruction for quantized values.
- PPL neutrality or task neutrality outside the exact receipt.
- Direct framework memory reduction from compressed storage size.
- Zero-copy/live shared cache.
- GPU throughput, latency, or memory results.
- Ollama/llama.cpp/vLLM/FlashInfer compatibility.
- Multi-user/network/production security.
- Production serving or enterprise readiness.
- Any QLoRA/dataset/training claim.

---

## 10. Recommended execution order

1. Task 0 isolated worktrees and source receipt.
2. Task 1 durable MSI evidence decision.
3. Task 2 CPU capability gate.
4. Task 3 canonical hybrid contract.
5. Tasks 4–5 manifest identity and lease authority.
6. Tasks 6–7 persistence/recovery/COW/concurrency/GC.
7. Task 8 raw-first policy.
8. Task 9 locked offline CPU environment.
9. Task 10 Qwen3.5 capture.
10. Task 11 raw persist/reopen/replay.
11. Task 12 branches/accounting.
12. Task 13 component calibration.
13. Task 14A invocation seam; stop if not adopted.
14. Tasks 14B–15 local graph/checkpoint integration.
15. Task 16 backend probes.
16. Task 17 claims/runbook/final audit.

No calendar estimate is included because no calibrated end-to-end hybrid-state implementation baseline exists.

---

## 11. Auditor-rerunnable completion checklist

- [ ] Exact source/worktree/dependency identities recorded.
- [ ] Unrelated sparse-prefill files untouched.
- [ ] Temporary MSI evidence promoted with a complete recipe or explicitly left non-durable.
- [ ] CPU/device/dtype probe passes; GPU remains out of MSI scope.
- [ ] `quant-codec-core` is the sole hybrid layout/ID/digest type owner.
- [ ] Existing v1 pool/shell/receipt readers remain green.
- [ ] State identity and authority lease are separate.
- [ ] Lease values never enter prompts, graph state, model output, logs, receipts, or filenames.
- [ ] Resource limits use checked arithmetic before allocation.
- [ ] Binary/Safetensors pages reject corruption, overflow, trailing bytes, and unsafe formats.
- [ ] Page/manifest commit ordering and directory fsync pass failure injection.
- [ ] Reachability rebuild and mark-sweep GC preserve all live states.
- [ ] Concurrent branch/expiry/release/GC tests pass.
- [ ] Python/model/runtime environment is locked and offline-reproducible.
- [ ] Full-attention, convolution, recurrent, flags, mask, and position state are captured.
- [ ] Raw bytes/digests reopen exactly.
- [ ] Acceptance profile was frozen from baseline-only data before replay.
- [ ] One-shot, token-decode, restart, mismatch, and divergent-branch gates pass.
- [ ] Accounting separates encoded storage, decoded/framework state, RSS/PSS, page cache, and latency.
- [ ] Recurrent/convolution codecs remain raw unless held-out admission passes.
- [ ] Agent Graph invocation seam exists before any bridge claim.
- [ ] Default graph/provider path remains compatible.
- [ ] Retry/checkpoint integration cannot duplicate non-idempotent tool effects.
- [ ] Backend capability entries are witnessed or `unknown`.
- [ ] `README.md`, `REPRODUCE.md`, `CLAIMS.json`, schemas, and receipts agree.
- [ ] Full workspace tests, strict clippy, `prove_audit.sh`, Python tests, and exact-host rerun pass.
- [ ] Changed files, commands, pass/fail/skip results, unresolved risks, rollback, and receipt digests are in the final handoff.

Completion is reached only when every required item is green on an exact source/model/runtime tuple. Everything else remains proposed, blocked, degraded, or quarantined.
