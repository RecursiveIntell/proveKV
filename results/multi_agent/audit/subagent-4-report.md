# Test coverage audit

Repository: `/home/sikmindz/Coding/agent-graph-mcp-release`

## Counts

Counted Rust test functions marked with `#[test]` or `#[tokio::test]` in `tests/` and inline `mod tests` blocks under `src/`.

- **Total: 183 tests**
  - **Integration tests (`tests/`): 112**
  - **Inline unit tests (`src/`): 71**

### By file

#### `tests/` (112)

| File | Tests |
|---|---:|
| `mcp_integration.rs` | 54 |
| `daemon_recovery.rs` | 16 |
| `tool_runtime.rs` | 15 |
| `proxy_confinement.rs` | 6 |
| `template_promotion.rs` | 6 |
| `operator_authority.rs` | 4 |
| `codex_app_server.rs` | 3 |
| `process_boundary.rs` | 2 |
| `proxy_stdio.rs` | 2 |
| `lifecycle.rs` | 1 |
| `migrations.rs` | 1 |
| `release_manifest.rs` | 1 |
| `terminal_projection.rs` | 1 |

#### Inline `src/` tests (71)

| File | Tests |
|---|---:|
| `cli.rs` | 22 |
| `templates.rs` | 7 |
| `fs_security.rs` | 8 |
| `provekv_executor.rs` | 5 |
| `run_manager.rs` | 5 |
| `checkpoint_binding.rs` | 4 |
| `evidence.rs` | 4 |
| `model_executor.rs` | 4 |
| `owner_lock.rs` | 3 |
| `server.rs` | 2 |
| `spec.rs` | 2 |
| `store.rs` | 2 |
| `lifecycle.rs` | 1 |
| `migrations.rs` | 1 |
| `nodes.rs` | 1 |

## Domains covered

- **Receipts, integrity, provenance, and evidence:** durable terminal receipts, receipt chains, typed invocation/terminal provenance, integrity keys, tamper/fail-closed behavior, source witnesses, witness projections, claim/source locator validation, checkpoint lineage.
- **Graph/node execution:** graph creation/execution/deletion, historical graph versions, terminal output selection, node budgets, parallel transform/join, ordered routing, conditional validation, cancellation boundaries, reducer requirements, templates and legacy aliases.
- **Lifecycle and run management:** canonical lifecycle mapping, persistent/interrupted/completed runs, restart readability, cancellation semantics, retention, capacity/admission, volatile persistence failures, terminal projection rollback.
- **Daemon/process lifecycle and recovery:** owner/process locks, duplicate daemon rejection, startup mode durability, concurrent startup, crash recovery, checkpoint-write/create crash handling, watchdog reacquisition, process-boundary timeout semantics, daemon socket proxy lifecycle.
- **Checkpoint/approval/resume:** deterministic-local checkpoints, exactly-once restart/resume, durable approval persistence, rejection/expiry/tampering, eligibility and linearity checks, budget continuation/exhaustion.
- **Tool runtime/security:** leases/HMAC/exact invocation binding, allowlists, effect classification, recursion/cycle detection, tool/effect/recursion budgets, idempotency/replay, atomic counters, receipt parent binding, summary limits.
- **Operator authority and retention:** nonce/version checks, authorization, stale digest/peer rejection, idempotent history-preserving retention, template-promotion eligibility/quarantine and nonce replay.
- **MCP/API and compatibility:** exact tool names and legacy contracts, status/capability reporting, API-key configuration, MCP integration, model-MCP mutation restrictions, README contract checks.
- **CLI/configuration:** flag/value validation, URL validation and credential stripping, ephemeral/data-dir rules, integrity-key requirements, help/default config, graph-capacity bounds.
- **Storage/migrations:** legacy-row quarantine/rewrite, migration idempotency, SQLite fault rollback, checkpoint persistence atomicity, storage mismatch handling.
- **Filesystem/proxy confinement:** private directory/file permissions, symlink rejection, WAL checks, proxy argument confinement, stdio framing and absent-daemon errors.
- **Model/executor integrations:** model executor acquisition/decline and Send/Sync behavior; proveKV executor state open/commit/fork/GC/release; Codex app-server stream and completion handling.
- **Release/template contracts:** release schema/scripts, executable template terminal outputs, routing/classifier field behavior, template registry security and bundle verification.

## Coverage shape and gaps

- Coverage is strongest in **integration-level durability, receipts, daemon recovery, MCP contracts, checkpoint/approval handling, and runtime safety**; `mcp_integration.rs` alone contains 54 tests.
- Unit coverage is concentrated in **CLI validation (22)**, templates, filesystem security, run management, executors, and persistence fault handling.
- Only one direct test exists for `src/nodes.rs` (cancellation primitive), and lifecycle/migrations each have one inline test plus one integration test; broader node behavior is primarily exercised indirectly through integration tests.
- The audit counts test attributes, not parameterized cases or assertions; it does not claim line/branch coverage. No tests were added or modified.
- A helper function in `tests/mcp_integration.rs` (`test_integrity_key`) is not counted because it is not a test attribute.
