# Agent Graph MCP tool-gating audit

## Result

- Repository checked: `/home/sikmindz/Coding/agent-graph-mcp-release`
- `src/gating.rs`: **missing** (no file at the requested path).
- No `gating` module is declared in `src/lib.rs` (the module list includes `auth`, `policy`, `tool_runtime`, etc., but not `gating`).

## Existing gating-related implementation

### `src/tool_runtime.rs` — closest dedicated tool-gating subsystem

This is the primary existing tool-policy/gating implementation:

- `ToolEffect` enum (`ReadOnly`, `LocalMutation`, `ExternalEffect`, `AuthorityChange`, `RecursiveOrchestration`) at lines 12–20.
- `ToolLease` with tool/effect allowlists, call/recursion/child/depth budgets, lineage and graph/run/node binding, and counters at lines 30–54.
- `SignedToolLease`, `LeaseBinding`, `ToolInvocation`, `ToolCallIntent`, and `ToolCallReceipt` at lines 56–130.
- `ToolPolicyError` includes lease/signature/scope failures, `ToolNotGranted`, `EffectNotGranted`, budget exhaustion, depth/cycle violations, and intent/receipt integrity failures at lines 138–164.
- Public enforcement functions: `issue_lease` (line 189), `verify_lease` (line 230), `reserve_call` (line 270), and `verify_receipt_chain` (line 426).
- `reserve_call` verifies the lease, checks tool allowlist/effect classification, and enforces tool, recursive, child, depth, and cycle budgets (implementation around lines 276–360).
- `classify_tool` is public at line 449.

Search found no production call sites outside `tool_runtime.rs` for `issue_lease`, `verify_lease`, `reserve_call`, or the related lease types; this suggests the subsystem may be standalone/not wired into the MCP server execution path (needs follow-up audit).

### `src/auth.rs` — capability gate

- `Capability` enum includes graph read/create/run/cancel, witness, checkpoint, approval, delete, migration, and config-install capabilities (lines 3–17).
- `Principal` enum distinguishes `ModelClient`, `StdioProxy`, `Daemon`, and `LocalOperator` (lines 19–25).
- `CapabilityPolicy::model()` grants only read/create/run/cancel, witness, and checkpoint capabilities; it excludes `ApprovalDecide`, `GraphDelete`, `DatabaseMigration`, and `ConfigInstall` (lines 33–49).
- `allows`/`require` provide capability checks (lines 51–63).

### `src/policy.rs` — non-authoritative graph preflight

- `PolicyReport`/`PolicyFinding` and `preflight` exist (lines 3–16).
- Checks graph version, iteration/output budgets, provider presence, unsupported node types, and evidence requirements (lines 17–65).
- Explicitly documented as “Non-authoritative admission preflight. It never grants authorization.”

### `src/server.rs` — exposed policy check

- `graph_policy_check` MCP tool exists at lines 2497–2529.
- It currently looks up the graph, reports node/edge and execution-budget stats, returns an empty `issues` vector, and reports model plus an empty `tools` capability list. It does not invoke `policy::preflight` in the shown implementation.

### `src/nodes.rs` — approval gate node, not tool gating

- `HumanApprovalNode` begins at line 547.
- It writes `__approval_request__` state, accepts a previously injected decision, otherwise returns `AgentGraphError::InterruptError` to suspend execution (lines 558–590).
- This is a human-approval execution interrupt, not a general MCP tool gate.

### Other related surfaces

- `src/templates.rs` marks `approval_gated_action` unavailable because authenticated HITL is not implemented (lines 232–238); tests around lines 304–322 enforce unavailable listing behavior.
- `src/spec.rs` validates node `fail_policy` values and policy-restricted fields, and supports a human-approval node type, but this is graph-spec validation rather than a dedicated tool gate.

## Assessment

There is no `gating.rs`. Tool-gating concepts are distributed across `tool_runtime.rs` (signed leases and runtime enforcement), `auth.rs` (principal capabilities), `policy.rs`/`server.rs` (non-authoritative graph preflight), and `nodes.rs` (human approval interrupt). The strongest concern is wiring: repository search found no callers of the public `tool_runtime` enforcement APIs outside that module, so the existence of the types/functions alone does not establish live MCP tool gating.