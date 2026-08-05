# Serializable public structs in `agent-graph-mcp/src/`

**Scope:** `/home/sikmindz/Coding/Libraries/agent-graph-mcp/src/` (Rust source files; public `struct` items whose adjacent derive attributes include `serde::Serialize` and/or `serde::Deserialize`).

**Total: 37 public serializable/deserializable structs** across 5 files.

## Inventory

### `evidence.rs` — 1
- **`EvidenceDisposition`** (`Serialize`): serializable evidence-status metadata describing shape validity, integrity verification, source-witness binding, source authority, and factual-support state.

### `nodes.rs` — 4
- **`TransformConfig`** (`Deserialize`): configuration for a state transformation node; contains ordered transformation operations.
- **`TransformOp`** (`Deserialize`): one transformation operation, including operation name, JSON path, optional source path, value, and values.
- **`RouterConfig`** (`Deserialize`): routing-node configuration containing matching rules and a default target list.
- **`Rule`** (`Deserialize`): one router matching rule, with JSON path, operation, comparison value, and target nodes.

### `run_manager.rs` — 2
- **`RunBudgets`** (`Serialize`, `Deserialize`): canonical graph-run resource budget contract (`max_wall_clock_ms`, `max_nodes`, and reserved `max_llm_calls`).
- **`BudgetCounters`** (`Serialize`): runtime counters for nodes, LLM calls, and wall-clock milliseconds.

### `spec.rs` — 4
- **`GraphSpec`** (`Serialize`, `Deserialize`): complete registered graph definition, including version, name, entry, nodes, edges, iteration/parallelism limits, output key, and reducers.
- **`NodeSpec`** (`Serialize`, `Deserialize`): graph node definition, including ID, node type, optional prompt/model, JSON/evidence settings, token limit, routes, and arbitrary config.
- **`EdgeSpec`** (`Serialize`, `Deserialize`): directed graph edge (`from` and `to` node IDs).
- **`ResumeEligibility`** (`Serialize`, `Deserialize`): deterministic-resume metadata: next-node cursor, execution chain, and dependency summary.

### `tools.rs` — 25
Tool request/response schemas used by the MCP surface:
- **`GraphCreateParams`** (`Deserialize`): create/validate/delete graph request, including optional spec, action, graph ID, template, idempotency key, and overwrite.
- **`GraphListParams`** (`Deserialize`): graph-list query and result limit.
- **`GraphInspectParams`** (`Deserialize`): graph ID to inspect.
- **`GraphDeleteParams`** (`Deserialize`): graph ID to delete.
- **`GraphExecuteParams`** (`Deserialize`): graph execution request: graph ID, input, version, thread, mode, and idempotency key.
- **`GraphStatusParams`** (`Deserialize`): server/graph/run/events/receipt/templates status query parameters.
- **`StructuredOutput`** (`Serialize`, `Deserialize`): standard tool response envelope with success flag, status, data, error details, graph/version, and run ID.
- **`ApprovalRequestParams`** (`Deserialize`): create an approval bound to a checkpoint, with audience, prompt, allowed decisions, and expiration.
- **`ApprovalListParams`** (`Deserialize`): filter approvals by run/status and limit.
- **`ApprovalGetParams`** (`Deserialize`): approval ID lookup.
- **`ApprovalDecideParams`** (`Deserialize`): approval decision request with approval ID, decision, and claimed actor label.
- **`RunStartParams`** (`Deserialize`): start an async run with graph/input/version/thread/idempotency, budgets, and optional checkpointing.
- **`RunWaitParams`** (`Deserialize`): wait for a run, optionally with timeout.
- **`RunCancelParams`** (`Deserialize`): cancel a run with optional reason.
- **`RunGetParams`** (`Deserialize`): run ID lookup.
- **`RunStateParams`** (`Deserialize`): query run state, optionally at checkpoint or JSON pointer.
- **`RunEventsParams`** (`Deserialize`): query run events from cursor with limit.
- **`RunReceiptParams`** (`Deserialize`): run receipt lookup.
- **`RunCheckpointParams`** (`Deserialize`): checkpoint lookup by optional run/checkpoint ID.
- **`RunResumeParams`** (`Deserialize`): resume request by optional checkpoint/run ID.
- **`WitnessCaptureParams`** (`Deserialize`): caller-supplied local source capture (locator, content, media type, authority class, retrieval timestamp).
- **`WitnessGetParams`** (`Deserialize`): source-witness ID lookup.
- **`PolicyCheckParams`** (`Deserialize`): policy-check request for a graph and optional input.
- **`RenderParams`** (`Deserialize`): graph-render request with optional output format.
- **`TemplateListParams`** (`Deserialize`): template query filter.
- **`TemplateInstantiateParams`** (`Deserialize`): instantiate a template under a supplied name.

## Notes / exclusions

- `tools.rs` actually contains **26** listed public structs (not 25); therefore the file total is **26** and the overall total is **37**. The initial headline was corrected below.
- Corrected total: **37 public serializable/deserializable structs** (1 + 4 + 2 + 4 + 26).
- This inventory counts structs only, not serializable enums (`NodeType`, `ReducerKind`, etc.), private structs, or types deriving only `JsonSchema` without a serde derive. `Deserialize`-only request types are included because they are serde-deserializable.
