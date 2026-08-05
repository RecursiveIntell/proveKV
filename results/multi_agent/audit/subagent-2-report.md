# Tool exposure audit: `src/nodes.rs`

## Scope
Inspected `/home/sikmindz/Coding/agent-graph-mcp-release/src/nodes.rs` (794 lines).

## Tool-related types counted
There is **1 tool-specific type** in this file:

1. `ToolNode` (declared at lines 599–607; `Node` implementation at 610–794)
   - Fields: `id`, `python`, `hermes_source`, `lease`, `receipt_dir`, `timeout_ms`, and `ctx`.
   - No tool-list, tool-schema, allowlist, denylist, capability, or per-node tool-set type is defined here.

Other node/config types (`RunContext`, `LlmNode`, `PassthroughNode`, `TransformNode`, `RouterNode`, `HumanApprovalNode`, plus their configs) are not tool-exposure types.

## How tools are currently exposed to LLM nodes
They are **not exposed directly to `LlmNode` at all**. `LlmNode` has no tool field, tool registry, tool definitions, tool-choice setting, filtering callback, or MCP client integration. Its provider invocation sends a rendered prompt through `LlmCall` and `ExecCtx`; the only controls visible here are model/provider configuration, JSON mode, token limit, timeout, cancellation, and the run-wide LLM-attempt budget.

Tool execution is a separate graph node path:

- `ToolNode` reads `__tool_name__` from graph state, defaulting to the string `"read_file"` (lines 625–630).
- It reads arbitrary JSON arguments from `__tool_args__`, defaulting to `null` (631–634).
- It spawns `agent.transports.hermes_tools_mcp_server` as a child process and performs MCP JSON-RPC over stdio (637–647, 659–721).
- It invokes `tools/call` with the state-derived name and arguments (700–708).
- Results/receipts/success are written to reserved state keys `__tool_result__`, `__tool_receipts__`, and `__tool_success__` (786–790).

## Filtering / gating / tool set
Within `nodes.rs`, there is **no filtering or tool-set selection**:

- No `tools/list` request is made.
- No validation that `__tool_name__` is one of an allowed set.
- No name allowlist/denylist or argument schema check.
- No LLM-generated tool call parsing or dispatch.
- No per-LLM-node tool gating.

The only apparent gating in this file is indirect/external: `ToolNode` passes a serialized `self.lease` to the broker via a temporary lease file and lineage environment variables (`AGENT_GRAPH_LINEAGE`, `AGENT_GRAPH_LINEAGE_LEASE_PATH`, `AGENT_GRAPH_LINEAGE_RECEIPT_DIR`). Any actual authorization or capability restriction therefore resides in the spawned Hermes MCP broker / lease validation code, not in `nodes.rs`.

`ToolNode` also comments that the call is read-only, but this file does not enforce read-only behavior; it forwards whichever `__tool_name__` is present in state.

## Conclusion
LLM nodes currently have **zero directly exposed tools**. Tool access is an explicit, separately scheduled `ToolNode` operation driven by graph state. At this layer, the selected tool name and arguments are effectively unrestricted (apart from downstream broker/lease enforcement), with a fallback to `read_file`; there is no local filtering, gating, or declared tool set.
