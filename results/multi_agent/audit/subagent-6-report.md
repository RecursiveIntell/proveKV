# agent-graph-mcp `src/spec.rs` validation audit

## Scope and counting method

Audited `validate()`, `validate_node()`, and `validate_state_write_conflicts()` in `/home/sikmindz/Coding/agent-graph-mcp-release/src/spec.rs`. Counted each distinct validation condition/error category (including per-item checks such as node/edge target validation as one rule category), not every loop iteration or every allowed enum value.

## Count

**38 validation rules** in the core graph-spec validation path:

- **8 graph-level rules:** valid graph name; node count 1..=128; edge count <=512; max_iterations 1..=64; max_parallelism 1..=32; unique node IDs; entry exists; non-empty output_key when supplied.
- **1 common node rule:** each node ID must satisfy `valid_id` (`[A-Za-z0-9_.-]`, max 64 bytes).
- **2 edge rules:** edge source exists; edge target is `END` or an existing node.
- **1 state-write conflict rule:** unordered parallel writers of the same key require a reducer.
- **25 node-specific rules:** Router 4, LLM 7, StateTransform 3, Join 2, Parallel 6, Subgraph 1, HumanApproval 2.

The 25 node-specific rules break down as:

- **Router (4):** routes/rules produce at least one target; config-based routers require an explicit default; predicates must be in the supported set; every target must be `END` or an existing node.
- **Llm (7):** evidence-required requires `json_mode`; evidence-required requires non-empty `config.output_key`; prompt <=16 KiB; max_tokens <=8192; model passes conservative alias validation; timeout_ms 1..=120000; retry.max_attempts 1..=5.
- **StateTransform (3):** `operations` array is present; operation count 1..=64; every operation has a supported `op`.
- **Join (2):** join mode is supported; `inputs` array and string `output` are present.
- **Parallel (7):** branches array is present; branch count 1..=16; each branch has an entry; each branch entry exists; join target is present; join target is `END` or an existing node; optional fail_policy is supported.
- **Subgraph (1):** non-null string `config.graph_name` is required.
- **HumanApproval (2):** `config.prompt_key` string is required; `config.audience` array is required.

The primary figure, **38**, is the literal validation-condition count for the core `validate()` path, treating each required-field/type/range/existence check as a separate rule.

## Node types validated

`NodeType` declares 10 variants: `Llm`, `Router`, `Passthrough`, `StateTransform`, `Join`, `Parallel`, `Subgraph`, `HumanApproval`, `External`, `Tool`, and `Loop`.

Node-specific validation logic exists for **7 node types**:

1. `Router`
2. `Llm`
3. `StateTransform`
4. `Join`
5. `Parallel`
6. `Subgraph`
7. `HumanApproval`

All 10 types are still accepted by deserialization and included in the common enum/execution classification. `Passthrough`, `External`, `Tool`, and `Loop` have no dedicated checks in `validate_node()`; `External` and `Loop` are rejected by `executable_node_type()`, while `Tool` is classified as executable there. `resume_eligibility()` separately handles all 10 variants, permitting only empty-config/non-evidence `Passthrough` and local `StateTransform` nodes for deterministic resume.

## Files

- Created: `/tmp/subagent-6-report.md`
- Modified repository files: none.
