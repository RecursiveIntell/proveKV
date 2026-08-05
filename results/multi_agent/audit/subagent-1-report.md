# Permit system audit — agent-graph-mcp-release

## Scope

Repository: `/home/sikmindz/Coding/agent-graph-mcp-release`

## Findings

- `src/permits.rs` is **absent**. The requested path does not exist, so there is no `PermitV2` definition to inspect.
- A repository-wide search found **no `PermitV2` symbol** and no permit-named source file.
- A repository-wide search for the literal term `permit` found only incidental prose/license matches; no implementation of a permit system.
- Therefore, `PermitV2` has **no fields in this checkout**, and there is no permit-related implementation code to enumerate.

## Related authorization/approval code (not permits)

The repository does contain an approval/authorization system under different names:

- `src/tools.rs`: `ApprovalDecision` and approval lifecycle tool definitions.
- `src/compiler.rs`: `NodeType::HumanApproval` compilation and approval interrupt/decision state handling.
- `src/auth.rs`: authorization action types, including approval-decision authorization.
- `src/operator_auth.rs`, `src/operator_ipc.rs`, and `src/operator.rs`: operator authorization/IPC handling.
- `src/store.rs` / migrations and tests: durable checkpoint-bound approval persistence and lifecycle behavior.
- Tests cover durable approval requests, restart persistence, checkpoint integrity, rejection/expiry, and model-facing decision restrictions.

These are approval/operator authorization mechanisms, not a `PermitV2` permit system.

## Files created/modified

- Created: `/tmp/subagent-1-report.md`
- No repository files modified.

## Issues

- The requested `src/permits.rs` file and `PermitV2` type are not present in this repository checkout.
