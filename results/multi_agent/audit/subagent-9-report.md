# Fresh test report

- Repository: `/home/sikmindz/Coding/agent-graph-mcp-release`
- Command: `cargo test --lib -p agent-graph-mcp 2>&1`
- Result: **PASS** (exit status 0)
- Tests: **66 passed, 0 failed, 0 ignored, 0 measured, 0 filtered out**
- Duration: 0.04s

Warnings observed:
- `src/nodes.rs:124`: variable `rendered` does not need to be mutable (`unused_mut`).
- `src/nodes.rs:124`: variable `rendered` is unused (`unused_variables`).
