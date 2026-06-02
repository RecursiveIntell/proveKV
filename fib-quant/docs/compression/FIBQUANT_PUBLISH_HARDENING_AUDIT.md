# FibQuant Publish-Hardening Audit

Date: 2026-05-16

This audit records the phase-0 blocker matrix required before code changes.

## Blocker Matrix

| Finding | Evidence | Required fix |
| --- | --- | --- |
| Workspace-inherited dependencies and lints | `Cargo.toml:10`, `Cargo.toml:16`, `Cargo.toml:17`, `Cargo.toml:19`, `Cargo.toml:21-22` | Replace all workspace inheritance with explicit standalone dependency versions and local lint config. |
| Missing package metadata | `Cargo.toml:1-22` lacks `readme`, `repository`, `documentation`, `keywords`, `categories`, package surface rules | Add conservative crates.io metadata and include/exclude package surface. |
| Release files missing | Preflight reported missing `LICENSE`, `CITATION.cff`, `CHANGELOG.md`, `RELEASE_CHECKLIST.md` | Add release and citation files before publish dry-run. |
| Public docs missing | Preflight reported missing `docs/compression/FIBQUANT_*` docs | Add source, conformance, benchmark-plan, non-claims, and readiness docs. |
| `z.py` in crate root | `z.py` exists in package root | Exclude from package list; do not ship generated context files or root sidecars. |
| `beta_d_k()` unsigned underflow risk | `src/spherical_beta.rs:12` contains `(d - k - 2) as f64` | Use signed/f64 arithmetic and validate positive finite beta shape. |
| Profile invariants are weak | `src/profile.rs:145-158` only validates shape plus `wire_index_bits` | Enforce schema marker, finite rates, exact paper/wire rate formulae, method compatibility, source mode, norm format, and Lloyd/training bounds. |
| Profile method fields are decorative | `src/codebook.rs:135-138` calls dispatch helpers by `k`; `src/directions.rs:73-84` dispatches by `k`; `src/spherical_beta.rs:25-30` dispatches radius by `k` | Dispatch from `radius_method` and `direction_method`, and reject unsupported combinations in profile validation. |
| Receipt source provenance missing | `src/receipt.rs:7-40` has profile/codebook/encoded digests but no source digest | Add `source_vector_digest` and schema/norm metadata to receipts. |
| Encoded digest fails open | `src/codec.rs:211-213` uses `serde_json::to_vec(code).unwrap_or_default()` | Return `Result<String>` and reject serialization failures. |
| Decode does not validate code schema marker | `src/codec.rs:182-206` validates digests and shape but not `FibCodeV1.schema_version` | Reject non-`fib_code_v1` schema markers. |
| README is publish-bundle text, not crate docs | `README.md:1-40` describes the Codex bundle rather than crate usage | Replace with public crate README including citation, status, non-claims, examples, API, deviations, and license. |

## Phase Gates

- Cargo package surface must exclude `z.py`, `docs/codex-runs/**`, archive zips, generated sidecars, and target output.
- Math/profile changes must be covered by underflow, tamper, method, and schema tests.
- Receipt changes must prove source digest and fail-closed encoded digest behavior.
- Final readiness cannot be claimed unless `cargo publish --dry-run --allow-dirty` succeeds.
