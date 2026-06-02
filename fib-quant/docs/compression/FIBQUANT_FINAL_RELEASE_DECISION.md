# FibQuant Final Release Decision

Date: 2026-05-16

Decision: ALLOW_ALPHA_RELEASE

Allowed version: `0.1.0-alpha.1` only.

## Scope

This decision applies only to a cautious experimental alpha release of `fib-quant` as a paper-core research crate. It does not approve production KV-cache compression claims, benchmark reproduction claims, `semantic-memory` integration, FEUT/SCR variants, or `turbo-quant` changes.

## Required Receipts

Final receipt paths:

- `target/release-receipts/fmt.txt`
- `target/release-receipts/test.txt`
- `target/release-receipts/clippy.txt`
- `target/release-receipts/examples.txt`
- `target/release-receipts/doc.txt`
- `target/release-receipts/package-list.txt`
- `target/release-receipts/package.txt`
- `target/release-receipts/publish-dry-run.txt`
- `target/release-receipts/deny.txt`
- `target/release-receipts/bench-encode-decode.txt`
- `target/release-receipts/static-audit.txt`
- `target/release-receipts/final-assert.txt`
- `target/release-receipts/final-assert.json`

## Waivers and Residual Risks

- `cargo-deny` is not installed in the local environment. `deny.toml` is present, and `target/release-receipts/deny.txt` records the exact skipped command state. This is waived for `0.1.0-alpha.1` only and should be resolved before any beta or stable release.
- `z.py` remains in the local crate root as a packaging tool, but it is excluded from the Cargo package by the manifest include list. Public release branch hygiene should remove or quarantine it.
- Criterion benchmark receipts are local smoke/performance receipts only. They are not paper benchmark reproduction evidence.
- Rotation reproducibility is asserted by stored matrix digest for this implementation and dependency set, not by a cross-implementation seed-only standard.

## Release Conditions

Release is allowed only if:

- final assert passes;
- `cargo publish --dry-run --allow-dirty` passes and its receipt is saved;
- `cargo package --list --allow-dirty` includes `CITATION.cff`;
- Cargo package contents exclude `z.py`, Codex prompts, overlays, `.agents`, `.codex`, zip files, and source-context sidecars;
- public language remains experimental and alpha-only.

Actual publish performed: NO.

NO actual cargo publish was performed.
