#!/usr/bin/env bash
set -euo pipefail

cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps

if command -v cargo-semver-checks >/dev/null 2>&1; then
  cargo semver-checks check-release || true
else
  echo "cargo-semver-checks not installed; skipped"
fi
