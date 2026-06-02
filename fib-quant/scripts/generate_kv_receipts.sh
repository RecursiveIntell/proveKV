#!/usr/bin/env bash
set -u
mkdir -p target/kv-production-receipts/final
{
  echo "# KV Production Receipt Run"
  date -u
  echo
  for cmd in \
    "python3 scripts/kv_static_preflight.py" \
    "cargo fmt --all --check" \
    "cargo test --all-features" \
    "cargo clippy --all-targets --all-features -- -D warnings" \
    "cargo test --examples" \
    "cargo bench --no-run" \
    "cargo doc --no-deps --all-features" \
    "cargo package --list --allow-dirty" \
    "cargo publish --dry-run --allow-dirty" \
    "python3 scripts/kv_final_assert.py"; do
      echo "## $cmd"
      bash -lc "$cmd" 2>&1
      echo "status=$?"
      echo
  done
} | tee target/kv-production-receipts/final/kv-production-validation.txt
