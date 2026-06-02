#!/usr/bin/env bash
set -euo pipefail
mkdir -p target/release-receipts

run() {
  local name="$1"; shift
  echo "===== $name ====="
  "$@" 2>&1 | tee "target/release-receipts/$name.txt"
}

run static-audit python3 scripts/fibquant_static_source_audit.py
run fmt cargo fmt --all --check
run test cargo test
run clippy cargo clippy --all-targets --all-features -- -D warnings
run examples cargo test --examples
run doc cargo doc --no-deps
run package-list cargo package --list --allow-dirty
run package cargo package --allow-dirty
run publish-dry-run cargo publish --dry-run --allow-dirty

if command -v cargo-deny >/dev/null 2>&1; then
  run deny cargo deny check advisories bans licenses sources
else
  echo "cargo-deny not installed" | tee target/release-receipts/deny.txt
fi

if grep -q '\[\[bench\]\]' Cargo.toml || [[ -d benches ]]; then
  cargo bench --bench encode_decode -- --sample-size 10 2>&1 | tee target/release-receipts/bench-encode-decode.txt || true
else
  echo "no benches configured" | tee target/release-receipts/bench-encode-decode.txt
fi

python3 scripts/fibquant_final_assert.py 2>&1 | tee target/release-receipts/final-assert.txt
