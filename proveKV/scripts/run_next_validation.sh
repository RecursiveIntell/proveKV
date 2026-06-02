#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-proveKV-next}"
RUN_DIR=".codex-runs/$RUN_ID"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/commands_run.log"
run() {
  echo "
$ $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}
run cargo fmt --all -- --check
run cargo check --workspace --all-targets
run cargo test --workspace --all-targets
run cargo clippy --workspace --all-targets -- -D warnings
run cargo doc --workspace --no-deps
run python3 scripts/validate_schemas.py
run python3 scripts/check_public_claims.py
run python3 scripts/validate_final_state.py
run python3 scripts/assert_no_boundary_drift.py
run python3 scripts/assert_receipt_integrity.py
run python3 scripts/assert_realized_accounting.py
if [ -d python ]; then
  run python -m compileall python
  run env PYTHONPATH=python python -m pytest -q python/tests
fi
