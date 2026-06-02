#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-proveKV-next}"
RUN_DIR=".codex-runs/$RUN_ID"
mkdir -p "$RUN_DIR"
{
  echo "== proveKV next-pass preflight =="
  echo "run_id: $RUN_ID"
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "pwd: $(pwd)"
  git rev-parse --show-toplevel || true
  git branch --show-current || true
  git rev-parse HEAD || true
  git status --short || true
  echo "-- rust --"
  rustc --version || true
  cargo --version || true
  echo "-- python --"
  python3 --version || true
  python --version || true
  echo "-- manifests --"
  find . -name Cargo.toml -print | sort
  find . -name Cargo.lock -print | sort
  echo "-- external path deps --"
  grep -R -n --include Cargo.toml 'path *= *"\.\.' . || true
  echo "-- codex/control files --"
  find . \( -path '*/.codex/*' -o -path '*/.agents/*' -o -path '*/.codex-runs/*' -o -name AGENTS.md \) -maxdepth 6 -print | sort | head -400
  echo "-- placeholders --"
  grep -RIn "TODO\|FIXME\|TBD\|@filename\|{feature}\|<placeholder>" AGENTS.md README.md crates docs scripts python pyproject.toml 2>/dev/null || true
} | tee "$RUN_DIR/startup_preflight.md"
