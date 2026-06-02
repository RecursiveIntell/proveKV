#!/usr/bin/env bash
set -euo pipefail
DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${OUT:-proveKV-generic-rust-next-codex-context-${DATE_TAG}.zip}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-handoff-package}"
mkdir -p ".codex-runs/$RUN_ID"
LOG=".codex-runs/$RUN_ID/commands_run.log"
run() {
  printf '\n$ %s\n' "$*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

run bash scripts/preflight_next_pass.sh
run python3 scripts/assert_python_sidecar_layout.py
run python3 scripts/assert_no_boundary_drift.py
run python3 scripts/assert_source_package_hygiene.py --repo-root . --mode prepackage || true

run python3 z.py \
  --root . \
  --profile generic-rust \
  --mode next-codex-context \
  --strict \
  --archive-codex-runs \
  --archive-root-markdown-noise \
  --archive-root-package-artifacts \
  --output "$OUT"

run python3 scripts/assert_source_package_hygiene.py --repo-root . --manifest "${OUT%.zip}.manifest.json" --mode manifest
