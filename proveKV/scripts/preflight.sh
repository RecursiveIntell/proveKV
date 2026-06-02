#!/usr/bin/env bash
set -euo pipefail

echo "== proveKV preflight =="
echo "cwd: $(pwd)"
echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "git_root: $(git rev-parse --show-toplevel)"
  echo "git_head: $(git rev-parse HEAD || true)"
  echo "git_status:"
  git status --short || true
else
  echo "WARN: not inside a git repository"
fi

echo "rustc: $(rustc --version 2>/dev/null || echo missing)"
echo "cargo: $(cargo --version 2>/dev/null || echo missing)"
echo "python: $(python3 --version 2>/dev/null || echo missing)"

echo "Cargo manifests:"
find . -name Cargo.toml -print | sort || true

echo "Path dependencies pointing outside workspace:"
grep -R "path *= *\"\\.\\." -n --include Cargo.toml . || true

echo "Codex/agents files:"
find . \( -path "*/.codex/*" -o -path "*/.agents/*" -o -name "AGENTS.md" \) -maxdepth 6 -print | sort || true

python3 scripts/validate_schemas.py || true
python3 scripts/check_public_claims.py || true

echo "Preflight complete."
