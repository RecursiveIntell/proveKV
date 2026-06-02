#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="${1:-}"
if [[ -z "$BUNDLE_DIR" || ! -d "$BUNDLE_DIR" ]]; then
  echo "usage: $0 /path/to/fibquant_publish_hardening_codex_bundle" >&2
  exit 2
fi
ROOT="$(pwd)"
mkdir -p "$ROOT/docs/codex-runs/fibquant-publish-hardening"
mkdir -p "$ROOT/scripts"
mkdir -p "$ROOT/.agents/skills/fibquant-publish-hardening"
mkdir -p "$ROOT/.codex/hooks"
cp "$BUNDLE_DIR"/*.md "$ROOT/docs/codex-runs/fibquant-publish-hardening/"
cp -R "$BUNDLE_DIR/phase_prompts" "$ROOT/docs/codex-runs/fibquant-publish-hardening/"
cp -R "$BUNDLE_DIR/manual_backstop_prompts" "$ROOT/docs/codex-runs/fibquant-publish-hardening/"
cp "$BUNDLE_DIR/scripts/publish_preflight.py" "$ROOT/scripts/publish_preflight.py"
cp "$BUNDLE_DIR/scripts/publish_final_assert.py" "$ROOT/scripts/publish_final_assert.py"
chmod +x "$ROOT/scripts/publish_preflight.py" "$ROOT/scripts/publish_final_assert.py"
cp -R "$BUNDLE_DIR/overlays/.agents/skills/fibquant-publish-hardening/"* "$ROOT/.agents/skills/fibquant-publish-hardening/"
cp -R "$BUNDLE_DIR/overlays/.codex/hooks/"* "$ROOT/.codex/hooks/"
echo "Installed FibQuant publish-hardening bundle into $ROOT"
echo "Paste docs/codex-runs/fibquant-publish-hardening/OPERATOR_PASTE_FIRST.md into Codex."
