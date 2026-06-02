#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="${1:-}"
if [[ -z "$BUNDLE_DIR" ]]; then
  echo "usage: $0 /path/to/fibquant_final_hardening_codex_bundle_2026-05-16" >&2
  exit 2
fi
if [[ ! -f Cargo.toml ]]; then
  echo "must be run from fib-quant crate root" >&2
  exit 2
fi
if ! grep -q 'name = "fib-quant"' Cargo.toml; then
  echo "Cargo.toml does not look like fib-quant" >&2
  exit 2
fi

DEST="docs/codex-runs/fibquant-final-hardening"
mkdir -p "$DEST"

copy_item() {
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  cp -R "$src" "$dst"
}

for f in README.md OPERATOR_PASTE_FIRST.md 01_CODEX_MASTER_PROMPT.md 02_PHASE_PLAN.md 03_CURRENT_HOSTILE_AUDIT.md 04_TARGET_PATCH_SPEC.md 05_ACCEPTANCE_GATES.md 06_VALIDATION_COMMANDS.md 07_DIAGRAMS.md 08_FINAL_AUDITOR_HANDOFF_TEMPLATE.md 09_PUBLISH_POSITIONING.md 10_NON_GOALS_AND_FORBIDDEN_STATES.md PACK_MANIFEST.json; do
  cp "$BUNDLE_DIR/$f" "$DEST/$f"
done

for d in phase_prompts manual_backstop_prompts matrices diagrams; do
  if [[ -d "$BUNDLE_DIR/$d" ]]; then
    rm -rf "$DEST/$d"
    cp -R "$BUNDLE_DIR/$d" "$DEST/$d"
  fi
done

mkdir -p scripts
cp "$BUNDLE_DIR/scripts/fibquant_static_source_audit.py" scripts/fibquant_static_source_audit.py
cp "$BUNDLE_DIR/scripts/fibquant_final_assert.py" scripts/fibquant_final_assert.py
cp "$BUNDLE_DIR/scripts/generate_release_receipts.sh" scripts/generate_release_receipts.sh
chmod +x scripts/fibquant_static_source_audit.py scripts/fibquant_final_assert.py scripts/generate_release_receipts.sh

mkdir -p .agents/skills/fibquant-final-hardening
cp "$BUNDLE_DIR/overlays/.agents/skills/fibquant-final-hardening/SKILL.md" .agents/skills/fibquant-final-hardening/SKILL.md

mkdir -p .codex/hooks
cp "$BUNDLE_DIR/overlays/.codex/hooks/fibquant_preflight_guard.py" .codex/hooks/fibquant_preflight_guard.py
cp "$BUNDLE_DIR/overlays/.codex/hooks/fibquant_release_claim_guard.py" .codex/hooks/fibquant_release_claim_guard.py
chmod +x .codex/hooks/fibquant_preflight_guard.py .codex/hooks/fibquant_release_claim_guard.py

cat > "$DEST/INSTALL_RECEIPT.txt" <<EOF
installed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
installed_from=$BUNDLE_DIR
installed_to=$PWD/$DEST
EOF

echo "Installed final hardening bundle to $DEST"
echo "Paste $DEST/OPERATOR_PASTE_FIRST.md into Codex."
echo "Optional hooks copied to .codex/hooks; approve only if expected."
