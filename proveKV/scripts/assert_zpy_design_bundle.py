#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

REQUIRED = [
    "docs/RESEARCH_SYNTHESIS.md",
    "docs/HIGH_ROI_CHANGE_MATRIX.md",
    "docs/ZPY_NEXT_ARCHITECTURE_SPEC.md",
    "docs/ECOSYSTEM_PARITY_CHECKS.md",
    "docs/SECURITY_AND_PORTABILITY_GATES.md",
    "docs/IMPLEMENTATION_PLAN.md",
    "docs/FINAL_STATE_AND_ACCEPTANCE.md",
    "schemas/PackagePolicyV1.schema.json",
    "scripts/test_zpy_universal_packager_regression.py",
    "codex/prompts/MASTER_PROMPT.md",
]

missing = [p for p in REQUIRED if not Path(p).exists()]
if missing:
    print("missing bundle files:")
    for p in missing:
        print(f" - {p}")
    raise SystemExit(2)
print("z.py universal packager design bundle is structurally complete")
