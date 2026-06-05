#!/usr/bin/env python3
"""Validate that public claim surfaces agree with CLAIMS.json.

This catches the exact failure mode where CLAIMS.json was corrected but README,
rustdoc, visuals, or reproduction docs still quote stale headline numbers.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIMS = json.loads((ROOT / "CLAIMS.json").read_text())
claims = CLAIMS["claims"]
lossless = claims["smollm2_wikitext2_n8_lossless_default"]
lossy = claims["smollm2_wikitext2_n8_lossy_default"]
legacy_lossless = claims["smollm2_wikitext2_n8_lossless_legacy_b8"]
legacy_lossy = claims["smollm2_wikitext2_n8_lossy_legacy_b8"]

EXPECTED = {
    "lossless_f32": f"{lossless['ratio_vs_f32_raw']:.2f}",
    "lossy_f32": f"{lossy['ratio_vs_f32_raw']:.2f}",
    "lossless_fp16": f"{lossless['ratio_vs_fp16_kv']:.2f}",
    "lossy_fp16": f"{lossy['ratio_vs_fp16_kv']:.2f}",
    "legacy_lossless_f32": f"{legacy_lossless['ratio_vs_f32_raw']:.2f}",
    "legacy_lossy_f32": f"{legacy_lossy['ratio_vs_f32_raw']:.2f}",
    "legacy_lossless_fp16": f"{legacy_lossless['ratio_vs_fp16_kv']:.2f}",
    "legacy_lossy_fp16": f"{legacy_lossy['ratio_vs_fp16_kv']:.2f}",
    "raw_total": f"{lossless['raw_total_bytes']:,}",
    "compressed_lossless": f"{lossless['compressed_total_bytes']:,}",
    "compressed_lossy": f"{lossy['compressed_total_bytes']:,}",
}

# Stale headline values from the superseded 2,604,662,784-byte denominator.
FORBIDDEN_PUBLIC_PATTERNS = [
    r"40\.50\s*[×x]",
    r"40\.53\s*[×x]",
    r"76\.54\s*[×x]",
    r"76\.55\s*[×x]",
    r"20\.25\s*[×x]",
    r"38\.27\s*[×x]",
    r"2,604,662,784",
    r"2\.43\s*GiB",
    r"2\.48\s*GB",
    r"1\.62\s*[×x]",
    r"2,484\s*MiB",
    r"--shell-bits",
]

# Legacy b=8 stale values also changed after the denominator correction.
FORBIDDEN_PUBLIC_PATTERNS += [
    r"37\.31\s*[×x]",
    r"65\.88\s*[×x]",
    r"18\.65\s*[×x]",
    r"32\.94\s*[×x]",
]

PUBLIC_SURFACES = [
    ROOT / "README.md",
    ROOT / "proveKV" / "README.md",
    ROOT / "REPRODUCE.md",
    ROOT / "CITATION.cff",
    ROOT / "docs" / "methodology" / "naive_computation.md",
    ROOT / "docs" / "img" / "_make_visuals.py",
    ROOT / "docs" / "img" / "architecture.svg",
    ROOT / "docs" / "img" / "n_scaling.svg",
    ROOT / "proveKV" / "docs" / "img" / "architecture.svg",
    ROOT / "proveKV" / "docs" / "img" / "n_scaling.svg",
    ROOT / "proveKV" / "src" / "lib.rs",
    ROOT / "proveKV" / "src" / "policy.rs",
    ROOT / "proveKV" / "examples" / "prove_kv_multi_agent_shell.rs",
    ROOT / "results" / "bench" / "decode_wallclock" / "decode_wallclock_smollm_shape_5reps.json",
    ROOT / "results" / "ppl_multi_agent" / "smollm2-1.7b" / "wikitext-2-n8" / "report.md",
]

# Validate public docs/source surfaces plus checked-in human-facing reports.

def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)

# Required canonical strings should appear on the main public surface.
readme = (ROOT / "README.md").read_text()
for key in ["lossless_f32", "lossy_f32", "lossless_fp16", "lossy_fp16", "raw_total"]:
    if EXPECTED[key] not in readme:
        fail(f"README.md missing canonical {key}={EXPECTED[key]}")

violations: list[str] = []
for path in PUBLIC_SURFACES:
    text = path.read_text(errors="replace")
    for pat in FORBIDDEN_PUBLIC_PATTERNS:
        for m in re.finditer(pat, text):
            line = text.count("\n", 0, m.start()) + 1
            violations.append(f"{path.relative_to(ROOT)}:{line}: stale public claim matches /{pat}/ -> {m.group(0)!r}")
if violations:
    print("\n".join(violations), file=sys.stderr)
    fail(f"{len(violations)} stale public claim surface(s)")

# CLAIMS receipts must exist, and validated PPL receipts must carry the headline
# ratio in a single field rather than forcing readers to derive it.
for name, claim in claims.items():
    for receipt in claim.get("receipts", []):
        p = ROOT / receipt
        if not p.exists():
            fail(f"{name}: missing receipt {receipt}")
    if claim.get("claim_status") == "PPL_validated" and "compressed_total_bytes" in claim:
        state_receipts = [ROOT / r for r in claim["receipts"] if Path(r).name.startswith("state")]
        if not state_receipts:
            fail(f"{name}: PPL claim has no state*.json receipt")
        for p in state_receipts:
            d = json.loads(p.read_text())
            phase1 = d.get("phase1", {})
            ratio = phase1.get("compression_ratio")
            if ratio is None:
                fail(f"{name}: {p.relative_to(ROOT)} phase1.compression_ratio is null")
            assert ratio is not None
            ratio_value = float(ratio)
            if not math.isclose(ratio_value, float(claim["ratio_vs_f32_raw"]), rel_tol=0, abs_tol=0.001):
                fail(f"{name}: receipt ratio {ratio} != CLAIMS ratio {claim['ratio_vs_f32_raw']}")
            if "ppl_window" in claim and d.get("ppl_window") != claim["ppl_window"]:
                fail(f"{name}: receipt ppl_window {d.get('ppl_window')} != CLAIMS {claim['ppl_window']}")
            if "raw_total_bytes" in phase1 and phase1["raw_total_bytes"] != claim["raw_total_bytes"]:
                fail(f"{name}: receipt raw_total_bytes {phase1['raw_total_bytes']} != CLAIMS {claim['raw_total_bytes']}")

print("OK: public claim surfaces and receipt headline fields agree with CLAIMS.json")
