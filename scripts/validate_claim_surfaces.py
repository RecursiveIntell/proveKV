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
    "oracle_ppl": f"{lossless['oracle_ppl']:.6f}",
    "roundtrip_ppl": f"{lossless['roundtrip_ppl']:.6f}",
    "delta_ppl_pct": f"{lossless['delta_ppl_pct']:+.2f}%",
}

# Stale headline values from the superseded denominator and window contract.
FORBIDDEN_PUBLIC_PATTERNS = [
    r"36\.00\s*[×x]",
    r"68\.04\s*[×x]",
    r"18\.00\s*[×x]",
    r"34\.02\s*[×x]",
    r"2,315,255,808",
    r"\[128,\s*1024\)",
    r"--shell-bits",
]

# Legacy b=8 stale values also changed after the denominator correction.
FORBIDDEN_PUBLIC_PATTERNS += [
    r"37\.31\s*[×x]",
    r"65\.88\s*[×x]",
    r"18\.65\s*[×x]",
    r"32\.94\s*[×x]",
]

# Stale language from the invalid full-cache/full-input aggregate evaluation.
# These patterns are narrow enough not to reject separately scoped historical
# single-pool and shell receipts that still report +0.00%.
FORBIDDEN_PUBLIC_PATTERNS += [
    r"Fresh aggregate-fixture check: oracle PPL 7\.203125, roundtrip PPL 7\.203125",
    r"Aggregate fixture ΔPPL=\+0\.00%",
    r"aggregate[^\n]{0,120}fixture matched oracle PPL",
    r"oracle and roundtrip PPL equal at reported precision",
    r"oracle and roundtrip 7\.2031",
    r"PPL neutrality on the measured configurations is the strongest claim",
    r"11\.13× compression with ΔPPL=\+0\.00%",
    r"validated at 11\.13× compression with ΔPPL=\+0\.00%",
    r"11\.13× lossless ΔPPL",
    r"(?:40\.50|20\.25)×\s+lossless",
    r"(?:76\.54|38\.27)×\s+lossy",
    r"40\.50[×x]\s*/\s*76\.54[×x]",
]

PUBLIC_SURFACES = [
    ROOT / "README.md",
    ROOT / "proveKV" / "README.md",
    ROOT / "REPRODUCE.md",
    ROOT / "CITATION.cff",
    ROOT / "docs" / "methodology" / "naive_computation.md",
    ROOT / "docs" / "SYSTEM_NAMING_AND_BRANDING.md",
    ROOT / "docs" / "img" / "_make_visuals.py",
    ROOT / "docs" / "img" / "architecture.svg",
    ROOT / "docs" / "img" / "cross_validation.svg",
    ROOT / "docs" / "img" / "n_scaling.svg",
    ROOT / "proveKV" / "docs" / "img" / "architecture.svg",
    ROOT / "proveKV" / "docs" / "img" / "cross_validation.svg",
    ROOT / "proveKV" / "docs" / "img" / "n_scaling.svg",
    ROOT / "proveKV" / "src" / "lib.rs",
    ROOT / "proveKV" / "src" / "policy.rs",
    ROOT / "proveKV" / "examples" / "prove_kv_multi_agent_shell.rs",
]

# Validate public docs/source surfaces plus checked-in human-facing reports.

def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)

# Required canonical strings should appear on the main public surface.
readme = (ROOT / "README.md").read_text()
for key in [
    "lossless_f32", "lossy_f32", "lossless_fp16", "lossy_fp16", "raw_total",
    "oracle_ppl", "roundtrip_ppl", "delta_ppl_pct",
]:
    if EXPECTED[key] not in readme:
        fail(f"README.md missing canonical {key}={EXPECTED[key]}")
for phrase in (
    "aggregate fixture",
    "does **not** establish eight independent-prompt PPL results",
    "shell K/V receipts support the size result",
    "PPL-neutrality claim is not publication-eligible",
    "2,604,662,784",
):
    if phrase not in readme:
        fail(f"README.md missing claim-boundary phrase {phrase!r}")

# Keep each visual on one estimand. The scaling chart is the historical
# Qwen0.5B size-only series at every N; the architecture diagram is the
# current SmolLM2 N=8 f32-radii size profile plus its degraded PPL note.
n_scaling_svg = (ROOT / "docs" / "img" / "n_scaling.svg").read_text()
if "Qwen2.5-0.5B synthetic size-only sweep at every N" not in n_scaling_svg:
    fail("n_scaling.svg does not identify one Qwen0.5B size-only series")
for current_ratio in (EXPECTED["lossless_f32"], EXPECTED["lossy_f32"]):
    if current_ratio in n_scaling_svg:
        fail(f"n_scaling.svg splices current SmolLM2 ratio {current_ratio} into the Qwen series")

architecture_svg = (ROOT / "docs" / "img" / "architecture.svg").read_text()
for phrase in (
    "codebook quantizer",
    "14.06 MiB",
    "5.91 MiB",
    "2.43 GiB at N=8",
    "PPL 7.2031 → 24.2344 (+236.44%)",
    "shell K/V unmeasured",
):
    if phrase not in architecture_svg:
        fail(f"architecture.svg missing current-profile evidence {phrase!r}")

for readme_path in (ROOT / "README.md", ROOT / "proveKV" / "README.md"):
    readme_text = readme_path.read_text()
    for phrase in (
        "Historical single-pool receipts (PPL outputs unadmitted)",
        "Historical standalone-shell receipt (PPL output unadmitted)",
        "full-cache/full-input pattern",
        "historical standalone-shell receipt whose PPL output is explicitly",
    ):
        if phrase not in readme_text:
            fail(f"{readme_path.relative_to(ROOT)} missing historical-PPL boundary {phrase!r}")

branding = (ROOT / "docs" / "SYSTEM_NAMING_AND_BRANDING.md").read_text()
if "Numeric PPL language below predates" not in branding or "CLAIMS.json` is the current claim" not in branding:
    fail("historical naming document lacks an explicit current-claim disclaimer")

cross_validation_svg = (ROOT / "docs" / "img" / "cross_validation.svg").read_text()
if "PPL values unadmitted (full-cache/full-input method)" not in cross_validation_svg:
    fail("cross_validation.svg does not label historical PPL outputs as unadmitted")

historical_ppl_reports = [
    ROOT / "results/ppl_shell/smollm2-1.7b/wikitext-2/report.md",
    ROOT / "results/ppl_multi_agent/smollm2-1.7b/wikitext-2-n8/report.md",
    ROOT / "results/ppl/smollm2-1.7b/wikitext-2-lossy/report.md",
    ROOT / "results/ppl/smollm2-1.7b/wikitext-2-lossless/report.md",
    ROOT / "results/bench/ppl/tinyllama-1.1b/wikitext-2/report.md",
    ROOT / "results/bench/ppl/qwen2.5-0.5b/wikitext-2/report.md",
    ROOT / "results/bench/ppl/smollm2-1.7b/code-source/report.md",
    ROOT / "results/bench/ppl/smollm2-1.7b/wikitext-2-n1280/report.md",
    ROOT / "results/bench/ppl/smollm2-1.7b/wikitext-2/report.md",
]
for report_path in historical_ppl_reports:
    if not report_path.is_file():
        fail(f"missing historical PPL report {report_path.relative_to(ROOT)}")
    report = report_path.read_text()
    if "Historical, unadmitted quality output" not in report or "not publication evidence" not in report:
        fail(f"{report_path.relative_to(ROOT)} lacks an unadmitted-quality banner")

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
    if claim.get("claim_status") in {
        "PPL_validated",
        "size_validated_shared_prefix_ppl_degraded",
    } and "compressed_total_bytes" in claim:
        state_receipts = [
            ROOT / r
            for r in claim["receipts"]
            if Path(r).name in {"state_lossless.json", "state_lossy.json"}
        ]
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
            for field in ("oracle_ppl", "roundtrip_ppl", "delta_ppl_pct"):
                if phase1.get(field) != claim.get(field):
                    fail(f"{name}: receipt {field} {phase1.get(field)} != CLAIMS {claim.get(field)}")
        if claim.get("claim_status") == "size_validated_shared_prefix_ppl_degraded":
            if claim.get("publication_eligible") is not True:
                fail(f"{name}: size publication eligibility must be explicit")
            if claim.get("ppl_publication_eligible") is not False:
                fail(f"{name}: PPL-neutrality publication must be explicitly ineligible")
            if not float(claim["roundtrip_ppl"]) > float(claim["oracle_ppl"]):
                fail(f"{name}: degraded claim must retain roundtrip_ppl > oracle_ppl")
    if claim.get("claim_status") in {"PPL_validated", "ppl_neutral"}:
        fail(f"{name}: deprecated PPL-neutral status remains active")
    if claim.get("claim_status") == "historical_unadmitted":
        if claim.get("publication_eligible") is not False:
            fail(f"{name}: historical claim must be publication-ineligible")
        if claim.get("ppl_publication_eligible") is not False:
            fail(f"{name}: historical PPL claim must be explicitly ineligible")

print("OK: public claim surfaces and receipt headline fields agree with CLAIMS.json")
