#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib, re, sys
ROOT = pathlib.Path.cwd()
findings = []

def add(sev, code, path, msg):
    findings.append({"severity": sev, "code": code, "path": str(path), "message": msg})

def read(path):
    p = ROOT / path
    return p.read_text(encoding="utf-8") if p.exists() else ""

cargo = read("Cargo.toml")
if not cargo:
    add("blocker", "NO_CARGO_TOML", "Cargo.toml", "missing Cargo.toml")
else:
    if "workspace = true" in cargo:
        add("blocker", "WORKSPACE_INHERITANCE", "Cargo.toml", "standalone crate still uses workspace inheritance")
    for key in ["description", "license", "readme", "repository", "documentation"]:
        if re.search(rf"^\s*{key}\s*=", cargo, re.M) is None:
            add("blocker", "MISSING_METADATA", "Cargo.toml", f"missing package metadata key: {key}")
    if "z.py" not in cargo and "include" not in cargo and "exclude" not in cargo:
        add("high", "PACKAGE_SURFACE_UNBOUNDED", "Cargo.toml", "no include/exclude package surface rule")

for required in ["LICENSE", "CITATION.cff", "CHANGELOG.md", "RELEASE_CHECKLIST.md"]:
    if not (ROOT / required).exists():
        add("blocker", "MISSING_RELEASE_FILE", required, f"missing {required}")

for required in [
    "docs/compression/FIBQUANT_SOURCE_BASIS.md",
    "docs/compression/FIBQUANT_MATH_CONFORMANCE.md",
    "docs/compression/FIBQUANT_BENCHMARK_PLAN.md",
    "docs/compression/FIBQUANT_PUBLICATION_NONCLAIMS.md",
]:
    if not (ROOT / required).exists():
        add("high", "MISSING_COMPRESSION_DOC", required, f"missing {required}")

if (ROOT / "z.py").exists():
    add("high", "ZPY_IN_REPO_ROOT", "z.py", "z.py exists in crate root; package include/exclude must prevent shipping it")

spherical = read("src/spherical_beta.rs")
if "d - k - 2" in spherical:
    add("blocker", "BETA_DK_USIZE_UNDERFLOW", "src/spherical_beta.rs", "beta_d_k still contains unsigned d-k-2 expression")

profile = read("src/profile.rs")
for token in ["paper_rate_bits_per_coord", "wire_bits_per_coord", "schema_version", "radius_method", "direction_method", "lloyd_restarts", "lloyd_iterations"]:
    if token not in profile:
        add("blocker", "PROFILE_FIELD_MISSING", "src/profile.rs", f"profile missing {token}")
# heuristic: validate() body should mention important invariants more than declarations do
if profile.count("paper_rate_bits_per_coord") < 3 or profile.count("wire_bits_per_coord") < 3:
    add("high", "PROFILE_RATE_NOT_ENFORCED", "src/profile.rs", "rate fields do not appear to be enforced in validate()")

receipt = read("src/receipt.rs")
if "source_vector_digest" not in receipt:
    add("blocker", "RECEIPT_SOURCE_DIGEST_MISSING", "src/receipt.rs", "receipt lacks source_vector_digest")

codec = read("src/codec.rs")
if "unwrap_or_default" in codec:
    add("blocker", "DIGEST_FAIL_OPEN", "src/codec.rs", "encoded_digest uses unwrap_or_default/fail-open behavior")
if "pub fn encoded_digest" in codec and "Result" not in codec[codec.find("pub fn encoded_digest"):codec.find("pub fn encoded_digest")+120]:
    add("high", "ENCODED_DIGEST_NOT_RESULT", "src/codec.rs", "encoded_digest likely does not return Result")
if "fib_code_v1" not in codec or "schema_version" not in codec:
    add("high", "CODE_SCHEMA_NOT_VALIDATED", "src/codec.rs", "decode/header path may not validate schema_version")

readme = read("README.md")
for term in ["arXiv", "experimental", "does not", "benchmark"]:
    if term.lower() not in readme.lower():
        add("medium", "README_THIN", "README.md", f"README missing public-positioning term: {term}")

out = {"root": str(ROOT), "finding_count": len(findings), "findings": findings}
print(json.dumps(out, indent=2))
pathlib.Path("target").mkdir(exist_ok=True)
(pathlib.Path("target") / "fibquant-publish-preflight.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
# Preflight may find blockers before patch. Do not fail hard; final assert will.
sys.exit(0)
