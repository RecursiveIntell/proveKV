#!/usr/bin/env python3
from __future__ import annotations
import json, pathlib, re, subprocess, sys
ROOT = pathlib.Path.cwd()
failures = []
warnings = []

def fail(code, msg): failures.append({"code": code, "message": msg})
def warn(code, msg): warnings.append({"code": code, "message": msg})
def read(path):
    p = ROOT / path
    return p.read_text(encoding="utf-8") if p.exists() else ""

cargo = read("Cargo.toml")
if "workspace = true" in cargo:
    fail("WORKSPACE_INHERITANCE", "Cargo.toml still uses workspace inheritance")
for key in ["description", "license", "readme", "repository", "documentation", "keywords", "categories"]:
    if re.search(rf"^\s*{key}\s*=", cargo, re.M) is None:
        fail("MISSING_METADATA", f"Cargo.toml missing {key}")
if "include" not in cargo and "exclude" not in cargo:
    fail("PACKAGE_SURFACE", "Cargo.toml must include include/exclude rules")

for required in ["LICENSE", "CITATION.cff", "CHANGELOG.md", "RELEASE_CHECKLIST.md"]:
    if not (ROOT / required).exists(): fail("MISSING_RELEASE_FILE", f"missing {required}")
for required in [
    "docs/compression/FIBQUANT_SOURCE_BASIS.md",
    "docs/compression/FIBQUANT_MATH_CONFORMANCE.md",
    "docs/compression/FIBQUANT_BENCHMARK_PLAN.md",
    "docs/compression/FIBQUANT_PUBLICATION_NONCLAIMS.md",
    "docs/compression/FIBQUANT_PUBLISH_READINESS_REPORT.md",
]:
    if not (ROOT / required).exists(): fail("MISSING_DOC", f"missing {required}")

spherical = read("src/spherical_beta.rs")
if "d - k - 2" in spherical:
    fail("BETA_DK_USIZE_UNDERFLOW", "src/spherical_beta.rs still contains d-k-2 unsigned expression")
if "beta.is_finite()" not in spherical or "beta > 0.0" not in spherical:
    fail("BETA_DK_VALIDATION", "beta_d_k must validate positive finite beta shape")

profile = read("src/profile.rs")
for token in ["paper_rate_bits_per_coord", "wire_bits_per_coord", "radius_method", "direction_method", "lloyd_restarts", "lloyd_iterations", "schema_version"]:
    if profile.count(token) < 2:
        fail("PROFILE_INVARIANT_WEAK", f"profile invariant for {token} does not appear enforced")
if profile.count("paper_rate_bits_per_coord") < 3:
    fail("PAPER_RATE_UNCHECKED", "paper_rate_bits_per_coord must be checked in validate()")
if profile.count("wire_bits_per_coord") < 3:
    fail("WIRE_RATE_UNCHECKED", "wire_bits_per_coord must be checked in validate()")

receipt = read("src/receipt.rs")
if "source_vector_digest" not in receipt:
    fail("SOURCE_DIGEST_MISSING", "receipt lacks source_vector_digest")
if "norm_format" not in receipt:
    fail("NORM_FORMAT_MISSING", "receipt lacks norm_format")

codec = read("src/codec.rs")
if "unwrap_or_default" in codec:
    fail("FAIL_OPEN_DIGEST", "codec still uses unwrap_or_default")
if "pub fn encoded_digest" in codec and "Result" not in codec[codec.find("pub fn encoded_digest"):codec.find("pub fn encoded_digest")+160]:
    fail("ENCODED_DIGEST_NOT_RESULT", "encoded_digest must return Result<String>")
if "fib_code_v1" not in codec or "schema_version" not in codec:
    fail("SCHEMA_VALIDATION_WEAK", "codec decode path must validate schema_version")

readme = read("README.md")
for term in ["arXiv", "experimental", "benchmark", "does not claim", "Namyoon Lee", "Yongjune Kim"]:
    if term.lower() not in readme.lower():
        fail("README_PUBLIC_POSITIONING", f"README missing: {term}")

# Best-effort package-list check if cargo exists.
try:
    res = subprocess.run(["cargo", "package", "--list", "--allow-dirty"], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    (ROOT / "target").mkdir(exist_ok=True)
    (ROOT / "target" / "fibquant-package-list-final.txt").write_text(res.stdout + "\n--- stderr ---\n" + res.stderr, encoding="utf-8")
    if res.returncode != 0:
        fail("CARGO_PACKAGE_LIST_FAIL", "cargo package --list failed; see target/fibquant-package-list-final.txt")
    else:
        forbidden = ["z.py", "docs/codex-runs/", ".zip", ".manifest.json", ".findings.json", ".excluded.json", ".codex-archive.json"]
        for item in forbidden:
            if item in res.stdout:
                fail("PACKAGE_FORBIDDEN_FILE", f"cargo package --list includes forbidden pattern: {item}")
except FileNotFoundError:
    warn("CARGO_MISSING", "cargo not found; final publish readiness cannot be proven here")
except subprocess.TimeoutExpired:
    fail("CARGO_PACKAGE_LIST_TIMEOUT", "cargo package --list timed out")

out = {"root": str(ROOT), "failures": failures, "warnings": warnings}
print(json.dumps(out, indent=2))
(ROOT / "target").mkdir(exist_ok=True)
(ROOT / "target" / "fibquant-publish-final-assert.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
if failures:
    sys.exit(1)
