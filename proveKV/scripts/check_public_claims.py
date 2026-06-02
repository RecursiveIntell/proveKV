#!/usr/bin/env python3
from pathlib import Path
import re
import sys

targets = [p for p in [Path("README.md"), Path("crates/proveKV/README.md"), Path("docs/README_DRAFT.md")] if p.exists()]
banned = [
    r"\bproduction[- ]ready\b",
    r"\bguaranteed\b",
    r"\blossless\b",
    r"\bzero[- ]loss\b",
    r"\boutperforms\b",
    r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bvLLM compatible\b",
    r"\bllama\.cpp compatible\b",
    r"\bCandle compatible\b",
    r"\bBurn compatible\b",
]
failures = []
for path in targets:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for pat in banned:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            failures.append((str(path), pat, m.start()))

if failures:
    print("Public claim boundary violations:")
    for f in failures:
        print(f"  {f[0]} matched {f[1]} at offset {f[2]}")
    sys.exit(1)

print("public claim boundary ok")
