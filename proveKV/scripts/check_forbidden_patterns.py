#!/usr/bin/env python3
from pathlib import Path
import sys, re

scan_roots = [Path("crates"), Path("src"), Path("tests"), Path("benches"), Path("examples")]
patterns = [
    ("todo", re.compile(r"\bTODO\b|\bFIXME\b|\bTBD\b")),
    ("placeholder", re.compile(r"@filename|\{feature\}|<placeholder>", re.IGNORECASE)),
    ("unsafe", re.compile(r"\bunsafe\b")),
]
failures = []
for root in scan_roots:
    if not root.exists():
        continue
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".rs", ".md", ".toml", ".json"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for name, pat in patterns:
                for m in pat.finditer(text):
                    failures.append((name, str(path), m.group(0), m.start()))
if failures:
    print("forbidden pattern findings:")
    for f in failures:
        print(f"  {f[0]} {f[1]} {f[2]} @ {f[3]}")
    sys.exit(1)
print("forbidden pattern check ok")
