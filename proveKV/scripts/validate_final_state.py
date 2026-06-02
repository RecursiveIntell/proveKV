#!/usr/bin/env python3
from pathlib import Path
import sys

required_any = [
    Path("Cargo.toml"),
]
for p in required_any:
    if not p.exists():
        print(f"missing required file: {p}")
        sys.exit(1)

expected = [
    "crates/quant-codec-core/Cargo.toml",
    "crates/proveKV/Cargo.toml",
    "crates/quant-codec-core/src/lib.rs",
    "crates/proveKV/src/lib.rs",
]
missing = [p for p in expected if not Path(p).exists()]
if missing:
    print("missing expected alpha files:")
    for p in missing:
        print(f"  {p}")
    sys.exit(1)

forbidden_dirs = [
    "crates/quant-governor",
    "crates/scr-runtime-compression",
    "crates/semantic-memory-compression",
]
present = [p for p in forbidden_dirs if Path(p).exists()]
if present:
    print("forbidden out-of-scope dirs present:")
    for p in present:
        print(f"  {p}")
    sys.exit(1)

print("final state shape ok")
