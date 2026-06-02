#!/usr/bin/env python3
import json
from pathlib import Path

paths = [
    Path("docs/POLY_KV_SCHEMA_PROPOSAL.json"),
    Path("docs/proveKV/POLY_KV_SCHEMA_PROPOSAL.json"),
]
found = False
for p in paths:
    if p.exists():
        found = True
        data = json.loads(p.read_text())
        assert "$schema" in data, f"{p}: missing $schema"
        assert "definitions" in data, f"{p}: missing definitions"
        required = [
            "KvPoolManifestV1",
            "PoolBuildReceiptV1",
            "ReaderInjectionReceiptV1",
            "DecodeReceiptV1",
            "FallbackReceiptV1",
            "QualityGateResultV1",
        ]
        missing = [r for r in required if r not in data["definitions"]]
        if missing:
            raise SystemExit(f"{p}: missing definitions {missing}")
        print(f"schema ok: {p}")
if not found:
    print("WARN: no schema proposal found yet")
