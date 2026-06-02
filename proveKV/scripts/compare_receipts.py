#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def latest_run_id() -> str:
    runs = sorted(Path(".codex-runs").glob("*-proveKV-next"))
    if not runs:
        raise SystemExit("no .codex-runs/*-proveKV-next directory found")
    return runs[-1].name


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=latest_run_id())
    args = parser.parse_args()

    out_dir = Path(".codex-runs") / args.run_id
    rust_path = out_dir / "rust_synthetic_bench.json"
    python_path = out_dir / "python_boundary_bench.json"
    missing = [str(p) for p in [rust_path, python_path] if not p.exists()]

    if missing:
        result = {
            "schema_version": 1,
            "status": "fail",
            "missing_inputs": missing,
            "checks": [],
        }
    else:
        rust = load_json(rust_path)
        python = load_json(python_path)
        checks = [
            {
                "name": "rust_synthetic_status",
                "status": rust.get("status"),
                "ok": rust.get("status") == "pass",
            },
            {
                "name": "python_boundary_status",
                "status": python.get("status"),
                "ok": python.get("status") in {"pass", "skip"},
                "skip_reason": python.get("skip_reason"),
            },
        ]
        result = {
            "schema_version": 1,
            "status": "pass" if all(item["ok"] for item in checks) else "fail",
            "checks": checks,
            "claims": [],
        }

    (out_dir / "receipt_parity_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
