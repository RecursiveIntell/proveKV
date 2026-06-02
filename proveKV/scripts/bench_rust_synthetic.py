#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


def latest_run_id() -> str:
    runs = sorted(Path(".codex-runs").glob("*-proveKV-next"))
    if not runs:
        raise SystemExit("no .codex-runs/*-proveKV-next directory found")
    return runs[-1].name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=latest_run_id())
    args = parser.parse_args()

    out_dir = Path(".codex-runs") / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cmd = [
        "cargo",
        "test",
        "-p",
        "proveKV",
        "synthetic_q8_key_drift_is_bounded_and_finite",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)

    result = {
        "schema_version": 1,
        "tier": "tier0-rust-synthetic",
        "status": "pass" if proc.returncode == 0 else "fail",
        "command": cmd,
        "elapsed_ms": elapsed_ms,
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "claims": [],
    }
    (out_dir / "rust_synthetic_bench.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
