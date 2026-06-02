#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
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
    env = os.environ.copy()
    env["PYTHONPATH"] = "python"
    started = time.perf_counter()
    cmd = [
        sys.executable,
        "-c",
        "import prove_kv, json; print(json.dumps({'native_available': prove_kv.native_available()}))",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env, check=False)
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)

    native_available = False
    if proc.returncode == 0:
        try:
            native_available = bool(json.loads(proc.stdout.strip())["native_available"])
        except (json.JSONDecodeError, KeyError, TypeError):
            native_available = False

    status = "pass" if native_available else "skip"
    result = {
        "schema_version": 1,
        "tier": "tier1-python-boundary",
        "status": status,
        "command": cmd,
        "elapsed_ms": elapsed_ms,
        "exit_code": proc.returncode,
        "native_available": native_available,
        "skip_reason": None
        if native_available
        else "prove_kv._native is not built; maturin unavailable or develop/build not run",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "claims": [],
    }
    (out_dir / "python_boundary_bench.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
