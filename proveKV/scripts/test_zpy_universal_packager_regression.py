#!/usr/bin/env python3
"""Reference regression fixture for the universal z.py packager pass.

This is intentionally small and self-contained. Copy it into the repo after the
implementation pass and wire it to the actual z.py CLI. It assumes the future CLI
supports the legacy package invocation and a portable verify command.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    repo = Path.cwd()
    zpy = repo / "z.py"
    if not zpy.exists():
        print("missing z.py in cwd", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="zpy-universal-fixture-") as td:
        root = Path(td) / "fixture"
        root.mkdir()
        shutil.copy2(zpy, root / "z.py")

        # Mixed repo fixture.
        write(root / "README.md", "# Fixture\n")
        write(root / "LICENSE", "MIT\n")
        write(root / "Cargo.toml", "[package]\nname='fixture'\nversion='0.1.0'\nedition='2021'\n")
        write(root / "src/lib.rs", "pub const DATA: &str = include_str!(\"../data/example.txt\");\n")
        write(root / "data/example.txt", "ok\n")
        write(root / "pyproject.toml", "[project]\nname='fixture-py'\nversion='0.1.0'\n")
        write(root / "python/fixture/__init__.py", "__version__='0.1.0'\n")
        write(root / "python/fixture/py.typed", "\n")
        write(root / "python/fixture/_native.pyi", "def hello() -> str: ...\n")
        write(root / "package.json", '{"name":"fixture-node","version":"0.1.0","files":["lib"]}\n')
        write(root / "lib/index.js", "module.exports = {};\n")
        write(root / ".dockerignore", "target\nnode_modules\ndist\n")
        write(root / "Dockerfile", "FROM scratch\nCOPY README.md /README.md\n")
        write(root / ".gitattributes", "dist/** export-ignore\n")
        write(root / ".codex-runs/run/commands_run.log", "echo ok\n")
        write(root / ".codex-runs/run/touched_diff.patch", "diff --git a/x b/x\n")
        write(root / "old-next-codex-context-20260101T000000Z.zip", "fake old package\n")
        write(root / "old-next-codex-context-20260101T000000Z.manifest.json", "{}\n")
        (root / "target/debug").mkdir(parents=True)
        write(root / "target/debug/noise", "noise\n")
        (root / "node_modules/pkg").mkdir(parents=True)
        write(root / "node_modules/pkg/noise.js", "noise\n")

        package = root / "fixture-next-codex-context.zip"
        cmd = [sys.executable, "z.py", "--root", ".", "--mode", "next-codex-context", "--output", str(package), "--strict"]
        result = run(cmd, root)
        if result.returncode != 0:
            print(result.stdout)
            return result.returncode

        manifest = package.with_suffix(".manifest.json")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        paths = {f["path"] for f in payload.get("files", [])}
        required = {
            "python/fixture/py.typed",
            "python/fixture/_native.pyi",
            ".codex-runs/run/commands_run.log",
            ".codex-runs/run/touched_diff.patch",
        }
        missing = sorted(required - paths)
        if missing:
            print("missing expected paths:", missing)
            return 3
        forbidden = [p for p in paths if p.startswith("target/") or p.startswith("node_modules/")]
        if forbidden:
            print("forbidden paths included:", forbidden)
            return 4
        root_archive = payload.get("root_package_archive") or payload.get("report", {}).get("root_package_archive")
        if not root_archive:
            print("missing root_package_archive report")
            return 5

        # Portable verify check. Accept either subcommand or legacy option.
        verify_cmds = [
            [sys.executable, "z.py", "verify", "--package", str(package), "--manifest", str(manifest), "--strict"],
            [sys.executable, "z.py", "--verify-package", str(package), "--manifest", str(manifest), "--strict"],
        ]
        ok = False
        for vcmd in verify_cmds:
            v = run(vcmd, root)
            if v.returncode == 0:
                ok = True
                break
        if not ok:
            print("portable verify command did not pass; implement one of the accepted CLI shapes")
            return 6

    print("zpy universal packager regression passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
