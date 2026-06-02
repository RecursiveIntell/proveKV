#!/usr/bin/env python3
"""Regression test design for z.py package hygiene.

This script is intentionally fixture-based. Codex should copy/adapt it into the repo
and make it pass after implementing root package hygiene in z.py.
"""
from __future__ import annotations
import json, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path

REPO_FILES = {
    "Cargo.toml": "[workspace]\nmembers = []\n",
    "README.md": "# fixture\n",
    "AGENTS.md": "# agents\n",
    "pyproject.toml": "[build-system]\nrequires=['maturin']\n",
    "python/prove_kv/__init__.py": "",
    "python/prove_kv/_native.pyi": "",
    "python/prove_kv/py.typed": "",
    ".codex-runs/current/commands_run.log": "$ cargo test\nok\n",
    "README_BUNDLE.md": "# old bundle\n",
    "BUNDLE_MANIFEST.json": "{}\n",
    "proveKV-generic-rust-next-codex-context-20260520.report.md": "# old report\n",
    "proveKV-generic-rust-next-codex-context-20260520.manifest.json": "{}\n",
    "proveKV-generic-rust-next-codex-context-20260520.zip": "fake zip bytes\n",
}


def main() -> int:
    source_z = Path("z.py").resolve()
    if not source_z.exists():
        print("z.py missing", file=sys.stderr)
        return 1
    source_assert = Path("scripts/assert_source_package_hygiene.py").resolve()
    if not source_assert.exists():
        print("scripts/assert_source_package_hygiene.py missing", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "fixture"
        root.mkdir()
        shutil.copy2(source_z, root / "z.py")
        (root / "scripts").mkdir()
        shutil.copy2(source_assert, root / "scripts/assert_source_package_hygiene.py")
        for rel, text in REPO_FILES.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        cmd = [
            sys.executable, "z.py", "--root", ".", "--profile", "generic-rust",
            "--mode", "next-codex-context", "--strict", "--archive-root-package-artifacts",
            "--archive-root-markdown-noise", "--output", "proveKV-generic-rust-next-codex-context-20990101.zip",
        ]
        result = subprocess.run(cmd, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(result.stdout)
            return result.returncode
        forbidden = ["README_BUNDLE.md", "BUNDLE_MANIFEST.json", "proveKV-generic-rust-next-codex-context-20260520.report.md"]
        still = [f for f in forbidden if (root / f).exists()]
        if still:
            print("root artifacts were not archived:", still)
            return 1
        manifest = root / "proveKV-generic-rust-next-codex-context-20990101.manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        paths = {item["path"] for item in payload.get("files", [])}
        required = {"python/prove_kv/_native.pyi", "python/prove_kv/py.typed", ".codex-runs/current/commands_run.log"}
        missing = sorted(required - paths)
        if missing:
            print("manifest missing required paths:", missing)
            return 1
        with zipfile.ZipFile(root / "proveKV-generic-rust-next-codex-context-20990101.zip") as zf:
            zip_paths = set(zf.namelist())
        zip_missing = sorted(required - zip_paths)
        if zip_missing:
            print("zip missing required paths:", zip_missing)
            return 1
        root_package_archive = payload.get("root_package_archive") or {}
        if root_package_archive.get("candidate_count", 0) < 3:
            print("root_package_archive did not record expected candidates:", root_package_archive)
            return 1
        archive_manifest = root_package_archive.get("manifest_path")
        if not archive_manifest or not Path(archive_manifest).exists():
            print("root package archive manifest missing:", archive_manifest)
            return 1
        hygiene = subprocess.run(
            [sys.executable, "scripts/assert_source_package_hygiene.py", "--repo-root", ".", "--manifest", str(manifest), "--mode", "manifest"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if hygiene.returncode != 0:
            print(hygiene.stdout)
            return hygiene.returncode
    print("z.py hygiene regression ok")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
