#!/usr/bin/env python3
"""Validate proveKV source package hygiene from repo root and/or z.py manifest."""
from __future__ import annotations
import argparse, glob, json, re, sys, zipfile
from pathlib import Path

ROOT_FORBIDDEN_EXACT = {
    "README_BUNDLE.md",
    "BUNDLE_MANIFEST.json",
}
ROOT_FORBIDDEN_RE = [
    re.compile(r".*-next-codex-context-[A-Za-z0-9T]+Z?\.(zip|manifest\.json|report\.md|excluded\.json|findings\.json|codex-archive\.json)$"),
    re.compile(r".*-generic-rust-next-codex-context-[A-Za-z0-9T]+Z?\.(zip|manifest\.json|report\.md|excluded\.json|findings\.json|codex-archive\.json)$"),
]
REQUIRED_PACKAGE_PATHS = {
    "python/prove_kv/_native.pyi",
    "python/prove_kv/py.typed",
}


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_manifest(pattern: str) -> Path | None:
    matches = sorted(Path('.').glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def check_root(repo: Path) -> list[str]:
    errors = []
    for p in sorted(repo.iterdir()):
        if not p.is_file():
            continue
        name = p.name
        if name in ROOT_FORBIDDEN_EXACT:
            errors.append(f"forbidden root package artifact remains: {name}")
        for rx in ROOT_FORBIDDEN_RE:
            if rx.match(name):
                errors.append(f"forbidden prior package sidecar/archive remains at root: {name}")
                break
    return errors


def check_manifest(manifest_path: Path) -> list[str]:
    payload = load_manifest(manifest_path)
    files = {item.get("path") for item in payload.get("files", []) if isinstance(item, dict)}
    errors = []
    for req in REQUIRED_PACKAGE_PATHS:
        if req not in files:
            errors.append(f"manifest missing required package path: {req}")

    has_command_evidence = any(
        str(path).endswith("commands_run.log") or str(path).endswith("commands_run.receipts.jsonl")
        for path in files
    )
    if not has_command_evidence:
        errors.append("manifest missing command evidence: commands_run.log or commands_run.receipts.jsonl")

    package_path = Path(str(payload.get("package", "")))
    if not package_path.is_absolute():
        package_path = manifest_path.parent / package_path
    if package_path.exists() and package_path.suffix == ".zip":
        try:
            with zipfile.ZipFile(package_path) as zf:
                zip_paths = set(zf.namelist())
        except zipfile.BadZipFile as exc:
            errors.append(f"package is not a readable zip: {package_path}: {exc}")
            zip_paths = set()
        for req in REQUIRED_PACKAGE_PATHS:
            if req not in zip_paths:
                errors.append(f"zip missing required package path: {req}")
        has_zip_command_evidence = any(
            path.endswith("commands_run.log") or path.endswith("commands_run.receipts.jsonl")
            for path in zip_paths
        )
        if not has_zip_command_evidence:
            errors.append("zip missing command evidence: commands_run.log or commands_run.receipts.jsonl")
    elif payload.get("report", {}).get("archive_written"):
        errors.append(f"manifest says archive was written but package is missing or not zip: {package_path}")

    report = payload.get("report", {})
    # Support the future field; do not fail old manifests solely for missing field if this validator is run pre-implementation.
    if "root_package_archive" in payload:
        rpa = payload.get("root_package_archive") or {}
        if rpa.get("errors"):
            errors.append(f"root package archive reported errors: {rpa.get('errors')}")
    elif "root_package_archive" in report:
        rpa = report.get("root_package_archive") or {}
        if rpa.get("errors"):
            errors.append(f"root package archive reported errors: {rpa.get('errors')}")
    else:
        errors.append("manifest/report missing root_package_archive summary")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--mode", choices=["prepackage", "manifest", "both"], default="both")
    args = ap.parse_args()
    repo = Path(args.repo_root).resolve()
    errors = []
    if args.mode in {"prepackage", "both"}:
        errors.extend(check_root(repo))
    if args.mode in {"manifest", "both"}:
        manifest = Path(args.manifest) if args.manifest else find_manifest("*-next-codex-context-*.manifest.json")
        if not manifest:
            errors.append("no manifest found")
        else:
            errors.extend(check_manifest(manifest))
    if errors:
        print("source package hygiene failures:")
        for e in errors:
            print(" -", e)
        return 1
    print("source package hygiene ok")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
