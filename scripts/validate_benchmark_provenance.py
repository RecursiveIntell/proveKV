#!/usr/bin/env python3
"""Fail-closed admission for public N=8 benchmark claims.

Historical receipts remain immutable. A claim is admissible only when its current
state, pool receipt, agent receipt, cache identity, denominator, and PPL window
form one recomputable contract.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import math
import re
import sys
import tarfile
from pathlib import Path
from pathlib import PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "CLAIMS.json"
DEFAULT_CLAIMS = (
    "smollm2_wikitext2_n8_lossless_default",
    "smollm2_wikitext2_n8_lossy_default",
)
PPL_WINDOW_KIND = "scored_target_token_indices_half_open"
SOURCE_EXACT_PATHS = {
    "Cargo.toml",
    "Cargo.lock",
    "proveKV/Cargo.toml",
    "proveKV/examples/prove_kv_multi_agent_shell.rs",
    "proveKV/scripts/ppl_validate_multi_agent.py",
    "proveKV/scripts/compute_system_ratio.py",
    "fib-quant/Cargo.toml",
    "turbo-quant/Cargo.toml",
    "gpu-backend/Cargo.toml",
    "quant-codec-core/Cargo.toml",
}
SOURCE_PREFIXES = (
    "proveKV/src/",
    "fib-quant/src/",
    "turbo-quant/src/",
    "gpu-backend/src/",
    "gpu-backend/kernels/",
    "quant-codec-core/src/",
)
SECRET_PREFIX = re.compile(
    rb"sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    rb"AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY"
)
PRIVATE_SOURCE_MARKERS = (b"/home/sikmindz/", b"/home/jstevenson/", b"192.168.")


def canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def finite_positive(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_json(path: Path, label: str, failures: list[str]) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"{label}: unreadable JSON: {exc}")
        return None
    if not isinstance(value, dict):
        failures.append(f"{label}: expected a JSON object")
        return None
    return value


def one_receipt(claim: dict, basename: str, name: str, failures: list[str]) -> Path | None:
    receipts = claim.get("receipts")
    if not isinstance(receipts, list):
        failures.append(f"{name}: receipts must be a list")
        return None
    matches = [ROOT / value for value in receipts if isinstance(value, str) and Path(value).name == basename]
    if len(matches) != 1:
        failures.append(f"{name}: expected exactly one {basename} receipt, found {len(matches)}")
        return None
    if not matches[0].is_file():
        failures.append(f"{name}: missing receipt {matches[0].relative_to(ROOT)}")
        return None
    return matches[0]


def ppl_state_receipt(claim: dict, name: str, failures: list[str]) -> Path | None:
    receipts = claim.get("receipts")
    if not isinstance(receipts, list):
        failures.append(f"{name}: receipts must be a list")
        return None
    matches = [
        ROOT / value
        for value in receipts
        if isinstance(value, str) and Path(value).name in {"state_lossless.json", "state_lossy.json"}
    ]
    if len(matches) != 1:
        failures.append(f"{name}: expected exactly one PPL state receipt, found {len(matches)}")
        return None
    if not matches[0].is_file():
        failures.append(f"{name}: missing receipt {matches[0].relative_to(ROOT)}")
        return None
    return matches[0]


def validate_claim(name: str, claim: object) -> list[str]:
    failures: list[str] = []
    if not isinstance(claim, dict):
        return [f"{name}: claim must be an object"]
    state_path = ppl_state_receipt(claim, name, failures)
    pool_path = one_receipt(claim, "shared_pool_receipt.json", name, failures)
    agents_path = one_receipt(claim, "agents_receipt.json", name, failures)
    source_manifest_path = one_receipt(claim, "source_manifest.json", name, failures)
    source_envelope_path = one_receipt(claim, "source_envelope.json", name, failures)
    token_ids_path = one_receipt(claim, "token_ids.json", name, failures)
    build_attestation_path = one_receipt(claim, "build_attestation.json", name, failures)
    if (
        state_path is None
        or pool_path is None
        or agents_path is None
        or source_manifest_path is None
        or source_envelope_path is None
        or token_ids_path is None
        or build_attestation_path is None
    ):
        return failures

    state = load_json(state_path, name, failures)
    pool = load_json(pool_path, name, failures)
    agents = load_json(agents_path, name, failures)
    source_manifest = load_json(source_manifest_path, name, failures)
    source_envelope = load_json(source_envelope_path, name, failures)
    token_ids_receipt = load_json(token_ids_path, name, failures)
    build_attestation = load_json(build_attestation_path, name, failures)
    shell_state_path = pool_path.parent / "state.json"
    if str(shell_state_path.relative_to(ROOT)) not in claim.get("receipts", []):
        failures.append(f"{name}: shell state is not listed as a claim receipt")
    shell_state = load_json(shell_state_path, name, failures)
    if (
        state is None
        or pool is None
        or agents is None
        or source_manifest is None
        or source_envelope is None
        or token_ids_receipt is None
        or build_attestation is None
        or shell_state is None
    ):
        return failures
    phase0 = state.get("phase0")
    phase1 = state.get("phase1")
    model_config = state.get("model_config")
    cache_identity = state.get("cache_identity")
    execution_identity = state.get("execution_identity")
    if not isinstance(phase0, dict) or phase0.get("status") != "complete":
        failures.append(f"{name}: phase0 is not complete")
    if not isinstance(phase1, dict) or phase1.get("status") != "complete":
        failures.append(f"{name}: phase1 is not complete")
    elif phase1.get("shell_output_reused") is not False:
        failures.append(f"{name}: publication state must generate its own shell artifacts")
    if not isinstance(model_config, dict):
        failures.append(f"{name}: model_config is missing")
    if not isinstance(cache_identity, dict):
        failures.append(f"{name}: cache_identity is missing")
    if not isinstance(execution_identity, dict):
        failures.append(f"{name}: execution_identity is missing")
    if failures:
        return failures
    assert isinstance(phase0, dict)
    assert isinstance(phase1, dict)
    assert isinstance(model_config, dict)
    assert isinstance(cache_identity, dict)
    assert isinstance(execution_identity, dict)

    integers = ("n_shared", "n_unique", "n_agents", "n_total")
    for field in integers:
        if type(state.get(field)) is not int or state[field] <= 0:
            failures.append(f"{name}: {field} must be a positive integer")
    for field in ("num_layers", "num_kv_heads", "head_dim"):
        if type(model_config.get(field)) is not int or model_config[field] <= 0:
            failures.append(f"{name}: model_config.{field} must be a positive integer")
    if failures:
        return failures
    if state["n_total"] != state["n_shared"] + state["n_agents"] * state["n_unique"]:
        failures.append(f"{name}: n_total does not match the aggregate fixture shape")

    window = state.get("ppl_window")
    if (
        not isinstance(window, list)
        or len(window) != 2
        or not all(type(value) is int for value in window)
        or not 1 <= window[0] < window[1] <= state["n_total"]
    ):
        failures.append(f"{name}: invalid PPL target window")
    else:
        if state.get("ppl_window_source") != "explicit_start":
            failures.append(f"{name}: PPL window is not explicit_start")
        if state.get("ppl_window_kind") != PPL_WINDOW_KIND:
            failures.append(f"{name}: PPL window kind is not exact")
        if state.get("ppl_scored_tokens") != window[1] - window[0]:
            failures.append(f"{name}: PPL scored-token count is inconsistent")
        if state.get("ppl_frac") is not None:
            failures.append(f"{name}: explicit-start receipt must set ppl_frac to null")
        if claim.get("ppl_window") != window:
            failures.append(f"{name}: claim PPL window differs from state")

    identity_digest = cache_identity.get("identity_sha256")
    identity_payload = {key: value for key, value in cache_identity.items() if key != "identity_sha256"}
    if identity_digest != canonical_digest(identity_payload):
        failures.append(f"{name}: cache identity digest mismatch")
    required_identity = {
        "schema", "model", "model_revision", "tokenizer", "tokenizer_revision",
        "corpus", "n_shared", "n_unique", "n_agents", "n_total", "seed",
        "model_config", "ppl_window", "ppl_window_kind", "token_ids_sha256",
        "benchmark_script_sha256", "identity_sha256",
    }
    if set(cache_identity) != required_identity:
        failures.append(f"{name}: cache identity fields are incomplete or widened")
    for field in ("model", "corpus", "n_shared", "n_unique", "n_agents", "n_total", "seed", "model_config", "ppl_window", "ppl_window_kind"):
        if cache_identity.get(field) != state.get(field):
            failures.append(f"{name}: cache identity {field} differs from state")
    if not isinstance(cache_identity.get("model_revision"), str) or not cache_identity["model_revision"]:
        failures.append(f"{name}: model revision is not pinned")
    if not sha256_digest(cache_identity.get("token_ids_sha256")):
        failures.append(f"{name}: token sequence digest is invalid")
    token_ids = token_ids_receipt.get("token_ids")
    if token_ids_receipt.get("schema") != "ProveKvTokenSequenceV1":
        failures.append(f"{name}: token sequence receipt schema is invalid")
    if not isinstance(token_ids, list) or len(token_ids) != state["n_total"] or not all(type(value) is int for value in token_ids):
        failures.append(f"{name}: token sequence receipt is incomplete")
    elif canonical_digest(token_ids) != cache_identity.get("token_ids_sha256"):
        failures.append(f"{name}: token sequence digest differs from cache identity")
    for field in ("model", "model_revision", "tokenizer", "tokenizer_revision", "token_ids_sha256", "n_total"):
        expected = cache_identity.get(field)
        if token_ids_receipt.get(field) != expected:
            failures.append(f"{name}: token sequence {field} differs from cache identity")
    if token_ids_receipt.get("dataset") != {
        "repository": "Salesforce/wikitext",
        "configuration": "wikitext-2-raw-v1",
        "split": "test",
    }:
        failures.append(f"{name}: token sequence dataset identity is invalid")
    if cache_identity.get("benchmark_script_sha256") != execution_identity.get("benchmark_script_sha256"):
        failures.append(f"{name}: cache and execution script digests differ")
    expected_execution_fields = {
        "benchmark_script_sha256",
        "multi_agent_cli_sha256",
        "source_manifest_sha256",
        "source_archive_sha256",
    }
    if set(execution_identity) != expected_execution_fields:
        failures.append(f"{name}: execution identity fields are incomplete or widened")
    for field in expected_execution_fields:
        if not sha256_digest(execution_identity.get(field)):
            failures.append(f"{name}: execution identity {field} is invalid")
    if build_attestation.get("schema") != "ProveKvCliBuildAttestationV1":
        failures.append(f"{name}: CLI build attestation schema is invalid")
    if build_attestation.get("source_manifest_sha256") != execution_identity.get("source_manifest_sha256"):
        failures.append(f"{name}: CLI build source manifest differs from execution identity")
    if build_attestation.get("source_archive_sha256") != execution_identity.get("source_archive_sha256"):
        failures.append(f"{name}: CLI build source archive differs from execution identity")
    if build_attestation.get("multi_agent_cli_sha256") != execution_identity.get("multi_agent_cli_sha256"):
        failures.append(f"{name}: CLI build binary differs from execution identity")
    for section in (
        "source_and_binary_verification",
        "fresh_shell_and_ppl_run",
        "public_source_identity_run",
    ):
        evidence = build_attestation.get(section)
        if not isinstance(evidence, dict) or evidence.get("status") != "passed":
            failures.append(f"{name}: CLI build attestation section {section} did not pass")
        elif not sha256_digest(evidence.get("receipt_sha256")):
            failures.append(f"{name}: CLI build attestation section {section} has an invalid receipt hash")
    for field in ("source_manifest_sha256", "source_archive_sha256"):
        if claim.get(field) != execution_identity.get(field):
            failures.append(f"{name}: claim {field} differs from execution identity")
    if source_manifest.get("schema") != "ProveKvSourceManifestV1":
        failures.append(f"{name}: source manifest schema is invalid")
    if source_manifest.get("manifest_sha256") != execution_identity.get("source_manifest_sha256"):
        failures.append(f"{name}: source manifest digest differs from execution identity")
    if source_manifest.get("archive_sha256") != execution_identity.get("source_archive_sha256"):
        failures.append(f"{name}: source archive digest differs from execution identity")
    manifest_entries = source_manifest.get("manifest")
    if not isinstance(manifest_entries, list) or not manifest_entries:
        failures.append(f"{name}: source manifest file list is missing")
    else:
        if canonical_digest(manifest_entries) != source_manifest.get("manifest_sha256"):
            failures.append(f"{name}: source manifest canonical digest mismatch")
        if build_attestation.get("verified_source_files") != len(manifest_entries):
            failures.append(f"{name}: CLI build attestation source-file count mismatch")
        if source_envelope.get("schema") != "cpu-router/cargo-snapshot-v1":
            failures.append(f"{name}: source envelope schema is invalid")
        if source_envelope.get("manifest") != manifest_entries:
            failures.append(f"{name}: source envelope manifest differs from checked-in manifest")
        if source_envelope.get("provenance") != source_manifest.get("provenance"):
            failures.append(f"{name}: source envelope provenance differs from checked-in manifest")
        if source_envelope.get("manifest_sha256") != source_manifest.get("manifest_sha256"):
            failures.append(f"{name}: source envelope manifest digest mismatch")
        provenance = source_manifest.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("file_selection") != "explicit-benchmark-build-and-run-inputs-v1":
            failures.append(f"{name}: source snapshot is not the public benchmark allowlist")
        archive_b64 = source_envelope.get("archive_b64")
        try:
            archive = base64.b64decode(archive_b64, validate=True) if isinstance(archive_b64, str) else b""
        except (binascii.Error, ValueError):
            archive = b""
            failures.append(f"{name}: source envelope archive is not valid base64")
        archive_digest = hashlib.sha256(archive).hexdigest() if archive else None
        if archive_digest != source_manifest.get("archive_sha256"):
            failures.append(f"{name}: preserved source archive digest mismatch")
        if source_envelope.get("archive_sha256") != source_manifest.get("archive_sha256"):
            failures.append(f"{name}: source envelope archive declaration mismatch")
        if archive:
            expected_files: dict[str, dict] = {}
            for entry in manifest_entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                    failures.append(f"{name}: source manifest contains an invalid entry")
                    continue
                if entry["path"] not in SOURCE_EXACT_PATHS and not entry["path"].startswith(SOURCE_PREFIXES):
                    failures.append(f"{name}: source manifest path is outside the public allowlist: {entry['path']}")
                if entry["path"] in expected_files:
                    failures.append(f"{name}: duplicate source manifest path {entry['path']}")
                expected_files[entry["path"]] = entry
            missing_inputs = SOURCE_EXACT_PATHS - set(expected_files)
            if missing_inputs:
                failures.append(f"{name}: source snapshot missing required inputs: {sorted(missing_inputs)}")
            observed_files: set[str] = set()
            try:
                with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
                    for member in bundle.getmembers():
                        member_path = PurePosixPath(member.name)
                        if member_path.is_absolute() or ".." in member_path.parts or not member.isfile():
                            failures.append(f"{name}: unsafe or non-file source archive member {member.name}")
                            continue
                        if member.name in observed_files:
                            failures.append(f"{name}: duplicate source archive member {member.name}")
                            continue
                        observed_files.add(member.name)
                        expected = expected_files.get(member.name)
                        if expected is None:
                            failures.append(f"{name}: undeclared source archive member {member.name}")
                            continue
                        extracted = bundle.extractfile(member)
                        payload = extracted.read() if extracted is not None else b""
                        if SECRET_PREFIX.search(payload):
                            failures.append(f"{name}: credential-like content in source archive member {member.name}")
                        if any(marker in payload for marker in PRIVATE_SOURCE_MARKERS):
                            failures.append(f"{name}: private path or topology marker in source archive member {member.name}")
                        if member.size != expected.get("bytes") or len(payload) != expected.get("bytes"):
                            failures.append(f"{name}: source archive byte count mismatch for {member.name}")
                        if member.mode != expected.get("mode"):
                            failures.append(f"{name}: source archive mode mismatch for {member.name}")
                        if hashlib.sha256(payload).hexdigest() != expected.get("sha256"):
                            failures.append(f"{name}: source archive content mismatch for {member.name}")
            except (tarfile.TarError, OSError) as exc:
                failures.append(f"{name}: unreadable preserved source archive: {exc}")
            missing_files = set(expected_files) - observed_files
            if missing_files:
                failures.append(f"{name}: source archive missing {len(missing_files)} manifest files")
    if phase0.get("cache_identity_sha256") != identity_digest:
        failures.append(f"{name}: phase0 cache identity binding mismatch")

    expected_pool = {
        "num_shared_tokens": state["n_shared"],
        "num_layers": model_config["num_layers"],
        "num_kv_heads": model_config["num_kv_heads"],
        "head_dim": model_config["head_dim"],
        "build_seed": state["seed"],
        "shared_codec": "fib_k4_n32_batched",
    }
    for field, expected in expected_pool.items():
        if pool.get(field) != expected:
            failures.append(f"{name}: pool receipt {field} mismatch")
    per_agent = agents.get("per_agent")
    if agents.get("n_agents") != state["n_agents"] or not isinstance(per_agent, list) or len(per_agent) != state["n_agents"]:
        failures.append(f"{name}: agent receipt count mismatch")
        per_agent = []
    expected_codec = "turbo_4bit_batched_lossy" if phase1.get("lossy") else "turbo_4bit_batched"
    shell_sum = 0
    for index, agent in enumerate(per_agent):
        if not isinstance(agent, dict):
            failures.append(f"{name}: agent {index} receipt is not an object")
            continue
        if agent.get("num_unique_tokens") != state["n_unique"]:
            failures.append(f"{name}: agent {index} unique-token count mismatch")
        if agent.get("shell_codec") != expected_codec:
            failures.append(f"{name}: agent {index} codec mismatch")
        if type(agent.get("shell_bytes")) is not int or agent["shell_bytes"] <= 0:
            failures.append(f"{name}: agent {index} shell bytes invalid")
        else:
            shell_sum += agent["shell_bytes"]
    if agents.get("total_shell_bytes") != shell_sum:
        failures.append(f"{name}: total_shell_bytes does not equal per-agent sum")

    bytes_per_token = model_config["num_layers"] * model_config["num_kv_heads"] * model_config["head_dim"] * 2 * 4
    baseline_tokens = state["n_shared"] + state["n_unique"]
    raw_total = state["n_agents"] * baseline_tokens * bytes_per_token
    pool_bytes = pool.get("pool_size_bytes")
    compressed_total = pool_bytes + shell_sum if type(pool_bytes) is int else None
    expected_phase1 = {
        "baseline_definition": (
            "N independent agent contexts, each containing n_shared + n_unique tokens; "
            "this size estimand is distinct from the aggregate PPL fixture."
        ),
        "baseline_tokens_per_agent": baseline_tokens,
        "baseline_bytes_per_token_fp32": bytes_per_token,
        "raw_total_bytes": raw_total,
        "pool_bytes": pool_bytes,
        "shells_size_bytes": shell_sum,
        "compressed_total_bytes": compressed_total,
    }
    for field, expected in expected_phase1.items():
        if phase1.get(field) != expected:
            failures.append(f"{name}: phase1.{field}={phase1.get(field)!r}, expected {expected!r}")
    if compressed_total and compressed_total > 0:
        ratio = raw_total / compressed_total
        for field, expected in (("compression_ratio", ratio), ("ratio_vs_f32_raw", ratio), ("ratio_vs_fp16_kv", ratio / 2)):
            if not finite_positive(phase1.get(field)) or not math.isclose(phase1[field], expected, rel_tol=0, abs_tol=0.001):
                failures.append(f"{name}: phase1.{field} is not byte-derived")
        for field, expected in (("raw_total_bytes", raw_total), ("compressed_total_bytes", compressed_total), ("ratio_vs_f32_raw", ratio), ("ratio_vs_fp16_kv", ratio / 2)):
            value = claim.get(field)
            if isinstance(expected, float):
                if not finite_positive(value) or not math.isclose(value, expected, rel_tol=0, abs_tol=0.001):
                    failures.append(f"{name}: claim {field} differs from admitted state")
            elif value != expected:
                failures.append(f"{name}: claim {field} differs from admitted state")

    for field in ("oracle_ppl", "roundtrip_ppl"):
        if not finite_positive(phase1.get(field)) or claim.get(field) != phase1.get(field):
            failures.append(f"{name}: {field} is invalid or differs from claim")
    if not isinstance(phase1.get("delta_ppl_pct"), (int, float)) or not math.isfinite(phase1["delta_ppl_pct"]):
        failures.append(f"{name}: delta_ppl_pct is invalid")
    else:
        derived_delta = (phase1["roundtrip_ppl"] - phase1["oracle_ppl"]) / phase1["oracle_ppl"] * 100
        if not math.isclose(phase1["delta_ppl_pct"], derived_delta, rel_tol=0, abs_tol=1e-9):
            failures.append(f"{name}: delta_ppl_pct is not PPL-derived")
        if claim.get("delta_ppl_pct") != phase1["delta_ppl_pct"]:
            failures.append(f"{name}: delta_ppl_pct differs from claim")
        if phase1["delta_ppl_pct"] <= 0:
            failures.append(f"{name}: degraded-PPL claim requires a positive measured delta")
    if claim.get("claim_status") != "size_validated_shared_prefix_ppl_degraded":
        failures.append(f"{name}: claim status does not preserve the degraded aggregate PPL result")
    if claim.get("publication_eligible") is not True:
        failures.append(f"{name}: receipt-backed size claim is not explicitly publication eligible")
    if claim.get("ppl_publication_eligible") is not False:
        failures.append(f"{name}: PPL-neutrality publication must be explicitly ineligible")
    if phase1.get("pool_receipt_sha256") != hashlib.sha256(pool_path.read_bytes()).hexdigest():
        failures.append(f"{name}: pool receipt digest binding mismatch")
    if phase1.get("agents_receipt_sha256") != hashlib.sha256(agents_path.read_bytes()).hexdigest():
        failures.append(f"{name}: agents receipt digest binding mismatch")
    if phase1.get("shell_state_sha256") != hashlib.sha256(shell_state_path.read_bytes()).hexdigest():
        failures.append(f"{name}: shell state digest binding mismatch")
    expected_shell_state = {
        "n_agents": state["n_agents"],
        "shared_pool_bytes": pool_bytes,
        "total_shell_bytes": shell_sum,
        "total_with_sharing_bytes": compressed_total,
        "naive_total_bytes": raw_total,
        "shell_codec": expected_codec,
        "shared_codec": "fib_k4_n32_batched",
        "lossy": phase1.get("lossy"),
    }
    for field, expected in expected_shell_state.items():
        if shell_state.get(field) != expected:
            failures.append(f"{name}: shell state {field} differs from receipts")
    return failures


def main() -> int:
    failures: list[str] = []
    claims_document = load_json(CLAIMS_PATH, "CLAIMS.json", failures)
    if claims_document is None:
        return 1
    claims = claims_document.get("claims")
    if not isinstance(claims, dict):
        print("BENCHMARK PROVENANCE BLOCKED\n- CLAIMS.json: claims must be an object", file=sys.stderr)
        return 1
    for name in DEFAULT_CLAIMS:
        failures.extend(validate_claim(name, claims.get(name)))
    if failures:
        print("BENCHMARK PROVENANCE BLOCKED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(
        "OK: benchmark baseline, cache identity, token witness, public source archive, "
        "receipts, and PPL window are explicit"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
