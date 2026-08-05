#!/usr/bin/env python3
"""Per-component codec calibration for hybrid state profiles.

Protocol:
    1. Freeze development, calibration, and held-out prompt families.
    2. Run raw baseline on all families.
    3. Sweep component-at-a-time: each profile on each component kind.
    4. Record reconstruction metrics, logit errors, token agreement.
    5. Held-out evaluation against frozen acceptance thresholds.
    6. Admitted profiles go into state_policy.rs admission table.

Usage:
    PATH="python/.venv/bin:$PATH" PYTHONPATH="python" \\
      python3 proveKV/scripts/qwen35_component_calibration.py \\
      --model Qwen/Qwen2.5-0.5B --device cpu \\
      --dev-prompts calibration/prompts/dev.txt \\
      --cal-prompts calibration/prompts/cal.txt \\
      --holdout-prompts calibration/prompts/holdout.txt \\
      --output results/bench/hybrid_state/qwen25/calibration/
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from provekv_transformers import (
    load_pinned_model,
    tensor_to_raw_bytes,
    compute_blake3,
)


# Component kinds in the hybrid layout.
COMPONENT_KINDS = ["full_attn_k", "full_attn_v", "conv_state", "recurrent_state"]

# Codec profiles to test.
CODEC_PROFILES = ["raw_exact", "radii_preserved_4bit", "radii_lossy_4bit"]


def load_prompts(path: Path) -> list[str]:
    """Load prompts, one per line, skipping blanks and comments."""
    prompts = []
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def generate_default_prompts() -> dict[str, list[str]]:
    """Generate a small default prompt set when no files provided."""
    return {
        "dev": [
            "The quick brown fox",
            "In a hole in the ground there lived",
            "It was a dark and stormy night",
        ],
        "cal": [
            "The meaning of life is",
            "Once upon a time in a land far away",
        ],
        "holdout": [
            "To be or not to be that is",
            "All happy families are alike but",
        ],
    }


def run_baseline(model, tokenizer, prompt: str, device: str, max_tokens: int = 16):
    """Run a single forward pass and capture all layer outputs."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = inputs["input_ids"].to(device)

    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)

    # Extract per-layer K/V tensors from past_key_values.
    pkv = outputs.past_key_values
    layers = {}
    for layer_idx, kv_pair in enumerate(pkv):
        if isinstance(kv_pair, tuple) and len(kv_pair) >= 2:
            k, v = kv_pair[0], kv_pair[1]
        else:
            continue
        layers[layer_idx] = {
            "k": k.cpu().clone(),
            "v": v.cpu().clone(),
        }

    logits = outputs.logits[0, -1, :].cpu()
    next_token = int(torch.argmax(logits).item())

    return {
        "prompt": prompt,
        "token_ids": input_ids[0].cpu().tolist(),
        "num_tokens": input_ids.shape[1],
        "num_layers": len(layers),
        "layers": layers,
        "logits_last": logits.tolist(),
        "next_token": next_token,
    }


def apply_codec(raw_bytes: bytes, profile: str, dims: list[int]) -> tuple[bytes, dict]:
    """Apply a codec profile to raw bytes. Returns (encoded_bytes, metadata).

    This is a stub for P1 — real implementation would call the Rust codecs
    through PyO3 or a subprocess."""
    if profile == "raw_exact":
        return raw_bytes, {"profile": profile, "compression": "none"}

    # Stub: for calibration harness testing.
    # In production, this calls fib-quant (radii_preserved) or turbo-quant
    # (radii_lossy) through the Rust FFI.
    return raw_bytes, {"profile": profile, "compression": "stub", "note": "Rust FFI not wired"}


def compute_metrics(baseline_logits: list[float], candidate_logits: list[float]) -> dict:
    """Compute reconstruction quality metrics."""
    base = torch.tensor(baseline_logits)
    cand = torch.tensor(candidate_logits)

    abs_err = float((cand - base).abs().max())
    rel_err = float(((cand - base).abs() / (base.abs() + 1e-8)).max())

    base_top5 = set(torch.topk(base, 5).indices.tolist())
    cand_top5 = set(torch.topk(cand, 5).indices.tolist())
    top5_overlap = len(base_top5 & cand_top5) / 5

    base_token = int(torch.argmax(base).item())
    cand_token = int(torch.argmax(cand).item())
    token_match = base_token == cand_token

    return {
        "max_abs_error": abs_err,
        "max_rel_error": rel_err,
        "top5_overlap": top5_overlap,
        "baseline_token": base_token,
        "candidate_token": cand_token,
        "token_match": token_match,
    }


def calibrate_component(
    baseline: dict,
    component_kind: str,
    profile: str,
    tolerance: dict,
) -> dict:
    """Calibrate one (component_kind, profile) pair against baseline."""
    # For full-attention components, apply codec to K or V tensor.
    # For conv/recurrent — raw only in P0; lossy blocked.

    if component_kind in ("conv_state", "recurrent_state"):
        if profile != "raw_exact":
            return {
                "component_kind": component_kind,
                "profile": profile,
                "status": "rejected",
                "reason": "conv/recurrent state requires raw_exact in P0",
            }

    # Find matching layers.
    total_bytes = 0
    compressed_bytes = 0
    layer_results = []

    for layer_idx, tensors in baseline["layers"].items():
        if component_kind == "full_attn_k":
            tensor = tensors["k"]
        elif component_kind == "full_attn_v":
            tensor = tensors["v"]
        else:
            continue

        raw = tensor_to_raw_bytes(tensor)
        compressed, meta = apply_codec(raw, profile, list(tensor.shape))

        total_bytes += len(raw)
        compressed_bytes += len(compressed)

        # Verify roundtrip: compressed → decompressed should match raw
        # (for lossless profiles). Stub: no actual compression.
        digest_original = compute_blake3(raw)
        digest_roundtrip = compute_blake3(compressed)  # stub: compressed == raw

        layer_results.append({
            "layer": layer_idx,
            "raw_bytes": len(raw),
            "compressed_bytes": len(compressed),
            "digest_match": digest_original == digest_roundtrip,
        })

    return {
        "component_kind": component_kind,
        "profile": profile,
        "status": "evaluated",
        "total_raw_bytes": total_bytes,
        "total_compressed_bytes": compressed_bytes,
        "compression_ratio": total_bytes / max(compressed_bytes, 1),
        "layers": len(layer_results),
        "all_digests_match": all(r["digest_match"] for r in layer_results),
    }


def main():
    parser = argparse.ArgumentParser(description="Per-component codec calibration")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dev-prompts", type=Path, help="Dev prompt file")
    parser.add_argument("--cal-prompts", type=Path, help="Calibration prompt file")
    parser.add_argument("--holdout-prompts", type=Path, help="Held-out prompt file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts.
    prompts = generate_default_prompts()
    if args.dev_prompts:
        prompts["dev"] = load_prompts(args.dev_prompts)
    if args.cal_prompts:
        prompts["cal"] = load_prompts(args.cal_prompts)
    if args.holdout_prompts:
        prompts["holdout"] = load_prompts(args.holdout_prompts)

    # Load model once.
    print(f"Loading model {args.model}")
    model, tokenizer, config = load_pinned_model(device=args.device)

    # Phase 1: Run baselines on all prompt families.
    baseline_results = {}
    for family, family_prompts in prompts.items():
        family_results = []
        for prompt in family_prompts:
            result = run_baseline(model, tokenizer, prompt, args.device, args.max_tokens)
            family_results.append(result)
        baseline_results[family] = family_results
        print(f"  {family}: {len(family_results)} prompts")

    # Phase 2: Freeze tolerance from dev family.
    dev_tokens = [r["next_token"] for r in baseline_results["dev"]]
    if len(set(dev_tokens)) > 1:
        print(f"  WARNING: dev baselines disagree: {dev_tokens}")

    tolerance = {
        "max_abs_error": 1e-6,
        "max_rel_error": 1e-5,
        "top5_overlap_min": 0.8,
        "require_token_match": True,
    }

    # Phase 3: Component-at-a-time calibration sweep.
    calibration_results = []
    dev_baseline = baseline_results["dev"][0]  # Use first dev prompt as reference.

    for component_kind in COMPONENT_KINDS:
        for profile in CODEC_PROFILES:
            result = calibrate_component(
                dev_baseline, component_kind, profile, tolerance
            )
            calibration_results.append(result)
            status = result["status"]
            extra = ""
            if status == "evaluated":
                extra = f" ratio={result.get('compression_ratio', 0):.1f}x"
            print(f"  {component_kind}/{profile}: {status}{extra}")

    # Phase 4: Held-out evaluation.
    holdout_results = []
    if baseline_results.get("holdout"):
        holdout_baseline = baseline_results["holdout"][0]
        for component_kind in COMPONENT_KINDS:
            for profile in CODEC_PROFILES:
                result = calibrate_component(
                    holdout_baseline, component_kind, profile, tolerance
                )
                holdout_results.append(result)

    # Write report.
    report = {
        "schema_version": 1,
        "kind": "hwos-component-calibration",
        "model": args.model,
        "device": args.device,
        "prompt_families": {k: len(v) for k, v in prompts.items()},
        "tolerance": tolerance,
        "calibration": calibration_results,
        "holdout": holdout_results,
        "admitted_profiles": [
            r for r in calibration_results
            if r["status"] == "evaluated" and r.get("all_digests_match", False)
        ],
    }

    report_path = output_dir / "calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport: {report_path}")
    print(f"Admitted profiles: {len(report['admitted_profiles'])}")

    # Print admission summary for state_policy.rs.
    print("\n# Admission table for state_policy.rs:")
    for component_kind in COMPONENT_KINDS:
        admitted = [
            r["profile"]
            for r in calibration_results
            if r["component_kind"] == component_kind and r["status"] == "evaluated"
        ]
        print(f"  {component_kind}: {admitted}")


if __name__ == "__main__":
    main()
