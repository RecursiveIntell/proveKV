#!/usr/bin/env python3
"""Capture Qwen3.5-2B KV cache state into proveKV binary pages.

Usage:
    HF_HUB_OFFLINE=1 python proveKV/scripts/qwen35_state_capture.py \
        --model Qwen/Qwen3.5-2B \
        --revision <immutable-revision> \
        --device cpu --dtype float32 --batch-size 1 --tokens 64 \
        --output results/bench/hybrid_state/qwen35/capture/<run-id>/
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

# Allow running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from provekv_transformers import (
    QWEN35_MODEL_ID,
    QWEN35_REVISION,
    capture_prefix,
    component_shape,
    compute_blake3,
    load_pinned_model,
    tensor_to_raw_bytes,
)


def main():
    parser = argparse.ArgumentParser(description="Capture Qwen3.5 KV state")
    parser.add_argument("--model", default=QWEN35_MODEL_ID)
    parser.add_argument("--revision", default=QWEN35_REVISION)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--output", required=True, help="Output directory for captured state")
    parser.add_argument("--prompt", default="The quick brown fox", help="Text to encode")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{run_id}] Loading model {args.model} @ {args.revision} on {args.device}/{args.dtype}")

    model, tokenizer, config = load_pinned_model(
        device=args.device,
        dtype=args.dtype,
    )

    print(f"[{run_id}] Running capture: '{args.prompt}' ({args.tokens} tokens max)")
    captured = capture_prefix(
        model, tokenizer, args.prompt, device=args.device, max_tokens=args.tokens
    )

    # Write raw byte pages per layer.
    pages_dir = run_dir / "pages"
    pages_dir.mkdir(exist_ok=True)

    page_manifest = []
    for layer_idx, tensors in captured["layers"].items():
        for kind in ("k", "v"):
            tensor = tensors[kind]
            raw = tensor_to_raw_bytes(tensor)
            digest = compute_blake3(raw)
            shape = component_shape(tensor)

            page_file = pages_dir / f"layer{layer_idx:03d}_{kind}.page"
            # Write JSON header + raw payload.
            header = {
                "magic": list(b"PKVP"),
                "schema_version": 1,
                "component_kind": f"full_attn_{kind}",
                "layer": layer_idx,
                "rank": len(shape),
                "dims": shape,
                "dtype": "float32",
                "endianness": "l",
                "model_digest": f"sha256:{hashlib.sha256(args.model.encode()).hexdigest()}",
                "layout_digest": captured["config_digest"],
                "position_start": 0,
                "position_end": captured["num_tokens"],
                "codec_profile": "raw_exact",
                "payload_len": len(raw),
                "payload_digest": digest,
            }
            header_json = json.dumps(header, sort_keys=True)
            with open(page_file, "wb") as f:
                f.write(header_json.encode())
                f.write(b"\n")
                f.write(raw)

            page_manifest.append(
                {
                    "page_id": f"layer{layer_idx:03d}_{kind}",
                    "digest": digest,
                    "bytes": len(raw),
                    "shape": shape,
                }
            )

    # Write state manifest.
    manifest = {
        "schema": "provekv_hybrid_state_manifest_v1",
        "model_id": args.model,
        "revision": args.revision,
        "config_digest": captured["config_digest"],
        "tokenizer_digest": f"sha256:{hashlib.sha256(tokenizer.__class__.__name__.encode()).hexdigest()}",
        "token_ids": captured["token_ids"],
        "num_layers": captured["num_layers"],
        "num_tokens": captured["num_tokens"],
        "dtype": captured["dtype"],
        "device": args.device,
        "page_manifest": page_manifest,
        "captured_at": run_id,
    }

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Write receipt.
    receipt = {
        "schema_version": 1,
        "kind": "hwos-qwen35-capture",
        "run_id": run_id,
        "model": args.model,
        "revision": args.revision,
        "device": args.device,
        "dtype": args.dtype,
        "num_layers": captured["num_layers"],
        "num_tokens": captured["num_tokens"],
        "page_count": len(page_manifest),
        "total_bytes": sum(p["bytes"] for p in page_manifest),
        "output_dir": str(run_dir),
        "status": "completed",
    }

    receipt_path = run_dir / "receipt.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    print(f"[{run_id}] Captured {len(page_manifest)} pages, {receipt['total_bytes']} bytes")
    print(f"[{run_id}] Output: {run_dir}")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
