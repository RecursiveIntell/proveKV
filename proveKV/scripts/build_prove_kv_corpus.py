#!/usr/bin/env python3
"""
build_prove_kv_corpus.py — convert a HuggingFace DynamicCache to a
proveKV-compatible JSON corpus.

The proveKV HuggingFace roundtrip CLI
(`prove_kv_dynamic_cache_roundtrip`) takes a JSON file in this shape:

    {
      "shape": {
        "attention_type": "MHA" | "GQA" | "MQA",
        "num_layers": <int>,
        "num_heads": <int>,
        "num_kv_heads": <int>,
        "head_dim": <int>,
        "hidden_size": <int>
      },
      "tokens": [
        {"id": "tok_<i>", "vector": [f32; num_layers * num_kv_heads * head_dim * 2]},
        ...
      ],
      "seed": <int>  (optional, default 42)
    }

Each token's `vector` is the concatenation of that token's K and V
values across all layers, flattened per-layer. Concretely:
    [K_layer0_head0_t0..., K_layer0_head0_t0..., K_layer0_head1_t0...,
     V_layer0_head0_t0..., V_layer0_head0_t0..., V_layer0_head1_t0...,
     K_layer1_head0_t0..., ..., V_layerN-1_headH-1_tT-1...]

This script reads a saved DynamicCache (from `ppl_validate.py` Phase 0),
extracts K and V tensors per layer, flattens them per token, and writes
the JSON corpus in the layout above.

The hard part: in HuggingFace transformers, `DynamicCache` stores K/V as
per-layer tensors of shape `(batch, num_kv_heads, seq_len, head_dim)`. To
get per-token vectors, we iterate the sequence dimension and for each
token index t, gather K[layer, :, t, :] and V[layer, :, t, :] for all
layers, then concatenate them in a fixed order.

Usage:
    python scripts/build_prove_kv_corpus.py \
        --cache cache_oracle.pt \
        --output prove_kv_corpus.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch


def attention_type_for(num_heads: int, num_kv_heads: int) -> str:
    if num_heads == num_kv_heads:
        return "MHA"
    if num_kv_heads == 1:
        return "MQA"
    return "GQA"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="Path to cache_oracle.pt (saved DynamicCache tensors dict)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON corpus path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the proveKV pool build (default: 42)",
    )
    args = parser.parse_args()

    t0 = time.time()
    if not args.cache.exists():
        print(f"FAIL: cache file not found: {args.cache}", file=sys.stderr)
        return 1

    print(f"[build_prove_kv_corpus] loading cache: {args.cache}", flush=True)
    blob = torch.load(args.cache, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        print(f"FAIL: cache blob is not a dict, got {type(blob)}", file=sys.stderr)
        return 1

    keys_layers = blob.get("keys")
    values_layers = blob.get("values")
    if keys_layers is None or values_layers is None:
        print(
            f"FAIL: cache blob missing 'keys' or 'values'. keys={list(blob.keys())}",
            file=sys.stderr,
        )
        return 1

    num_layers = len(keys_layers)
    print(f"  num_layers: {num_layers}", flush=True)

    # Sanity check: all layers have the same shape
    first_k = keys_layers[0]
    if first_k.ndim != 4:
        print(
            f"FAIL: expected K tensor shape (batch, num_kv_heads, seq_len, head_dim), got {tuple(first_k.shape)}",
            file=sys.stderr,
        )
        return 1
    batch_size, num_kv_heads, seq_len, head_dim = first_k.shape
    if batch_size != 1:
        print(f"FAIL: expected batch=1, got {batch_size}", file=sys.stderr)
        return 1
    print(
        f"  shape: batch={batch_size} num_kv_heads={num_kv_heads} seq_len={seq_len} head_dim={head_dim}",
        flush=True,
    )

    # num_heads is not stored in the cache; we pass it through separately
    num_heads = blob.get("num_heads", num_kv_heads)
    hidden_size = blob.get("hidden_size", num_heads * head_dim)
    attn_type = attention_type_for(num_heads, num_kv_heads)
    print(
        f"  attention_type={attn_type} num_heads={num_heads} hidden_size={hidden_size}",
        flush=True,
    )

    # Constraint check: head_dim % k == 0 (k=4 default)
    # The fib codec requires head_dim % 4 == 0. SmolLM2 head_dim=64 ✓.
    if head_dim % 4 != 0:
        print(
            f"WARN: head_dim={head_dim} is not divisible by 4 (fib codec default k=4). "
            f"Pool build will fail. Use a different model or codec.",
            file=sys.stderr,
        )

    # Convert K/V tensors to fp32 (small corpus, precision matters)
    print("[build_prove_kv_corpus] extracting per-token vectors", flush=True)
    print(f"  target vector length per token: {num_layers * num_kv_heads * head_dim * 2} floats", flush=True)

    tokens: list = []
    for t in range(seq_len):
        # Per-token vector: for each layer, concat K[layer, :, t, :] and V[layer, :, t, :]
        # Order: K_layer_0, V_layer_0, K_layer_1, V_layer_1, ...
        # Within a layer: head 0 first, then head 1, ...
        # Within a head: head_dim floats
        vec_parts: list = []
        for layer_idx in range(num_layers):
            k_t = keys_layers[layer_idx][0, :, t, :]  # (num_kv_heads, head_dim)
            v_t = values_layers[layer_idx][0, :, t, :]  # (num_kv_heads, head_dim)
            # Flatten to (num_kv_heads * head_dim,)
            vec_parts.append(k_t.reshape(-1).to(torch.float32))
            vec_parts.append(v_t.reshape(-1).to(torch.float32))
        full_vec = torch.cat(vec_parts).tolist()
        tokens.append({"id": f"tok_{t}", "vector": full_vec})
        if t % 256 == 0 and t > 0:
            print(f"  ... {t}/{seq_len} tokens extracted", flush=True)

    payload = {
        "shape": {
            "attention_type": attn_type,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
        },
        "tokens": tokens,
        "seed": args.seed,
    }

    print(f"[build_prove_kv_corpus] writing JSON: {args.output}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f)
    size_mb = args.output.stat().st_size / 1e6
    print(f"  wrote {size_mb:.1f} MB, {len(tokens)} tokens", flush=True)

    elapsed = time.time() - t0
    print(f"\nOK: corpus built in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
