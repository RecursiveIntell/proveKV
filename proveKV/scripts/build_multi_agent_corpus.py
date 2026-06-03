#!/usr/bin/env python3
"""
build_multi_agent_corpus.py — build a synthetic corpus for multi-agent bench.

Generates a JSON file in the format expected by `prove_kv_multi_agent_shell`:
  {
    "shape": { attention_type, num_layers, num_heads, num_kv_heads, head_dim, hidden_size },
    "shared_tokens": [ {id, vector}, ... ],
    "agents": [ {id, tokens: [{id, vector}, ...]}, ... ],
    "seed": 42
  }

The vectors are random f32 values shaped like real K/V data (K and V concatenated
across all layers and heads). The shared portion is identical across agents, and
each agent has a small unique tail (the "agent-specific" tokens).

Usage:
  python build_multi_agent_corpus.py --output /tmp/corpus.json --n-shared 819 \
        --n-unique 28 --n-agents 8 --num-layers 24 --num-kv-heads 2 --head-dim 64
"""
import argparse
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--n-shared", type=int, default=819)
    p.add_argument("--n-unique", type=int, default=28)
    p.add_argument("--n-agents", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--num-heads", type=int, default=14)  # Qwen2.5-0.5B
    p.add_argument("--num-kv-heads", type=int, default=2)  # Qwen2.5-0.5B GQA
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=896)  # Qwen2.5-0.5B
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attention-type", default="GQA")
    args = p.parse_args()

    rng = random.Random(args.seed)
    vec_len = args.num_layers * args.num_kv_heads * args.head_dim * 2
    print(f"Building corpus: n_shared={args.n_shared} n_unique={args.n_unique} "
          f"n_agents={args.n_agents} vec_len={vec_len} "
          f"({args.num_layers} layers x {args.num_kv_heads} kv heads x {args.head_dim} head_dim x 2 K/V)")

    shared_tokens = [
        {"id": f"shared_{i}", "vector": [rng.gauss(0, 1) for _ in range(vec_len)]}
        for i in range(args.n_shared)
    ]
    # Each agent gets its own unique token set (non-overlapping with shared).
    agents = []
    for a in range(args.n_agents):
        tokens = [
            {"id": f"agent_{a}_tok_{i}",
             "vector": [rng.gauss(0, 1) for _ in range(vec_len)]}
            for i in range(args.n_unique)
        ]
        agents.append({"id": f"agent_{a}", "tokens": tokens})

    out = {
        "shape": {
            "attention_type": args.attention_type,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "hidden_size": args.hidden_size,
        },
        "shared_tokens": shared_tokens,
        "agents": agents,
        "seed": args.seed,
    }
    Path(args.output).write_text(json.dumps(out))
    print(f"Wrote corpus to {args.output} "
          f"({Path(args.output).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
