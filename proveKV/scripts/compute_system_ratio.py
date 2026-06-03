#!/usr/bin/env python3
"""
Compute the system-level N-agent compression ratio from the per-tier
receipts and patch the state.json files with the correct number.

System ratio formula:
  raw_total = N_total_tokens * num_layers * num_kv_heads * head_dim * 2 * 4  (fp32 bytes)
  compressed_total = pool_size_bytes + sum(per-agent shell_bytes)
  ratio = raw_total / compressed_total

Where:
  N_total_tokens = n_shared + N_agents * n_unique  (naive: each agent has full cache)
  pool_size_bytes = shell_output_*_shared_pool_receipt.json
  per-agent shell_bytes = shell_output_*_agents_receipt.json

The "naive" baseline assumes each agent has its own full cache of
n_shared + n_unique tokens (no sharing). The compressed system has ONE
pool for the shared prefix + N small shells for the unique tails.
"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("usage: compute_system_ratio.py <bench_dir>")
        sys.exit(1)

    bench_dir = Path(sys.argv[1])

    for mode in ("lossless", "lossy"):
        state_path = bench_dir / f"state_{mode}.json"
        pool_path = bench_dir / f"shell_output_{mode}_shared_pool_receipt.json"
        agents_path = bench_dir / f"shell_output_{mode}_agents_receipt.json"

        with state_path.open() as f:
            state = json.load(f)
        with pool_path.open() as f:
            pool = json.load(f)
        with agents_path.open() as f:
            agents = json.load(f)

        n_shared = state["n_shared"]
        n_unique = state["n_unique"]
        n_agents = state["n_agents"]
        num_layers = state["model_config"]["num_layers"]
        num_kv_heads = state["model_config"]["num_kv_heads"]
        head_dim = state["model_config"]["head_dim"]

        # Per-token K+V vector in fp32
        bytes_per_token_fp32 = (
            num_layers * num_kv_heads * head_dim * 2 * 4
        )

        # Naive baseline: each agent has its own full cache
        n_total_per_agent = n_shared + n_unique
        raw_total = n_agents * n_total_per_agent * bytes_per_token_fp32

        # Compressed: 1 pool (shared prefix) + N small shells (unique tails)
        pool_bytes = pool["pool_size_bytes"]
        shells_bytes = agents["total_shell_bytes"]
        compressed_total = pool_bytes + shells_bytes

        system_ratio = raw_total / compressed_total
        pool_ratio = pool["compression_ratio"]

        # Write back to state.json
        state["phase1"]["compression_ratio"] = system_ratio
        state["phase1"]["raw_total_bytes"] = raw_total
        state["phase1"]["compressed_total_bytes"] = compressed_total
        state["phase1"]["pool_size_bytes"] = pool_bytes
        state["phase1"]["shells_size_bytes"] = shells_bytes
        state["phase1"]["pool_compression_ratio"] = pool_ratio
        state["phase1"]["naive_per_agent_full_cache"] = True

        with state_path.open("w") as f:
            json.dump(state, f, indent=2)

        print(f"[{mode}] system compression ratio: {system_ratio:.4f}x")
        print(f"[{mode}]   pool: {pool_bytes:,} B ({pool_ratio:.2f}x)")
        print(f"[{mode}]   shells: {shells_bytes:,} B ({shells_bytes/n_agents:,} B/agent)")
        print(f"[{mode}]   compressed: {compressed_total:,} B = {compressed_total/1e6:.2f} MB")
        print(f"[{mode}]   raw (naive): {raw_total:,} B = {raw_total/1e6:.2f} MB")
        print(f"[{mode}]   ratio: {raw_total/compressed_total:.2f}x")
        print(f"[{mode}] state updated: {state_path}")
        print()


if __name__ == "__main__":
    main()
