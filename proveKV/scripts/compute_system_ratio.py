#!/usr/bin/env python3
"""
Independently recompute the system-level N-agent compression ratio from the
per-tier receipts and verify that the benchmark state contains the same
derivation. This script never repairs or rewrites a benchmark receipt.

System ratio formula:
  raw_total = N_agents * baseline_tokens_per_agent * bytes_per_token_fp32  (fp32 bytes)
  compressed_total = pool_size_bytes + sum(per-agent shell_bytes)
  ratio = raw_total / compressed_total

Where:
  baseline_tokens_per_agent = n_shared + n_unique
    (one independent agent context contains the shared prefix plus its own tail)
  raw_total = N_agents * baseline_tokens_per_agent * bytes_per_token_fp32
  pool_size_bytes = shell_output_*/shared_pool_receipt.json
  per-agent shell_bytes = shell_output_*/agents_receipt.json

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
        output_dir = bench_dir / f"shell_output_{mode}"
        pool_path = output_dir / "shared_pool_receipt.json"
        agents_path = output_dir / "agents_receipt.json"

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

        # Naive baseline: each agent has its own independent context made of
        # the shared prefix plus that agent's unique tail. Keep this derivation
        # explicit in the receipt so it cannot be confused with the aggregate
        # concatenated PPL fixture (n_shared + n_agents * n_unique).
        baseline_tokens_per_agent = n_shared + n_unique
        raw_total = n_agents * baseline_tokens_per_agent * bytes_per_token_fp32

        # Compressed: 1 pool (shared prefix) + N small shells (unique tails)
        pool_bytes = pool["pool_size_bytes"]
        shells_bytes = agents["total_shell_bytes"]
        compressed_total = pool_bytes + shells_bytes

        system_ratio = raw_total / compressed_total
        pool_ratio = pool["compression_ratio"]

        expected = {
            "compression_ratio": system_ratio,
            "raw_total_bytes": raw_total,
            "compressed_total_bytes": compressed_total,
            "pool_size_bytes": pool_bytes,
            "shells_size_bytes": shells_bytes,
            "pool_compression_ratio": pool_ratio,
            "naive_per_agent_full_cache": True,
            "baseline_definition": (
                "N independent agent contexts, each containing n_shared + n_unique tokens; "
                "this size estimand is distinct from the aggregate PPL fixture."
            ),
            "baseline_tokens_per_agent": baseline_tokens_per_agent,
            "baseline_bytes_per_token_fp32": bytes_per_token_fp32,
        }
        phase1 = state.get("phase1")
        if not isinstance(phase1, dict) or phase1.get("status") != "complete":
            raise SystemExit(f"{state_path}: phase1 is not complete")
        for field, value in expected.items():
            observed = phase1.get(field)
            if isinstance(value, float):
                if not isinstance(observed, (int, float)) or abs(observed - value) > 0.001:
                    raise SystemExit(f"{state_path}: phase1.{field} does not match receipts")
            elif observed != value:
                raise SystemExit(f"{state_path}: phase1.{field} does not match receipts")

        print(f"[{mode}] system compression ratio: {system_ratio:.4f}x")
        print(f"[{mode}]   pool: {pool_bytes:,} B ({pool_ratio:.2f}x)")
        print(f"[{mode}]   shells: {shells_bytes:,} B ({shells_bytes/n_agents:,} B/agent)")
        print(f"[{mode}]   compressed: {compressed_total:,} B = {compressed_total/1e6:.2f} MB")
        print(f"[{mode}]   raw (naive): {raw_total:,} B = {raw_total/1e6:.2f} MB")
        print(f"[{mode}]   ratio: {raw_total/compressed_total:.2f}x")
        print(f"[{mode}] state verified: {state_path}")
        print()


if __name__ == "__main__":
    main()
