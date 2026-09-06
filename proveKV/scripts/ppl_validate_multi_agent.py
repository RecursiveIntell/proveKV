#!/usr/bin/env python3
"""
ppl_validate_multi_agent.py — System-level PPL validation for the
proveKV multi-agent two-tier compression.

The existing `ppl_validate.py` and `ppl_validate_shell.py` test a
single agent's pool + shell roundtrip. This script tests one aggregate
fixture: a shared prefix followed by N contiguous unique slices, compressed
as one pool plus N shells and reconstructed into the same aggregate cache.
Its PPL result applies only to that aggregate fixture; it does not establish
per-agent PPL equivalence for N independent prompts.

Methodology (receipt-bound by the emitted cache and state contracts):

  Phase 0 (oracle):
    Forward pass on N_total = N_shared + N_agents * N_unique tokens
    with use_cache=True. Save oracle K/V cache.

  Phase 1 (lossless + lossy, per mode):
    1. Extract the oracle K/V at positions [N_shared, N_total):
       split into N_agents slices of length N_unique each.
    2. Build a corpus JSON with shape + shared_tokens (first
       N_shared tokens) + N_agents agents each with its own
       N_unique-token slice.
    3. Invoke prove_kv_multi_agent_shell to build a pool from the
       shared prefix, materialize N shells (lossless or lossy per
       --lossy), decompress each shell to f32.
    4. Patch the oracle cache at positions [N_shared, N_total) with
       the decompressed shared K/V and the N shell K/V slices.
    5. Forward pass with the patched cache. Compute PPL over an
       explicitly recorded eval window. Use --ppl-window-start for
       an exact start token, or --ppl-frac to derive the start from
       the end of the aggregate fixture.

  Report: aggregate-fixture PPL delta plus an independent-agent size
  baseline derived from N contexts of n_shared + n_unique tokens.

Usage:
  python3 ppl_validate_multi_agent.py \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --corpus wikitext-2 \
    --n-shared 800 --n-unique 28 --n-agents 8 \
    --ppl-window-start 800 \
    --output /path/to/output/state.json \
    --multi-agent-cli /path/to/prove_kv_multi_agent_shell \
    --source-manifest-sha256 <snapshot-manifest-sha256> \
    --source-archive-sha256 <snapshot-archive-sha256>
"""
import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

CACHE_SCHEMA = "ProveKvOracleCacheV1"
PPL_WINDOW_KIND = "scored_target_token_indices_half_open"


def canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def ppl_slice_bounds(target_start: int, target_end: int, input_length: int) -> tuple[int, int]:
    if not 1 <= target_start < target_end <= input_length:
        raise ValueError("PPL target window must satisfy 1 <= start < end <= input length")
    return target_start - 1, target_end - 1


def roundtrip_context_bounds(
    target_start: int, target_end: int, input_length: int
) -> tuple[int, int, int]:
    slice_start, slice_end = ppl_slice_bounds(target_start, target_end, input_length)
    return slice_start, slice_start, slice_end


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.new")
    if temporary.exists():
        raise SystemExit(f"refusing existing cache temporary: {temporary}")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_json_write(value: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.new")
    if temporary.exists():
        raise SystemExit(f"refusing existing receipt temporary: {temporary}")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def cache_identity(
    *, model_name: str, model_revision: str | None,
    tokenizer_name: str, tokenizer_revision: str | None,
    corpus: str, n_shared: int, n_unique: int, n_agents: int,
    n_total: int, seed: int, model_config: dict,
    ppl_window: list[int], tokens: list[int], benchmark_script_sha256: str,
) -> dict:
    identity = {
        "schema": CACHE_SCHEMA,
        "model": model_name,
        "model_revision": model_revision,
        "tokenizer": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "corpus": corpus,
        "n_shared": n_shared,
        "n_unique": n_unique,
        "n_agents": n_agents,
        "n_total": n_total,
        "seed": seed,
        "model_config": model_config,
        "ppl_window": ppl_window,
        "ppl_window_kind": PPL_WINDOW_KIND,
        "token_ids_sha256": canonical_digest(tokens),
        "benchmark_script_sha256": benchmark_script_sha256,
    }
    return {**identity, "identity_sha256": canonical_digest(identity)}


def validate_shell_receipts(
    output_dir: Path, *, n_shared: int, n_unique: int, n_agents: int,
    model_config: dict, seed: int, lossy: bool,
) -> dict:
    pool_path = output_dir / "shared_pool_receipt.json"
    agents_path = output_dir / "agents_receipt.json"
    shell_state_path = output_dir / "state.json"
    pool = json.loads(pool_path.read_text())
    agents = json.loads(agents_path.read_text())
    shell_state = json.loads(shell_state_path.read_text())
    expected_pool = {
        "num_shared_tokens": n_shared,
        "num_layers": model_config["num_layers"],
        "num_kv_heads": model_config["num_kv_heads"],
        "head_dim": model_config["head_dim"],
        "build_seed": seed,
        "shared_codec": "fib_k4_n32_batched",
    }
    for field, expected in expected_pool.items():
        if pool.get(field) != expected:
            raise SystemExit(f"shared-pool receipt {field}={pool.get(field)!r}, expected {expected!r}")
    per_agent = agents.get("per_agent")
    if agents.get("n_agents") != n_agents or not isinstance(per_agent, list) or len(per_agent) != n_agents:
        raise SystemExit("agents receipt count does not match n_agents")
    expected_codec = "turbo_4bit_batched_lossy" if lossy else "turbo_4bit_batched"
    shell_sum = 0
    for index, agent in enumerate(per_agent):
        if agent.get("num_unique_tokens") != n_unique:
            raise SystemExit(f"agent {index} unique-token count mismatch")
        if agent.get("shell_codec") != expected_codec:
            raise SystemExit(f"agent {index} shell codec mismatch")
        shell_bytes = agent.get("shell_bytes")
        if type(shell_bytes) is not int or shell_bytes <= 0:
            raise SystemExit(f"agent {index} shell byte count is invalid")
        shell_sum += shell_bytes
    if agents.get("total_shell_bytes") != shell_sum:
        raise SystemExit("agents receipt total_shell_bytes does not equal per-agent sum")
    pool_bytes = pool.get("pool_size_bytes")
    if type(pool_bytes) is not int or pool_bytes <= 0:
        raise SystemExit("shared-pool receipt byte count is invalid")
    bytes_per_token_fp32 = (
        model_config["num_layers"]
        * model_config["num_kv_heads"]
        * model_config["head_dim"]
        * 2
        * 4
    )
    baseline_tokens_per_agent = n_shared + n_unique
    raw_total_bytes = n_agents * baseline_tokens_per_agent * bytes_per_token_fp32
    compressed_total_bytes = pool_bytes + shell_sum
    expected_shell_state = {
        "n_agents": n_agents,
        "shared_pool_bytes": pool_bytes,
        "total_shell_bytes": shell_sum,
        "total_with_sharing_bytes": compressed_total_bytes,
        "naive_total_bytes": raw_total_bytes,
        "shell_codec": expected_codec,
        "shared_codec": "fib_k4_n32_batched",
        "lossy": lossy,
    }
    for field, expected in expected_shell_state.items():
        if shell_state.get(field) != expected:
            raise SystemExit(f"shell state {field}={shell_state.get(field)!r}, expected {expected!r}")
    return {
        "baseline_definition": (
            "N independent agent contexts, each containing n_shared + n_unique tokens; "
            "this size estimand is distinct from the aggregate PPL fixture."
        ),
        "baseline_tokens_per_agent": baseline_tokens_per_agent,
        "baseline_bytes_per_token_fp32": bytes_per_token_fp32,
        "raw_total_bytes": raw_total_bytes,
        "pool_bytes": pool_bytes,
        "pool_size_bytes": pool_bytes,
        "shells_size_bytes": shell_sum,
        "compressed_total_bytes": compressed_total_bytes,
        "compression_ratio": raw_total_bytes / compressed_total_bytes,
        "ratio_vs_f32_raw": raw_total_bytes / compressed_total_bytes,
        "ratio_vs_fp16_kv": (raw_total_bytes / 2) / compressed_total_bytes,
        "pool_compression_ratio": pool["compression_ratio"],
        "naive_per_agent_full_cache": True,
        "pool_receipt_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
        "agents_receipt_sha256": hashlib.sha256(agents_path.read_bytes()).hexdigest(),
        "shell_state_sha256": hashlib.sha256(shell_state_path.read_bytes()).hexdigest(),
    }


def load_wikitext2_tokens(tokenizer, n_tokens: int) -> list[int]:
    """Load the first n_tokens from wikitext-2 test split."""
    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True
    )
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return ids[:n_tokens].tolist()


def phase0_oracle(
    model, input_ids: torch.Tensor, target_start: int, target_end: int
):
    """Forward pass with use_cache=True, save cache, compute oracle PPL.

    Returns (cache_dict, oracle_ppl).
    cache_dict: {layer_idx: {"k": tensor, "v": tensor}}
    target_start and target_end are the exact half-open token-index interval
    whose targets are scored. Token zero cannot be scored because it has no
    preceding logit.
    """
    cache_dict = {}
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
    pkvs = out.past_key_values
    for i, layer in enumerate(pkvs.layers):
        cache_dict[i] = {
            "k": layer.keys.detach().cpu().clone(),
            "v": layer.values.detach().cpu().clone(),
        }
    logits = out.logits[:, :-1, :].contiguous()  # (1, T-1, V)
    targets = input_ids[:, 1:].contiguous()  # (1, T-1)
    slice_start, slice_end = ppl_slice_bounds(target_start, target_end, input_ids.shape[1])
    log_win = logits[:, slice_start:slice_end, :].contiguous()
    tgt_win = targets[:, slice_start:slice_end].contiguous()
    nll = torch.nn.functional.cross_entropy(
        log_win.view(-1, log_win.size(-1)),
        tgt_win.view(-1),
        reduction="mean",
    )
    ppl = torch.exp(nll).item()
    return cache_dict, ppl


def extract_kv_for_range(
    cache_dict: dict, num_layers: int, num_kv_heads: int, head_dim: int,
    start: int, end: int
) -> dict:
    """Extract K/V slices from oracle cache for tokens [start, end).

    Returns {layer_idx: (k_vec, v_vec)} where each vec is a flat
    list of (end - start) * num_kv_heads * head_dim floats in
    [t, h, d] order (matching the corpus format expected by
    prove_kv_multi_agent_shell).
    """
    out = {}
    for layer_idx in range(num_layers):
        k = cache_dict[layer_idx]["k"]  # (1, num_kv_heads, T, head_dim)
        v = cache_dict[layer_idx]["v"]
        k_slice = k[0, :, start:end, :].contiguous()  # (kv, T_slice, d)
        v_slice = v[0, :, start:end, :].contiguous()
        # Reorder to (T_slice, kv, d) and flatten
        k_re = k_slice.permute(1, 0, 2).contiguous()  # (T, kv, d)
        v_re = v_slice.permute(1, 0, 2).contiguous()
        out[layer_idx] = (
            k_re.view(-1).tolist(),
            v_re.view(-1).tolist(),
        )
    return out


def build_corpus_json(
    shape: dict, shared_kv: dict, agents_kv: list[dict], output_path: Path
) -> None:
    """Build the corpus JSON that prove_kv_multi_agent_shell expects.

    The pool expects per-token vectors of length
    num_layers * num_kv_heads * head_dim * 2 (K and V interleaved
    per layer).
    """
    num_layers = shape["num_layers"]
    num_kv_heads = shape["num_kv_heads"]
    head_dim = shape["head_dim"]
    n_shared = len(next(iter(shared_kv.values()))[0]) // (num_kv_heads * head_dim)
    assert n_shared == len(next(iter(shared_kv.values()))[0]) // (num_kv_heads * head_dim)

    def build_token(per_layer_kv: dict, token_idx: int) -> list[float]:
        """Build a single token's per-layer K/V vector (all layers
        concatenated, each layer has [K_vec, V_vec])."""
        out = []
        for layer_idx in range(num_layers):
            k, v = per_layer_kv[layer_idx]
            # token_idx * num_kv_heads * head_dim to (token_idx+1) * num_kv_heads * head_dim
            stride = num_kv_heads * head_dim
            k_t = k[token_idx * stride : (token_idx + 1) * stride]
            v_t = v[token_idx * stride : (token_idx + 1) * stride]
            out.extend(k_t)
            out.extend(v_t)
        return out

    shared_tokens = [
        {"id": f"shared_{i}", "vector": build_token(shared_kv, i)}
        for i in range(n_shared)
    ]

    agents = []
    for agent_idx, agent_kv in enumerate(agents_kv):
        n_unique = len(next(iter(agent_kv.values()))[0]) // (num_kv_heads * head_dim)
        tokens = [
            {"id": f"agent{agent_idx}_{i}", "vector": build_token(agent_kv, i)}
            for i in range(n_unique)
        ]
        agents.append({"id": f"agent_{agent_idx}", "tokens": tokens})

    payload = {
        "shape": {
            "attention_type": shape["attention_type"],
            "num_layers": num_layers,
            "num_heads": shape.get("num_heads", num_kv_heads),
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "hidden_size": shape.get("hidden_size", num_kv_heads * head_dim),
        },
        "shared_tokens": shared_tokens,
        "agents": agents,
        "seed": shape.get("seed", 42),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f)


def read_kv_binary(path: Path):
    """Read a kv binary file produced by prove_kv_multi_agent_shell.

    Returns (manifest, layers_dict, num_layers, num_kv_heads, head_dim)
    where layers_dict is {layer_idx: (k, v)} and k, v are flat lists
    in [t, h, d] order.
    """
    with open(path, "rb") as f:
        manifest_len = struct.unpack("<Q", f.read(8))[0]
        manifest_bytes = f.read(manifest_len)
        manifest = json.loads(manifest_bytes.decode())
        num_layers = manifest["num_layers"]
        num_kv_heads = manifest["num_kv_heads"]
        head_dim = manifest["head_dim"]
        layers = {}
        for layer_idx in range(num_layers):
            k_len = struct.unpack("<I", f.read(4))[0]
            k_floats = struct.unpack(f"<{k_len}f", f.read(k_len * 4))
            v_len = struct.unpack("<I", f.read(4))[0]
            v_floats = struct.unpack(f"<{v_len}f", f.read(v_len * 4))
            layers[layer_idx] = (k_floats, v_floats)
    return manifest, layers, num_layers, num_kv_heads, head_dim


def patch_cache_with_multi_agent_kv(
    cache_dict: dict,
    shared_decompressed: dict,
    agents_decompressed: list[dict],
    n_shared: int,
    n_unique: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict:
    """Patch the oracle cache with decompressed K/V.

    shared_decompressed replaces cache[0, n_shared) (the shared
    prefix that was in the pool).
    agents_decompressed[i] replaces cache[n_shared + i*n_unique, n_shared + (i+1)*n_unique).
    """
    patched = {}
    for layer_idx in range(num_layers):
        k = cache_dict[layer_idx]["k"].clone()  # (1, kv, T, d)
        v = cache_dict[layer_idx]["v"].clone()

        # Shared slice
        sd_k, sd_v = shared_decompressed[layer_idx]
        stride = num_kv_heads * head_dim
        for t in range(n_shared):
            k[0, :, t, :] = torch.tensor(
                sd_k[t * stride : (t + 1) * stride], dtype=k.dtype
            ).view(num_kv_heads, head_dim)
            v[0, :, t, :] = torch.tensor(
                sd_v[t * stride : (t + 1) * stride], dtype=v.dtype
            ).view(num_kv_heads, head_dim)

        # Agent slices
        for agent_idx, ad in enumerate(agents_decompressed):
            ad_k, ad_v = ad[layer_idx]
            for t in range(n_unique):
                pos = n_shared + agent_idx * n_unique + t
                k[0, :, pos, :] = torch.tensor(
                    ad_k[t * stride : (t + 1) * stride], dtype=k.dtype
                ).view(num_kv_heads, head_dim)
                v[0, :, pos, :] = torch.tensor(
                    ad_v[t * stride : (t + 1) * stride], dtype=v.dtype
                ).view(num_kv_heads, head_dim)

        patched[layer_idx] = {"k": k, "v": v}
    return patched


def forward_ppl(
    model, input_ids: torch.Tensor, cache_dict: dict,
    target_start: int, target_end: int,
):
    """Score targets using reconstructed K/V only for preceding context.

    For target interval [start, end), cache contains tokens [0, start-1),
    while model input is [start-1, end-1). Each output logit therefore
    predicts exactly one token in [start, end).
    """
    from transformers import DynamicCache, DynamicLayer

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    fresh_cache = DynamicCache()
    while len(fresh_cache.layers) < num_layers:
        fresh_cache.layers.append(DynamicLayer())
    cache_end, input_start, input_end = roundtrip_context_bounds(
        target_start, target_end, input_ids.shape[1]
    )
    for layer_idx in range(num_layers):
        k = cache_dict[layer_idx]["k"][:, :, :cache_end, :].to(input_ids.device)
        v = cache_dict[layer_idx]["v"][:, :, :cache_end, :].to(input_ids.device)
        fresh_cache.layers[layer_idx].keys = k
        fresh_cache.layers[layer_idx].values = v

    evaluation_input = input_ids[:, input_start:input_end]
    targets = input_ids[:, target_start:target_end].contiguous()
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model(
            input_ids=evaluation_input,
            past_key_values=fresh_cache,
            use_cache=True,
            return_dict=True,
        )
    torch.cuda.synchronize()
    fwd_s = time.time() - t0

    log_win = out.logits.contiguous()
    if log_win.shape[1] != targets.shape[1]:
        raise ValueError("roundtrip logits and target counts differ")
    nll = torch.nn.functional.cross_entropy(
        log_win.view(-1, log_win.size(-1)),
        targets.view(-1),
        reduction="mean",
    )
    ppl = torch.exp(nll).item()
    return ppl, fwd_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--n-shared", type=int, default=800)
    p.add_argument("--n-unique", type=int, default=28)
    p.add_argument("--n-agents", type=int, default=8)
    p.add_argument(
        "--ppl-frac", type=float, default=0.3,
        help="fraction of the aggregate fixture evaluated from the end; "
             "ignored when --ppl-window-start is supplied",
    )
    p.add_argument(
        "--ppl-window-start", type=int, default=None,
        help="exact inclusive PPL window start token; end is n_total",
    )
    p.add_argument("--output", required=True, help="state.json output path")
    p.add_argument("--multi-agent-cli", required=True)
    p.add_argument("--source-manifest-sha256", required=True)
    p.add_argument("--source-archive-sha256", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lossy", action="store_true", help="use lossy turbo shell")
    p.add_argument("--cache-path", default=None,
                   help="re-use an existing oracle cache (skips phase 0)")
    p.add_argument(
        "--reuse-shell-output",
        action="store_true",
        help="verify and reuse an existing shell_output_<mode> directory",
    )
    args = p.parse_args()

    for name, value in (
        ("source-manifest-sha256", args.source_manifest_sha256),
        ("source-archive-sha256", args.source_archive_sha256),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            sys.exit(f"{name} must be a lowercase SHA-256 digest")

    output_state = Path(args.output)
    output_state.parent.mkdir(parents=True, exist_ok=True)
    if args.corpus != "wikitext-2":
        sys.exit("this benchmark currently admits only corpus=wikitext-2")
    if args.n_shared <= 0 or args.n_unique <= 0 or args.n_agents <= 0:
        sys.exit("n-shared, n-unique, and n-agents must be positive")
    n_total = args.n_shared + args.n_agents * args.n_unique
    if n_total < 64:
        sys.exit("n_total must be >= 64")
    if args.ppl_window_start is not None:
        ppl_window_start = args.ppl_window_start
        ppl_window_source = "explicit_start"
    else:
        if not 0 < args.ppl_frac <= 1:
            sys.exit("ppl-frac must be in (0, 1]")
        ppl_window_start = n_total - int(n_total * args.ppl_frac)
        ppl_window_source = "fraction_from_end"
    if not 1 <= ppl_window_start < n_total:
        sys.exit("PPL target window start must satisfy 1 <= start < n_total")
    ppl_window = [ppl_window_start, n_total]
    benchmark_script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    multi_agent_cli = Path(args.multi_agent_cli).resolve(strict=True)
    multi_agent_cli_sha256 = hashlib.sha256(multi_agent_cli.read_bytes()).hexdigest()

    state = {
        "schema_version": "1.0.0",
        "model": args.model,
        "corpus": args.corpus,
        "n_shared": args.n_shared,
        "n_unique": args.n_unique,
        "n_agents": args.n_agents,
        "n_total": args.n_shared + args.n_agents * args.n_unique,
        "ppl_frac": None if args.ppl_window_start is not None else args.ppl_frac,
        "requested_ppl_frac": args.ppl_frac,
        "ppl_window": ppl_window,
        "ppl_window_source": ppl_window_source,
        "ppl_window_kind": PPL_WINDOW_KIND,
        "ppl_scored_tokens": n_total - ppl_window_start,
        "seed": args.seed,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "execution_identity": {
            "benchmark_script_sha256": benchmark_script_sha256,
            "multi_agent_cli_sha256": multi_agent_cli_sha256,
            "source_manifest_sha256": args.source_manifest_sha256,
            "source_archive_sha256": args.source_archive_sha256,
        },
    }

    print(f"[setup] n_total={n_total}, n_shared={args.n_shared}, "
          f"n_unique={args.n_unique}, n_agents={args.n_agents}, "
          f"lossy={args.lossy}", flush=True)

    # Load model + tokenizer
    print(f"[phase0] loading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to("cuda").eval()
    cfg = model.config
    state["model_config"] = {
        "num_layers": cfg.num_hidden_layers,
        "num_heads": cfg.num_attention_heads,
        "num_kv_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim if hasattr(cfg, "head_dim") else (
            cfg.hidden_size // cfg.num_attention_heads
        ),
        "hidden_size": cfg.hidden_size,
        "attention_type": "GQA" if cfg.num_attention_heads != cfg.num_key_value_heads else "MHA",
    }
    print(f"[phase0] model config: {state['model_config']}", flush=True)

    # Phase 0: oracle forward pass
    expected_tokens = load_wikitext2_tokens(tokenizer, n_total)
    if len(expected_tokens) != n_total:
        sys.exit(f"tokenized corpus returned {len(expected_tokens)} tokens, expected {n_total}")
    model_revision = getattr(cfg, "_commit_hash", None)
    tokenizer_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if not isinstance(model_revision, str) or not model_revision:
        sys.exit("model revision is unavailable; refuse an unpinned benchmark receipt")
    expected_cache_identity = cache_identity(
        model_name=args.model,
        model_revision=model_revision,
        tokenizer_name=getattr(tokenizer, "name_or_path", args.model),
        tokenizer_revision=tokenizer_revision,
        corpus=args.corpus,
        n_shared=args.n_shared,
        n_unique=args.n_unique,
        n_agents=args.n_agents,
        n_total=n_total,
        seed=args.seed,
        model_config=state["model_config"],
        ppl_window=ppl_window,
        tokens=expected_tokens,
        benchmark_script_sha256=benchmark_script_sha256,
    )
    state["cache_identity"] = expected_cache_identity
    cache_cache_path = Path(args.cache_path) if args.cache_path else output_state.parent / "cache_oracle.pt"
    if cache_cache_path.exists():
        print(f"[phase0] reusing oracle cache at {cache_cache_path}", flush=True)
        ckpt = torch.load(cache_cache_path, weights_only=False)
        if not isinstance(ckpt, dict) or ckpt.get("cache_identity") != expected_cache_identity:
            raise SystemExit("cached oracle identity is missing or does not match this run")
        required_cache_fields = {"cache_dict", "oracle_ppl", "tokens", "cache_identity"}
        if set(ckpt) != required_cache_fields:
            raise SystemExit("cached oracle fields do not match the closed cache contract")
        cache_dict = ckpt["cache_dict"]
        oracle_ppl = ckpt["oracle_ppl"]
        tokens = ckpt["tokens"]
        if tokens != expected_tokens:
            raise SystemExit("cached oracle token sequence does not match this run")
        if not isinstance(oracle_ppl, (int, float)) or not math.isfinite(oracle_ppl) or oracle_ppl <= 0:
            raise SystemExit("cached oracle PPL is invalid")
    else:
        print(f"[phase0] tokenizing wikitext-2 ({n_total} tokens)...", flush=True)
        tokens = expected_tokens
        input_ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
        print(f"[phase0] input_ids.shape={tuple(input_ids.shape)}", flush=True)

        torch.cuda.synchronize()
        t0 = time.time()
        cache_dict, oracle_ppl = phase0_oracle(
            model, input_ids, target_start=ppl_window_start, target_end=n_total
        )
        fwd_s = time.time() - t0
        print(f"[phase0] oracle forward done in {fwd_s:.1f}s, oracle_ppl={oracle_ppl:.4f} "
              f"(window [{ppl_window_start}, {n_total}))", flush=True)
        atomic_torch_save(
            {
                "cache_dict": cache_dict,
                "oracle_ppl": oracle_ppl,
                "tokens": tokens,
                "cache_identity": expected_cache_identity,
            },
            cache_cache_path,
        )

    state["phase0"] = {
        "status": "complete",
        "ppl": oracle_ppl,
        "cache_path": str(cache_cache_path),
        "cache_identity_sha256": expected_cache_identity["identity_sha256"],
        "ppl_window": ppl_window,
        "ppl_window_kind": PPL_WINDOW_KIND,
        "ppl_scored_tokens": n_total - ppl_window_start,
    }

    num_layers = state["model_config"]["num_layers"]
    num_kv_heads = state["model_config"]["num_kv_heads"]
    head_dim = state["model_config"]["head_dim"]

    output_dir = output_state.parent / (
        f"shell_output_{'lossy' if args.lossy else 'lossless'}"
    )
    if args.reuse_shell_output:
        if not output_dir.is_dir():
            raise SystemExit(f"shell output directory is unavailable: {output_dir}")
        build_s = None
        print(f"[phase1] verifying reused shell output at {output_dir}", flush=True)
    else:
        print(f"[phase1] extracting shared K/V [0, {args.n_shared})...", flush=True)
        shared_kv = extract_kv_for_range(
            cache_dict, num_layers, num_kv_heads, head_dim, 0, args.n_shared
        )
        print(f"[phase1] extracting {args.n_agents} agent K/V slices...", flush=True)
        agents_kv = []
        for agent_idx in range(args.n_agents):
            start = args.n_shared + agent_idx * args.n_unique
            end = start + args.n_unique
            agent_kv = extract_kv_for_range(
                cache_dict, num_layers, num_kv_heads, head_dim, start, end
            )
            agents_kv.append(agent_kv)
        corpus_path = output_state.parent / (
            f"corpus_{'lossy' if args.lossy else 'lossless'}.json"
        )
        shape_dict = dict(state["model_config"], seed=args.seed)
        build_corpus_json(shape_dict, shared_kv, agents_kv, corpus_path)
        print(f"[phase1] wrote corpus to {corpus_path}", flush=True)
        output_dir.mkdir(exist_ok=True)
        print(f"[phase1] running {args.multi_agent_cli} ...", flush=True)
        cmd = [str(multi_agent_cli), str(corpus_path), str(output_dir)]
        if args.lossy:
            cmd.append("--lossy")
        torch.cuda.synchronize()
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        build_s = time.time() - t0
        if proc.returncode != 0:
            print(f"[phase1] CLI failed: rc={proc.returncode}", flush=True)
            print(f"[phase1] stderr: {proc.stderr[-2000:]}", flush=True)
            sys.exit(1)
        print(f"[phase1] CLI ok in {build_s:.1f}s", flush=True)
    receipt_facts = validate_shell_receipts(
        output_dir,
        n_shared=args.n_shared,
        n_unique=args.n_unique,
        n_agents=args.n_agents,
        model_config=state["model_config"],
        seed=args.seed,
        lossy=args.lossy,
    )

    # Read the decompressed K/V from the CLI output
    shared_manifest, shared_dec, _, _, _ = read_kv_binary(
        output_dir / "shared_kv.bin"
    )
    expected_shared_manifest = {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "num_tokens": args.n_shared,
    }
    for field, expected in expected_shared_manifest.items():
        if shared_manifest.get(field) != expected:
            raise SystemExit(f"decoded shared K/V manifest {field} mismatch")
    agents_dec = []
    for agent_idx in range(args.n_agents):
        _, agent_dec, _, _, _ = read_kv_binary(
            output_dir / f"agent_{agent_idx}_kv.bin"
        )
        agents_dec.append(agent_dec)

    # Free the model and reload to ensure clean state for the second forward
    del model
    torch.cuda.empty_cache()
    print(f"[phase1] reloading model fresh...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to("cuda").eval()
    input_ids = torch.tensor([tokens], dtype=torch.long, device="cuda")

    # Patch the cache with decompressed K/V
    print(f"[phase1] patching cache with decompressed K/V...", flush=True)
    patched_cache = patch_cache_with_multi_agent_kv(
        cache_dict, shared_dec, agents_dec,
        args.n_shared, args.n_unique, num_layers, num_kv_heads, head_dim,
    )
    # Forward pass with patched cache
    print(f"[phase1] forward pass with patched cache...", flush=True)
    roundtrip_ppl, fwd_s = forward_ppl(
        model,
        input_ids,
        patched_cache,
        target_start=ppl_window_start,
        target_end=n_total,
    )
    print(f"[phase1] roundtrip_ppl={roundtrip_ppl:.4f} (forward in {fwd_s:.1f}s, "
          f"window [{ppl_window_start}, {n_total}))", flush=True)

    # This is aggregate-fixture PPL. The per-agent receipt below supports the
    # size calculation only and is not evidence of independent-agent PPL.
    if not math.isfinite(roundtrip_ppl) or roundtrip_ppl <= 0:
        raise SystemExit("roundtrip PPL is invalid")
    if oracle_ppl > 0:
        delta_pct = (roundtrip_ppl - oracle_ppl) / oracle_ppl * 100.0
    else:
        print("[warn] oracle_ppl is 0, delta_pct set to 0.0 (unreliable)", flush=True)
        delta_pct = 0.0

    state["phase1"] = {
        "lossy": args.lossy,
        "ppl": roundtrip_ppl,
        "roundtrip_seconds": build_s,
        "shell_output_reused": args.reuse_shell_output,
        "forward_seconds": fwd_s,
        "delta_ppl_pct": delta_pct,
        "status": "complete",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "oracle_ppl": oracle_ppl,
        "roundtrip_ppl": roundtrip_ppl,
        **receipt_facts,
    }

    # ============ FINAL ============
    print("\n========== FINAL ==========", flush=True)
    print(f"oracle_ppl={oracle_ppl:.4f}", flush=True)
    print(f"roundtrip_ppl={roundtrip_ppl:.4f} (lossy={args.lossy})", flush=True)
    print(f"delta_ppl_pct={delta_pct:+.2f}%", flush=True)

    atomic_json_write(state, output_state)
    print(f"\n[done] state written to {output_state}", flush=True)


if __name__ == "__main__":
    main()
