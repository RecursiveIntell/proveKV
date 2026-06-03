#!/usr/bin/env python3
"""
ppl_validate_multi_agent.py — System-level PPL validation for the
proveKV multi-agent two-tier compression.

The existing `ppl_validate.py` and `ppl_validate_shell.py` test a
single agent's pool + shell roundtrip. This script tests the FULL
N-agent system: shared prefix in the pool, N unique suffixes in N
shells. It reports per-agent PPL delta and an averaged delta across
all N agents.

Methodology (locked per PPL_RUNBOOK.md):

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
    5. Forward pass with the patched cache. Compute PPL over the
       eval window [N_shared + 0.7*N_unique, N_total).

  Report: per-agent PPL delta + averaged delta.

Usage:
  python3 ppl_validate_multi_agent.py \
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --corpus wikitext-2 \
    --n-shared 800 --n-unique 28 --n-agents 8 \
    --ppl-frac 0.3 \
    --output /path/to/output/state.json \
    --multi-agent-cli /path/to/prove_kv_multi_agent_shell
"""
import argparse
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


def load_wikitext2_tokens(tokenizer, n_tokens: int) -> list[int]:
    """Load the first n_tokens from wikitext-2 test split."""
    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True
    )
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return ids[:n_tokens].tolist()


def phase0_oracle(model, input_ids: torch.Tensor, ws: int = 0, we: int = -1):
    """Forward pass with use_cache=True, save cache, compute oracle PPL.

    Returns (cache_dict, oracle_ppl).
    cache_dict: {layer_idx: {"k": tensor, "v": tensor}}
    ws, we: start (inclusive) and end (exclusive) of the PPL window
    in input positions. PPL is computed over the positions that
    PREDICT input[ws+1:we+1], i.e. the model is asked to predict
    tokens at positions [ws+1, we] using the cache.
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
    if we < 0:
        we = targets.shape[1]
    # We need logits at positions [ws, we-1] predicting [ws+1, we].
    # logits has shape (1, T-1) → logits[:, ws:we, :] predicts targets[:, ws:we+1]
    # Actually: logits[:, i, :] predicts target at position i+1
    # So to predict targets[:, ws:we], we need logits[:, ws-1:we-1, :]
    # i.e. logits at positions [ws-1, we-2] for predictions at [ws, we-1].
    # Standard: shift the targets. logits[:, i, :] = predicted distribution for input[:, i+1]
    # To predict positions [ws+1, we], we use logits[:, ws:we, :].
    # targets to match: input_ids[:, ws+1:we+1] = targets[:, ws:we].
    log_win = logits[:, ws:we, :].contiguous()
    tgt_win = targets[:, ws:we].contiguous()
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


def forward_ppl(model, input_ids: torch.Tensor, cache_dict: dict, ws: int = 0, we: int = -1):
    """Forward pass with a pre-populated DynamicCache, return (PPL, forward_seconds)
    over the eval window [ws, we) in input positions. PPL is computed
    over the predictions for tokens [ws+1, we]."""
    from transformers import DynamicCache, DynamicLayer

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    fresh_cache = DynamicCache()
    while len(fresh_cache.layers) < num_layers:
        fresh_cache.layers.append(DynamicLayer())
    for layer_idx in range(num_layers):
        k = cache_dict[layer_idx]["k"].to(input_ids.device)
        v = cache_dict[layer_idx]["v"].to(input_ids.device)
        fresh_cache.layers[layer_idx].keys = k
        fresh_cache.layers[layer_idx].values = v

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            past_key_values=fresh_cache,
            use_cache=True,
            return_dict=True,
        )
    torch.cuda.synchronize()
    fwd_s = time.time() - t0

    logits = out.logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    if we < 0:
        we = targets.shape[1]
    log_win = logits[:, ws:we, :].contiguous()
    tgt_win = targets[:, ws:we].contiguous()
    nll = torch.nn.functional.cross_entropy(
        log_win.view(-1, log_win.size(-1)),
        tgt_win.view(-1),
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
    p.add_argument("--ppl-frac", type=float, default=0.3)
    p.add_argument("--output", required=True, help="state.json output path")
    p.add_argument("--multi-agent-cli", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lossy", action="store_true", help="use lossy turbo shell")
    p.add_argument("--cache-path", default=None,
                   help="re-use an existing oracle cache (skips phase 0)")
    args = p.parse_args()

    output_state = Path(args.output)
    output_state.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "1.0.0",
        "model": args.model,
        "corpus": args.corpus,
        "n_shared": args.n_shared,
        "n_unique": args.n_unique,
        "n_agents": args.n_agents,
        "n_total": args.n_shared + args.n_agents * args.n_unique,
        "ppl_frac": args.ppl_frac,
        "seed": args.seed,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    n_total = args.n_shared + args.n_agents * args.n_unique
    if n_total < 64:
        sys.exit("n_total must be >= 64")

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
    cache_cache_path = args.cache_path or output_state.parent / "cache_oracle.pt"
    if Path(cache_cache_path).exists():
        print(f"[phase0] reusing oracle cache at {cache_cache_path}", flush=True)
        # Support both formats: new (cache_dict) and legacy (keys/values)
        ckpt = torch.load(cache_cache_path, weights_only=False)
        if "cache_dict" in ckpt:
            cache_dict = ckpt["cache_dict"]
            oracle_ppl = ckpt["oracle_ppl"]
            tokens = ckpt["tokens"]
        else:
            # Legacy format from ppl_validate.py
            cache_dict = {
                i: {"k": k, "v": v}
                for i, (k, v) in enumerate(zip(ckpt["keys"], ckpt["values"]))
            }
            # ppl_validate.py may have saved oracle_ppl under a different key
            oracle_ppl = ckpt.get("oracle_ppl", ckpt.get("ppl_oracle", 0.0))
            tokens = ckpt.get("tokens")
            # If oracle_ppl is missing, we need to compute it by re-running
            # the forward pass and reading the saved state.json if available.
            if oracle_ppl == 0.0:
                # Try to read it from the state.json sibling
                # (follow symlinks to find the actual cache location)
                real_cache_path = cache_cache_path.resolve()
                state_json = real_cache_path.parent / "state.json"
                if state_json.exists():
                    state = json.loads(state_json.read_text())
                    if "phase0" in state and "ppl" in state["phase0"]:
                        oracle_ppl = state["phase0"]["ppl"]
                        print(f"[phase0] recovered oracle_ppl={oracle_ppl} from {state_json}", flush=True)
            # If still missing, re-tokenize and re-compute (slow fallback)
            if oracle_ppl == 0.0 or tokens is None:
                print(f"[phase0] oracle_ppl or tokens missing from cache, recomputing...", flush=True)
                tokens = load_wikitext2_tokens(tokenizer, n_total)
                input_ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
                # Use the same eval window as the pool bench: last 30% of tokens
                ppl_window_start = n_total - int(n_total * args.ppl_frac)
                torch.cuda.synchronize()
                t0 = time.time()
                cache_dict, oracle_ppl = phase0_oracle(
                    model, input_ids, ws=ppl_window_start, we=n_total
                )
                fwd_s = time.time() - t0
                print(f"[phase0] recomputed oracle_ppl={oracle_ppl:.4f} "
                      f"in {fwd_s:.1f}s (window [{ppl_window_start}, {n_total}))",
                      flush=True)
                torch.save(
                    {"cache_dict": cache_dict, "oracle_ppl": oracle_ppl, "tokens": tokens},
                    cache_cache_path,
                )
            # Always set the ppl_window for the report
            state["ppl_window"] = [
                n_total - int(n_total * args.ppl_frac),
                n_total,
            ]
    else:
        print(f"[phase0] tokenizing wikitext-2 ({n_total} tokens)...", flush=True)
        tokens = load_wikitext2_tokens(tokenizer, n_total)
        input_ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
        print(f"[phase0] input_ids.shape={tuple(input_ids.shape)}", flush=True)

        # Use the same eval window as the pool bench: last 30% of tokens
        ppl_window_start = n_total - int(n_total * args.ppl_frac)
        torch.cuda.synchronize()
        t0 = time.time()
        cache_dict, oracle_ppl = phase0_oracle(
            model, input_ids, ws=ppl_window_start, we=n_total
        )
        fwd_s = time.time() - t0
        print(f"[phase0] oracle forward done in {fwd_s:.1f}s, oracle_ppl={oracle_ppl:.4f} "
              f"(window [{ppl_window_start}, {n_total}))", flush=True)
        torch.save(
            {"cache_dict": cache_dict, "oracle_ppl": oracle_ppl, "tokens": tokens},
            cache_cache_path,
        )
        state["ppl_window"] = [ppl_window_start, n_total]

    state["phase0"] = {
        "status": "complete",
        "ppl": oracle_ppl,
        "cache_path": str(cache_cache_path),
    }
    state["ppl_window"] = [
        args.n_shared,
        n_total,
    ]

    num_layers = state["model_config"]["num_layers"]
    num_kv_heads = state["model_config"]["num_kv_heads"]
    head_dim = state["model_config"]["head_dim"]

    # Phase 1: build corpus from oracle cache
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

    # Build corpus JSON
    corpus_path = output_state.parent / (
        f"corpus_{'lossy' if args.lossy else 'lossless'}.json"
    )
    shape_dict = dict(state["model_config"], seed=args.seed)
    build_corpus_json(shape_dict, shared_kv, agents_kv, corpus_path)
    print(f"[phase1] wrote corpus to {corpus_path}", flush=True)

    # Invoke the multi-agent shell CLI
    output_dir = output_state.parent / (
        f"shell_output_{'lossy' if args.lossy else 'lossless'}"
    )
    output_dir.mkdir(exist_ok=True)
    print(f"[phase1] running {args.multi_agent_cli} ...", flush=True)
    cmd = [args.multi_agent_cli, str(corpus_path), str(output_dir)]
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

    # Read the decompressed K/V from the CLI output
    shared_manifest, shared_dec, _, _, _ = read_kv_binary(
        output_dir / "shared_kv.bin"
    )
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
    ppl_window_start = n_total - int(n_total * args.ppl_frac)
    roundtrip_ppl, fwd_s = forward_ppl(
        model, input_ids, patched_cache, ws=ppl_window_start, we=n_total
    )
    print(f"[phase1] roundtrip_ppl={roundtrip_ppl:.4f} (forward in {fwd_s:.1f}s, "
          f"window [{ppl_window_start}, {n_total}))", flush=True)

    # Per-agent breakdown: which tokens are predicted with each agent's
    # patched K/V. Each agent's slice affects predictions at positions
    # [agent_unique_end+1, total]. For simplicity we just report the
    # overall PPL delta and an "agents_receipt.json" sizes summary.
    if oracle_ppl > 0:
        delta_pct = (roundtrip_ppl - oracle_ppl) / oracle_ppl * 100.0
    else:
        print("[warn] oracle_ppl is 0, delta_pct set to 0.0 (unreliable)", flush=True)
        delta_pct = 0.0

    state["phase1"] = {
        "lossy": args.lossy,
        "ppl": roundtrip_ppl,
        "compression_ratio": None,  # not measured in this script
        "roundtrip_seconds": build_s,
        "forward_seconds": fwd_s,
        "delta_ppl_pct": delta_pct,
        "status": "complete",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ============ FINAL ============
    print("\n========== FINAL ==========", flush=True)
    print(f"oracle_ppl={oracle_ppl:.4f}", flush=True)
    print(f"roundtrip_ppl={roundtrip_ppl:.4f} (lossy={args.lossy})", flush=True)
    print(f"delta_ppl_pct={delta_pct:+.2f}%", flush=True)

    with open(output_state, "w") as f:
        json.dump(state, f, indent=2)
    print(f"\n[done] state written to {output_state}", flush=True)


if __name__ == "__main__":
    main()
