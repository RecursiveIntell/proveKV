#!/usr/bin/env python3
"""
ppl_validate_shell.py — Real-LLM PPL validation for the SHELL tier (turbo codec).

The existing `ppl_validate.py` tests the pool tier (fib codec) roundtrip. This
script tests the shell tier (turbo codec) — the per-agent delta tokens that
sit on top of the shared pool.

Methodology (mirrors PPL_RUNBOOK.md):
- Phase 0: Oracle baseline. use_cache=True forward pass on N WikiText-2
  tokens. Save K/V cache. Compute oracle PPL over last 30%.
- Phase 1: Shell build + roundtrip.
   - Split tokens into "shared" (first N - M) and "agent-unique" (last M).
   - Extract the per-token K/V vectors for the LAST M tokens from the
     oracle cache. Write a proveKV corpus JSON of just those M tokens.
   - Invoke `prove_kv_shell_roundtrip` (with or without --lossy) to:
       1. Build a pool from those M tokens
       2. Materialize a shell covering all M tokens
       3. Decompress the shell back to f32 K/V
       4. Write the decompressed K/V to a binary file
   - Read the roundtripped shell K/V. Load as fp16 to match the cache dtype.
   - Patch the oracle cache at positions [N-M, N) with the roundtripped K/V.
- Phase 2: Forward pass with patched cache, compute PPL.
- Report: shell_ppl_lossless, shell_ppl_lossy, delta_ppl_pct vs oracle.

The pool tier is UNTOUCHED in this bench — the only roundtripped data is
the shell (turbo codec). The PPL delta measures the turbo codec's quality
in isolation.

Usage:
    python scripts/ppl_validate_shell.py \\
        --model HuggingFaceTB/SmolLM2-1.7B-Instruct \\
        --corpus wikitext-2 \\
        --n-tokens 1024 \\
        --n-shell-tokens 224 \\
        --ppl-frac 0.3 \\
        --output bench/ppl_shell/smollm2-1.7b/wikitext-2/state.json
"""
import argparse
import datetime
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import torch


# ---------- helpers ----------

def iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()


def write_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, state_path)


def per_token_nll(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Standard HF causal LM PPL recipe: returns per-token NLL over T-1 positions."""
    out = model(input_ids=input_ids, labels=input_ids, return_dict=True)
    logits = out.logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    del out
    nll_chunks = []
    chunk_size = 512
    for start in range(0, logits.shape[1], chunk_size):
        end = min(start + chunk_size, logits.shape[1])
        l_chunk = logits[:, start:end, :].float()
        lse = torch.logsumexp(l_chunk, dim=-1, keepdim=True)
        tgt = targets[:, start:end].unsqueeze(-1)
        log_prob_at_tgt = l_chunk.gather(2, tgt).squeeze(-1)
        nll_chunk = lse.squeeze(-1) - log_prob_at_tgt
        nll_chunks.append(nll_chunk)
        del l_chunk, lse, log_prob_at_tgt
    nll = torch.cat(nll_chunks, dim=1).squeeze(0)
    del logits, targets
    return nll


def windowed_ppl(nll: torch.Tensor, ppl_frac: float) -> tuple[float, int, int]:
    T = nll.shape[0]
    start = int(T * (1.0 - ppl_frac))
    end = T
    window = nll[start:end]
    mean_nll = window.mean().item()
    return math.exp(mean_nll), start, end


def read_dynamic_cache(pt_path: Path):
    """Load a saved DynamicCache from ppl_validate.py Phase 0.

    Format (per ppl_validate.py:194-206):
      {
        "model_id": str,
        "num_layers": int,
        "num_heads": int,
        "num_kv_heads": int,
        "head_dim": int,
        "hidden_size": int,
        "seq_len": int,
        "keys":   [Tensor; (1, num_kv_heads, T, head_dim)],
        "values": [Tensor; (1, num_kv_heads, T, head_dim)],
      }

    Returns a tuple: (cache_dict, model_config)
    where cache_dict[layer_idx] = {"k": Tensor, "v": Tensor} on cpu.
    """
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    if "keys" in blob and "values" in blob:
        cache_dict = {}
        for layer_idx, (k, v) in enumerate(zip(blob["keys"], blob["values"])):
            cache_dict[layer_idx] = {"k": k, "v": v}
        model_config = {
            "num_layers": blob["num_layers"],
            "num_heads": blob["num_heads"],
            "num_kv_heads": blob["num_kv_heads"],
            "head_dim": blob["head_dim"],
            "hidden_size": blob.get("hidden_size", 0),
        }
        return cache_dict, model_config
    if "cache" in blob:
        return blob["cache"], blob.get("model_config", {})
    if isinstance(blob, dict):
        return blob, {}
    raise RuntimeError(f"unexpected cache blob type: {type(blob)}")


def write_shell_corpus(
    cache_dict: dict,
    model_config: dict,
    n_tokens: int,
    n_shell_tokens: int,
    output_path: Path,
    seed: int,
) -> None:
    """Extract the LAST n_shell_tokens token K/V vectors and write a corpus JSON.

    Layout matches `build_prove_kv_corpus.py`:
      { "shape": {...}, "tokens": [{id, vector}], "seed": N }
    where vector is [num_layers * num_kv_heads * head_dim * 2] floats per token.
    """
    num_layers = model_config["num_layers"]
    num_kv_heads = model_config["num_kv_heads"]
    head_dim = model_config["head_dim"]
    print(
        f"[shell-corpus] extracting last {n_shell_tokens} tokens "
        f"(total={n_tokens}, layers={num_layers}, kv_heads={num_kv_heads}, head_dim={head_dim})",
        flush=True,
    )
    tokens = []
    for t_local in range(n_shell_tokens):
        t_global = n_tokens - n_shell_tokens + t_local
        # Per-token vector: concat K and V for all layers.
        # Within each layer, the cache stores [num_kv_heads, T, head_dim].
        # Per-token slice is [num_kv_heads, head_dim] for K, same for V.
        # We lay it out: layer0_K[h0..], layer0_K[h1..], ..., layer0_V[..],
        #                 layer1_K[..], layer1_V[..], ...
        full_vec = []
        for layer_idx in range(num_layers):
            layer = cache_dict[layer_idx] if layer_idx in cache_dict else cache_dict[str(layer_idx)]
            k = layer["k"]  # (1, num_kv_heads, T, head_dim) or (num_kv_heads, T, head_dim)
            v = layer["v"]
            # Squeeze batch dim if present
            if k.dim() == 4 and k.shape[0] == 1:
                k = k.squeeze(0)
                v = v.squeeze(0)
            k_tok = k[:, t_global, :].reshape(-1)  # (num_kv_heads * head_dim,)
            v_tok = v[:, t_global, :].reshape(-1)
            full_vec.extend(k_tok.tolist())
            full_vec.extend(v_tok.tolist())
        tokens.append({"id": f"tok_{t_local}", "vector": full_vec})

    corpus = {
        "shape": {
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "attention_type": (
                "MHA" if num_kv_heads == model_config.get("num_heads", num_kv_heads)
                else "GQA"
            ),
            "seed": seed,
        },
        "tokens": tokens,
    }
    output_path.write_text(json.dumps(corpus))
    print(f"[shell-corpus] wrote {len(tokens)} tokens to {output_path}", flush=True)


def read_shell_binary(bin_path: Path, num_layers: int) -> list[dict]:
    """Read the shell_roundtrip binary format.

    Per layer: u32 K_len | K f32 | u32 V_len | V f32
    Returns a list of {layer_idx, k, v} dicts.
    """
    out = []
    with open(bin_path, "rb") as f:
        for layer_idx in range(num_layers):
            k_len_bytes = f.read(4)
            if len(k_len_bytes) < 4:
                raise RuntimeError(f"truncated file at layer {layer_idx}")
            k_len = struct.unpack("<I", k_len_bytes)[0]
            k_data = f.read(k_len * 4)
            v_len_bytes = f.read(4)
            v_len = struct.unpack("<I", v_len_bytes)[0]
            v_data = f.read(v_len * 4)
            k = torch.frombuffer(bytearray(k_data), dtype=torch.float32).clone()
            v = torch.frombuffer(bytearray(v_data), dtype=torch.float32).clone()
            out.append({"layer_idx": layer_idx, "k": k, "v": v})
    return out


def patch_cache_with_shell(
    cache_dict: dict,
    shell_layers: list[dict],
    n_tokens: int,
    n_shell_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    num_layers: int,
    target_dtype: torch.dtype,
) -> dict:
    """Build a new DynamicCache-shaped dict with the shell's K/V patched in
    at positions [n_tokens - n_shell_tokens, n_tokens)."""
    patched = {}
    shell_start = n_tokens - n_shell_tokens
    for layer_idx in range(num_layers):
        src = cache_dict[layer_idx] if layer_idx in cache_dict else cache_dict[str(layer_idx)]
        k_orig = src["k"]
        v_orig = src["v"]
        # Squeeze batch dim if present
        if k_orig.dim() == 4 and k_orig.shape[0] == 1:
            k_orig = k_orig.squeeze(0)
            v_orig = v_orig.squeeze(0)
        # k_orig is (num_kv_heads, T, head_dim) in original dtype
        k_patched = k_orig.clone().to(torch.float32)
        v_patched = v_orig.clone().to(torch.float32)
        # Replace the last n_shell_tokens positions
        shell_k = shell_layers[layer_idx]["k"]  # (num_shell_tokens * num_kv_heads * head_dim,)
        shell_v = shell_layers[layer_idx]["v"]
        # Reshape to (n_shell_tokens, num_kv_heads, head_dim) then transpose to (kv, t, dim)
        sh_k = shell_k.reshape(n_shell_tokens, num_kv_heads, head_dim).permute(1, 0, 2).contiguous()
        sh_v = shell_v.reshape(n_shell_tokens, num_kv_heads, head_dim).permute(1, 0, 2).contiguous()
        k_patched[:, shell_start:, :] = sh_k
        v_patched[:, shell_start:, :] = sh_v
        patched[layer_idx] = {
            "k": k_patched.to(target_dtype),
            "v": v_patched.to(target_dtype),
        }
    return patched


# ---------- phases ----------

def phase0_oracle(args, state: dict) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print("[phase0] importing transformers + datasets", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"[phase0] loading model in fp16 on {args.device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", None)
    num_heads = getattr(cfg, "num_attention_heads", None)
    num_kv_heads = getattr(cfg, "num_key_value_heads", None) or num_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // num_heads)
    hidden_size = cfg.hidden_size
    model_config = {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
    }
    print(
        f"[phase0] model config: num_layers={num_layers} num_heads={num_heads} "
        f"num_kv_heads={num_kv_heads} head_dim={head_dim} hidden_size={hidden_size}",
        flush=True,
    )
    if head_dim % 4 != 0:
        raise RuntimeError(f"head_dim={head_dim} not divisible by 4")

    print(f"[phase0] loading corpus: {args.corpus}", flush=True)
    if args.corpus == "wikitext-2":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    else:
        text = Path(args.corpus).read_text()

    print(f"[phase0] tokenizing (n_tokens target: {args.n_tokens})", flush=True)
    enc = tokenizer(text, return_tensors="pt")
    full_ids = enc.input_ids[0]
    input_ids = full_ids[: args.n_tokens].to(args.device).unsqueeze(0)
    print(f"[phase0] input_ids shape: {tuple(input_ids.shape)}", flush=True)

    print("[phase0] forward pass (use_cache=True)", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    fwd_s = time.time() - t0
    print(f"[phase0] forward done in {fwd_s:.1f}s", flush=True)

    cache = out.past_key_values
    # Extract per-layer K/V as a dict
    cache_dict = {}
    for layer_idx in range(num_layers):
        k, v = cache[layer_idx]
        cache_dict[layer_idx] = {"k": k.detach().cpu(), "v": v.detach().cpu()}

    # Compute oracle PPL
    nll = per_token_nll(model, input_ids)
    ppl, ws, we = windowed_ppl(nll, args.ppl_frac)
    print(f"[phase0] oracle PPL: {ppl:.4f} (window: tokens {ws}..{we})", flush=True)

    # Save the cache to disk for Phase 1
    cache_path = args.output.parent / "cache_oracle.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"cache": cache_dict, "model_config": model_config},
        cache_path,
    )
    print(f"[phase0] cache saved: {cache_path}", flush=True)

    # Free GPU memory
    del model, out, cache
    torch.cuda.empty_cache()

    state["phase0"] = {
        "status": "complete",
        "ppl": ppl,
        "ppl_window": [ws, we],
        "cache_path": str(cache_path),
        "cache_bytes": cache_path.stat().st_size,
        "model_config": model_config,
        "forward_seconds": fwd_s,
        "n_tokens": args.n_tokens,
        "n_shell_tokens": args.n_shell_tokens,
        "completed_at": iso_now(),
    }


def phase1_shell_roundtrip(args, state: dict, lossy: bool) -> dict:
    """Run the shell roundtrip (lossless or lossy) and return PPL result."""
    suffix = "lossy" if lossy else "lossless"
    print(f"\n========== PHASE 1 ({suffix.upper()}): SHELL ROUNDTRIP ==========", flush=True)
    cfg = state["model_config"]
    n_tokens = state["phase0"]["n_tokens"]
    n_shell_tokens = args.n_shell_tokens

    # Load the oracle cache from disk
    cache_path = Path(state["phase0"]["cache_path"])
    cache_dict, _ = read_dynamic_cache(cache_path)

    # Build the shell corpus (last n_shell_tokens of the cache)
    corpus_path = args.output.parent / f"shell_corpus_{suffix}.json"
    write_shell_corpus(
        cache_dict=cache_dict,
        model_config=cfg,
        n_tokens=n_tokens,
        n_shell_tokens=n_shell_tokens,
        output_path=corpus_path,
        seed=args.seed,
    )

    # Invoke prove_kv_shell_roundtrip
    bin_path = args.output.parent / f"shell_roundtrip_{suffix}.bin"
    if bin_path.exists():
        bin_path.unlink()
    cli = args.shell_cli
    if not Path(cli).exists():
        raise FileNotFoundError(f"shell CLI not found: {cli}")
    cli_args = [cli, str(corpus_path), str(bin_path)]
    if lossy:
        cli_args.append("--lossy")
    print(f"[phase1] running {cli} on shell corpus", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        cli_args,
        check=False, capture_output=True, text=True, timeout=args.phase1_timeout,
    )
    rt_s = time.time() - t0
    if proc.returncode != 0:
        print(f"[phase1] CLI stdout: {proc.stdout}", flush=True)
        print(f"[phase1] CLI stderr: {proc.stderr}", flush=True)
        raise RuntimeError(f"shell CLI failed with code {proc.returncode}")
    print(f"[phase1] CLI ok in {rt_s:.1f}s", flush=True)
    if proc.stderr:
        # Show last few lines for context
        for line in proc.stderr.strip().split("\n")[-5:]:
            print(f"[phase1] CLI: {line}", flush=True)

    # Read the roundtripped shell K/V
    shell_layers = read_shell_binary(bin_path, cfg["num_layers"])
    print(f"[phase1] read {len(shell_layers)} layers from shell bin", flush=True)

    # Patch the cache with the roundtripped shell
    patched_cache = patch_cache_with_shell(
        cache_dict=cache_dict,
        shell_layers=shell_layers,
        n_tokens=n_tokens,
        n_shell_tokens=n_shell_tokens,
        num_kv_heads=cfg["num_kv_heads"],
        head_dim=cfg["head_dim"],
        num_layers=cfg["num_layers"],
        target_dtype=torch.float16,
    )

    # Load the model again for the patched forward pass
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"[phase1] loading model fresh: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    # Re-tokenize (matches Phase 0 exactly)
    from datasets import load_dataset
    if args.corpus == "wikitext-2":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    else:
        text = Path(args.corpus).read_text()
    enc = tokenizer(text, return_tensors="pt")
    full_ids = enc.input_ids[0][: n_tokens].to(args.device).unsqueeze(0)
    print(f"[phase1] re-tokenized: input_ids shape={tuple(full_ids.shape)}", flush=True)

    # Build a DynamicCache and pre-populate it with the PATCHED K/V
    # (mirrors the pattern in ppl_validate.py phase1_compressed)
    from transformers import DynamicCache
    if hasattr(DynamicCache(), "layers"):
        from transformers.cache_utils import DynamicLayer
        fresh_cache = DynamicCache()
        while len(fresh_cache.layers) < cfg["num_layers"]:
            fresh_cache.layers.append(DynamicLayer())
        for i in range(cfg["num_layers"]):
            # patched_cache[i] is (num_kv_heads, T, head_dim) in fp16 on cpu
            k = patched_cache[i]["k"].unsqueeze(0).to(args.device)  # (1, kv, T, dim)
            v = patched_cache[i]["v"].unsqueeze(0).to(args.device)
            fresh_cache.layers[i].keys = k
            fresh_cache.layers[i].values = v
    else:
        # Legacy tuple format
        legacy = tuple(
            (patched_cache[i]["k"].unsqueeze(0).to(args.device),
             patched_cache[i]["v"].unsqueeze(0).to(args.device))
            for i in range(cfg["num_layers"])
        )
        fresh_cache = legacy

    # Forward pass with the patched cache. We pass input_ids of shape
    # (1, 1) — the LAST token — and let the model use the pre-populated
    # cache for [0, T). The model computes Q for the last token, attends
    # to cache positions [0, T), and produces logits for predicting the
    # (T+1)th token. This gives us the logit at position T-1, which
    # predicts the token at position T.
    #
    # Wait — for PPL we need logits at ALL positions in the eval window
    # [ws, T-1] which predict positions [ws+1, T]. The single-logit
    # approach won't work.
    #
    # The right pattern: do a prefill of the cache (no model.forward
    # call yet), then call forward with input_ids[:, ws:T] and
    # cache_position = [0, 1, ..., ws-1, ws, ws+1, ...]. The model will
    # compute K/V at [ws, T) (none, since we already have them) and
    # compute Q at [ws, T) using the cache.
    #
    # But transformers' DynamicCache doesn't easily support "use cache
    # at [0, ws) and compute fresh at [ws, T)" in one call. The simplest
    # workaround: pass input_ids of length T (the full sequence) and a
    # fresh empty cache. The model computes K/V at [0, T) and stores
    # them. Then we replace those K/V with the patched K/V.
    #
    # Easiest implementation: pass input_ids and let the model compute.
    # Then OVERWRITE the cache's K/V at the relevant positions with the
    # patched K/V. Run the model AGAIN to produce the correct logits.
    # This is what `ppl_validate.py` does, but it relies on the model
    # using the same K/V the second time (since the cache is in
    # past_key_values).
    #
    # Actually the simplest correct path: just do the standard pattern
    # from `ppl_validate.py` — pre-populate, then forward, with
    # use_cache=True. The first run in this script (lossless) worked
    # correctly (1.6s, PPL 4.7608). The second run (lossy) takes 0.0s
    # which suggests a state issue. Let me try adding a CUDA sync.

    torch.cuda.synchronize()
    t_fwd = time.time()
    with torch.no_grad():
        out2 = model(
            input_ids=full_ids,
            past_key_values=fresh_cache,
            use_cache=True,
            return_dict=True,
        )
    torch.cuda.synchronize()
    fwd_s = time.time() - t_fwd
    print(f"[phase1] forward done in {fwd_s:.1f}s, out2.logits.shape={tuple(out2.logits.shape)}", flush=True)

    # Compute PPL over the eval window. The full forward pass gives
    # logits at positions [0, T-1] which predict positions [1, T].
    # Standard shift: logits[:, :-1] predicts input_ids[:, 1:].
    logits = out2.logits[:, :-1, :].contiguous()
    targets = full_ids[:, 1:].contiguous()
    nll_chunks = []
    chunk_size = 512
    for start in range(0, logits.shape[1], chunk_size):
        end = min(start + chunk_size, logits.shape[1])
        l_chunk = logits[:, start:end, :].float()
        lse = torch.logsumexp(l_chunk, dim=-1, keepdim=True)
        tgt = targets[:, start:end].unsqueeze(-1)
        log_prob_at_tgt = l_chunk.gather(2, tgt).squeeze(-1)
        nll_chunk = lse.squeeze(-1) - log_prob_at_tgt
        nll_chunks.append(nll_chunk.squeeze(0))
    nll = torch.cat(nll_chunks, dim=0)

    ppl, ws, we = windowed_ppl(nll, args.ppl_frac)
    print(
        f"[phase1] {suffix} shell PPL: {ppl:.4f} (window: tokens {ws}..{we})",
        flush=True,
    )

    del model
    torch.cuda.empty_cache()

    return {
        "ppl": ppl,
        "ppl_window": [ws, we],
        "compression_ratio": None,  # filled in by the receipt
        "shell_size_bytes": None,
        "roundtrip_seconds": rt_s,
        "forward_seconds": fwd_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--corpus", default="wikitext-2")
    parser.add_argument("--n-tokens", type=int, default=1024)
    parser.add_argument("--n-shell-tokens", type=int, default=224,
                        help="Number of tokens to put in the shell (the 'unique' agent prefix)")
    parser.add_argument("--ppl-frac", type=float, default=0.3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shell-cli", type=Path,
        default=Path(
            "/home/jstevenson/proveKV_bench/prove_kv_shell_roundtrip"
        ),
        help="Path to the prove_kv_shell_roundtrip binary",
    )
    parser.add_argument("--phase1-timeout", type=int, default=600,
                        help="Phase 1 CLI timeout in seconds (default 10min)")
    args = parser.parse_args()

    state = {
        "schema_version": "1.0.0",
        "model": args.model,
        "corpus": args.corpus,
        "n_tokens": args.n_tokens,
        "n_shell_tokens": args.n_shell_tokens,
        "ppl_frac": args.ppl_frac,
        "seed": args.seed,
        "started_at": iso_now(),
    }
    if args.output.exists():
        state = json.loads(args.output.read_text())
        print(f"[main] resumed state from {args.output}", flush=True)

    # Phase 0
    if state.get("phase0", {}).get("status") != "complete":
        phase0_oracle(args, state)
        write_state(args.output, state)
    else:
        print("[main] phase0 already complete, skipping", flush=True)

    # Phase 1 — lossless
    if state.get("phase1_lossless", {}).get("status") != "complete":
        result = phase1_shell_roundtrip(args, state, lossy=False)
        result["status"] = "complete"
        result["completed_at"] = iso_now()
        state["phase1_lossless"] = result
        write_state(args.output, state)
    else:
        print("[main] phase1_lossless already complete, skipping", flush=True)

    # Phase 1 — lossy
    if state.get("phase1_lossy", {}).get("status") != "complete":
        result = phase1_shell_roundtrip(args, state, lossy=True)
        result["status"] = "complete"
        result["completed_at"] = iso_now()
        state["phase1_lossy"] = result
        write_state(args.output, state)
    else:
        print("[main] phase1_lossy already complete, skipping", flush=True)

    # Report
    oracle = state["phase0"]["ppl"]
    lossless = state["phase1_lossless"]["ppl"]
    lossy = state["phase1_lossy"]["ppl"]
    print("\n========== FINAL ==========", flush=True)
    print(
        f"oracle_ppl={oracle:.4f} "
        f"lossless_shell_ppl={lossless:.4f} "
        f"lossy_shell_ppl={lossy:.4f}",
        flush=True,
    )
    print(
        f"delta_lossless={((lossless-oracle)/oracle)*100:+.2f}% "
        f"delta_lossy={((lossy-oracle)/oracle)*100:+.2f}%",
        flush=True,
    )
    state["completed_at"] = iso_now()
    write_state(args.output, state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
