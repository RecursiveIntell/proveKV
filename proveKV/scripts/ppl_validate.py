#!/usr/bin/env python3
"""
ppl_validate.py — Real-LLM PPL validation for proveKV.

Methodology (locked per PPL_RUNBOOK.md):
- Phase 0: Oracle baseline. use_cache=True forward pass on N WikiText-2 tokens.
  Save K/V cache to cache_oracle.pt. Compute PPL over last 30%.
- Phase 1: Compressed roundtrip. Convert oracle cache to proveKV corpus JSON,
  invoke the prove_kv_fast_roundtrip CLI, read the decompressed layers,
  build a fresh DynamicCache, run a SECOND forward pass, compute PPL over last 30%.
- Phase 2: Report. delta_ppl_pct, compression_ratio, per-layer stats. Write report.md
  and append summary to state.json.

Hard constraints (per runbook):
- head_dim % 4 == 0 (SmolLM2 head_dim=64 ✓)
- head_dim > 4 (block_dim default)
- seed=42 on decode (v1 limitation, fine because encode+decode in same process)
- Don't run PPL in main session — use nohup, write state.json after every phase

Usage:
    python scripts/ppl_validate.py \
        --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
        --corpus wikitext-2 \
        --n-tokens 1024 \
        --ppl-frac 0.3 \
        --output bench/ppl/smollm2-1.7b/wikitext-2/state.json

NOTE: fib-quant decode is O(blocks × indices_per_block) per call and the current
single-block decode path does not vectorize. At n_tokens=1024 the roundtrip takes
~30+ minutes per layer × 24 layers. For the Phase 1 demo, use n_tokens=256
(default for the fast roundtrip path) to keep the demo under 5 minutes total.
The oracle PPL is computed at the requested n_tokens; the compressed PPL uses
the same prefix.
"""
import argparse
import datetime
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


# ---------- helpers ----------

def iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()


def write_state(state_path: Path, state: dict) -> None:
    """Atomically write state.json so a crash mid-write doesn't corrupt the file."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, state_path)


def per_token_nll(model, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Standard HF causal LM PPL recipe:
      Forward pass with labels=input_ids, return the per-token negative log-likelihood.
    We compute PPL = exp(mean(NLL over the eval window)).

    Memory-conscious: we don't materialize the full (T, V) log_softmax in fp32.
    Instead, we do a chunked logsumexp over the vocab dim.
    """
    out = model(input_ids=input_ids, labels=input_ids, return_dict=True)
    logits = out.logits[:, :-1, :].contiguous()  # predict next token (1, T-1, V)
    targets = input_ids[:, 1:].contiguous()       # (1, T-1)
    del out
    nll_chunks = []
    chunk_size = 512
    for start in range(0, logits.shape[1], chunk_size):
        end = min(start + chunk_size, logits.shape[1])
        l_chunk = logits[:, start:end, :].float()  # (1, chunk, V)
        lse = torch.logsumexp(l_chunk, dim=-1, keepdim=True)  # (1, chunk, 1)
        tgt = targets[:, start:end].unsqueeze(-1)             # (1, chunk, 1)
        log_prob_at_tgt = l_chunk.gather(2, tgt).squeeze(-1)  # (1, chunk)
        nll_chunk = lse.squeeze(-1) - log_prob_at_tgt         # (1, chunk)
        nll_chunks.append(nll_chunk)
        del l_chunk, lse, log_prob_at_tgt
    nll = torch.cat(nll_chunks, dim=1).squeeze(0)  # (T-1,)
    del logits, targets
    return nll


def windowed_ppl(nll: torch.Tensor, ppl_frac: float) -> tuple[float, int, int]:
    """
    Return (ppl, start, end) where start..end is the eval window
    (last `ppl_frac` of the sequence).
    """
    T = nll.shape[0]
    start = int(T * (1.0 - ppl_frac))
    end = T
    window = nll[start:end]
    mean_nll = window.mean().item()
    return math.exp(mean_nll), start, end


# ---------- phases ----------

def phase0_oracle(args, state: dict) -> None:
    """
    Load model + corpus. Run use_cache=True forward pass on first n_tokens.
    Save the resulting K/V cache. Compute oracle PPL over last ppl_frac.
    """
    print("[phase0] importing transformers + datasets", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"[phase0] loading tokenizer: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"[phase0] loading model in fp16 on {args.device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    if args.device == "cuda":
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"[phase0] model on cuda, {alloc:.2f} GB allocated", flush=True)
        if alloc == 0:
            raise RuntimeError("model is on CPU despite --device cuda")

    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", None)
    num_heads = getattr(cfg, "num_attention_heads", None)
    num_kv_heads = getattr(cfg, "num_key_value_heads", None) or num_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // num_heads)
    hidden_size = cfg.hidden_size
    print(
        f"[phase0] model config: num_layers={num_layers} num_heads={num_heads} "
        f"num_kv_heads={num_kv_heads} head_dim={head_dim} hidden_size={hidden_size}",
        flush=True,
    )
    if head_dim % 4 != 0:
        raise RuntimeError(f"head_dim={head_dim} is not divisible by 4 (fib codec default k=4)")
    if head_dim <= 4:
        raise RuntimeError(f"head_dim={head_dim} must be > 4 (block_dim default)")

    print(f"[phase0] loading corpus: {args.corpus}", flush=True)
    if args.corpus == "wikitext-2":
        ds = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
        )
        text = "\n\n".join(ds["text"])
    elif args.corpus.startswith("file:"):
        text = Path(args.corpus[5:]).read_text()
    else:
        raise ValueError(f"unknown corpus: {args.corpus} (use 'wikitext-2' or 'file:/path')")

    print(f"[phase0] tokenizing (n_tokens target: {args.n_tokens})", flush=True)
    enc = tokenizer(text, return_tensors="pt")
    full_ids = enc.input_ids[0]
    if full_ids.shape[0] < args.n_tokens:
        raise RuntimeError(
            f"corpus has only {full_ids.shape[0]} tokens, need {args.n_tokens}"
        )
    input_ids = full_ids[: args.n_tokens].to(args.device).unsqueeze(0)
    print(f"[phase0] input_ids shape: {tuple(input_ids.shape)}", flush=True)

    # Forward pass with use_cache=True to capture K/V
    print("[phase0] forward pass (use_cache=True)", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    fwd_s = time.time() - t0
    print(f"[phase0] forward done in {fwd_s:.1f}s", flush=True)

    cache = out.past_key_values
    print(f"[phase0] cache type: {type(cache).__name__}", flush=True)
    if hasattr(cache, "layers"):
        keys_list = [layer.keys for layer in cache.layers]
        vals_list = [layer.values for layer in cache.layers]
    else:
        keys_list = [k for k, v in cache]
        vals_list = [v for k, v in cache]
    print(
        f"[phase0] cache: {len(keys_list)} layers, "
        f"K[0] shape={tuple(keys_list[0].shape)} dtype={keys_list[0].dtype}",
        flush=True,
    )

    cache_path = args.output.parent / "cache_oracle.pt"
    print(f"[phase0] saving cache to {cache_path}", flush=True)
    torch.save(
        {
            "model_id": args.model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "seq_len": int(input_ids.shape[1]),
            "keys": [k.detach().cpu() for k in keys_list],
            "values": [v.detach().cpu() for v in vals_list],
        },
        cache_path,
    )
    cache_bytes = cache_path.stat().st_size
    print(f"[phase0] cache saved: {cache_bytes/1e6:.1f} MB", flush=True)

    print("[phase0] computing oracle PPL", flush=True)
    nll = per_token_nll(model, input_ids)
    ppl, start, end = windowed_ppl(nll, args.ppl_frac)
    print(
        f"[phase0] oracle PPL: {ppl:.4f} (window: tokens {start}..{end} of {nll.shape[0]})",
        flush=True,
    )

    state["phase0"] = {
        "status": "complete",
        "ppl": ppl,
        "ppl_window": [start, end],
        "cache_path": str(cache_path),
        "cache_bytes": cache_bytes,
        "model_config": {
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
        },
        "forward_seconds": fwd_s,
        "completed_at": iso_now(),
    }
    state["model_config"] = state["phase0"]["model_config"]
    state["model_slug"] = args.model_slug
    write_state(args.output, state)

    del model, out, cache
    torch.cuda.empty_cache() if args.device == "cuda" else None


def phase1_compressed(args, state: dict) -> None:
    """
    Read cache_oracle.pt, convert to proveKV corpus JSON, invoke the
    prove_kv_fast_roundtrip CLI to compress+decompress, read
    the decompressed layers, run a fresh forward pass with the new cache,
    compute PPL over last ppl_frac, and write delta_ppl_pct.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from datasets import load_dataset

    cfg = state["model_config"]
    cache_path = Path(state["phase0"]["cache_path"])
    if not cache_path.exists():
        raise FileNotFoundError(f"oracle cache missing: {cache_path}")

    # 1) Convert DynamicCache to proveKV corpus JSON
    print("[phase1] building proveKV corpus from oracle cache", flush=True)
    corpus_path = args.output.parent / "prove_kv_corpus.json"
    cmd_build = [
        sys.executable, str(args.script_dir / "build_prove_kv_corpus.py"),
        "--cache", str(cache_path),
        "--output", str(corpus_path),
        "--seed", "42",
    ]
    r = subprocess.run(cmd_build, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"build_prove_kv_corpus.py failed with code {r.returncode}")
    corpus_bytes = corpus_path.stat().st_size
    print(f"[phase1] corpus: {corpus_path} ({corpus_bytes/1e6:.1f} MB)", flush=True)

    # 2) Invoke the prove_kv_fast_roundtrip CLI
    roundtrip_bin = args.output.parent / "roundtrip.bin"
    if roundtrip_bin.exists():
        roundtrip_bin.unlink()
    cli = args.prove_kv_cli
    if not Path(cli).exists():
        raise FileNotFoundError(f"prove_kv CLI not found: {cli}")
    print(f"[phase1] running {cli} on corpus", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [cli, str(corpus_path), str(roundtrip_bin)],
        check=False, capture_output=True, text=True, timeout=args.phase1_timeout,
    )
    rt_s = time.time() - t0
    if proc.returncode != 0:
        print(f"[phase1] CLI stdout: {proc.stdout}", flush=True)
        print(f"[phase1] CLI stderr: {proc.stderr}", flush=True)
        raise RuntimeError(f"prove_kv CLI failed with code {proc.returncode}")
    print(f"[phase1] CLI ok in {rt_s:.1f}s", flush=True)
    if proc.stderr:
        print(f"[phase1] CLI stderr (tail): {proc.stderr[-500:]}", flush=True)

    # 3) Read the binary file: [manifest_len:u64][manifest_json][layer_0_len:u64][layer_0_json]...
    print(f"[phase1] reading {roundtrip_bin}", flush=True)
    with open(roundtrip_bin, "rb") as f:
        data = f.read()
    offset = 0
    manifest_len = int.from_bytes(data[offset:offset + 8], "little"); offset += 8
    manifest_bytes = data[offset:offset + manifest_len]; offset += manifest_len
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    print(
        f"[phase1] manifest: pool_id={manifest['pool_id'][:12]} "
        f"compression_ratio={manifest['compression_ratio']:.2f}x "
        f"pool_size_bytes={manifest['pool_size_bytes']} "
        f"total_compressed_bytes={manifest['total_compressed_bytes']}",
        flush=True,
    )

    # 4) Reconstruct a DynamicCache from the decompressed layer blobs
    # Build new_keys/new_vals directly on GPU as fp16 (cuts memory in half
    # vs fp32-on-CPU + move-to-GPU).
    print(f"[phase1] rebuilding {cfg['num_layers']} layer tensors directly on {args.device}", flush=True)
    new_keys: list[torch.Tensor] = []
    new_vals: list[torch.Tensor] = []
    for i in range(cfg["num_layers"]):
        layer_len = int.from_bytes(data[offset:offset + 8], "little"); offset += 8
        layer_bytes = data[offset:offset + layer_len]; offset += layer_len
        layer_data = json.loads(layer_bytes.decode("utf-8"))
        T = layer_data["num_tokens"]
        H = layer_data["num_heads"]
        D = layer_data["head_dim"]
        k_per_head = layer_data["keys"]   # length H, each of length T*D
        v_per_head = layer_data["values"]
        # Allocate destination directly on target device, fp16
        k = torch.empty((1, H, T, D), dtype=torch.float16, device=args.device)
        v = torch.empty((1, H, T, D), dtype=torch.float16, device=args.device)
        for h in range(H):
            # Build per-head slice on CPU as fp16 (small), then move+copy to GPU
            kh = torch.tensor(k_per_head[h], dtype=torch.float32).reshape(T, D).to(
                args.device, dtype=torch.float16, non_blocking=True
            )
            vh = torch.tensor(v_per_head[h], dtype=torch.float32).reshape(T, D).to(
                args.device, dtype=torch.float16, non_blocking=True
            )
            k[0, h, :, :] = kh
            v[0, h, :, :] = vh
        new_keys.append(k)
        new_vals.append(v)
        # Free the per-layer dict (large Python list of lists)
        del layer_data, k_per_head, v_per_head, kh, vh
    assert offset == len(data), f"read {offset} of {len(data)} bytes"
    print(f"[phase1] rebuilt {len(new_keys)} layer tensors on {args.device}", flush=True)
    print(
        f"[phase1] shape per layer: K={tuple(new_keys[0].shape)} V={tuple(new_vals[0].shape)}",
        flush=True,
    )
    # Free the raw binary blob
    del data
    import gc; gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # 5) Reconstruct input_ids for the same prefix
    # Free everything we don't need before re-loading the model. The
    # roundtrip.bin JSON parse leaves a ~1GB Python list of dicts
    # around that we have to release before the second model load.
    import gc
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
        # Print current GPU usage for diagnostics
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        print(
            f"[phase1] pre-second-model-load: GPU alloc={mem_alloc:.2f}GB reserved={mem_reserved:.2f}GB",
            flush=True,
        )

    # 5b) Move the new_keys/new_vals to CPU temporarily so the second
    # model load has room. We'll move them back before the forward pass.
    # (The cache is only 200MB so this is cheap.)
    new_keys_cpu = [k.cpu() for k in new_keys]
    new_vals_cpu = [v.cpu() for v in new_vals]
    del new_keys, new_vals
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        print(
            f"[phase1] after-cache-offload: GPU alloc={mem_alloc:.2f}GB",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    full_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    input_ids = full_ids[: args.n_tokens].to(args.device).unsqueeze(0)
    print(f"[phase1] re-tokenized: input_ids shape={tuple(input_ids.shape)}", flush=True)

    # 6) Load the model FRESH, pre-populate a fresh DynamicCache with our K/V, forward
    print(f"[phase1] loading model fresh: {args.model}", flush=True)
    # NOTE: low_cpu_mem_usage=True uses a streaming loader that keeps
    # intermediate fp32 copies on GPU during materialization — it eats
    # ~4GB of extra VRAM on the 7.91GB msi host and OOMs at the .to(cuda)
    # step. Default loading puts the model in CPU RAM first then moves to
    # GPU in one shot. We have 64GB RAM on msi so this is fine.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    ).to(args.device)
    model.eval()

    fresh_cache = DynamicCache()
    if hasattr(fresh_cache, "layers"):
        # Pre-allocate the per-layer DynamicLayer objects (transformers 5.x
        # DynamicCache starts with 0 layers and grows them on update()).
        from transformers.cache_utils import DynamicLayer
        while len(fresh_cache.layers) < cfg["num_layers"]:
            fresh_cache.layers.append(DynamicLayer())
        for i in range(cfg["num_layers"]):
            # new_keys_cpu/new_vals_cpu are on CPU; move to GPU now
            fresh_cache.layers[i].keys = new_keys_cpu[i].to(args.device)
            fresh_cache.layers[i].values = new_vals_cpu[i].to(args.device)
        t0 = time.time()
        with torch.no_grad():
            out2 = model(
                input_ids=input_ids,
                past_key_values=fresh_cache,
                use_cache=True,
                return_dict=True,
            )
        fwd2_s = time.time() - t0
        print(f"[phase1] forward with pre-populated cache done in {fwd2_s:.1f}s", flush=True)
    else:
        legacy = tuple(
            (new_keys_cpu[i].to(args.device), new_vals_cpu[i].to(args.device))
            for i in range(cfg["num_layers"])
        )
        t0 = time.time()
        with torch.no_grad():
            out2 = model(
                input_ids=input_ids,
                past_key_values=legacy,
                use_cache=True,
                return_dict=True,
            )
        fwd2_s = time.time() - t0

    # 7) Compute roundtrip PPL
    print("[phase1] computing roundtrip PPL", flush=True)
    logits2 = out2.logits[:, :-1, :].contiguous()
    targets2 = input_ids[:, 1:].contiguous()
    nll2_chunks = []
    for start in range(0, logits2.shape[1], 512):
        end = min(start + 512, logits2.shape[1])
        l_chunk = logits2[:, start:end, :].float()
        lse = torch.logsumexp(l_chunk, dim=-1, keepdim=True)
        tgt = targets2[:, start:end].unsqueeze(-1)
        log_prob_at_tgt = l_chunk.gather(2, tgt).squeeze(-1)
        nll2_chunks.append(lse.squeeze(-1) - log_prob_at_tgt)
        del l_chunk, lse, log_prob_at_tgt
    nll2 = torch.cat(nll2_chunks, dim=1).squeeze(0)
    del logits2, targets2
    ppl2, start2, end2 = windowed_ppl(nll2, args.ppl_frac)
    print(f"[phase1] roundtrip PPL: {ppl2:.4f} (window: tokens {start2}..{end2})", flush=True)

    ppl_oracle = state["phase0"]["ppl"]
    delta_ppl_pct = (ppl2 - ppl_oracle) / ppl_oracle * 100.0
    print(
        f"[phase1] DELTA: oracle={ppl_oracle:.4f} roundtrip={ppl2:.4f} "
        f"delta_ppl_pct={delta_ppl_pct:+.2f}%",
        flush=True,
    )

    state["phase1"] = {
        "status": "complete",
        "ppl": ppl2,
        "ppl_window": [start2, end2],
        "roundtrip_bin": str(roundtrip_bin),
        "roundtrip_bin_bytes": roundtrip_bin.stat().st_size,
        "manifest": manifest,
        "compression_ratio": manifest["compression_ratio"],
        "pool_size_bytes": manifest["pool_size_bytes"],
        "total_compressed_bytes": manifest["total_compressed_bytes"],
        "delta_ppl_pct": delta_ppl_pct,
        "roundtrip_cli_seconds": rt_s,
        "forward_with_overwritten_cache_seconds": fwd2_s,
        "completed_at": iso_now(),
    }
    write_state(args.output, state)

    del model, out2, fresh_cache
    torch.cuda.empty_cache() if args.device == "cuda" else None


def phase2_report(args, state: dict) -> None:
    """Compute per-layer stats and write the final report.md."""
    p0 = state["phase0"]
    p1 = state["phase1"]
    cfg = state["model_config"]

    # Per-layer bytes from the roundtrip binary
    per_layer: list[dict] = []
    rt_path = Path(p1["roundtrip_bin"])
    if rt_path.exists():
        with open(rt_path, "rb") as f:
            data = f.read()
        offset = 0
        mlen = int.from_bytes(data[offset:offset + 8], "little"); offset += 8 + mlen
        for i in range(cfg["num_layers"]):
            llen = int.from_bytes(data[offset:offset + 8], "little"); offset += 8
            layer_bytes = data[offset:offset + llen]; offset += llen
            T = p1["manifest"]["num_shared_tokens"]
            H = cfg["num_kv_heads"]
            D = cfg["head_dim"]
            oracle_bytes = 2 * H * T * D * 2  # fp16 = 2 bytes
            per_layer.append({
                "layer": i,
                "oracle_bytes": oracle_bytes,
                "roundtrip_layer_bytes": len(layer_bytes) + 8,
            })

    summary = (
        f"Oracle PPL {p0['ppl']:.4f} | Roundtrip PPL {p1['ppl']:.4f} | "
        f"delta_ppl_pct {p1['delta_ppl_pct']:+.2f}% | "
        f"compression_ratio {p1['compression_ratio']:.2f}x | "
        f"model {args.model} | corpus {args.corpus} | "
        f"n_tokens {args.n_tokens} | ppl_frac {args.ppl_frac}"
    )
    state["report"] = {
        "summary": summary,
        "per_layer": per_layer,
        "completed_at": iso_now(),
    }
    write_state(args.output, state)

    # Write report.md
    report_path = args.output.parent / "report.md"
    lines = [
        f"# PPL Validation Report — {args.model} on {args.corpus}",
        "",
        f"- **Generated:** {iso_now()}",
        f"- **Model:** `{args.model}`",
        f"- **Corpus:** `{args.corpus}` (n_tokens={args.n_tokens}, ppl_frac={args.ppl_frac})",
        f"- **Seed:** 42",
        "",
        "## Headline",
        "",
        f"- **Oracle PPL:** {p0['ppl']:.4f}",
        f"- **Roundtrip PPL:** {p1['ppl']:.4f}",
        f"- **Δ PPL:** {p1['delta_ppl_pct']:+.2f}%",
        f"- **Compression ratio:** {p1['compression_ratio']:.2f}x",
        f"- **Pool size:** {p1['pool_size_bytes']:,} bytes",
        f"- **Total compressed:** {p1['total_compressed_bytes']:,} bytes",
        "",
        "## Methodology",
        "",
        f"1. **Phase 0 (oracle):** use_cache=True forward pass over first n_tokens of "
        f"WikiText-2 test split. fp16, cuda. PPL computed over last {args.ppl_frac*100:.0f}% "
        "of tokens (HF causal-LM recipe: shift, logsumexp, gather, exp(mean)).",
        "2. **Phase 1 (compressed roundtrip):**",
        "   - Extract per-token K/V vectors from the DynamicCache",
        "   - Build proveKV corpus JSON",
        f"   - Invoke `prove_kv_fast_roundtrip` CLI (composite build + parallel decompress) "
        f"({p1.get('roundtrip_cli_seconds', 0):.1f}s)",
        "   - Read decompressed layers from the roundtrip.bin output",
        "   - Pre-populate a fresh `DynamicCache` and forward with it",
        "   - PPL over the same window as Phase 0",
        "3. **Phase 2 (report):** per-layer byte accounting; this file.",
        "",
        "## Per-layer accounting",
        "",
        "| Layer | Oracle bytes (fp16 KV) | Roundtrip layer bytes (JSON+len) |",
        "|------:|-----------------------:|---------------------------------:|",
    ]
    for entry in per_layer:
        l = entry["layer"]
        o = entry.get("oracle_bytes")
        d = entry.get("roundtrip_layer_bytes")
        o_s = f"{o:,}" if o is not None else "—"
        d_s = f"{d:,}" if d is not None else "—"
        lines.append(f"| {l} | {o_s} | {d_s} |")
    lines += [
        "",
        "## Receipts",
        "",
        f"- `state.json` — full machine-readable state",
        f"- `cache_oracle.pt` — Phase 0 DynamicCache (fp16 K/V tensors)",
        f"- `prove_kv_corpus.json` — Phase 1 input to the proveKV CLI",
        f"- `roundtrip.bin` — Phase 1 binary output (manifest + 24 layer blobs)",
        f"- `manifest` (in `roundtrip.bin`) — pool manifest from proveKV",
        "",
        "## Caveats",
        "",
        "- The fib-quant decoder is single-block per call; for n_tokens=1024 × 24 layers "
        "the decode work is ~24M codeword lookups, which serial-decoded in Rust takes >30 min. "
        "For this initial validation, n_tokens can be reduced to keep roundtrip time under 5 min; "
        "for a public release the codec needs a vectorized batched decode implementation.",
        f"- transformers {__import__('transformers').__version__}, "
        f"torch {torch.__version__}, device {args.device}",
        f"- Model config: num_layers={cfg['num_layers']} num_heads={cfg['num_heads']} "
        f"num_kv_heads={cfg['num_kv_heads']} head_dim={cfg['head_dim']} "
        f"hidden_size={cfg['hidden_size']}",
        f"- Phase 0 forward: {p0.get('forward_seconds', 0):.1f}s",
        f"- Phase 1 roundtrip CLI: {p1.get('roundtrip_cli_seconds', 0):.1f}s",
        f"- Phase 1 forward with pre-populated cache: "
        f"{p1.get('forward_with_overwritten_cache_seconds', 0):.1f}s",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    print(f"[phase2] report written: {report_path}", flush=True)


# ---------- main ----------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    p.add_argument("--model-slug", default="smollm2-1.7b")
    p.add_argument("--corpus", default="wikitext-2",
                   help="wikitext-2 or file:/path/to/text")
    p.add_argument("--n-tokens", type=int, default=1024)
    p.add_argument("--ppl-frac", type=float, default=0.3)
    p.add_argument("--output", type=Path, required=True,
                   help="state.json path (other artifacts sit next to it)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--proveKV-cli", type=Path,
        default=Path(
            "/home/jstevenson/Coding/Libraries/proveKV/target/release/examples/"
            "prove_kv_fast_roundtrip"
        ),
        help="Path to the prove_kv_fast_roundtrip binary",
    )
    p.add_argument(
        "--script-dir", type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the helper scripts (build_prove_kv_corpus.py)",
    )
    p.add_argument("--phase1-timeout", type=int, default=2700,
                   help="Phase 1 CLI timeout in seconds (default 45min)")
    p.add_argument(
        "--skip-phase0", action="store_true",
        help="Skip Phase 0 (assume cache_oracle.pt + state.phase0 already exist)",
    )
    p.add_argument(
        "--skip-phase1", action="store_true",
        help="Skip Phase 1 (assume roundtrip.bin + state.phase1 already exist)",
    )
    args = p.parse_args()

    if args.output.exists():
        state = json.loads(args.output.read_text())
        print(f"[main] resumed state from {args.output}", flush=True)
    else:
        state = {
            "schema_version": "1.0.0",
            "model": args.model,
            "model_slug": args.model_slug,
            "corpus": args.corpus,
            "corpus_slug": "wikitext-2",
            "n_tokens": args.n_tokens,
            "ppl_frac": args.ppl_frac,
            "started_at": iso_now(),
        }
        write_state(args.output, state)
        print(f"[main] initialized state at {args.output}", flush=True)

    if state.get("phase0", {}).get("status") != "complete":
        if args.skip_phase0 and (args.output.parent / "cache_oracle.pt").exists():
            print("[main] --skip-phase0 set and cache exists; skipping", flush=True)
        else:
            print("\n========== PHASE 0: ORACLE BASELINE ==========", flush=True)
            phase0_oracle(args, state)
    else:
        print("[main] phase0 already complete, skipping", flush=True)

    if state.get("phase1", {}).get("status") != "complete":
        if args.skip_phase1 and (args.output.parent / "roundtrip.bin").exists():
            print("[main] --skip-phase1 set and roundtrip.bin exists; skipping", flush=True)
        else:
            print("\n========== PHASE 1: COMPRESSED ROUNDTRIP ==========", flush=True)
            phase1_compressed(args, state)
    else:
        print("[main] phase1 already complete, skipping", flush=True)

    if state.get("report", {}).get("summary") is None:
        print("\n========== PHASE 2: REPORT ==========", flush=True)
        phase2_report(args, state)
    else:
        print("[main] phase2 already complete, skipping", flush=True)

    p1 = state.get("phase1", {})
    p0 = state.get("phase0", {})
    if p1 and p0:
        print("\n========== FINAL ==========", flush=True)
        print(
            f"oracle_ppl={p0['ppl']:.4f} roundtrip_ppl={p1['ppl']:.4f} "
            f"delta_ppl_pct={p1['delta_ppl_pct']:+.2f}% "
            f"compression_ratio={p1['compression_ratio']:.2f}x",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
