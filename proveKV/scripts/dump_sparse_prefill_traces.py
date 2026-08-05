#!/usr/bin/env python3
"""
dump_sparse_prefill_traces.py — export real attention traces for sparse-prefill gates.

The Rust sparse-prefill benchmark expects a JSON array of SparsePrefillTraceV1:

[
  {
    "trace_id": "layer0_head0_q127",
    "layer": 0,
    "head": 0,
    "scores": [...]
  }
]

Transformers exposes attention probabilities, not raw pre-softmax logits, so
this script stores log(probability + eps). The sparse-prefill gate softmaxes the
stored scores, which reconstructs the original attention distribution closely
enough for token/block selection receipts.

Usage:
    python proveKV/scripts/dump_sparse_prefill_traces.py \
      --model HuggingFaceTB/SmolLM2-1.7B-Instruct \
      --device cuda \
      --n-tokens 256 \
      --output /tmp/sparse_prefill_traces.json

Then run:
    cargo run -p turbo-quant --example sparse_prefill_trace_bench -- \
      /tmp/sparse_prefill_traces.json
"""

import argparse
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        help="HF causal LM model id",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Execution device",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Sparse attention kernels should be validated with real model "
            "attention distributions before CUDA work begins. "
        ),
        help="Prompt text. Repeated until --n-tokens is reached.",
    )
    parser.add_argument("--text-file", type=Path, help="Optional text file prompt source")
    parser.add_argument("--n-tokens", type=int, default=256)
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--max-heads", type=int, default=8)
    parser.add_argument(
        "--query-count",
        type=int,
        default=8,
        help="Number of late query positions to sample",
    )
    parser.add_argument(
        "--query-start-frac",
        type=float,
        default=0.5,
        help="Start sampling query positions after this fraction of the sequence",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Probability floor before log()",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def dtype_from_name(torch, name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def load_model(transformers, model_id: str, torch_dtype):
    model_cls = transformers.AutoModelForCausalLM
    kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        # output_attentions with SDPA/flash implementations often returns
        # nothing or rejects attention capture. Eager is slower but explicit.
        "attn_implementation": "eager",
    }
    try:
        return model_cls.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        return model_cls.from_pretrained(model_id, **kwargs)


def repeated_text(args: argparse.Namespace) -> str:
    if args.text_file:
        text = args.text_file.read_text()
    else:
        text = args.prompt
    if not text.strip():
        raise ValueError("prompt/text-file is empty")
    return (text + "\n") * max(1, math.ceil(args.n_tokens / 16))


def query_positions(seq_len: int, start_frac: float, count: int) -> list[int]:
    if seq_len <= 1 or count <= 0:
        return []
    start = max(1, min(seq_len - 1, int(seq_len * start_frac)))
    if count == 1:
        return [seq_len - 1]
    span = seq_len - 1 - start
    if span <= 0:
        return [seq_len - 1]
    positions = {
        start + round((span * idx) / (count - 1))
        for idx in range(count)
    }
    positions.add(seq_len - 1)
    return sorted(pos for pos in positions if 0 <= pos < seq_len)


def main() -> int:
    args = parse_args()
    if args.n_tokens <= 1:
        print("--n-tokens must be > 1", file=sys.stderr)
        return 2
    if not (0.0 <= args.query_start_frac < 1.0):
        print("--query-start-frac must be in [0.0, 1.0)", file=sys.stderr)
        return 2

    try:
        import torch
    except ImportError as error:
        print(f"torch import failed: {error}", file=sys.stderr)
        print("install/use the same Python environment as the PPL scripts, then rerun this dumper", file=sys.stderr)
        return 1
    try:
        import transformers
    except ImportError as error:
        print(f"transformers import failed: {error}", file=sys.stderr)
        print("install/use the same Python environment as the PPL scripts, then rerun this dumper", file=sys.stderr)
        return 1

    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but torch.cuda.is_available() is false", file=sys.stderr)
        return 1

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = load_model(transformers, args.model, dtype_from_name(torch, args.dtype))
    model.to(args.device)
    model.eval()

    text = repeated_text(args)
    encoded = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encoded.input_ids[:, : args.n_tokens]
    if input_ids.shape[1] < args.n_tokens:
        print(
            f"warning: tokenizer produced only {input_ids.shape[1]} tokens; dumping available tokens",
            file=sys.stderr,
        )
    input_ids = input_ids.to(args.device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )

    attentions = outputs.attentions
    if not attentions:
        print("model returned no attentions; try a model/config that supports output_attentions", file=sys.stderr)
        return 1

    seq_len = input_ids.shape[1]
    positions = query_positions(seq_len, args.query_start_frac, args.query_count)
    traces = []
    for layer_idx, layer_attn in enumerate(attentions[: args.max_layers]):
        # Expected shape: [batch, heads, query_tokens, key_tokens].
        if layer_attn.ndim != 4:
            print(f"skipping layer {layer_idx}: unexpected attention shape {tuple(layer_attn.shape)}", file=sys.stderr)
            continue
        head_count = min(args.max_heads, layer_attn.shape[1])
        for head_idx in range(head_count):
            for q_pos in positions:
                probs = layer_attn[0, head_idx, q_pos, : q_pos + 1]
                scores = torch.log(probs.float().clamp_min(args.eps)).cpu().tolist()
                traces.append(
                    {
                        "trace_id": f"layer{layer_idx}_head{head_idx}_q{q_pos}",
                        "layer": layer_idx,
                        "head": head_idx,
                        "scores": scores,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(traces, indent=2))
    tmp.replace(args.output)
    print(
        json.dumps(
            {
                "schema": "SparsePrefillTraceDumpV1",
                "model": args.model,
                "device": args.device,
                "dtype": args.dtype,
                "seq_len": seq_len,
                "trace_count": len(traces),
                "output": str(args.output),
                "query_positions": positions,
                "warning": "scores are log(attention_probability + eps), not raw pre-softmax logits",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
