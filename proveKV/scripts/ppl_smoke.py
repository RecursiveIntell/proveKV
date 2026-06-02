#!/usr/bin/env python3
"""
ppl_smoke.py — fast environment + model load test.

Goal: detect broken env / wrong model id / OOM / cudnn init issues in <60s
BEFORE committing to a 5-minute Phase 0 forward pass. If this fails,
don't bother with ppl_validate.py.

Checks:
1. torch.cuda.is_available() + GPU name
2. transformers version
3. SmolLM2-1.7B-Instruct tokenizer load
4. SmolLM2-1.7B-Instruct model load (fp16, on cuda)
5. torch.cuda.memory_allocated() > 0 after model load
   (catches the "receipt says GPU, code did CPU" failure mode)
6. 1-token forward on cuda (catches dtype/device mismatch)

Usage:
    python scripts/ppl_smoke.py
    python scripts/ppl_smoke.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct

Exit code 0 on success, 1 on any check failure.
"""
import argparse
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        help="HF model id (default: SmolLM2-1.7B-Instruct)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)",
    )
    args = parser.parse_args()

    t0 = time.time()

    # Check 1: torch
    print("[1/6] torch import + cuda check", flush=True)
    try:
        import torch
    except ImportError as e:
        print(f"  FAIL: torch import failed: {e}", file=sys.stderr)
        return 1
    print(f"  torch {torch.__version__}", flush=True)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("  FAIL: torch.cuda.is_available() is False", file=sys.stderr)
            print("  hint: check CUDA_VISIBLE_DEVICES, nvidia driver, or pass --device cpu", file=sys.stderr)
            return 1
        print(f"  cuda device: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  cuda capability: {torch.cuda.get_device_capability(0)}", flush=True)

    # Check 2: transformers
    print("[2/6] transformers version", flush=True)
    try:
        import transformers
    except ImportError as e:
        print(f"  FAIL: transformers import failed: {e}", file=sys.stderr)
        return 1
    print(f"  transformers {transformers.__version__}", flush=True)

    # Check 3: tokenizer
    print(f"[3/6] tokenizer load: {args.model}", flush=True)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"  FAIL: tokenizer load failed: {e}", file=sys.stderr)
        return 1
    print(f"  tokenizer ok: vocab={tokenizer.vocab_size}", flush=True)

    # Check 4: model
    print(f"[4/6] model load (fp16, {args.device}): {args.model}", flush=True)
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f"  FAIL: model load failed: {e}", file=sys.stderr)
        return 1
    model = model.to(args.device)
    model.eval()
    print(f"  model ok: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params", flush=True)

    # Check 5: GPU memory check
    print(f"[5/6] GPU memory check", flush=True)
    if args.device == "cuda":
        alloc_bytes = torch.cuda.memory_allocated()
        if alloc_bytes == 0:
            print(
                "  FAIL: torch.cuda.memory_allocated() is 0 — model is on CPU despite --device cuda",
                file=sys.stderr,
            )
            return 1
        print(f"  cuda memory allocated: {alloc_bytes / 1e9:.2f} GB", flush=True)

    # Check 6: 1-token forward
    print(f"[6/6] 1-token forward (sanity)", flush=True)
    try:
        with torch.no_grad():
            inputs = tokenizer("Hello", return_tensors="pt").to(args.device)
            outputs = model(**inputs)
        print(
            f"  forward ok: logits shape={tuple(outputs.logits.shape)} dtype={outputs.logits.dtype}",
            flush=True,
        )
        if args.device == "cuda":
            print(
                f"  cuda peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
                flush=True,
            )
    except Exception as e:
        print(f"  FAIL: 1-token forward failed: {e}", file=sys.stderr)
        return 1

    elapsed = time.time() - t0
    print(f"\nOK: smoke test passed in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
