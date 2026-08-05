"""Qwen3.5-specific adapter: capture KV cache state into proveKV binary pages."""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .versions import QWEN35_MODEL_ID, QWEN35_REVISION


def load_pinned_model(device: str = "cpu", dtype: str = "float32"):
    """Load the pinned Qwen3.5-2B with exact revision."""
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]

    config = AutoConfig.from_pretrained(
        QWEN35_MODEL_ID,
        revision=QWEN35_REVISION,
        trust_remote_code=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        QWEN35_MODEL_ID,
        revision=QWEN35_REVISION,
        torch_dtype=torch_dtype,
        device_map=None if device == "cpu" else "auto",
        trust_remote_code=False,
    )
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        QWEN35_MODEL_ID,
        revision=QWEN35_REVISION,
        trust_remote_code=False,
    )

    return model, tokenizer, config


def capture_prefix(
    model,
    tokenizer,
    text: str,
    device: str = "cpu",
    max_tokens: int = 64,
) -> dict:
    """Run a forward pass and capture all KV cache tensors plus metadata.

    Returns a dict with:
      - token_ids: list[int]
      - layers: dict mapping layer index to {"k": tensor, "v": tensor}
      - metadata: model config digest, tokenizer digest, position info
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=False,
        )

    past_key_values = outputs.past_key_values
    layers = {}

    # Handle both legacy tuple format and DynamicCache.
    if hasattr(past_key_values, 'key_cache'):
        # DynamicCache (transformers >= 5.x)
        for layer_idx in range(len(past_key_values.key_cache)):
            k = past_key_values.key_cache[layer_idx]
            v = past_key_values.value_cache[layer_idx]
            if k is not None and v is not None:
                layers[layer_idx] = {
                    "k": k.cpu().clone(),
                    "v": v.cpu().clone(),
                }
    else:
        # Legacy tuple of (k, v) pairs.
        for layer_idx, (k, v) in enumerate(past_key_values):
            layers[layer_idx] = {
                "k": k.cpu().clone(),
                "v": v.cpu().clone(),
            }

    # Compute model config digest.
    config_json = json.dumps(model.config.to_dict(), sort_keys=True)
    config_digest = hashlib.sha256(config_json.encode()).hexdigest()

    # Weight digest from safetensors index if available.
    weight_digest = "unavailable"  # computed lazily during full capture

    return {
        "model_id": QWEN35_MODEL_ID,
        "revision": QWEN35_REVISION,
        "config_digest": f"sha256:{config_digest}",
        "weight_digest": weight_digest,
        "token_ids": input_ids[0].cpu().tolist(),
        "attention_mask": attention_mask[0].cpu().tolist(),
        "num_layers": len(layers),
        "num_tokens": input_ids.shape[1],
        "dtype": str(model.dtype),
        "device": device,
        "layers": layers,
    }


def tensor_to_raw_bytes(tensor: torch.Tensor) -> bytes:
    """Convert a float32 CPU tensor to contiguous little-endian raw bytes."""
    return tensor.contiguous().cpu().to(torch.float32).numpy().tobytes()


def component_shape(tensor: torch.Tensor) -> list[int]:
    """Return the shape as a list of ints."""
    return list(tensor.shape)


def compute_blake3(data: bytes) -> str:
    """Compute BLAKE3 digest of bytes."""
    import blake3  # local import so adapter works without blake3 for structure checks

    return blake3.blake3(data).hexdigest()
