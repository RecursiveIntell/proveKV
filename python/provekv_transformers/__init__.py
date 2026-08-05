"""proveKV Transformers adapter — pinned Qwen3.5 capture and replay."""

from .qwen35_adapter import (
    capture_prefix,
    component_shape,
    compute_blake3,
    load_pinned_model,
    tensor_to_raw_bytes,
)
from .versions import (
    QWEN35_MODEL_ID,
    QWEN35_REVISION,
    check_environment,
)

__all__ = [
    "QWEN35_MODEL_ID",
    "QWEN35_REVISION",
    "capture_prefix",
    "check_environment",
    "component_shape",
    "compute_blake3",
    "load_pinned_model",
    "tensor_to_raw_bytes",
]
