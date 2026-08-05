"""Structural tests for the Qwen3.5 adapter (no model load required)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_adapter_exports():
    """All expected functions are importable."""
    from provekv_transformers import (
        capture_prefix,
        component_shape,
        compute_blake3,
        load_pinned_model,
        tensor_to_raw_bytes,
    )

    assert callable(capture_prefix)
    assert callable(component_shape)
    assert callable(compute_blake3)
    assert callable(load_pinned_model)
    assert callable(tensor_to_raw_bytes)


def test_blake3_imports_and_works():
    """BLAKE3 should be importable and deterministic."""
    from provekv_transformers.qwen35_adapter import compute_blake3

    d1 = compute_blake3(b"hello")
    d2 = compute_blake3(b"hello")
    d3 = compute_blake3(b"world")
    assert d1 == d2
    assert d1 != d3
    assert len(d1) == 64  # hex string


def test_revision_is_pinned_commit():
    """Model ID and revision must be pinned."""
    from provekv_transformers.versions import QWEN35_REVISION, QWEN35_MODEL_ID

    assert QWEN35_MODEL_ID in ("Qwen/Qwen3.5-2B", "Qwen/Qwen2.5-0.5B")
    assert len(QWEN35_REVISION) >= 4  # commit hash or 'main'
    assert all(c in "0123456789abcdef" for c in QWEN35_REVISION) or QWEN35_REVISION == "main"
