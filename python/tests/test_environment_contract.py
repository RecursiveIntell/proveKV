"""Tests for the proveKV Transformers environment contract."""
import sys
from pathlib import Path

# Allow running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from provekv_transformers.versions import check_environment, REQUIRED_PACKAGES


def test_environment_contract():
    """The environment must match the pinned package versions."""
    report = check_environment()
    assert report["status"] == "ok", (
        f"Environment mismatch: {report['mismatches']}"
    )
    assert report["cpu_available"], "CPU must be available"
    assert not report.get("cuda_available"), (
        "CUDA should not be required for P0-C CPU-only slice"
    )


def test_required_packages_listed():
    """Every required package must be listed."""
    assert "torch" in REQUIRED_PACKAGES
    assert "transformers" in REQUIRED_PACKAGES
    assert "safetensors" in REQUIRED_PACKAGES
    assert "blake3" in REQUIRED_PACKAGES


def test_model_revision_is_pinned():
    """The model revision must be an immutable commit hash."""
    from provekv_transformers.versions import QWEN35_REVISION

    # Must be a 40-char hex string.
    assert len(QWEN35_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in QWEN35_REVISION)
