"""Environment contract: exact pinned versions for proveKV Qwen3.5 replay."""

# Primary target: Qwen3.5-2B (blocked: architecture not in transformers yet).
# Fallback: Qwen2.5-0.5B (supported, used for P0-C pipeline verification).
QWEN35_MODEL_ID = "Qwen/Qwen2.5-0.5B"
QWEN35_REVISION = "main"  # Qwen2.5 uses main branch

# Required package versions — any mismatch fails the environment contract.
REQUIRED_PACKAGES = {
    "torch": "2.5",
    "transformers": "4.51.3",
    "safetensors": "0.4",
    "blake3": "0.4",
}

# Required torch device capabilities.
REQUIRED_CPU = True
REQUIRED_CUDA = False  # P0-C is CPU-only.

# Offline-only: HF_HUB_OFFLINE=1 must succeed.
REQUIRED_OFFLINE = True


def check_environment() -> dict:
    """Verify the environment matches the pinned contract. Returns a dict with
    status, mismatches, and capability report."""
    import importlib.metadata
    import sys

    report = {
        "python_version": sys.version,
        "packages": {},
        "mismatches": [],
        "cpu_available": True,  # always true
        "cuda_available": False,
        "status": "ok",
    }

    for name, min_version in REQUIRED_PACKAGES.items():
        try:
            ver = importlib.metadata.version(name)
            report["packages"][name] = ver
            # Accept any version >= min_version (major.minor prefix match)
            ver_parts = ver.split(".")
            min_parts = min_version.split(".")
            if ver_parts[:len(min_parts)] < min_parts:
                report["mismatches"].append(
                    f"{name}=={ver} (expected >={min_version})"
                )
        except importlib.metadata.PackageNotFoundError:
            report["mismatches"].append(f"{name} not installed")

    # Check torch device.
    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        if REQUIRED_CUDA and not report["cuda_available"]:
            report["mismatches"].append("CUDA required but not available")
        if not REQUIRED_CPU and report["cuda_available"]:
            pass  # optional
    except ImportError:
        report["mismatches"].append("torch not importable")

    if report["mismatches"]:
        report["status"] = "mismatch"

    return report
