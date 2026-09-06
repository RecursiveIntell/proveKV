#!/usr/bin/env python3
"""Focused unit tests for the benchmark provenance helpers."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "proveKV/scripts/ppl_validate_multi_agent.py"


torch = types.ModuleType("torch")
setattr(torch, "Tensor", object)
sys.modules.setdefault("torch", torch)
transformers = types.ModuleType("transformers")
setattr(transformers, "AutoModelForCausalLM", object)
setattr(transformers, "AutoTokenizer", object)
sys.modules.setdefault("transformers", transformers)
datasets = types.ModuleType("datasets")
setattr(datasets, "load_dataset", None)
sys.modules.setdefault("datasets", datasets)
spec = importlib.util.spec_from_file_location("ppl_validate_multi_agent_contract", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load benchmark module")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class PplWindowContract(unittest.TestCase):
    def test_target_window_maps_to_shifted_logit_slice(self):
        self.assertEqual(benchmark.ppl_slice_bounds(800, 1024, 1024), (799, 1023))
        self.assertEqual(1023 - 799, 224)
        self.assertEqual(
            benchmark.roundtrip_context_bounds(800, 1024, 1024),
            (799, 799, 1023),
        )
        cache_end, input_start, input_end = benchmark.roundtrip_context_bounds(
            800, 1024, 1024
        )
        self.assertLess(cache_end, 1024)
        self.assertEqual(input_end - input_start, 224)

    def test_target_window_rejects_unscorable_or_out_of_range_bounds(self):
        for bounds in ((0, 10, 10), (10, 10, 10), (9, 11, 10)):
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):
                    benchmark.ppl_slice_bounds(*bounds)


class CacheIdentityContract(unittest.TestCase):
    def identity(self, **overrides):
        values = {
            "model_name": "model",
            "model_revision": "a" * 40,
            "tokenizer_name": "tokenizer",
            "tokenizer_revision": None,
            "corpus": "wikitext-2",
            "n_shared": 800,
            "n_unique": 28,
            "n_agents": 8,
            "n_total": 1024,
            "seed": 42,
            "model_config": {"num_layers": 24, "num_kv_heads": 32, "head_dim": 64},
            "ppl_window": [800, 1024],
            "tokens": list(range(1024)),
            "benchmark_script_sha256": "b" * 64,
        }
        values.update(overrides)
        return benchmark.cache_identity(**values)

    def test_identity_is_deterministic_and_bound_to_tokens_and_shape(self):
        one = self.identity()
        self.assertEqual(one, self.identity())
        self.assertNotEqual(one["identity_sha256"], self.identity(seed=43)["identity_sha256"])
        changed_tokens = list(range(1024))
        changed_tokens[-1] = 9999
        self.assertNotEqual(
            one["identity_sha256"],
            self.identity(tokens=changed_tokens)["identity_sha256"],
        )


class ShellReceiptContract(unittest.TestCase):
    def write_fixture(self, root: Path):
        pool = {
            "pool_id": "pool",
            "num_shared_tokens": 800,
            "num_layers": 24,
            "num_kv_heads": 32,
            "head_dim": 64,
            "shared_codec": "fib_k4_n32_batched",
            "compression_ratio": 21.33,
            "pool_size_bytes": 14_746_512,
            "build_seed": 42,
            "built_at_unix": 1,
        }
        per_agent = [
            {
                "agent_id": f"agent-{index}",
                "num_unique_tokens": 28,
                "shell_bytes": 6_194_976,
                "shell_codec": "turbo_4bit_batched",
            }
            for index in range(8)
        ]
        agents = {
            "n_agents": 8,
            "per_agent": per_agent,
            "total_shell_bytes": 8 * 6_194_976,
            "avg_shell_bytes": 6_194_976.0,
        }
        compressed = pool["pool_size_bytes"] + agents["total_shell_bytes"]
        state = {
            "model": "KV shape (24 layers, 32 kv heads, head_dim 64)",
            "n_agents": 8,
            "shared_pool_bytes": pool["pool_size_bytes"],
            "total_shell_bytes": agents["total_shell_bytes"],
            "total_with_sharing_bytes": compressed,
            "naive_total_bytes": 2_604_662_784,
            "memory_reduction_factor": 2_604_662_784 / compressed,
            "shell_codec": "turbo_4bit_batched",
            "shared_codec": "fib_k4_n32_batched",
            "radii_compression": "Lossless",
            "lossy": False,
        }
        (root / "shared_pool_receipt.json").write_text(json.dumps(pool))
        (root / "agents_receipt.json").write_text(json.dumps(agents))
        (root / "state.json").write_text(json.dumps(state))

    def test_receipts_recompute_the_independent_agent_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(root)
            facts = benchmark.validate_shell_receipts(
                root,
                n_shared=800,
                n_unique=28,
                n_agents=8,
                model_config={"num_layers": 24, "num_kv_heads": 32, "head_dim": 64},
                seed=42,
                lossy=False,
            )
            self.assertEqual(facts["raw_total_bytes"], 2_604_662_784)
            self.assertEqual(facts["compressed_total_bytes"], 64_306_320)

    def test_receipt_total_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_fixture(root)
            path = root / "agents_receipt.json"
            agents = json.loads(path.read_text())
            agents["total_shell_bytes"] += 1
            path.write_text(json.dumps(agents))
            with self.assertRaises(SystemExit):
                benchmark.validate_shell_receipts(
                    root,
                    n_shared=800,
                    n_unique=28,
                    n_agents=8,
                    model_config={"num_layers": 24, "num_kv_heads": 32, "head_dim": 64},
                    seed=42,
                    lossy=False,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
