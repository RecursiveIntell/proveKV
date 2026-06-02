# state.json schema (v1.0.0)

This is the schema for `results/bench/ppl/<model_slug>/<corpus_slug>/state.json`
written by `ppl_validate.py`. It is the canonical receipt for a single
validation run.

```jsonc
{
  "schema_version": "1.0.0",
  "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",   // HF model id
  "model_slug": "smollm2-1.7b",                    // filesystem-safe slug
  "corpus": "wikitext-2",                          // HF dataset config
  "corpus_slug": "wikitext-2",                     // filesystem-safe slug
  "n_tokens": 1024,                                // context length fed to model
  "ppl_frac": 0.3,                                 // fraction of input used as PPL eval window
  "started_at": "2026-06-02T12:52:34.205895-05:00",// ISO-8601 with timezone

  "phase0": {                                      // ORACLE FORWARD PASS
    "status": "complete",
    "ppl": 4.760762087094494,                      // oracle perplexity (full precision)
    "ppl_window": [716, 1023],                     // token index range for PPL
    "cache_path": "bench/ppl/.../cache_oracle.pt", // raw fp16 K/V cache
    "cache_bytes": 201341281,                      // raw cache size on disk
    "model_config": {                              // frozen config snapshot
      "num_layers": 24,
      "num_heads": 32,
      "num_kv_heads": 32,
      "head_dim": 64,
      "hidden_size": 2048
    },
    "forward_seconds": 1.5647552013397217,         // single forward pass timing
    "completed_at": "2026-06-02T12:52:47.637473-05:00"
  },

  "model_config": { ... },                         // same as phase0.model_config
                                                   // (top-level for downstream tools)

  "phase1": {                                      // COMPRESSED ROUNDTRIP
    "status": "complete",
    "ppl": 4.760762087094494,                      // roundtrip PPL (compare to phase0.ppl)
    "ppl_window": [716, 1023],
    "roundtrip_bin": "bench/ppl/.../roundtrip.bin",// compressed cache artifact
    "roundtrip_bin_bytes": 1102338606,             // file size
    "manifest": {                                  // proveKV pool manifest
      "pool_id": "fcaece76b1...5f99",              // blake3 content-address
      "num_shared_tokens": 1024,
      "num_layers": 24,
      "num_kv_heads": 32,
      "head_dim": 64,
      "shared_codec": "\"fib_k4_n32\"",            // codec identity
      "compression_ratio": 11.130434782608695,     // 11.13x (vs fp32 raw)
      "pool_size_bytes": 36175872,                 // 36 MB actual pool
      "total_compressed_bytes": 36175872,
      "backend": "cpu",                            // decode backend
      "fib_build_ms": 2784,                        // build wall time
      "build_seed": 42,                            // deterministic seed
      "built_at_unix": 1780422894                  // unix timestamp
    },
    "compression_ratio": 11.130434782608695,       // duplicate of manifest (for grep)
    "pool_size_bytes": 36175872,                   // duplicate of manifest
    "total_compressed_bytes": 36175872,
    "delta_ppl_pct": 0.0,                          // (roundtrip - oracle) / oracle * 100
    "roundtrip_cli_seconds": 76.65067481994629,    // CLI wall time
    "forward_with_overwritten_cache_seconds": 0.02699565887451172, // second forward
    "completed_at": "2026-06-02T12:56:36.599920-05:00"
  },

  "report": {                                      // human-readable report summary
    "summary": "Oracle PPL 4.7608 | Roundtrip PPL 4.7608 | ... | compression_ratio 11.13x | ...",
    "per_layer": [                                 // per-layer byte accounting
      {
        "layer": 0,
        "oracle_bytes": 8388608,                   // 8 MB raw fp16 K+V at 1024 tokens
        "roundtrip_layer_bytes": 49147904         // decompressed layer (incl. JSON serialization overhead)
      },
      // ... 24 layers total
    ],
    "completed_at": "2026-06-02T12:56:38.127977-05:00"
  }
}
```

## Invariants

The following invariants are checked at the end of each run and any
violation aborts the run with a non-zero exit code:

1. `phase0.ppl > 1.0` (sanity: a working model forward pass)
2. `phase0.cache_bytes == num_layers * num_kv_heads * n_tokens * head_dim * 2 * 2` (raw cache size = K+V)
3. `phase1.ppl > 0` (sanity)
4. `phase1.manifest.compression_ratio > 1.0` (a negative compression ratio
   indicates a wire-format regression and is a hard failure)
5. `phase1.manifest.pool_size_bytes > 0`
6. `len(phase1.report.per_layer) == phase0.model_config.num_layers`

## Determinism

Re-running with the same `(model, corpus, n_tokens, ppl_frac, seed)`
yields the same `phase0.ppl`, `phase1.ppl`, `phase1.manifest.pool_id`,
and `phase1.manifest.pool_size_bytes` to the precision recorded. The
`phase0.ppl` is deterministic at fp16 because the model weights and
input are fixed. The `phase1.ppl` is deterministic because the
`build_seed = 42` is fixed.

Wall times (`forward_seconds`, `roundtrip_cli_seconds`, `forward_with_overwritten_cache_seconds`)
vary by host.
