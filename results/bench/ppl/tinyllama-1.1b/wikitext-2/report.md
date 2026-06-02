# PPL Validation Report — TinyLlama/TinyLlama-1.1B-Chat-v1.0 on wikitext-2

- **Generated:** 2026-06-02T14:14:12.557257-05:00
- **Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Corpus:** `wikitext-2` (n_tokens=1024, ppl_frac=0.3)
- **Seed:** 42

## Headline

- **Oracle PPL:** 2.7018
- **Roundtrip PPL:** 2.7018
- **Δ PPL:** +0.00%
- **Compression ratio:** 11.13x
- **Pool size:** 4,145,152 bytes
- **Total compressed:** 4,145,152 bytes

## Methodology

1. **Phase 0 (oracle):** use_cache=True forward pass over first n_tokens of WikiText-2 test split. fp16, cuda. PPL computed over last 30% of tokens (HF causal-LM recipe: shift, logsumexp, gather, exp(mean)).
2. **Phase 1 (compressed roundtrip):**
   - Extract per-token K/V vectors from the DynamicCache
   - Build poly-kv corpus JSON
   - Invoke `poly_kv_fast_roundtrip` CLI (composite build + parallel decompress) (9.1s)
   - Read decompressed layers from the roundtrip.bin output
   - Pre-populate a fresh `DynamicCache` and forward with it
   - PPL over the same window as Phase 0
3. **Phase 2 (report):** per-layer byte accounting; this file.

## Per-layer accounting

| Layer | Oracle bytes (fp16 KV) | Roundtrip layer bytes (JSON+len) |
|------:|-----------------------:|---------------------------------:|
| 0 | 1,048,576 | 6,346,543 |
| 1 | 1,048,576 | 6,087,772 |
| 2 | 1,048,576 | 6,036,549 |
| 3 | 1,048,576 | 5,917,407 |
| 4 | 1,048,576 | 5,892,223 |
| 5 | 1,048,576 | 5,880,628 |
| 6 | 1,048,576 | 5,881,450 |
| 7 | 1,048,576 | 5,876,076 |
| 8 | 1,048,576 | 5,853,206 |
| 9 | 1,048,576 | 5,861,598 |
| 10 | 1,048,576 | 5,861,657 |
| 11 | 1,048,576 | 5,837,816 |
| 12 | 1,048,576 | 5,848,605 |
| 13 | 1,048,576 | 5,836,671 |
| 14 | 1,048,576 | 5,829,178 |
| 15 | 1,048,576 | 5,822,905 |
| 16 | 1,048,576 | 5,823,823 |
| 17 | 1,048,576 | 5,807,912 |
| 18 | 1,048,576 | 5,792,746 |
| 19 | 1,048,576 | 5,775,340 |
| 20 | 1,048,576 | 5,777,550 |
| 21 | 1,048,576 | 5,764,678 |

## Receipts

- `state.json` — full machine-readable state
- `cache_oracle.pt` — Phase 0 DynamicCache (fp16 K/V tensors)
- `poly_kv_corpus.json` — Phase 1 input to the poly-kv CLI
- `roundtrip.bin` — Phase 1 binary output (manifest + 24 layer blobs)
- `manifest` (in `roundtrip.bin`) — pool manifest from poly-kv

## Caveats

- The fib-quant decoder is single-block per call; for n_tokens=1024 × 24 layers the decode work is ~24M codeword lookups, which serial-decoded in Rust takes >30 min. For this initial validation, n_tokens can be reduced to keep roundtrip time under 5 min; for a public release the codec needs a vectorized batched decode implementation.
- transformers 5.1.0, torch 2.10.0+cu126, device cuda
- Model config: num_layers=22 num_heads=32 num_kv_heads=4 head_dim=64 hidden_size=2048
- Phase 0 forward: 0.9s
- Phase 1 roundtrip CLI: 9.1s
- Phase 1 forward with pre-populated cache: 0.2s
