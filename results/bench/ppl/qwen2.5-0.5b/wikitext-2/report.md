# PPL Validation Report — Qwen/Qwen2.5-0.5B-Instruct on wikitext-2

- **Generated:** 2026-06-02T14:37:44.157463-05:00
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Corpus:** `wikitext-2` (n_tokens=1024, ppl_frac=0.3)
- **Seed:** 42

## Headline

- **Oracle PPL:** 7.6123
- **Roundtrip PPL:** 7.6123
- **Δ PPL:** +0.00%
- **Compression ratio:** 11.13x
- **Pool size:** 2,260,992 bytes
- **Total compressed:** 2,260,992 bytes

## Methodology

1. **Phase 0 (oracle):** use_cache=True forward pass over first n_tokens of WikiText-2 test split. fp16, cuda. PPL computed over last 30% of tokens (HF causal-LM recipe: shift, logsumexp, gather, exp(mean)).
2. **Phase 1 (compressed roundtrip):**
   - Extract per-token K/V vectors from the DynamicCache
   - Build poly-kv corpus JSON
   - Invoke `poly_kv_fast_roundtrip` CLI (composite build + parallel decompress) (5.0s)
   - Read decompressed layers from the roundtrip.bin output
   - Pre-populate a fresh `DynamicCache` and forward with it
   - PPL over the same window as Phase 0
3. **Phase 2 (report):** per-layer byte accounting; this file.

## Per-layer accounting

| Layer | Oracle bytes (fp16 KV) | Roundtrip layer bytes (JSON+len) |
|------:|-----------------------:|---------------------------------:|
| 0 | 524,288 | 2,981,705 |
| 1 | 524,288 | 2,887,142 |
| 2 | 524,288 | 2,860,324 |
| 3 | 524,288 | 2,862,164 |
| 4 | 524,288 | 2,874,263 |
| 5 | 524,288 | 2,875,266 |
| 6 | 524,288 | 2,888,611 |
| 7 | 524,288 | 2,869,059 |
| 8 | 524,288 | 2,851,294 |
| 9 | 524,288 | 2,862,913 |
| 10 | 524,288 | 2,865,444 |
| 11 | 524,288 | 2,848,732 |
| 12 | 524,288 | 2,865,038 |
| 13 | 524,288 | 2,852,703 |
| 14 | 524,288 | 2,861,104 |
| 15 | 524,288 | 2,855,962 |
| 16 | 524,288 | 2,838,580 |
| 17 | 524,288 | 2,865,706 |
| 18 | 524,288 | 2,862,120 |
| 19 | 524,288 | 2,848,631 |
| 20 | 524,288 | 2,829,024 |
| 21 | 524,288 | 2,809,709 |
| 22 | 524,288 | 2,825,632 |
| 23 | 524,288 | 2,813,947 |

## Receipts

- `state.json` — full machine-readable state
- `cache_oracle.pt` — Phase 0 DynamicCache (fp16 K/V tensors)
- `poly_kv_corpus.json` — Phase 1 input to the poly-kv CLI
- `roundtrip.bin` — Phase 1 binary output (manifest + 24 layer blobs)
- `manifest` (in `roundtrip.bin`) — pool manifest from poly-kv

## Caveats

- The fib-quant decoder is single-block per call; for n_tokens=1024 × 24 layers the decode work is ~24M codeword lookups, which serial-decoded in Rust takes >30 min. For this initial validation, n_tokens can be reduced to keep roundtrip time under 5 min; for a public release the codec needs a vectorized batched decode implementation.
- transformers 5.1.0, torch 2.10.0+cu126, device cuda
- Model config: num_layers=24 num_heads=14 num_kv_heads=2 head_dim=64 hidden_size=896
- Phase 0 forward: 0.3s
- Phase 1 roundtrip CLI: 5.0s
- Phase 1 forward with pre-populated cache: 0.1s
