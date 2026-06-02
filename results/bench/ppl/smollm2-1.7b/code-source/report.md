# PPL Validation Report — HuggingFaceTB/SmolLM2-1.7B-Instruct on file:/tmp/code_corpus.txt

- **Generated:** 2026-06-02T14:19:51.760367-05:00
- **Model:** `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **Corpus:** `file:/tmp/code_corpus.txt` (n_tokens=1024, ppl_frac=0.3)
- **Seed:** 42

## Headline

- **Oracle PPL:** 5.1379
- **Roundtrip PPL:** 4.7608
- **Δ PPL:** -7.34%
- **Compression ratio:** 11.13x
- **Pool size:** 36,175,872 bytes
- **Total compressed:** 36,175,872 bytes

## Methodology

1. **Phase 0 (oracle):** use_cache=True forward pass over first n_tokens of WikiText-2 test split. fp16, cuda. PPL computed over last 30% of tokens (HF causal-LM recipe: shift, logsumexp, gather, exp(mean)).
2. **Phase 1 (compressed roundtrip):**
   - Extract per-token K/V vectors from the DynamicCache
   - Build poly-kv corpus JSON
   - Invoke `poly_kv_fast_roundtrip` CLI (composite build + parallel decompress) (90.5s)
   - Read decompressed layers from the roundtrip.bin output
   - Pre-populate a fresh `DynamicCache` and forward with it
   - PPL over the same window as Phase 0
3. **Phase 2 (report):** per-layer byte accounting; this file.

## Per-layer accounting

| Layer | Oracle bytes (fp16 KV) | Roundtrip layer bytes (JSON+len) |
|------:|-----------------------:|---------------------------------:|
| 0 | 8,388,608 | 49,080,288 |
| 1 | 8,388,608 | 46,915,017 |
| 2 | 8,388,608 | 46,425,393 |
| 3 | 8,388,608 | 46,310,191 |
| 4 | 8,388,608 | 46,248,046 |
| 5 | 8,388,608 | 46,144,610 |
| 6 | 8,388,608 | 46,127,991 |
| 7 | 8,388,608 | 46,044,474 |
| 8 | 8,388,608 | 45,980,655 |
| 9 | 8,388,608 | 46,058,346 |
| 10 | 8,388,608 | 45,902,842 |
| 11 | 8,388,608 | 45,869,026 |
| 12 | 8,388,608 | 45,763,461 |
| 13 | 8,388,608 | 45,832,107 |
| 14 | 8,388,608 | 45,830,081 |
| 15 | 8,388,608 | 45,607,062 |
| 16 | 8,388,608 | 45,624,840 |
| 17 | 8,388,608 | 45,469,663 |
| 18 | 8,388,608 | 45,429,970 |
| 19 | 8,388,608 | 45,320,548 |
| 20 | 8,388,608 | 45,280,991 |
| 21 | 8,388,608 | 45,196,974 |
| 22 | 8,388,608 | 45,203,001 |
| 23 | 8,388,608 | 45,040,633 |

## Receipts

- `state.json` — full machine-readable state
- `cache_oracle.pt` — Phase 0 DynamicCache (fp16 K/V tensors)
- `poly_kv_corpus.json` — Phase 1 input to the poly-kv CLI
- `roundtrip.bin` — Phase 1 binary output (manifest + 24 layer blobs)
- `manifest` (in `roundtrip.bin`) — pool manifest from poly-kv

## Caveats

- The fib-quant decoder is single-block per call; for n_tokens=1024 × 24 layers the decode work is ~24M codeword lookups, which serial-decoded in Rust takes >30 min. For this initial validation, n_tokens can be reduced to keep roundtrip time under 5 min; for a public release the codec needs a vectorized batched decode implementation.
- transformers 5.1.0, torch 2.10.0+cu126, device cuda
- Model config: num_layers=24 num_heads=32 num_kv_heads=32 head_dim=64 hidden_size=2048
- Phase 0 forward: 1.6s
- Phase 1 roundtrip CLI: 90.5s
- Phase 1 forward with pre-populated cache: 0.1s
