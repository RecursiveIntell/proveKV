# PPL Validation Report — HuggingFaceTB/SmolLM2-1.7B-Instruct on wikitext-2

- **Generated:** 2026-06-02T12:56:38.128655-05:00
- **Model:** `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **Corpus:** `wikitext-2` (n_tokens=1024, ppl_frac=0.3)
- **Seed:** 42

## Headline

- **Oracle PPL:** 4.7608
- **Roundtrip PPL:** 4.7608
- **Δ PPL:** +0.00%
- **Compression ratio:** 11.13x
- **Pool size:** 36,175,872 bytes
- **Total compressed:** 36,175,872 bytes

## Methodology

1. **Phase 0 (oracle):** use_cache=True forward pass over first n_tokens of WikiText-2 test split. fp16, cuda. PPL computed over last 30% of tokens (HF causal-LM recipe: shift, logsumexp, gather, exp(mean)).
2. **Phase 1 (compressed roundtrip):**
   - Extract per-token K/V vectors from the DynamicCache
   - Build poly-kv corpus JSON
   - Invoke `poly_kv_fast_roundtrip` CLI (composite build + parallel decompress) (76.7s)
   - Read decompressed layers from the roundtrip.bin output
   - Pre-populate a fresh `DynamicCache` and forward with it
   - PPL over the same window as Phase 0
3. **Phase 2 (report):** per-layer byte accounting; this file.

## Per-layer accounting

| Layer | Oracle bytes (fp16 KV) | Roundtrip layer bytes (JSON+len) |
|------:|-----------------------:|---------------------------------:|
| 0 | 8,388,608 | 49,147,904 |
| 1 | 8,388,608 | 46,821,189 |
| 2 | 8,388,608 | 46,417,491 |
| 3 | 8,388,608 | 46,275,148 |
| 4 | 8,388,608 | 46,197,928 |
| 5 | 8,388,608 | 46,155,023 |
| 6 | 8,388,608 | 46,131,034 |
| 7 | 8,388,608 | 46,024,014 |
| 8 | 8,388,608 | 45,960,281 |
| 9 | 8,388,608 | 46,049,700 |
| 10 | 8,388,608 | 45,900,074 |
| 11 | 8,388,608 | 45,885,443 |
| 12 | 8,388,608 | 45,761,729 |
| 13 | 8,388,608 | 45,850,022 |
| 14 | 8,388,608 | 45,847,096 |
| 15 | 8,388,608 | 45,653,514 |
| 16 | 8,388,608 | 45,649,504 |
| 17 | 8,388,608 | 45,500,313 |
| 18 | 8,388,608 | 45,430,397 |
| 19 | 8,388,608 | 45,288,311 |
| 20 | 8,388,608 | 45,221,160 |
| 21 | 8,388,608 | 45,108,525 |
| 22 | 8,388,608 | 45,100,198 |
| 23 | 8,388,608 | 44,962,238 |

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
- Phase 1 roundtrip CLI: 76.7s
- Phase 1 forward with pre-populated cache: 0.0s
