# PPL Validation Report — HuggingFaceTB/SmolLM2-1.7B-Instruct on wikitext-2

- **Generated:** 2026-06-02T14:28:42.516693-05:00
- **Model:** `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **Corpus:** `wikitext-2` (n_tokens=1280, ppl_frac=0.3)
- **Seed:** 42

## Headline

- **Oracle PPL:** 4.8249
- **Roundtrip PPL:** 4.8249
- **Δ PPL:** +0.00%
- **Compression ratio:** 11.13x
- **Pool size:** 45,219,840 bytes
- **Total compressed:** 45,219,840 bytes

## Methodology

1. **Phase 0 (oracle):** use_cache=True forward pass over first n_tokens of WikiText-2 test split. fp16, cuda. PPL computed over last 30% of tokens (HF causal-LM recipe: shift, logsumexp, gather, exp(mean)).
2. **Phase 1 (compressed roundtrip):**
   - Extract per-token K/V vectors from the DynamicCache
   - Build poly-kv corpus JSON
   - Invoke `poly_kv_fast_roundtrip` CLI (composite build + parallel decompress) (97.6s)
   - Read decompressed layers from the roundtrip.bin output
   - Pre-populate a fresh `DynamicCache` and forward with it
   - PPL over the same window as Phase 0
3. **Phase 2 (report):** per-layer byte accounting; this file.

## Per-layer accounting

| Layer | Oracle bytes (fp16 KV) | Roundtrip layer bytes (JSON+len) |
|------:|-----------------------:|---------------------------------:|
| 0 | 10,485,760 | 61,435,016 |
| 1 | 10,485,760 | 58,525,857 |
| 2 | 10,485,760 | 58,005,970 |
| 3 | 10,485,760 | 57,828,036 |
| 4 | 10,485,760 | 57,731,235 |
| 5 | 10,485,760 | 57,679,457 |
| 6 | 10,485,760 | 57,649,137 |
| 7 | 10,485,760 | 57,516,779 |
| 8 | 10,485,760 | 57,439,835 |
| 9 | 10,485,760 | 57,558,834 |
| 10 | 10,485,760 | 57,368,739 |
| 11 | 10,485,760 | 57,345,681 |
| 12 | 10,485,760 | 57,189,615 |
| 13 | 10,485,760 | 57,305,575 |
| 14 | 10,485,760 | 57,294,914 |
| 15 | 10,485,760 | 57,047,483 |
| 16 | 10,485,760 | 57,047,729 |
| 17 | 10,485,760 | 56,862,607 |
| 18 | 10,485,760 | 56,777,565 |
| 19 | 10,485,760 | 56,608,423 |
| 20 | 10,485,760 | 56,521,362 |
| 21 | 10,485,760 | 56,384,247 |
| 22 | 10,485,760 | 56,378,038 |
| 23 | 10,485,760 | 56,205,913 |

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
- Phase 0 forward: 0.3s
- Phase 1 roundtrip CLI: 97.6s
- Phase 1 forward with pre-populated cache: 0.1s
