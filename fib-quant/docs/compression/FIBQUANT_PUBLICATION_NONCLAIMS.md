# FibQuant Publication Non-Claims

Date: 2026-05-16

The allowed public positioning for `fib-quant` is narrow:

`fib-quant` is an experimental Rust implementation of the core FibQuant radial-angular vector quantization math for research and integration experiments.

## This Crate Does Not Claim

- production KV-cache compressor readiness;
- default-on compression in any parent workspace crate;
- integration with `semantic-memory`;
- replacement or mutation of `turbo-quant`;
- fused attention-kernel decompression;
- local reproduction of GPT-2, TinyLlama, or other paper benchmark numbers;
- superiority over TurboQuant, KIVI, KVQuant, CommVQ, or other systems on local workloads;
- safety for permanent crates.io publication without the release checklist and dry-run receipts.

## Required Language

Use "experimental", "research implementation", and "paper-faithful core math" when describing the release. Keep benchmark references clearly attributed to the arXiv paper unless local benchmark receipts exist.
