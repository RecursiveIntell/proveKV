# FibQuant Benchmark Plan

Date: 2026-05-16

This crate does not include local benchmark receipts reproducing the FibQuant paper's results. Paper-reported results must remain attributed to the paper until this repository contains repeatable benchmark artifacts.

## Required Local Receipts Before Benchmark Claims

- Exact model, dataset, sequence length, batch size, context length, and hardware.
- Exact crate version, compiler version, feature flags, and git revision.
- Codebook profile: `d`, `k`, `N`, seeds, Lloyd restarts, iterations, training samples, norm format, and source mode.
- Quality metrics: MSE/cosine for codec fixtures and task-level metrics for model workloads.
- Throughput and memory metrics with raw command output.
- Baseline comparisons using the same environment and measurement scripts.

## Suggested Command Classes

- Unit correctness: `cargo test`.
- Example compilation: `cargo test --examples`.
- Criterion or harness-based codec microbenchmarks for encode/decode latency.
- End-to-end KV-cache benchmarks only after an explicit integration crate exists outside this release boundary.

## Publication Rule

Do not write that this crate reproduces the FibQuant paper's performance, memory, perplexity, or throughput numbers unless those results are generated locally and committed with enough metadata for another operator to rerun them.
