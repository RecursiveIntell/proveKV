#![cfg(feature = "kv")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fib_quant::kv::compare_attention_fixture;

fn bench_kv_attention_ref(c: &mut Criterion) {
    let head_dim = 8usize;
    let tokens = 8usize;
    let query: Vec<f32> = (0..head_dim).map(|idx| (idx as f32 + 1.0) * 0.1).collect();
    let keys: Vec<f32> = (0..tokens * head_dim)
        .map(|idx| ((idx as f32 + 1.0) * 0.03).cos())
        .collect();
    let decoded_keys: Vec<f32> = keys.iter().map(|value| value * 0.99).collect();
    let values: Vec<f32> = (0..tokens * head_dim)
        .map(|idx| ((idx as f32 + 1.0) * 0.02).sin())
        .collect();
    let decoded_values: Vec<f32> = values.iter().map(|value| value * 1.01).collect();

    c.bench_with_input(
        BenchmarkId::new("kv_attention_ref", tokens),
        &keys,
        |b, _| {
            b.iter(|| {
                compare_attention_fixture(
                    &query,
                    &keys,
                    &decoded_keys,
                    &values,
                    &decoded_values,
                    head_dim,
                    3,
                )
                .unwrap()
            });
        },
    );
}

criterion_group!(benches, bench_kv_attention_ref);
criterion_main!(benches);
