use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fib_quant::{FibCodebookV1, FibQuantProfileV1};

fn profile(seed: u64) -> FibQuantProfileV1 {
    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, seed).unwrap();
    profile.training_samples = 64;
    profile.lloyd_restarts = 1;
    profile.lloyd_iterations = 1;
    profile
}

fn bench_codebook_build(c: &mut Criterion) {
    c.bench_with_input(BenchmarkId::new("codebook_build", 8), &601u64, |b, seed| {
        b.iter(|| FibCodebookV1::build(profile(*seed)).unwrap());
    });
}

criterion_group!(benches, bench_codebook_build);
criterion_main!(benches);
