use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fib_quant::{FibQuantProfileV1, FibQuantizer};

fn quantizer(seed: u64) -> FibQuantizer {
    let mut profile = FibQuantProfileV1::paper_default(8, 2, 8, seed).unwrap();
    profile.training_samples = 64;
    profile.lloyd_restarts = 1;
    profile.lloyd_iterations = 1;
    FibQuantizer::new(profile).unwrap()
}

fn bench_encode_decode(c: &mut Criterion) {
    let q = quantizer(501);
    let input = vec![0.25, -0.5, 0.75, 1.0, -1.25, 0.5, 0.125, -0.875];

    c.bench_with_input(BenchmarkId::new("encode_decode", 8), &input, |b, input| {
        b.iter(|| {
            let code = q.encode(input).unwrap();
            q.decode(&code).unwrap()
        });
    });
}

criterion_group!(benches, bench_encode_decode);
criterion_main!(benches);
