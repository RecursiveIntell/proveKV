#![cfg(feature = "kv")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fib_quant::{
    kv::{
        decode_kv_pages, encode_kv_tensor, KvAttentionKind, KvAxisPolicyV1, KvCacheLayoutV1,
        KvCompressionProfileV1, KvDType, KvPageGeometryV1, KvRole, KvRopeState, KvTensorShapeV1,
    },
    FibQuantProfileV1, FibQuantizer,
};

fn fixture() -> (
    KvTensorShapeV1,
    KvCacheLayoutV1,
    KvCompressionProfileV1,
    Vec<f32>,
) {
    let shape = KvTensorShapeV1::new(
        KvRole::Value,
        KvAttentionKind::Mha,
        1,
        1,
        1,
        1,
        4,
        8,
        KvDType::F32,
        KvRopeState::NotApplicable,
    );
    let layout = KvCacheLayoutV1::canonical(&shape).unwrap();
    let mut fib_profile = FibQuantProfileV1::paper_default(8, 2, 8, 7).unwrap();
    fib_profile.training_samples = 64;
    fib_profile.lloyd_restarts = 1;
    fib_profile.lloyd_iterations = 1;
    let quantizer = FibQuantizer::new(fib_profile.clone()).unwrap();
    let geometry = KvPageGeometryV1::new(2, 8, 64);
    let profile = KvCompressionProfileV1::from_parts(
        "bench-value",
        &shape,
        fib_profile,
        quantizer.codebook().codebook_digest.clone(),
        KvAxisPolicyV1::PerToken,
        geometry,
    )
    .unwrap();
    let values = (0..shape.element_count().unwrap())
        .map(|idx| ((idx as f32 + 1.0) * 0.03125).sin())
        .collect();
    (shape, layout, profile, values)
}

fn bench_kv_encode_decode(c: &mut Criterion) {
    let (shape, layout, profile, values) = fixture();
    c.bench_with_input(
        BenchmarkId::new("kv_encode_decode", 4),
        &values,
        |b, values| {
            b.iter(|| {
                let encoded =
                    encode_kv_tensor(shape.clone(), layout.clone(), profile.clone(), values)
                        .unwrap();
                decode_kv_pages(&encoded).unwrap()
            });
        },
    );
}

criterion_group!(benches, bench_kv_encode_decode);
criterion_main!(benches);
