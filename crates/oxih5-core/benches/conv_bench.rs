use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxih5_core::{ByteOrder, Dataset, Dtype};
use std::hint::black_box;

fn make_f32_dataset(n: usize) -> Dataset {
    let values: Vec<f32> = (0..n).map(|i| i as f32 * 0.001_f32).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Dataset {
        data,
        shape: vec![n],
        dtype: Dtype::Float {
            size: 4,
            order: ByteOrder::Little,
        },
        attributes: vec![],
    }
}

fn bench_eager_vs_lazy_f32(c: &mut Criterion) {
    let n = 1_000_000usize;
    let ds = make_f32_dataset(n);
    let bytes = (n * 4) as u64;

    let mut group = c.benchmark_group("typed_conversion_f32");
    group.throughput(Throughput::Bytes(bytes));

    group.bench_function(BenchmarkId::new("eager_as_f32", n), |b| {
        b.iter(|| {
            let v = ds.as_f32().unwrap();
            black_box(v.len())
        });
    });

    group.bench_function(BenchmarkId::new("lazy_iter_f32_sum", n), |b| {
        b.iter(|| {
            let sum: f32 = ds
                .iter_f32()
                .unwrap()
                .fold(0.0_f32, |a, x| a + black_box(x));
            black_box(sum)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_eager_vs_lazy_f32);
criterion_main!(benches);
