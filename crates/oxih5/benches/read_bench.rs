use criterion::{criterion_group, criterion_main, Criterion};
use oxih5::File;
use std::io::Write;
use std::path::PathBuf;

/// Write the embedded fixture bytes to a temp file and return its path.
///
/// The benchmark must delete the file after use; criterion does not own the
/// lifetime so we return the path explicitly.
fn write_fixture_to_tmp(name: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("oxih5_bench_{name}.h5"));
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(bytes).unwrap();
    path
}

// ---------------------------------------------------------------------------
// Embedded fixture bytes (same approach used by integration tests so we have
// no dependency on the on-disk fixtures dir path existing in CI).
// ---------------------------------------------------------------------------

const F4_1D_CONTIG: &[u8] = include_bytes!("../tests/fixtures/f4_1d_contig.h5");
const CHUNKED_GZIP_F4_1D: &[u8] = include_bytes!("../tests/fixtures/chunked_gzip_f4_1d.h5");

// ---------------------------------------------------------------------------
// Benchmark 1: heap-backed open + dataset lookup (contiguous float32)
// ---------------------------------------------------------------------------

fn bench_open_contiguous(c: &mut Criterion) {
    let path = write_fixture_to_tmp("bench_open_contig", F4_1D_CONTIG);

    c.bench_function("open_contiguous_f32", |b| {
        b.iter(|| {
            let f = File::open(&path).unwrap();
            let _ds = f.dataset("temperature").unwrap();
        });
    });

    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Benchmark 2: mmap-backed open + dataset lookup (contiguous float32)
// ---------------------------------------------------------------------------

fn bench_open_mmap(c: &mut Criterion) {
    let path = write_fixture_to_tmp("bench_open_mmap", F4_1D_CONTIG);

    c.bench_function("open_mmap_f32", |b| {
        b.iter(|| {
            let f = File::open_mmap(&path).unwrap();
            let _ds = f.dataset("temperature").unwrap();
        });
    });

    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Benchmark 3: heap-backed open + chunked gzip dataset read
// ---------------------------------------------------------------------------

fn bench_chunked_gzip(c: &mut Criterion) {
    let path = write_fixture_to_tmp("bench_chunked_gzip", CHUNKED_GZIP_F4_1D);

    c.bench_function("read_chunked_gzip_1d", |b| {
        b.iter(|| {
            let f = File::open(&path).unwrap();
            let _ds = f.dataset("data").unwrap();
        });
    });

    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Benchmark 4: dataset_names() on the root group (contiguous float32 fixture)
// ---------------------------------------------------------------------------

fn bench_dataset_names(c: &mut Criterion) {
    let path = write_fixture_to_tmp("bench_dataset_names", F4_1D_CONTIG);

    c.bench_function("dataset_names", |b| {
        b.iter(|| {
            let f = File::open(&path).unwrap();
            f.dataset_names().unwrap()
        });
    });

    let _ = std::fs::remove_file(&path);
}

criterion_group!(
    benches,
    bench_open_contiguous,
    bench_open_mmap,
    bench_chunked_gzip,
    bench_dataset_names
);
criterion_main!(benches);
