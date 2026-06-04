use criterion::{criterion_group, criterion_main, Criterion, Throughput};
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

// ---------------------------------------------------------------------------
// Throughput benchmark 5: contiguous f64 full read (heap-backed)
//
// Writes a fresh ~4 MB contiguous f64 dataset (500_000 elements × 8 B = 4 MB)
// to a temp file using FileWriter, then measures the full open + dataset +
// as_f64 cycle.  Throughput::Bytes lets criterion report MB/s.
// ---------------------------------------------------------------------------

const THROUGHPUT_N_ELEMS: usize = 500_000; // 500_000 × 8 = 4_000_000 bytes

fn throughput_contiguous_f64(c: &mut Criterion) {
    let n = THROUGHPUT_N_ELEMS;
    let n_bytes = (n * 8) as u64;

    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let tmp = std::env::temp_dir().join("oxih5_bench_throughput_contig_f64.h5");

    oxih5::FileWriter::new()
        .write_dataset_f64("data", &data, &[n])
        .unwrap()
        .build(&tmp)
        .unwrap();

    let mut group = c.benchmark_group("throughput_contiguous_f64");
    group.throughput(Throughput::Bytes(n_bytes));

    group.bench_function("heap", |b| {
        b.iter(|| {
            let f = File::open(&tmp).unwrap();
            let ds = f.dataset("data").unwrap();
            let _vals = ds.as_f64().unwrap();
        });
    });

    group.finish();

    let _ = std::fs::remove_file(&tmp);
}

// ---------------------------------------------------------------------------
// Throughput benchmark 6: contiguous f64 full read (mmap-backed)
//
// Identical dataset as bench 5, but opened via File::open_mmap so the OS
// pages in only the touched regions.  Reports the same MB/s metric for
// direct comparison with the heap path.
// ---------------------------------------------------------------------------

fn throughput_mmap_f64(c: &mut Criterion) {
    let n = THROUGHPUT_N_ELEMS;
    let n_bytes = (n * 8) as u64;

    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let tmp = std::env::temp_dir().join("oxih5_bench_throughput_mmap_f64.h5");

    oxih5::FileWriter::new()
        .write_dataset_f64("data", &data, &[n])
        .unwrap()
        .build(&tmp)
        .unwrap();

    let mut group = c.benchmark_group("throughput_mmap_f64");
    group.throughput(Throughput::Bytes(n_bytes));

    group.bench_function("mmap", |b| {
        b.iter(|| {
            let f = File::open_mmap(&tmp).unwrap();
            let ds = f.dataset("data").unwrap();
            let _vals = ds.as_f64().unwrap();
        });
    });

    group.finish();

    let _ = std::fs::remove_file(&tmp);
}

// ---------------------------------------------------------------------------
// Throughput benchmark 7: chunked + gzip full read
//
// Reads the committed chunked_gzip_f4_1d.h5 fixture (float32, dataset "data").
// The element count is determined at setup from the dataset shape so that
// Throughput::Bytes always matches the actual uncompressed payload even if the
// fixture is regenerated with different dimensions.
// ---------------------------------------------------------------------------

fn throughput_chunked_gzip(c: &mut Criterion) {
    let path = write_fixture_to_tmp("bench_throughput_chunked_gzip", CHUNKED_GZIP_F4_1D);

    // Determine the uncompressed byte count once at setup time.
    let n_elems: usize = {
        let f = File::open(&path).unwrap();
        let ds = f.dataset("data").unwrap();
        ds.len()
    };
    // float32 = 4 bytes per element
    let n_bytes = (n_elems * 4) as u64;

    let mut group = c.benchmark_group("throughput_chunked_gzip");
    group.throughput(Throughput::Bytes(n_bytes));

    group.bench_function("heap", |b| {
        b.iter(|| {
            let f = File::open(&path).unwrap();
            let ds = f.dataset("data").unwrap();
            let _vals = ds.as_f32().unwrap();
        });
    });

    group.finish();

    let _ = std::fs::remove_file(&path);
}

criterion_group!(
    benches,
    bench_open_contiguous,
    bench_open_mmap,
    bench_chunked_gzip,
    bench_dataset_names,
    throughput_contiguous_f64,
    throughput_mmap_f64,
    throughput_chunked_gzip,
);
criterion_main!(benches);
