//! Memory-allocation tests using DHAT heap profiling.
//!
//! These tests are only compiled when the `dhat-heap` feature is enabled.
//! Run with:
//!   cargo test -p oxih5-core --features dhat-heap -- --test-threads=1
//!
//! The `--test-threads=1` flag is required because `#[global_allocator]`
//! is a per-binary singleton and `dhat::Profiler` uses a global mutex.
//! Sequential execution avoids races between test threads.

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[cfg(feature = "dhat-heap")]
#[test]
fn lazy_iter_allocates_less_than_eager() {
    use oxih5_core::{ByteOrder, Dataset, Dtype};

    let n = 1_000_000_usize;
    let data: Vec<u8> = (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect();

    let ds = Dataset {
        data,
        shape: vec![n],
        dtype: Dtype::Float {
            size: 4,
            order: ByteOrder::Little,
        },
        attributes: vec![],
        max_dims: None,
    };

    // ---------------------------------------------------------------
    // Measure eager allocation: as_f32() materialises a Vec<f32>
    // ---------------------------------------------------------------
    let eager_total_bytes = {
        let _profiler = dhat::Profiler::new_heap();
        let v = ds.as_f32().expect("as_f32 must succeed");
        let stats = dhat::HeapStats::get();
        drop(v);
        stats.total_bytes
    };

    // ---------------------------------------------------------------
    // Measure lazy allocation: iter_f32() streams without a Vec<f32>
    // ---------------------------------------------------------------
    let lazy_total_bytes = {
        let _profiler = dhat::Profiler::new_heap();
        let sum: f32 = ds
            .iter_f32()
            .expect("iter_f32 must succeed")
            .fold(0.0_f32, |acc, x| acc + x);
        let stats = dhat::HeapStats::get();
        // Prevent the compiler from eliding the fold.
        std::hint::black_box(sum);
        stats.total_bytes
    };

    assert!(
        lazy_total_bytes < eager_total_bytes,
        "lazy iterator should allocate fewer total bytes than eager path: \
         lazy={lazy_total_bytes} bytes, eager={eager_total_bytes} bytes",
    );
}
