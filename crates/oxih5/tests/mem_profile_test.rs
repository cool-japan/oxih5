// Run with: cargo test -p oxih5 --features dhat-heap --test mem_profile_test -- --test-threads=1
//
//! Memory-allocation tests using DHAT heap profiling.
//!
//! These tests are only compiled/executed when the `dhat-heap` feature is enabled.
//! Run with:
//!   cargo test -p oxih5 --features dhat-heap --test mem_profile_test -- --test-threads=1
//!
//! The `--test-threads=1` flag is required because `#[global_allocator]`
//! is a per-binary singleton and `dhat::Profiler` uses a global mutex.
//! Sequential execution avoids races between test threads.

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Profile scenario 1: compare peak allocation between heap-based File::open
/// and memory-mapped File::open_mmap on a large (~32 MB) f64 dataset.
///
/// Assertions are generous multiples to remain non-flaky across platforms,
/// while still catching catastrophic regressions.
#[cfg(feature = "dhat-heap")]
#[test]
fn profile_heap_vs_mmap_large_read() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = 4_000_000; // 4M × 8 bytes = 32 MB
    let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let tmp = std::env::temp_dir().join("oxih5_mem_profile_large.h5");

    oxih5::FileWriter::new()
        .write_dataset_f64("big_data", &data, &[n])?
        .build(&tmp)?;

    let file_len = std::fs::metadata(&tmp)?.len();

    // Profile heap-based open + dataset read.
    let heap_stats = {
        let _profiler = dhat::Profiler::new_heap();
        let f = oxih5::File::open(&tmp)?;
        let _ = f.dataset("big_data")?;
        dhat::HeapStats::get()
    };

    // Profile mmap-based open + dataset read.
    let mmap_stats = {
        let _profiler = dhat::Profiler::new_heap();
        let f = oxih5::File::open_mmap(&tmp)?;
        let _ = f.dataset("big_data")?;
        dhat::HeapStats::get()
    };

    // Non-flaky bounds:
    // Peak allocation must stay within a small constant multiple of file size.
    assert!(
        heap_stats.max_bytes < file_len as usize * 3,
        "heap open peak {}, file_len {}",
        heap_stats.max_bytes,
        file_len
    );

    // mmap avoids the whole-file Vec, so its peak should be lower.
    assert!(
        mmap_stats.max_bytes < heap_stats.max_bytes,
        "mmap peak {} should be less than heap peak {}",
        mmap_stats.max_bytes,
        heap_stats.max_bytes
    );

    // Total block count must be bounded (not per-element allocation).
    assert!(
        heap_stats.total_blocks < 10_000,
        "too many allocations: {} blocks",
        heap_stats.total_blocks
    );

    std::fs::remove_file(&tmp).ok();
    Ok(())
}

/// Profile scenario 2: compare allocation profiles between a full dataset read
/// and a lazy slice on a 1 M-element f64 dataset (~8 MB).
///
/// NOTE: `FileWriter` writes contiguous datasets; `dataset_slice` on a
/// contiguous layout falls back to full-read + in-memory slice (lazy loading
/// only kicks in for chunked layouts).  This test documents the current
/// contiguous-path baseline so that future chunked-write support can show
/// the expected lazy-chunk allocation win.
#[cfg(feature = "dhat-heap")]
#[test]
fn profile_full_vs_lazy_slice() -> Result<(), Box<dyn std::error::Error>> {
    let n: usize = 1_000_000; // 1M × 8 bytes = 8 MB
    let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();
    let tmp = std::env::temp_dir().join("oxih5_mem_profile_slice.h5");

    oxih5::FileWriter::new()
        .write_dataset_f64("slice_data", &data, &[n])?
        .build(&tmp)?;

    // Profile full read.
    let full_stats = {
        let _profiler = dhat::Profiler::new_heap();
        let f = oxih5::File::open(&tmp)?;
        let _ = f.dataset("slice_data")?;
        dhat::HeapStats::get()
    };

    // Profile partial slice (0..1000 out of 1M).
    let slice_stats = {
        let _profiler = dhat::Profiler::new_heap();
        let f = oxih5::File::open(&tmp)?;
        let r: std::ops::Range<usize> = 0..1000;
        let _ = f.dataset_slice("slice_data", std::slice::from_ref(&r))?;
        dhat::HeapStats::get()
    };

    // For contiguous datasets, dataset_slice falls back to full read + slice,
    // so allocation profiles are similar.  Document this as a non-flaky baseline:
    // the key assertion is that neither path causes catastrophic block counts.
    assert!(
        full_stats.total_blocks < 10_000,
        "full read: too many allocations: {} blocks",
        full_stats.total_blocks
    );
    assert!(
        slice_stats.total_blocks < 10_000,
        "slice read: too many allocations: {} blocks",
        slice_stats.total_blocks
    );

    // Report numbers for informational purposes (not assertions).
    eprintln!(
        "Full read: max_bytes={}, total_blocks={}",
        full_stats.max_bytes, full_stats.total_blocks
    );
    eprintln!(
        "Slice read: max_bytes={}, total_blocks={}",
        slice_stats.max_bytes, slice_stats.total_blocks
    );

    std::fs::remove_file(&tmp).ok();
    Ok(())
}
