# oxih5 TODO (facade)

## Status
Full read-only facade with hierarchical group navigation. `Arc<Vec<u8>>` shared data, `File::group(path)` / `File::root()` / `Group::datasets()` / `Group::groups()` / `Group::attrs()` / `Group::dataset()`, `Dataset::attrs()` / `Dataset::attr(name)`, `File::info()`, `Debug for File`, `version()`. ~450 SLOC production code.

## Core Implementation
- [x] Implement hierarchical path navigation: `file.dataset("/group1/subgroup/data")` traversing nested groups
- [x] Implement `File::group(path)` returning a `Group` handle with iteration over children
- [x] Implement `File::root()` returning root group handle
- [x] Implement `Group::datasets()` listing datasets within a specific group
- [x] Implement `Group::groups()` listing sub-groups
- [x] Implement `Group::attrs()` listing attributes on a group
- [x] Implement `Dataset::attrs()` listing attributes on a dataset
- [x] Implement `Dataset::attr(name)` reading a single attribute by name
- [x] Implement dataset slicing: `file.dataset_slice(name, &[0..10, 5..15])` for sub-region reads (150-200 SLOC)
  **Done:** File::dataset_slice() + Group::dataset_slice() delegating to Dataset::slice() — 2026-05-25
- [x] Implement chunked dataset reading (decompress + reassemble chunks) (100-150 SLOC)
  - **Done:** `read_dataset_from_group` now routes `LayoutInfo::Chunked` through `oxih5_format::chunked::read_chunked` (B-tree v1 index + scatter); verified end-to-end against h5py chunked fixtures — 2026-05-25
- [x] Implement compressed dataset reading with filter pipeline application (80-100 SLOC)
  - **Done:** facade extracts the filter-pipeline message (0x000B) and applies it (gzip via oxiarc-deflate, shuffle, fletcher32) per chunk — 2026-05-25
- [x] Add `File::info()` returning file-level metadata (superblock version, file size, creation time)
- [x] Add streaming/lazy mode: `open_mmap(path)` using memory-mapped I/O instead of full read (60-80 SLOC)
  - **Done:** `FileData` enum (`Heap`/`Mapped`) with `Deref<Target=[u8]>` + `Clone` + `Debug`; `File.data: FileData`; `Group.file_data: FileData`; free fn `oxih5::open_mmap()` + associated `File::open_mmap()` + `File::open()`; `#![deny(unsafe_code)]` + `#[allow(unsafe_code)]` on mmap fn; 2 new integration tests (mmap_f4_1d, mmap_i4_2d) — 2026-05-25
- [x] Implement write support: `FileWriter` — flat HDF5 file creation, superblock v0 + old-style group + contiguous layout, ≤8 datasets, 5 element types (f32/f64/i32/i64/u8), h5py-verified (done 2026-05-25)
- [x] Implement `Dataset::to_ndarray<T>()` behind `ndarray` feature returning `ArrayD<T>` (done 2026-05-25)
  - **Result:** Added `ndarray = ["oxih5-core/ndarray"]` feature to crates/oxih5/Cargo.toml; no new lib.rs code needed — Dataset re-export gains to_array_f32/f64/i32() from oxih5-core when feature enabled — 2026-05-25

## API Improvements
- [x] Add `File::walk(visitor)` for recursive traversal of the entire file tree
  - **Done:** `File::walk(&mut impl FnMut(&str, bool))` — pre-order traversal, bool=true for groups — 2026-05-25
- [x] Add `File::contains(path)` predicate for checking existence of groups/datasets
  - **Done:** `File::contains(path)` — delegates to dataset()/group() — 2026-05-25
- [x] Remove `pure` feature gate once format implementation is complete (facade should work without feature flags)
  **Done:** Removed `pure` from `[features]` and `default` in `crates/oxih5/Cargo.toml`; dropped `optional = true` from `oxih5-format` and `memmap2` deps; removed all 24 `#[cfg(feature = "pure")]` guards from `lib.rs` — 2026-05-25
- [x] Add builder pattern for file creation: `FileWriter::new().write_dataset_f32("data", &vals, &shape).build(&path)` (done 2026-05-25)
- [x] Implement `std::fmt::Debug` for `File` showing file structure summary
- [x] Add `oxih5::version()` returning crate version string

## Testing
- [x] Integration test: read h5py-generated contiguous f32/f64/i32 datasets and verify values
- [x] Integration test: read big-endian datasets
- [x] Integration test: read multi-dimensional datasets and verify shape
- [x] Integration test: navigate nested groups `/a/b/c/data`
  - **Done:** tests 27-29 (nested group navigation 2 levels deep, group listing, leaf group datasets) against h5py libver='earliest' fixture `nested_groups.h5` — 2026-05-25
- [x] Integration test: read attributes from groups and datasets
  - **Done:** tests 30-32 (dataset attrs, attr by name, group attrs) against h5py libver='earliest' fixture `with_attrs.h5` — 2026-05-25
- [x] Integration test: read chunked + gzip-compressed datasets
  - **Done:** read_contig.rs tests 19-24 (chunked uncompressed, gzip 1-D/2-D, gzip+shuffle 2-D, fletcher32, partial edge chunks, group-handle path) against real h5py libver='earliest' fixtures — 2026-05-25
- [x] Test error handling: missing dataset, corrupt file, truncated file
- [x] Test that `dataset_names()` returns correct names for files with many datasets

## Performance
- [x] Benchmark full-file-read vs mmap for files of various sizes
  **Done:** `benches/read_bench.rs` — `open_contiguous_f32` vs `open_mmap_f32` criterion benchmarks using `f4_1d_contig.h5` fixture — 2026-05-25
- [x] Benchmark dataset read throughput for contiguous vs chunked layouts
  **Done:** `benches/read_bench.rs` — `open_contiguous_f32` (contiguous) and `read_chunked_gzip_1d` (chunked+gzip) criterion benchmarks — 2026-05-25
- [x] Profile memory allocation for large dataset reads — dhat-based memory profile test for the facade large-read path; non-flaky allocation regression guard. (done 2026-06-02)
  - **Done:** Created `oxih5/tests/mem_profile_test.rs` with two dhat heap-profile tests: `profile_heap_vs_mmap_large_read` (32 MB f64, asserts mmap peak < heap peak and total_blocks < 10_000) and `profile_full_vs_lazy_slice` (8 MB f64 baseline). Added `dhat-heap = ["dep:dhat"]` feature + `dhat = { workspace = true, optional = true }` dep to `oxih5/Cargo.toml`. Run with `cargo test -p oxih5 --features dhat-heap --test mem_profile_test -- --test-threads=1`.
- [x] Consider pre-parsed index cache for repeated access to the same file

## Integration
- [ ] Ensure compatibility with SciRS2 for reading ML model weights (PyTorch .h5, Keras .h5) — blocked on SciRS2 defining its API; oxih5 already supports float32/float64/int32 arrays needed for ML weights
  - **Refinement (2026-06-03):** oxih5-side prerequisite (typed Attribute accessors + vlen-string decode) lands in 0.1.1 (items A1/A4); SciRS2 API coordination remains upstream-blocked.
- [x] Ensure compatibility with NumRS2 ndarray bridge — Already implemented — ndarray feature gate in oxih5 and oxih5-core provides Dataset::to_array_f32/f64/i32()
- [x] Coordinate with OxiARC for decompression filters (oxiarc-deflate for gzip) — Already implemented — oxih5-format uses oxiarc-deflate 0.3.0 for gzip/deflate decompression in filters.rs (COOLJAPAN policy compliant)
- [x] Test interoperability with h5py-generated files across libver versions ('earliest', 'latest')
  - **Done:** libver_latest_chunked.h5 fixture (superblock v3) + tests 40-42 cover chunked gzip/shuffle/plain datasets with extensible array chunk index; VDS parsing also added (NotImplemented error path, test_vds_returns_not_implemented) — 2026-05-25
