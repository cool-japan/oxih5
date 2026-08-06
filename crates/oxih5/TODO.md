# oxih5 TODO (facade)

## Status
Full read/write facade with hierarchical group navigation and Virtual Dataset (VDS) resolution. The `FileWriter` builder now covers contiguous, compact (`set_compact`) and chunked/tiled datasets — unlimited (`create_dataset_unlimited`) or fixed-maxshape (`set_chunking`) — with composable DEFLATE/shuffle/Fletcher32 filters (`set_deflate`/`set_shuffle`/`set_fletcher32`), custom fill values (`set_fill_value_*`), fixed-length (`create_fixed_string_dataset`) and vlen strings, booleans (`write_dataset_bool`), nested groups at any depth, the full fixed-width + array + vlen-objref + compound attribute set, and in-place overwrite (`write_dataset_in_place`). 0.2.3 adds compound records (`create_compound_dataset`), ragged vlen sequences (`create_vlen_sequence_dataset` and friends), array / opaque / bitfield element types, big-endian and half-precision numerics on both datasets and attributes (`write_dataset_numeric` / `write_numeric_attr` / `write_numeric_scalar_attr`), soft / external / hard-alias links (`create_soft_link` / `create_external_link` / `create_hard_link`), and new-style link-message groups in both compact and dense form — a fractal-heap writer plus a type-5 version-2 B-tree name index with libhdf5-exact Jenkins lookup3 checksums — with `set_link_storage` / `set_track_order` selecting the style and preserving creation order. `Arc<Vec<u8>>` shared data, `File::group`/`root`/`Group::datasets`/`groups`/`attrs`/`dataset`, `Dataset::attrs`/`attr`, `File::info()` (true superblock version 0/1/2/3 + extension address), `File::superblock_extension()`, `Debug for File`, `version()`. ~12,700 SLOC (tokei, `crates/oxih5/src`, incl. the `write/*` module family) across `lib.rs`, `file.rs`, `group_handle.rs`, `reader.rs`, `slicing.rs`, `links.rs`, `attr_view.rs`, and `write/*`. 528 unit/integration tests + 15 doc tests passing (`--all-features`; 526 with default features), 0 failures, 0 clippy/rustdoc warnings. Writer output verified against h5py 3.16 / libhdf5 2.0.0. Status as of v0.2.3 — 2026-08-06.

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
- [x] Expose superblock version 1 support and superblock-extension parsing at the facade level
  - **Done:** `File::info()` now reports the actual on-disk superblock version (`FileInfo.superblock_version`: `0`/`1`/`2`/`3`, previously hardcoded to always report `0`) plus the new `FileInfo.superblock_extension_address: Option<u64>` field. New `File::superblock_extension(&self) -> Result<Option<SuperblockExtension>, OxiH5Error>` decodes the B-tree 'K' Values (0x0013), Shared Message Table (0x000F), File Space Info (0x0018), and Driver Info (0x0014) messages from the superblock extension object header (versions 2/3 only), via `oxih5_format::superblock::read_superblock_extension`; `SuperblockExtension` and `BtreeKValues` re-exported as `oxih5::SuperblockExtension` / `oxih5::BtreeKValues`. Files written with HDF5 superblock v1 now open successfully instead of failing with `OxiH5Error::UnsupportedSuperblock(1)` — the fix lives in `oxih5-format`'s new `superblock::parse_v1`, and required no facade code changes since `File`/`Group` already call `superblock::parse` uniformly regardless of version. Covered by `lib.rs::test_superblock_extension_none_on_v0_file` — 2026-07-18
- [x] Add streaming/lazy mode: `open_mmap(path)` using memory-mapped I/O instead of full read (60-80 SLOC)
  - **Done:** `FileData` enum (`Heap`/`Mapped`) with `Deref<Target=[u8]>` + `Clone` + `Debug`; `File.data: FileData`; `Group.file_data: FileData`; free fn `oxih5::open_mmap()` + associated `File::open_mmap()` + `File::open()`; `#![deny(unsafe_code)]` + `#[allow(unsafe_code)]` on mmap fn; 2 new integration tests (mmap_f4_1d, mmap_i4_2d) — 2026-05-25
- [x] Implement write support: `FileWriter` — flat HDF5 file creation, superblock v0 + old-style group + contiguous layout, ≤8 datasets, 5 element types (f32/f64/i32/i64/u8), h5py-verified (done 2026-05-25)
- [x] Implement `Dataset::to_ndarray<T>()` behind `ndarray` feature returning `ArrayD<T>` (done 2026-05-25)
  - **Result:** Added `ndarray = ["oxih5-core/ndarray"]` feature to crates/oxih5/Cargo.toml; no new lib.rs code needed — Dataset re-export gains to_array_f32/f64/i32() from oxih5-core when feature enabled — 2026-05-25
- [x] Implement Virtual Dataset (VDS, layout class 3) reading end-to-end instead of returning `NotImplemented`
  - **Done:** `read_virtual_dataset` in the new `src/links.rs` resolves each VDS mapping entry (same file, or an external file resolved relative to the containing file's directory), reads the selected source region and scatters it into the virtual dataset's buffer. Fixed-size element types only (a vlen-typed VDS still returns `OxiH5Error::NotImplemented`); unmapped regions read as zero (non-zero fill values not yet applied). Covered by `tests/vds_tests.rs` (`vds_simple_full_mapping`, `vds_concat_two_sources`, `vds_main_fixture`) and the rewritten `read_contig.rs::test_vds_resolves_source` (formerly `test_vds_returns_not_implemented`) — 2026-07-17
- [x] Add `File::dataset_vlen_sequences(path)` decoding a variable-length *sequence* dataset (datatype class 9) into `Vec<oxih5_format::values::Value>`
  - **Done:** works for both contiguous and chunked layouts by delegating to `oxih5_format::values::decode_vlen_sequences`; returns `OxiH5Error::TypeMismatch` for non-`VarLen` dtypes (vlen *strings* remain on `File::dataset_strings`) — 2026-07-17
- [x] Resolve a soft link whose target is itself an external link (soft → external chain) when reading a dataset, instead of failing with `NotImplemented`
  - **Done:** `resolve_soft_link_target` / `SoftTarget` in `src/links.rs`. Link-resolution helpers (soft-link and external-link navigation) were also extracted from `lib.rs` into the new `src/links.rs` (426 lines) to keep `lib.rs` under the 2000-line-per-file policy. Covered by `tests/vds_tests.rs::soft_link_through_external_link` (`/soft` → `/ext` → `other.h5:/payload`). Note: a soft link to an external-file *group* (as opposed to a dataset) still returns `NotImplemented` — only the dataset-read path was fixed — 2026-07-17
- [x] Fix `write_vlen_ref`'s on-disk byte layout to match real HDF5 (`H5T__vlen_disk_write`): `[seq_len:4][heap_addr:8][obj_idx:4]`, not the previous (incorrect) `[seq_len:4][obj_idx:2][reserved:2][heap_addr:8]`
  - **Done:** `src/write/mod.rs`; files written by `FileWriter` are now byte-layout-compatible with other HDF5 implementations for vlen references — 2026-07-17
- [x] Validate that `write_dataset_f32/f64/i32/i64/u8` data length matches `shape.iter().product() × element_size`, returning `OxiH5Error::Format` on mismatch instead of risking a corrupt file or a later out-of-bounds panic
  - **Done:** validation added to `add_dataset` in `src/write/mod.rs`; `tests/write_tests.rs::test_write_shape_data_mismatch_rejected` — 2026-07-17
- [x] Add a `szip` feature flag forwarding to `oxih5-format/szip` for szip (compression id 4) chunk decoding, including RAW mode
  - **Done:** `szip = ["oxih5-format/szip"]` in `crates/oxih5/Cargo.toml`; decoding itself lives in `oxih5-format` (via `oxiarc-szip`), this facade only forwards the feature — 2026-07-17

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
- [x] Integration test: Virtual Dataset (VDS) resolution — full mapping, two-source concat, pre-existing multi-source fixture, and a soft-link-through-external-link chain
  - **Done:** `tests/vds_tests.rs` (4 tests: `vds_simple_full_mapping`, `vds_concat_two_sources`, `vds_main_fixture`, `soft_link_through_external_link`); each skips gracefully if its h5py fixture wasn't generated — 2026-07-17
- [x] Integration test: chunked variable-length-string dataset reads, both full and via a hyperslab selection straddling a chunk boundary
  - **Done:** `tests/vlen_chunked_tests.rs` (2 tests) against `vlen_str_chunked.h5` (h5py, `chunks=(3,)` over a length-10 dataset) — 2026-07-17
- [x] Integration test: `write_dataset_*` rejects a data/shape length mismatch instead of writing a corrupt file
  - **Done:** `tests/write_tests.rs::test_write_shape_data_mismatch_rejected` — 2026-07-17
- [x] Add runnable crate-level doctests to `src/lib.rs` (`//!` docs): a write/read round-trip and a hyperslab slice read
  - **Done:** 2 new doctests; `cargo test --all-features` now runs 3 doctests total (incl. the pre-existing `FileWriter` struct-level example) — 2026-07-17

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
