# Changelog

All notable changes to the OxiH5 workspace are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.4] - Unreleased

## [0.1.3] - 2026-06-19

### Changed

- Workspace version bumped from 0.1.2 to 0.1.3; internal workspace dependency
  references for `oxih5-core`, `oxih5-format`, `oxih5`, and `oxinetcdf` updated
  accordingly. No public API changes.

---

## [0.1.2] - 2026-06-10

### Added

- **`oxinetcdf` — deep group hierarchy**: `NcGroup.children: Vec<NcGroup>`; `resolve_group_deep` with `MAX_GROUP_DEPTH=64` and cycle detection via visited-path set; 4 new unit tests.
- **`oxinetcdf` — cross-group shared dimensions**: two-phase scan — `collect_global_dims` (Phase 1) walks all groups and builds an addr→`GlobalDim` registry; `resolve_dim_list` (Phase 2) resolves refs via local cache → global registry → lazy `attrs_of` → phony; `NcAxis` gains `group_path` and `is_unlimited` fields.
- **`Dataset::max_dims` / `is_unlimited` / `unlimited_axes`**: `DataspaceInfo::max_dims` reads the HDF5 max-dims block; 8 new tests; all 350 tests pass.
- **`File::attrs_of(addr)`**: metadata-only attribute accessor (avoids loading variable data); 2 new unit tests.
- **`NcType` enum** in `oxinetcdf::types`: `From<&Dtype>` covers all 11 Dtype variants; `NcVariable::nc_type()` convenience method; 12 unit tests.
- **NC_STRING variable support**: `NcVariable::read_strings(nc)` delegates to `File::dataset_strings`; `NcAttribute::new_with_view` eagerly decodes vlen strings; `NcFile::h5()` public accessor.
- **`_FillValue`-aware masked reads**: `apply_fill_mask<T>`, `apply_fill_mask_f32/f64` (bit-exact NaN safe); `NcVariable::read_f64_masked` (NaN for fill), `read_i64_masked` (Option for fill); 5 unit tests.
- **CF conventions**: `NcGroup::coordinates_of/bounds_of/grid_mapping_of`; `cf.rs` module with `parse_cf_name_list/cf_group_prefix/cf_var_name`; supports CF-1.7 `group:var` form; 9 unit tests in `cf.rs` + 7 in `model.rs`.
- **`NcFileWriter`** (NetCDF-4 writing): `def_dim/def_var/put_var_f64/put_var_i32/put_att_str/close`; 7 round-trip tests. Backed by `FileWriter` attribute-writing infrastructure: `write_string_attr`, `write_f64_attr`, `write_i64_attr`, `write_i32_attr`, `write_obj_ref_list_attr`.
- **Sub-group creation**: `FileWriter::create_group` + `write_group_dataset_f64/i32` + `write_group_string_attr`; SNOD cache_type=1 scratch-pad; round-trip tests `w0b_create_group_and_dataset_roundtrip` + `w0b_group_groups_listing`.
- **Unlimited/chunked dataset layout**: `FileWriter::create_dataset_unlimited`; B-tree v1 type-1 single-chunk node; chunked layout v3 with `max_dim[0]=u64::MAX`; 1-D and 2-D round-trip tests.
- **Root group string attributes**: `FileWriter::write_root_str_attr`; dynamic OH size via `compute_root_oh_size`; 2 round-trip tests.
- **Unlimited-dimension append in `NcFileWriter`**: `def_dim_unlimited` + `put_vara_f64/i32`; rewrite-on-append strategy; 2 round-trip tests.
- **NETCDF4_CLASSIC strict-mode**: `NcFileWriter::set_classic_mode`; writes `_nc3_strict = ""` on root group; 2 tests.
- **`GlobalHeapWriter`** (`oxih5-format`): GCOL serialiser; `FileWriter::create_vlen_string_dataset`; `NcFileWriter::def_var_strings` + `put_var_strings`; 12 new tests.
- **`FileWriter` write module refactored** into sub-modules: `write/mod.rs`, `write/chunked.rs`, `write/format.rs`, `write/messages.rs`.
- **`oxih5-format` — region reference handling**: `decode_region_refs` added to `values.rs`.

### Changed

- `oxih5-core/src/dataset_convert.rs`: region reference decode path reworked for correctness.
- `oxinetcdf` resolver refactored into `resolver.rs` (extracted from `file.rs`); `file.rs` substantially slimmed.
- SNOD capacity increased from 8 to 64 entries to support groups with more datasets.
- `oxiarc-szip` bumped to `0.3.3` (registry dependency, no path override).

---

## [0.1.1] - 2026-06-04

### Added

- **`oxinetcdf` crate** — new workspace member providing a NetCDF-4 reader built
  atop OxiH5: `NcFile::open` / `open_from_bytes`, `NcFile::root_group()`, full
  `NcGroup` / `NcVariable` / `NcDimension` / `NcAxis` / `NcAttribute` model,
  NetCDF-4 convention resolution (DIMENSION_SCALE, `_Netcdf4Dimid`,
  DIMENSION_LIST object-reference axis linkage), reserved-attribute filtering,
  pure-dimension sentinel parsing, and phony-dimension naming.
- **`AttrView<'a>`** (new public type in `oxih5`) — file-context-aware attribute
  accessor that owns the `Attribute` data and borrows the file bytes; exposes
  `as_strings()` (fixed-length and vlen), `as_object_refs()`,
  `as_compound()`, `as_vlen_sequence()`, and all scalar helpers.
- **`File::attr_views(path)`** — returns `Vec<AttrView<'_>>` for all attributes on
  any dataset or group path.
- **`File::object_at(addr)`** — resolves an HDF5 object-reference address
  (obtained from `AttrView::as_object_refs()`) to an `ObjectKind::Dataset` or
  `ObjectKind::Group`; returns `OxiH5Error::NotFound` for null references
  (`u64::MAX`).
- **`File::dataset_at(addr)`** — convenience wrapper around `object_at` that
  returns `TypeMismatch` when the referenced object is a group.
- **`File::dataset_hyperslab(path, selection)`** and free function
  **`read_dataset_hyperslab`** — strided HDF5 hyperslab selection
  (`DimSelection` + `Hyperslab`); only chunks overlapping the bounding box are
  decompressed; non-selected elements inside chunks are dropped without
  allocation.
- **`Attribute` scalar accessors** (`as_i64`, `as_u64`, `as_f64`,
  `as_str_fixed`, `is_scalar`, `shape`) — decode fixed-width integer/float and
  fixed-length string attributes directly on the `Attribute` type in
  `oxih5-core`.
- **`f16_to_f32`** exposed as a public function from `oxih5-core`; correctly
  handles subnormals, ±infinity, and NaN.
- **`ndarray` bridge extended** — `to_array_u8`, `to_array_u16`, `to_array_u32`,
  `to_array_u64`, `to_array_i8`, `to_array_i16`, `to_array_i64`,
  `to_array_f16` added (feature-gated behind `ndarray`).
- **Criterion benchmarks** for `oxih5-format`: `parse_bench` (superblock v0/v2/v3
  and object-header parsing) and `traverse_bench` (group traversal throughput).
- **`oxih5-format` hyperslab and values modules** — `hyperslab.rs` and `values.rs`
  implementing strided selection logic and typed value decoding (`Value` enum,
  vlen-string decode, object-ref decode, compound decode, vlen-sequence decode).
- **`oxih5-format` chunked hyperslab module** — `chunked_hyperslab.rs` providing
  per-chunk hyperslab intersection for efficient partial-read of chunked datasets.
- Tests for all new APIs: `test_attribute_scalar_accessors`, `test_to_array_u8/u16/u32/u64/i8/i16/i64/f16`, `AttrView` unit tests, hyperslab integration tests.

### Changed

- `Dataset` typed-accessor methods (`as_f32`, `as_f64`, `as_i32`, etc.) and lazy
  iterators (`iter_f32`, …) extracted into a dedicated `dataset_convert` module in
  `oxih5-core`; public API is unchanged.
- `File::dataset_slice` now uses lazy per-chunk loading for chunked datasets
  (previously loaded the full dataset first, then sliced in memory).
- `oxih5` re-exports `Value` from `oxih5_format::values` and `DimSelection` /
  `Hyperslab` from `oxih5_format`.

---

## [0.1.0] — 2026-06-01

### Added

**Core types (`oxih5-core`)**

- `Dataset` — primary data container with raw bytes, shape, dtype, and
  attached attributes; provides typed accessors (`as_f32`, `as_f64`, `as_i32`,
  `as_u8`, `as_u16`, `as_u32`, `as_u64`, `as_i8`, `as_i16`, `as_i64`,
  `as_f16`, `as_string`) and lazy iterators (`iter_f32`, `iter_f64`, etc.).
- `Dtype` — full HDF5 datatype hierarchy: `Int`, `Float`, `String`, `Compound`,
  `Array`, `Enum`, `Opaque`, `Reference`, `VarLen`, `Bitfield`; all 11 classes.
- `Attribute`, `Dataspace`, `FilterPipeline`, `FilterInfo`, `PropertyList`,
  `Link`, `Group` core structs.
- `OxiH5Error` — comprehensive error enum covering I/O, format violations,
  type mismatches, unsupported features, and checksum failures.
- `Dataset::slice` — multi-dimensional sub-region extraction without copying
  the full dataset.
- `Dataset::reshape` — zero-copy shape reinterpretation with element-count
  validation.
- `ndarray` feature gate — `Dataset::to_array_f32/f64/i32` bridge to
  `ndarray::ArrayD` when the `ndarray` feature is enabled.

**Format parsers (`oxih5-format`)**

- Superblock parser: v0 (libver='earliest'), v2, and v3 (libver='latest').
- Object header parsers: v1 (message list with continuation blocks) and v2
  (OHDR + OCHK continuation, creation-order index, modification-time
  timestamps, phase-change flags).
- Message parsers for all standard HDF5 message types: Dataspace (0x0001),
  Datatype (0x0003), Layout (0x0008), SymbolTable (0x0011), FilterPipeline
  (0x000B), Attribute (0x000C v1/v2/v3), FillValue, ModificationTime,
  LinkInfo (0x0002), Link (0x0006).
- Contiguous, compact, and chunked data layout support.
- Chunked dataset reads via B-tree v1 (libver='earliest') and B-tree v2
  (libver='latest') chunk indices, plus extensible array and fixed array
  chunk indices.
- Filter pipeline: gzip/deflate (via `oxiarc-deflate`, COOLJAPAN policy),
  shuffle (byte unshuffle), fletcher32 checksum verification, nbit
  (integer bit-packing), and scaleoffset (integer precision reduction).
- SZIP filter support behind the `szip` feature (via `oxiarc-szip`).
- Parallel chunk decompression behind the `parallel` feature (via `rayon`).
- B-tree v1 group traversal (TREE signature), local heap name resolution
  (HEAP), SNOD symbol-table node parsing.
- B-tree v2 type-5 (name-indexed link) traversal.
- Fractal heap (FRHP + FHDB direct blocks + FHIB indirect blocks) for
  large new-style groups exceeding the inline link threshold.
- Global heap (GCOL) for variable-length and string dataset resolution.
- New-style group support: Link Info + Link messages, fractal heap traversal,
  B-tree v2 name index — covers HDF5 files written with `libver='latest'`.
- All 11 datatype class parsers: fixed-point int, float, string (fixed/VL),
  compound, array, enum, opaque, reference, variable-length, bitfield.
- Fuzz harness integration test suite (`fuzz_parsers`): random bytes,
  uniform bytes, empty input, bit-flipped real fixtures — all must not panic.

**Facade crate (`oxih5`)**

- `open(path)` — heap-backed file open.
- `open_mmap(path)` — memory-mapped file open (read-only, zero-copy for
  large files).
- `read_dataset(path, name)` — one-shot convenience wrapper.
- `File` handle with `dataset(path)`, `dataset_names()`, `dataset_slice()`,
  `group(path)`, `root()`, `contains(path)`, `walk(visitor)`, `info()`.
- `Group` handle with `datasets()`, `groups()`, `dataset(name)`,
  `dataset_slice(name, ranges)`, `attrs()`.
- Hierarchical path navigation (`/group/subgroup/dataset`).
- External link resolution (opens the referenced file and navigates to the
  target path).
- `ChunkIndexCache` — shared cache of pre-parsed chunk index structures,
  reused across multiple dataset reads from the same `File` handle.
- `FileWriter` — flat HDF5 file creation (write support): contiguous layout,
  multiple datasets, float32/float64/int32/uint8 dtypes; verified against
  h5py round-trip.
- `version()` — returns crate version string.

**Testing**

- 273 unit and integration tests across all three crates; all pass.
- Integration tests verify real h5py-generated HDF5 fixtures: superblock
  v0/v2/v3, old-style and new-style groups, large groups (20+ datasets),
  fractal heap traversal, B-tree v2 name index, chunked + gzip + shuffle
  datasets, compound/string/enum/opaque/array/reference/bitfield datatypes.
- Fuzz corpus (4 `cargo-fuzz` targets in `fuzz/`): `fuzz_superblock`,
  `fuzz_header`, `fuzz_message`, `fuzz_file_open`.

### Architecture

```
HDF5 file bytes
      │
      ▼
superblock.rs       — v0/v2/v3 root group address
      │
      ▼
header.rs           — object header v1/v2 message list + continuation
      │
      ▼
message.rs          — decode all standard message types
      │
      ├── btree.rs            — B-tree v1 group-node traversal
      ├── btree_v1_chunk.rs   — B-tree v1 chunk index
      ├── btree_v2.rs         — B-tree v2 (new-style groups + chunks)
      ├── ea_index.rs         — extensible array chunk index
      ├── fa_index.rs         — fixed array chunk index
      ├── snod.rs             — symbol-table node entries
      ├── heap.rs             — local heap name resolution
      ├── global_heap.rs      — global heap (VL/string data)
      ├── fractal_heap.rs     — fractal heap (large new-style groups)
      ├── link_msg.rs         — Link Info + Link message parsing
      ├── group.rs            — name → object-header resolution
      ├── chunked.rs          — full chunked dataset assembly
      ├── filters.rs          — filter pipeline (deflate/shuffle/fletcher32/nbit/scaleoffset)
      └── datatype.rs         — all 11 HDF5 datatype class parsers
```

### Policy compliance

- Pure Rust default features: no libhdf5 FFI, no C/C++ dependencies.
- DEFLATE via `oxiarc-deflate` (COOLJAPAN policy; never flate2/miniz/zlib-ng).
- SZIP via `oxiarc-szip` (feature-gated; COOLJAPAN policy).
- HDF5 FFI crates banned workspace-wide via `deny.toml`.
- `#![forbid(unsafe_code)]` on `oxih5-core`; `#[deny(unsafe_code)]` on the
  facade (only `open_mmap` uses `unsafe` for the mmap call, documented).

---

[0.1.3]: https://github.com/cool-japan/oxih5/releases/tag/v0.1.3
[0.1.2]: https://github.com/cool-japan/oxih5/releases/tag/v0.1.2
[0.1.1]: https://github.com/cool-japan/oxih5/releases/tag/v0.1.1
[0.1.0]: https://github.com/cool-japan/oxih5/releases/tag/v0.1.0
