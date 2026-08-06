# OxiH5

**OxiH5** is the COOLJAPAN Pure-Rust HDF5 reader/writer. It parses and creates
real HDF5 files (as written by h5py / libhdf5) from scratch using only `std`
byte parsing — no `*-sys`, no C libhdf5, no unsafe code in production paths.

OxiH5 replaces `hdf5-sys` / `hdf5` / `netcdf-sys` on the **read** path and
provides a **write** path — contiguous, compact, and chunked/tiled datasets
with DEFLATE/shuffle/Fletcher32 filters, custom fill values, fixed-length /
vlen strings, booleans, nested groups and the full attribute set — whose output
is verified against h5py 3.16 and netCDF4-python 1.7.4.

---

## Release: 0.2.3 — 2026-08-06

987 unit + integration tests (`--all-features`; 966 with default features) plus
21 doc tests; all pass.  Full workspace (~31.1 k SLOC of Rust across four crates,
`crates/*/src`).

0.2.3 closes six writer roadmap gaps — compound records, ragged vlen sequences,
array / opaque / bitfield element types, big-endian datasets and attributes,
half-precision floats, and soft / external / hard-alias links — adds new-style
(link-message) groups in both compact and dense form (fractal-heap + B-tree-v2
name index, with `track_order` creation-order preservation), makes
**extensible-array chunk indexes readable** instead of a typed `NotImplemented`,
and fixes a mis-parsed array datatype plus five crash/OOM classes on crafted or
corrupted input (see `CHANGELOG.md`).

**Interop-verified.** Every file `FileWriter` and `NcFileWriter` produce is
opened and read back by **h5py 3.16 (libhdf5 2.0.0)** *and* **netCDF4-python
1.7.4** in the test suite — not only by OxiH5's own reader. 0.2.2 was the product
of a 44-agent differential interop audit that fixed 33 confirmed
libhdf5/netCDF-C conformance defects and closed 9 writer capability gaps (see
`CHANGELOG.md`).

---

## Crates

| Crate | Purpose |
|---|---|
| `oxih5-core` | Public types: `Dataset`, `Dtype`, `ByteOrder`, `OxiH5Error`, `Attribute`, `FilterPipeline`, `Link`, `Group` |
| `oxih5-format` | Low-level binary parsers: superblock, headers, messages, heap, B-tree v1/v2, SNOD, fractal heap, EA/FA index, filters, global heap, chunked assembly |
| `oxih5` | User-facing facade: `open()`, `open_mmap()`, `read_dataset()`, `File`, `Group`, `FileWriter` |
| `oxinetcdf` | Pure-Rust NetCDF-4 conventions reader/writer atop OxiH5: `NcFile`, `NcGroup`, `NcVariable`, `NcDimension`, `NcFileWriter` |

---

## Architecture

```
HDF5 file bytes
      │
      ▼
superblock.rs       — v0/v1/v2/v3 root group address + superblock extension
      │
      ▼
header.rs           — object header v1/v2 message list + continuation
      │
      ▼
message.rs          — decode all standard message types
      │
      ├── btree.rs            — B-tree v1 group-node traversal
      ├── btree_v1_chunk.rs   — B-tree v1 chunk index (libver='earliest')
      ├── btree_v2.rs         — B-tree v2 (new-style groups + chunks)
      ├── ea_index/           — extensible array chunk index (header, blocks, elements)
      ├── fa_index.rs         — fixed array chunk index
      ├── snod.rs             — symbol-table node entries
      ├── heap.rs             — local heap name resolution
      ├── global_heap.rs      — global heap (VL/string data)
      ├── fractal_heap.rs     — fractal heap (large new-style groups)
      ├── link_msg.rs         — Link Info + Link message parsing
      ├── group.rs            — name → object-header resolution
      ├── chunked/            — full chunked dataset assembly (cache, index, read, slice, geometry)
      ├── filters.rs          — filter pipeline (deflate/shuffle/fletcher32/nbit/scaleoffset)
      └── datatype.rs         — all 11 HDF5 datatype class parsers
```

---

## What Works (v0.2.3)

### Superblock

- v0 (`libver='earliest'`)
- v1 (transitional format; parses the extra 4 bytes inserted after the File
  Consistency Flags before Base Address / Root Group Symbol Table Entry)
- v2 and v3 (`libver='latest'`)
- Superblock extension (v2/v3): `File::superblock_extension()` decodes the
  extension object header's B-tree 'K' Values (0x0013), Shared Message Table
  (0x000F), File Space Info (0x0018), and Driver Info (0x0014) messages into
  `SuperblockExtension` / `BtreeKValues`

### Object Headers

- v1 (message list + continuation)
- v2 (OHDR + OCHK, creation-order, timestamps, phase-change)

### Groups

- Old-style: B-tree v1 + local heap + SNOD
- New-style: Link Info / Link messages + fractal heap (large groups) + B-tree v2 name index

### Data Layouts

- Contiguous
- Compact (inline data)
- Chunked: B-tree v1, B-tree v2, extensible array, fixed array indices
- Virtual (VDS): layout class 3 — resolves source-dataset mappings (same-file
  or external, `None`/`All`/`Hyperslab` selections) into the virtual buffer

### Filters (chunked)

| Filter | ID | Status |
|---|---|---|
| Deflate / gzip | 1 | DONE (via `oxiarc-deflate`) |
| Shuffle | 2 | DONE |
| Fletcher32 | 3 | DONE |
| SZIP / AEC | 4 | DONE (via `oxiarc-szip`, `szip` feature; RAW-mode chunks via `apply_pipeline_sized`) |
| Nbit | 5 | DONE (integer bit-packing) |
| Scaleoffset | 6 | DONE (integer precision reduction) |

### Datatypes (all 11 HDF5 classes)

| Class | Variants |
|---|---|
| Fixed-point integer | `Int`: i8/u8/i16/u16/i32/u32/i64/u64, LE/BE |
| Floating-point | `Float`: f16/f32/f64, LE/BE |
| String | `String`: fixed-length (ASCII/UTF-8) |
| Bitfield | `Bitfield`: size + byte order |
| Opaque | `Opaque`: raw bytes + tag |
| Compound | `Compound`: named fields at offsets |
| Reference | `Reference`: object / region |
| Enumerated | `Enum`: base type + member table |
| Variable-length | `VarLen`: global-heap-backed sequences |
| Array | `Array`: base type + dimension array |

### Attributes

- Message type 0x000C, versions 1, 2, and 3
- All datatype classes supported in attribute data

### Variable-length data

Chunked vlen and vlen-string datasets now read in full and via hyperslab
slicing (previously unsupported for chunked layouts).
`File::dataset_vlen_sequences(path)` decodes vlen *sequence* datasets
(datatype class 9) for both contiguous and chunked layouts.

### ndarray bridge

Enable the `ndarray` feature for `Dataset::to_array_f32/f64/i32` returning
`ndarray::ArrayD<T>`.

### Parallel decompression

Enable the `parallel` feature for concurrent chunk decompression via Rayon.

### Write support

`FileWriter` — creates valid HDF5 files verified readable by h5py 3.16 /
libhdf5 2.0.0.

- **Element types:** float16/32/64, int8/16/32/64, uint8/16/32/64, in either
  byte order (`write_dataset_numeric`); fixed-length strings
  (`create_fixed_string_dataset`, numpy `S<n>` / `NC_CHAR`); variable-length
  strings backed by the global heap (`create_vlen_string_dataset`); numpy
  `bool` as a class-8 enumeration (`write_dataset_bool`).
- **Structured datatypes:** compound records with caller-declared member
  offsets (`create_compound_dataset`, backing pandas HDFStore tables and event
  logs); ragged variable-length sequences over any fixed base type
  (`create_vlen_sequence_dataset`, `create_vlen_i32_dataset`,
  `create_vlen_f64_dataset`); array, opaque and bitfield element types
  (`create_array_dataset`, `create_opaque_dataset`, `create_bitfield_dataset`).
- **Layouts:** contiguous; compact / inline (`set_compact`); chunked and tiled,
  either unlimited (`create_dataset_unlimited`) or fixed-maxshape
  (`set_chunking`), with a real N-chunk multi-level B-tree index.
- **Filters (composable in one pipeline):** DEFLATE / gzip (`set_deflate`),
  shuffle (`set_shuffle`), Fletcher32 (`set_fletcher32`).
- **Fill values:** custom per-dataset fill via `set_fill_value_{f32,f64,i8,i16,i32,i64,u8,u16,u32,u64}`.
- **Groups:** nested at any depth (`create_group("a/b/c")`, intermediate groups
  auto-created); attributes on any group. Both HDF5 storage styles per group —
  old-style symbol tables by default, or link messages (`set_link_storage`)
  and, past eight members, a fractal heap with a version-2 name index.
  `set_track_order` records and preserves creation order, h5py's
  `track_order=True`.
- **Links:** soft (`create_soft_link`), external (`create_external_link`) and
  hard aliases (`create_hard_link`, which also raises the target object
  header's reference count as libhdf5 does).
- **Attributes:** the full fixed-width scalar set
  (`write_{i8,i16,i32,i64,u8,u16,u32,u64,f32,f64}_attr`), their 1-D `_array_attr`
  siblings, fixed / NUL-terminated strings, object-reference lists
  (`write_obj_ref_list_attr`), variable-length reference attributes
  (`write_vlen_obj_ref_attr`, the type of a netCDF `DIMENSION_LIST`), and the
  `{ dataset, dimension }` compound (`write_ref_index_list_attr`, a netCDF
  `REFERENCE_LIST`).
- **In-place overwrite:** `write_dataset_in_place` rewrites a dataset's bytes
  where they sit, preserving every other byte (e.g. a MATLAB v7.3 `.mat` file's
  `#refs#` group and `MATLAB_class` attributes).

`NcFileWriter` (in `oxinetcdf`) — creates NetCDF-4 files verified readable by
netCDF4-python 1.7.4, with conformant `DIMENSION_SCALE` / `_Netcdf4Dimid`
encoding, `DIMENSION_LIST` as `H5T_VLEN{H5T_REFERENCE}`, `REFERENCE_LIST` on
every dimension scale, `_Netcdf4Coordinates` on multidimensional variables, and
true **coordinate variables** (a dimension and variable sharing a name emitted as
one dimension-scale dataset — no phantom fabricated coordinates). Supports
`def_dim`, `def_dim_unlimited`, `def_var`, `put_var_f64/i32`, `put_vara_f64/i32`
(unlimited append), `def_var_strings` / `put_var_strings`, `put_att_str`, and
`set_classic_mode`.

### Memory-mapped I/O

`open_mmap(path)` / `File::open_mmap(path)` — the OS pages in only touched
regions; opening a 1 GB file is essentially free.

### Dataset utilities

- `Dataset::slice(&ranges)` — multi-dimensional sub-region extraction
- `Dataset::reshape(&shape)` — zero-copy shape reinterpretation
- Lazy iterators: `iter_f32`, `iter_f64`, `iter_i32`, `iter_u8`, `iter_i8`,
  `iter_u16`, `iter_i16`, `iter_u32`, `iter_i64`, `iter_u64`, `iter_f16`

---

## Usage

```rust
use oxih5::{open, read_dataset};

// One-shot convenience
let ds = read_dataset("data.h5", "/temperature")?;
let values: Vec<f32> = ds.as_f32()?;
println!("shape: {:?}, {} elements", ds.shape, ds.len());

// File handle (for multiple datasets)
let f = open("data.h5")?;
for name in f.dataset_names()? {
    println!("{name}");
}
let ds = f.dataset("/pressure")?;
let values: Vec<f64> = ds.as_f64()?;

// Hierarchical groups
let grp = f.group("/sensors/imu")?;
let names = grp.datasets()?;
let ds = grp.dataset("accel_x")?;

// Dataset slicing
let region = f.dataset_slice("/image", &[100..200, 50..150])?;

// Memory-mapped I/O for large files
let f = oxih5::open_mmap("large_file.h5")?;

// Write a new HDF5 file
use oxih5::FileWriter;
let path = std::env::temp_dir().join("output.h5");
let mut writer = FileWriter::new();
writer.write_dataset_f32("temperature", &[1.0f32, 2.0, 3.0], &[3])?;
writer.write_dataset_i32("index", &[0i32, 1, 2], &[3])?;
writer.build(&path)?;
```

---

## Milestone Table

| Milestone | Status | Description |
|---|---|---|
| M0 | DONE | Compile-clean workspace skeleton |
| M1 | DONE | Full read chain for contiguous float/int datasets |
| M2 | DONE | Chunked layout + gzip/shuffle/fletcher32 + ndarray bridge |
| M3 | DONE | Superblock v2/v3, object header v2, strings, compound types, attributes, new-style groups |
| M4 | DONE | mmap, lazy chunk reads, fuzz corpus, parallel decompression |
| M5 | DONE | Write support (FileWriter), full datatype coverage, nbit/scaleoffset filters |
| M6 | DONE | NetCDF-4 read conventions (oxinetcdf), hyperslab, AttrView, vlen/compound decode |
| M7 | DONE (0.1.2) | NcFileWriter, unlimited dims, sub-groups, GlobalHeap writer, CF conventions, fill masks, deep group hierarchy |
| M8 | DONE (0.1.4) | Virtual dataset (VDS) reads, chunked vlen/vlen-string reads, szip RAW-mode decoding, filtered fractal-heap root blocks, soft→external link chains |
| M9 | DONE (0.2.0) | Superblock v1 parsing, superblock v2/v3 extension parsing (B-tree K values, shared message table, file space info, driver info), object-header v2 OCHK continuation creation-order fix |
| M10 | DONE (0.2.1) | DEFLATE compression on write (`set_deflate`), real multi-chunk tiling with per-chunk compression, in-place dataset overwrite (`write_dataset_in_place`), nested groups at any depth (`create_group("a/b/c")`, auto-created intermediate groups), attributes on sub-groups and non-string/array-valued root-group attributes; fixed 8 write-path correctness bugs (chunk B-tree sized for libhdf5's real `2×K` node width instead of the dataset's own chunk count, 2-D unlimited variables reading back mostly zero, groups with 9+ links being unreadable by libhdf5, links declared out of name order being silently invisible, a dataset able to shadow a group of the same name, dangling object references silently becoming the undefined-address sentinel, soft links inside old-style/default-libver groups not resolving, and HDF5 layout-message version 4 — used by `libver='latest'` files — being entirely unsupported on read) |
| M11 | DONE (0.2.2) | Full h5py-3.16 / libhdf5-2.0.0 **and** netCDF4-python-1.7.4 interop conformance (44-agent differential audit): 33 confirmed defects fixed — global-heap `H5HG_MINSIZE` 4096 floor + real free-space object + 32-bit object index + `strlen` vlen lengths, size-0 datatype / defined-zero-length-address / `NULLTERM`-vs-`NULLPAD` / ASCII-vs-UTF-8 attribute-encoding fixes, terminal chunk-B-tree key, fill-value read + `Incremental` allocation, embedded-NUL / attr-padding / phantom-element reader fixes, and the netCDF `DIMENSION_LIST` `H5T_VLEN{REFERENCE}` segfault. 9 writer gaps closed — shuffle + Fletcher32 pipelines (`set_shuffle`/`set_fletcher32`), fixed-maxshape tiling (`set_chunking`), vlen-objref + compound attributes, widened attribute types, custom fill values, fixed-length strings, boolean datasets, compact layout, and netCDF coordinate variables |
| M12 | DONE (0.2.3) | Writer capability wave + extensible-array reader: compound records (`create_compound_dataset`), ragged vlen sequences (`create_vlen_sequence_dataset`), array / opaque / bitfield element types, big-endian datasets and attributes plus half-precision floats (`write_dataset_numeric`, `write_numeric_attr`), soft / external / hard-alias links (`create_soft_link` / `create_external_link` / `create_hard_link`), new-style link-message groups in both compact and dense form (fractal-heap writer + type-5 version-2 B-tree name index + Jenkins lookup3 metadata checksums) with `set_track_order` creation-order preservation; extensible-array chunk indexes now read in full (header, index block, super blocks, unpaged and paged data blocks, both element clients, rotated coordinate mapping) instead of returning `NotImplemented`; class-10 array datatype read fix (4-byte extents, version-2 reserved bytes and permutation indices); five crash/OOM classes closed on crafted input (zero chunk dimension, zero hyperslab stride, vlen/global-heap size overflow, B-tree-v1 chunk address overflow, unbounded VDS block count), six new byte-level fuzz targets |

---

## Testing

```bash
# Run all tests
cargo nextest run --all-features

# Run fuzz targets (requires nightly)
cargo +nightly fuzz run fuzz_superblock
cargo +nightly fuzz run fuzz_header
cargo +nightly fuzz run fuzz_message
cargo +nightly fuzz run fuzz_file_open
cargo +nightly fuzz run fuzz_fa_index
cargo +nightly fuzz run fuzz_ea_index
cargo +nightly fuzz run fuzz_btree_v1_chunk
cargo +nightly fuzz run fuzz_filters
cargo +nightly fuzz run fuzz_vds
cargo +nightly fuzz run fuzz_vlen_values
```

---

## Policy Compliance

- Pure Rust default features: no libhdf5 FFI, no C/C++ dependencies in the
  default build.
- `#![forbid(unsafe_code)]` on `oxih5-core`; `#[deny(unsafe_code)]` on the
  facade (only `open_mmap` uses `unsafe` for the mmap call, documented).
- DEFLATE via `oxiarc-deflate` (COOLJAPAN policy; never flate2/miniz/zlib-ng).
- SZIP via `oxiarc-szip` (feature-gated; COOLJAPAN policy).
- HDF5 FFI crates banned workspace-wide via `deny.toml`.
- Zero `unwrap()`/`expect()` in production code paths: a full workspace audit
  (re-verified 2026-08-06 with `cargo clippy --workspace --all-features --
  -W clippy::unwrap_used -W clippy::expect_used -W clippy::panic`, which lints
  library targets only — the exact set of code that ships — and reports
  nothing) finds 338 `.unwrap()` call sites under `crates/*/src`, every one of
  them inside a `#[cfg(test)]` module, a test-only source file (e.g.
  `write/golden_tests.rs`, `write/elem_tests.rs`), or `///`/`//!` rustdoc
  example code — none in shipped library logic. A further 430 sites live in
  the `crates/*/tests/*` integration-test binaries and 34 in the
  `crates/*/benches/*.rs` Criterion benchmarks; neither is compiled into the
  published library. That audit also flagged the one production `.expect()` site
  (`chunked.rs`'s parallel chunk-slice path, `parallel` feature): it has since
  been removed by carrying the already-resolved chunk-record index through
  the pipeline instead of re-deriving it with a second, panic-on-miss map
  lookup. Two narrow, non-`unwrap`/`expect` assertions remain by design: a
  compile-time `assert!` inside a `const fn` (`write/format.rs`) that can only
  fail the *build* if a hardcoded layout-geometry constant stops fitting its
  field width, and one `debug_assert_eq!` (`global_heap_writer.rs`, compiled
  out of release builds).
- Zero clippy warnings: `cargo clippy --workspace --all-features --all-targets`
  is clean (verified 2026-08-06).

---

## License

Apache-2.0 — Copyright COOLJAPAN OU (Team Kitasan)
