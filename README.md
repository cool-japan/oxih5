# OxiH5

**OxiH5** is the COOLJAPAN Pure-Rust HDF5 reader/writer. It parses and creates
real HDF5 files (as written by h5py / libhdf5) from scratch using only `std`
byte parsing — no `*-sys`, no C libhdf5, no unsafe code in production paths.

OxiH5 replaces `hdf5-sys` / `hdf5` / `netcdf-sys` on the **read** path and
provides a minimal write path for flat contiguous datasets.

---

## Release: 0.2.1 (2026-07-21)

682 unit + integration tests (`--all-features`; 661 with default features);
all pass.  Full workspace (~27.9 k SLOC of Rust across four crates).

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

---

## What Works (v0.2.1)

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

`FileWriter` — creates valid HDF5 files readable by h5py and libhdf5.
Supported dtypes: float32, float64, int8, int16, int32, int64, uint8,
uint16, uint32, uint64 (all ten fixed-width element types).  v0.1.2 additions:
- Multi-group HDF5 files (`create_group`, `write_group_dataset_f64/i32`)
- Unlimited / chunked datasets (`create_dataset_unlimited`)
- Root group string attributes (`write_root_str_attr`)
- Variable-length string datasets backed by GlobalHeap (`create_vlen_string_dataset`)

`NcFileWriter` (in `oxinetcdf`) — creates NetCDF-4-compliant HDF5 files with
full DIMENSION_SCALE / DIMENSION_LIST / `_Netcdf4Dimid` convention encoding.
Supports `def_dim`, `def_dim_unlimited`, `def_var`, `put_var_f64/i32`,
`put_vara_f64/i32` (unlimited append), `def_var_strings/put_var_strings`,
`put_att_str`, and `set_classic_mode`.

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
- Zero `unwrap()` in production code paths: a full workspace audit
  (2026-07-21) found 404 total `.unwrap()` call sites, and every one is
  confined to test modules (`#[cfg(test)]`), the `oxih5/tests/write_tests.rs`
  integration-test binary, `benches/*.rs` Criterion benchmarks, or
  `///`/`//!` rustdoc example code — none in shipped library logic.

---

## License

Apache-2.0 — Copyright COOLJAPAN OU (Team Kitasan)
