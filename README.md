# OxiH5

**OxiH5** is the COOLJAPAN Pure-Rust HDF5 reader/writer. It parses and creates
real HDF5 files (as written by h5py / libhdf5) from scratch using only `std`
byte parsing — no `*-sys`, no C libhdf5, no unsafe code in production paths.

OxiH5 replaces `hdf5-sys` / `hdf5` / `netcdf-sys` on the **read** path and
provides a minimal write path for flat contiguous datasets.

---

## Release: 0.1.3 (2026-06-19)

459 unit + integration tests; all pass.  Full workspace
(~20.4 k SLOC of Rust across four crates).

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
superblock.rs       — v0/v2/v3 root group address
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

## What Works (v0.1.3)

### Superblock

- v0 (`libver='earliest'`)
- v2 and v3 (`libver='latest'`)

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

### Filters (chunked)

| Filter | ID | Status |
|---|---|---|
| Deflate / gzip | 1 | DONE (via `oxiarc-deflate`) |
| Shuffle | 2 | DONE |
| Fletcher32 | 3 | DONE |
| SZIP / AEC | 4 | DONE (via `oxiarc-szip`, `szip` feature) |
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

### ndarray bridge

Enable the `ndarray` feature for `Dataset::to_array_f32/f64/i32` returning
`ndarray::ArrayD<T>`.

### Parallel decompression

Enable the `parallel` feature for concurrent chunk decompression via Rayon.

### Write support

`FileWriter` — creates valid HDF5 files readable by h5py and libhdf5.
Supported dtypes: float32, float64, int32, uint8.  v0.1.2 additions:
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
FileWriter::new("output.h5")?
    .write_dataset_f32("temperature", &[1.0f32, 2.0, 3.0], &[3])?
    .write_dataset_i32("index", &[0i32, 1, 2], &[3])?
    .finish()?;
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
- No `unwrap()` in production code paths.

---

## License

Apache-2.0 — Copyright COOLJAPAN OU (Team Kitasan)
