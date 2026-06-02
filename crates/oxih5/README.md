# oxih5 — The COOLJAPAN Pure-Rust HDF5 facade

[![Crates.io](https://img.shields.io/crates/v/oxih5.svg)](https://crates.io/crates/oxih5)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`oxih5` is the top-level façade crate of **OxiH5**, the COOLJAPAN Pure-Rust HDF5 reader/writer. It reads real HDF5 files — exactly as written by h5py / libhdf5 — and provides a minimal write path for flat, contiguous datasets, all with **no libhdf5 FFI, no `*-sys` crates, and no C/Fortran dependencies**. It replaces `hdf5-sys` / `hdf5` / `netcdf-sys` on the read path.

This crate is the recommended entry point: it wires together [`oxih5-core`] (the data model) and [`oxih5-format`] (the binary parsers) behind a small, ergonomic surface — `open`, `open_mmap`, `read_dataset`, the [`File`] and [`Group`] navigation handles, and the [`FileWriter`] builder. Production read paths are entirely safe Rust; memory-mapped opening uses one localized, audited `unsafe` block (see [`File::open_mmap`]).

## Installation

```toml
[dependencies]
oxih5 = "0.1.0"

# With the ndarray bridge (Dataset::to_array_f32 / _f64 / _i32):
oxih5 = { version = "0.1.0", features = ["ndarray"] }

# With rayon-parallel chunk assembly:
oxih5 = { version = "0.1.0", features = ["parallel"] }
```

## Quick Start

### Read a dataset

```rust,no_run
use oxih5::File;

fn main() -> Result<(), oxih5::OxiH5Error> {
    let file = File::open("data.h5")?;

    // List datasets in the root group.
    for name in file.dataset_names()? {
        println!("dataset: {name}");
    }

    // Read a dataset by flat name or hierarchical path.
    let ds = file.dataset("/group1/temperature")?;
    let values: Vec<f32> = ds.as_f32()?;
    println!("{} elements, shape {:?}", values.len(), ds.shape);
    Ok(())
}
```

### One-shot read

```rust,no_run
let ds = oxih5::read_dataset("data.h5", "temperature")?;
let values = ds.as_f64()?;
# Ok::<(), oxih5::OxiH5Error>(())
```

### Write a flat file

```rust,no_run
use oxih5::FileWriter;

let path = std::env::temp_dir().join("out.h5");
FileWriter::new()
    .write_dataset_f32("signal", &[1.0f32, 2.0, 3.0], &[3])?
    .write_dataset_i32("labels", &[0i32, 1, 1], &[3])?
    .build(&path)?;
# Ok::<(), oxih5::OxiH5Error>(())
```

## Entry Points

### `open(path)` → `File`

Read a file into memory (file bytes held in a heap `Vec<u8>`).

```rust,no_run
let file = oxih5::open("data.h5")?;
# Ok::<(), oxih5::OxiH5Error>(())
```

### `open_mmap(path)` → `File`

Memory-map the file so the OS pages in only the regions actually touched — opening a 100 MB+ file is essentially free. The mapping is read-only; the file must not be modified for the lifetime of the handle.

```rust,no_run
let file = oxih5::open_mmap("huge.h5")?;
# Ok::<(), oxih5::OxiH5Error>(())
```

### `read_dataset(path, name)` → `Dataset`

One-shot convenience wrapper around `open` + `File::dataset`.

### `version()` → `&'static str`

Returns the crate version (`env!("CARGO_PKG_VERSION")`).

## `File` — open HDF5 file handle

| Method | Description |
|--------|-------------|
| `File::open(path)` | Open into memory (same as `oxih5::open`) |
| `File::open_mmap(path)` | Open via memory-mapped I/O |
| `File::open_from_bytes(&[u8])` | Open from in-memory bytes (tests / fuzzing) |
| `dataset_names()` | Root-level dataset names → `Vec<String>` |
| `dataset(path)` | Read a dataset by flat name or `/a/b/c` path → `Dataset` |
| `dataset_slice(path, ranges)` | Read a dataset sub-region (`&[Range<usize>]`) |
| `root()` | Root [`Group`] handle |
| `group(path)` | Navigate to a group by hierarchical path |
| `contains(path)` | Whether a dataset or group exists at `path` → `bool` |
| `walk(visitor)` | Pre-order traversal; `visitor(full_path, is_group)` |
| `info()` | File-level metadata → [`FileInfo`] |

### `FileInfo`

Returned by `File::info()`.

| Field | Type | Description |
|-------|------|-------------|
| `superblock_version` | `u8` | Superblock version (currently always 0) |
| `file_size` | `u64` | Byte size of the file as loaded |
| `offset_size` | `u8` | Superblock `size_of_offsets` (typically 8) |
| `length_size` | `u8` | Superblock `size_of_lengths` (typically 8) |

## `Group` — group navigation handle

Obtained from `File::root()` or `File::group(path)`. The `name` field holds the last path segment (`"/"` for root).

| Method | Description |
|--------|-------------|
| `datasets()` | Dataset names directly in this group → `Vec<String>` |
| `groups()` | Sub-group names → `Vec<String>` |
| `dataset(name)` | Read a dataset in this group (one level, no traversal) → `Dataset` |
| `dataset_slice(name, ranges)` | Read a dataset sub-region within this group |
| `attrs()` | Attributes attached to the group → `Vec<Attribute>` |

Both old-style groups (B-tree v1 + SNOD + local heap) and new-style groups (Link messages / fractal heap) are handled transparently; hard links and external file links are followed automatically.

## `FileWriter` — flat-file writer

A builder that produces minimal, valid HDF5 files (superblock v0, old-style root group, contiguous layout, no compression). Constraints: flat files only (no nested groups), up to **8 datasets** per file. Each `write_dataset_*` returns `&mut Self` for chaining.

| Method | Element type |
|--------|--------------|
| `FileWriter::new()` | Create an empty writer |
| `write_dataset_f32(name, &[f32], shape)` | 32-bit float |
| `write_dataset_f64(name, &[f64], shape)` | 64-bit float |
| `write_dataset_i32(name, &[i32], shape)` | signed 32-bit int |
| `write_dataset_i64(name, &[i64], shape)` | signed 64-bit int |
| `write_dataset_u8(name, &[u8], shape)` | unsigned 8-bit int |
| `build(path)` | Serialize and write the file to disk |

Adding a 9th dataset, an empty name, a name containing `/`, or a duplicate name returns `OxiH5Error::Format`.

## Re-exported types

The data-model types from [`oxih5-core`] are re-exported at the crate root, so most programs need only `use oxih5::...`:

- `Dataset` — fully-decoded N-dimensional array with typed accessors (`as_f32`, `iter_i64`, `slice`, `reshape`, …)
- `Dtype` — HDF5 datatype enum
- `ByteOrder` — `Little` / `Big`
- `Attribute` — named attribute on a dataset or group
- `OxiH5Error` — the crate-wide error enum

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `ndarray` | off | Enables `Dataset::to_array_f32` / `_f64` / `_i32` (forwards to `oxih5-core/ndarray`) |
| `parallel` | off | rayon-parallel chunked-dataset assembly (forwards to `oxih5-format/parallel`) |

## What is supported

- **Read:** superblock v0/v2/v3; object headers v1/v2; old- and new-style groups; hierarchical paths; hard / external links; contiguous, compact, and chunked layouts; B-tree v1/v2, fixed-array and extensible-array chunk indices; deflate / shuffle / fletcher32 / nbit / scaleoffset filters; all 11 datatype classes; dataset and group attributes; sub-region slicing.
- **Write:** flat files with up to 8 contiguous, uncompressed datasets (`f32`, `f64`, `i32`, `i64`, `u8`).
- **Not yet implemented:** following soft links, variable-length string decode, and virtual-dataset layout return `OxiH5Error::NotImplemented`.

## Errors

All fallible APIs return `Result<_, OxiH5Error>`. See [`oxih5-core`] for the complete variant list (`BadSignature`, `NotFound`, `TypeMismatch`, `DataTruncated`, `Format`, `Corrupted`, `NotImplemented`, …).

## Related crates

- [`oxih5-core`](https://crates.io/crates/oxih5-core) — shared data-model types and the `OxiH5Error` enum.
- [`oxih5-format`](https://crates.io/crates/oxih5-format) — low-level HDF5 binary-format parsers.

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)
