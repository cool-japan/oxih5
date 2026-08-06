# oxih5 — The COOLJAPAN Pure-Rust HDF5 facade

[![Crates.io](https://img.shields.io/crates/v/oxih5.svg)](https://crates.io/crates/oxih5)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`oxih5` is the top-level façade crate of **OxiH5**, the COOLJAPAN Pure-Rust HDF5 reader/writer. It reads real HDF5 files — exactly as written by h5py / libhdf5 — and provides a full write path (contiguous, compact and chunked/tiled datasets; DEFLATE/shuffle/Fletcher32 filters; custom fill values; fixed-length/vlen strings; booleans; nested groups; the full attribute set), all with **no libhdf5 FFI, no `*-sys` crates, and no C/Fortran dependencies**. Writer output is verified against h5py 3.16 / libhdf5 2.0.0. It replaces `hdf5-sys` / `hdf5` / `netcdf-sys` on the read path.

This crate is the recommended entry point: it wires together [`oxih5-core`] (the data model) and [`oxih5-format`] (the binary parsers) behind a small, ergonomic surface — `open`, `open_mmap`, `read_dataset`, the [`File`] and [`Group`] navigation handles, and the [`FileWriter`] builder. Production read paths are entirely safe Rust; memory-mapped opening uses one localized, audited `unsafe` block (see [`File::open_mmap`]).

## Installation

```toml
[dependencies]
oxih5 = "0.2.3"

# With the ndarray bridge (Dataset::to_array_f32 / _f64 / _i32):
oxih5 = { version = "0.2.3", features = ["ndarray"] }

# With rayon-parallel chunk assembly:
oxih5 = { version = "0.2.3", features = ["parallel"] }

# With szip (compression id 4) chunk decoding:
oxih5 = { version = "0.2.3", features = ["szip"] }
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
| `dataset_strings(path)` | Decode a vlen- or fixed-length string dataset → `Vec<String>` |
| `dataset_vlen_sequences(path)` | Decode a variable-length *sequence* dataset (datatype class 9) → `Vec<Value>` |
| `root()` | Root [`Group`] handle |
| `group(path)` | Navigate to a group by hierarchical path |
| `contains(path)` | Whether a dataset or group exists at `path` → `bool` |
| `walk(visitor)` | Pre-order traversal; `visitor(full_path, is_group)` |
| `info()` | File-level metadata → [`FileInfo`] |
| `superblock_extension()` | Parse the file's superblock-extension messages (B-tree 'K' Values, Shared Message Table, File Space Info, Driver Info) → `Option<SuperblockExtension>`; only superblock v2/v3 files can have one |

### `FileInfo`

Returned by `File::info()`.

| Field | Type | Description |
|-------|------|-------------|
| `superblock_version` | `u8` | Actual on-disk superblock version, as parsed: `0`, `1`, `2`, or `3` |
| `file_size` | `u64` | Byte size of the file as loaded |
| `offset_size` | `u8` | Superblock `size_of_offsets` (typically 8) |
| `length_size` | `u8` | Superblock `size_of_lengths` (typically 8) |
| `superblock_extension_address` | `Option<u64>` | Address of the superblock extension object header, if the file has one; only versions 2/3 can carry one, so this is always `None` for v0/v1 |

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

A builder that produces minimal, valid HDF5 files (superblock v0, old-style root group). `write_dataset_*` / `create_dataset` write contiguous data; `create_dataset_unlimited` writes a chunked dataset with an unlimited first dimension, tiled into `ceil(shape[d] / chunk_shape[d])` chunks per dimension exactly as libhdf5 tiles one; `set_deflate` DEFLATE-compresses a dataset's chunks, converting a contiguous dataset to chunked storage first if needed; `create_vlen_string_dataset` writes variable-length UTF-8 strings; the `write_*_attr` family attaches scalar or 1-D attributes to a dataset **or** a group at any depth, `"/"` naming the root group. Every method takes a path: a leading `/` is optional, and writing to `"/a/b/x"` creates the groups above `x`, matching h5py's `create_intermediate_group=True` — `create_group` is only needed for a group that holds nothing yet. There is no fixed link limit and no nesting limit short of 64 path components: symbol table nodes chain and the group B-tree grows a level as needed, exactly as libhdf5 does. Each `write_dataset_*` returns `&mut Self` for chaining; most other `create_*` / `set_deflate` / `write_*_attr` methods return `Result<(), OxiH5Error>` (`write_root_str_attr` is infallible and returns `()`).

| Method | Element type / purpose |
|--------|--------------|
| `FileWriter::new()` | Create an empty writer |
| `write_dataset_f32(name, &[f32], shape)` | 32-bit float |
| `write_dataset_f64(name, &[f64], shape)` | 64-bit float |
| `write_dataset_i32(name, &[i32], shape)` | signed 32-bit int |
| `write_dataset_i64(name, &[i64], shape)` | signed 64-bit int |
| `write_dataset_u8(name, &[u8], shape)` | unsigned 8-bit int |
| `write_dataset_i8(name, &[i8], shape)` | signed 8-bit int |
| `write_dataset_i16(name, &[i16], shape)` | signed 16-bit int |
| `write_dataset_u16(name, &[u16], shape)` | unsigned 16-bit int |
| `write_dataset_u32(name, &[u32], shape)` | unsigned 32-bit int |
| `write_dataset_u64(name, &[u64], shape)` | unsigned 64-bit int |
| `create_dataset(name, shape, &Dtype)` | Zero-filled dataset of any fixed-size dtype |
| `create_vlen_string_dataset(name, &[&str])` | Variable-length UTF-8 string dataset |
| `create_dataset_unlimited(name, shape, chunk_shape, &Dtype, data)` | Chunked dataset with an unlimited first dimension |
| `set_deflate(path, level)` | DEFLATE/gzip-compress a dataset (level 0–9); converts contiguous storage to chunked as needed |
| `create_group(path)` | Sub-group at `path`, creating any missing groups above it too (e.g. `create_group("a/b/c")` creates all three levels) |
| `write_group_dataset_f64` / `write_group_dataset_i32(group, name, data, shape)` | Add a dataset inside a sub-group |
| `write_string_attr` / `write_f64_attr` / `write_i64_attr` / `write_i32_attr` / `write_obj_ref_list_attr(obj_path, attr_name, value)` | Scalar attribute on a dataset or group |
| `write_i64_array_attr` / `write_f64_array_attr` / `write_string_array_attr(path, attr_name, values)` | 1-D array-valued attribute (`i64[]` / `f64[]` / string vector) on a dataset or group |
| `write_root_str_attr(name, value)` | String attribute on the root group |
| `write_group_string_attr(group, obj, attr_name, value)` | String attribute on a dataset inside a group |
| `build(path)` | Serialize and write the file to disk |
| `build_to_vec()` | Serialize to an in-memory `Vec<u8>` without touching disk |

Adding a duplicate or invalid name returns `OxiH5Error::Format`. A `write_dataset_*` call whose data length doesn't match `shape.iter().product() × element_size` also returns `OxiH5Error::Format`, instead of writing a corrupt file.

## `write_dataset_in_place` — non-destructive dataset overwrite

A free function — not a `FileWriter` method — that overwrites an existing **contiguous** dataset's raw bytes exactly where they already sit, leaving every other byte of the file untouched. `FileWriter` is the wrong tool for a file oxih5 did not author: rebuilding it from `FileWriter`'s in-memory model discards whatever that model doesn't represent, which is destructive for e.g. a MATLAB v7.3 `.mat` file — `MATLAB_class` attributes, object references, the `#refs#` group, cell arrays and compound structs would all be lost.

```rust,no_run
let bytes: Vec<u8> = [1.0f64, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect();
oxih5::write_dataset_in_place("measurements.mat", "/results/values", &bytes)?;
# Ok::<(), oxih5::OxiH5Error>(())
```

The supplied buffer must match the dataset's allocated size **exactly** — because the byte count never changes, no address recorded anywhere else in the file can shift. Typed wrappers `write_dataset_in_place_f32` / `_f64` / `_i8` / `_i16` / `_i32` / `_i64` / `_u8` / `_u16` / `_u32` / `_u64` additionally verify the dataset's on-disk element type and encode in the dataset's *own* byte order, so a big-endian file is written back big-endian. `dataset_data_extent(path, dataset_path)` returns a `DataExtent { address, size }` giving the writable byte range — useful for sizing a buffer up front, and for asking whether a dataset is overwritable at all, since it applies every precondition the overwrite itself applies except the length check.

Rejected with a typed error, and the file left completely unmodified: a non-contiguous layout (chunked, compact, or virtual — only a contiguous layout is a single flat run of bytes); a filter pipeline (compressed data cannot be overwritten with raw bytes at a fixed size); a variable-length datatype (its data area holds global-heap references, not values); and `dataset_path` naming a group.

## Re-exported types

The data-model types from [`oxih5-core`] are re-exported at the crate root (alongside [`Value`], `SuperblockExtension`, and `BtreeKValues` from [`oxih5-format`], and `DataExtent` defined in `oxih5` itself), so most programs need only `use oxih5::...`:

- `Dataset` — fully-decoded N-dimensional array with typed accessors (`as_f32`, `iter_i64`, `slice`, `reshape`, …)
- `Dtype` — HDF5 datatype enum
- `ByteOrder` — `Little` / `Big`
- `Attribute` — named attribute on a dataset or group
- `Value` — dynamically-typed decoded element (`Int`, `Uint`, `Float`, `Str`, `Sequence`, …) returned by `File::dataset_vlen_sequences`
- `SuperblockExtension` — decoded superblock-extension messages returned by `File::superblock_extension()` (`btree_k`, `shared_message_table_address`, `file_space_info_present`, `file_space_strategy`, `driver_info_present`)
- `BtreeKValues` — decoded "B-tree 'K' Values" message (`indexed_storage_internal_k`, `group_internal_k`, `group_leaf_k`), surfaced via `SuperblockExtension::btree_k`
- `DataExtent` — `{ address, size }` byte range of a dataset's on-disk data, returned by `dataset_data_extent()` to size a buffer for (or check the overwritability of) a `write_dataset_in_place()` call
- `OxiH5Error` — the crate-wide error enum

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `ndarray` | off | Enables `Dataset::to_array_f32` / `_f64` / `_i32` (forwards to `oxih5-core/ndarray`) |
| `parallel` | off | rayon-parallel chunked-dataset assembly (forwards to `oxih5-format/parallel`) |
| `szip` | off | szip (compression id 4) chunk decoding, including RAW mode, via `oxiarc-szip` (forwards to `oxih5-format/szip`) |
| `dhat-heap` | off | dhat heap-profiling instrumentation used by the crate's own memory-profile tests (dev-only) |

## What is supported

- **Read:** superblock v0/v1/v2/v3 (v2/v3 superblock-extension messages — B-tree 'K' Values, Shared Message Table, File Space Info, Driver Info — available via `File::superblock_extension()`); object headers v1/v2; old- and new-style groups; hierarchical paths; hard / soft / external links (including a soft link chained to an external link); contiguous, compact, chunked, and virtual-dataset (VDS) layouts; B-tree v1/v2 and fixed-array chunk indices (an extensible-array index's location is parsed correctly, but its chunk records are not yet decoded — see "Not yet implemented" below); deflate / shuffle / fletcher32 / nbit / scaleoffset filters, plus szip behind the `szip` feature; all 11 datatype classes, including chunked variable-length strings and sequences; dataset and group attributes; sub-region slicing.
- **Write:** contiguous datasets (all ten fixed-width element types), optionally DEFLATE/gzip-compressed via `set_deflate` (which converts them to chunked storage as needed), variable-length string datasets, chunked datasets with an unlimited first dimension genuinely tiled per a caller-supplied chunk shape, sub-groups nested to any depth, and scalar or array attributes on datasets, sub-groups and the root group — see `FileWriter` below for exact limits. Outside `FileWriter`, `write_dataset_in_place` overwrites an existing contiguous dataset's raw bytes in place without touching anything else in the file.
- **Not yet implemented:** a virtual dataset with variable-length elements returns `OxiH5Error::NotImplemented` (fixed-size element types are supported); unmapped virtual-dataset regions always read as zero (non-zero fill values are not applied yet); a soft link returns `OxiH5Error::NotImplemented` only if it targets an external-file *group* (an external-file *dataset* target — a soft → external chain — is resolved); a chunked dataset indexed by an extensible array (HDF5 layout-v4, one unlimited dimension) returns `OxiH5Error::NotImplemented` naming the index and element size — the index block itself is located correctly, but decoding its chunk records needs the dataset's chunk geometry, which is not yet threaded in.

## Errors

All fallible APIs return `Result<_, OxiH5Error>`. See [`oxih5-core`] for the complete variant list (`BadSignature`, `NotFound`, `TypeMismatch`, `DataTruncated`, `Format`, `Corrupted`, `NotImplemented`, …).

## Related crates

- [`oxih5-core`](https://crates.io/crates/oxih5-core) — shared data-model types and the `OxiH5Error` enum.
- [`oxih5-format`](https://crates.io/crates/oxih5-format) — low-level HDF5 binary-format parsers.

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)
