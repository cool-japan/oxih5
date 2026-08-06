# oxih5-core — Core types and error for OxiH5

[![Crates.io](https://img.shields.io/crates/v/oxih5-core.svg)](https://crates.io/crates/oxih5-core)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**34 tests passing** (`cargo nextest run -p oxih5-core --all-features`; 22 with default features) · 1 doc test · zero clippy / rustdoc warnings

`oxih5-core` defines the shared, parser-agnostic data model for **OxiH5**, the COOLJAPAN Pure-Rust HDF5 reader/writer. It contains the in-memory representations of an HDF5 file's contents — datatypes, datasets, attributes, groups, links, filter pipelines — plus the crate-wide [`OxiH5Error`] error enum. It deliberately contains **no binary-parsing logic**: the on-disk format readers live in `oxih5-format`, and the user-facing file API lives in the `oxih5` facade.

The headline type is [`Dataset`], a fully-decoded N-dimensional array (raw bytes + shape + [`Dtype`] + attributes + optional max-dims) with a large family of zero-copy typed accessors (`as_f32`, `iter_i64`, …), `slice`, `reshape`, unlimited-dimension introspection, and an optional `ndarray` bridge. This crate is 100% Pure Rust and sets `#![forbid(unsafe_code)]` — its only required dependency is `thiserror`.

## Installation

```toml
[dependencies]
oxih5-core = "0.2.3"

# Optional: enable the ndarray bridge (Dataset::to_array_* — full numeric coverage)
oxih5-core = { version = "0.2.3", features = ["ndarray"] }
```

## Quick Start

```rust
use oxih5_core::{ByteOrder, Dataset, Dtype};

// Build a little-endian f32 dataset from raw bytes (normally produced by oxih5-format).
let data: Vec<u8> = [1.0f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect();
let ds = Dataset {
    data,
    shape: vec![3],
    dtype: Dtype::Float { size: 4, order: ByteOrder::Little },
    attributes: vec![],
    max_dims: None,
};

// Decode to a typed Vec (validates the dtype, returns Err on mismatch).
let values: Vec<f32> = ds.as_f32()?;
assert_eq!(values, vec![1.0, 2.0, 3.0]);

// Or stream values lazily without an intermediate allocation.
let sum: f32 = ds.iter_f32()?.sum();
assert_eq!(sum, 6.0);
# Ok::<(), oxih5_core::OxiH5Error>(())
```

## API Overview

### `Dataset` — fully-decoded N-dimensional array

A public-field struct holding the dataset's raw bytes, shape, [`Dtype`], attributes, and optional max-dims metadata.

| Field | Type | Description |
|-------|------|-------------|
| `data` | `Vec<u8>` | Raw element bytes in row-major (C) order |
| `shape` | `Vec<usize>` | Dimensions in elements (empty ⇒ scalar) |
| `dtype` | `Dtype` | The element datatype |
| `attributes` | `Vec<Attribute>` | Attributes attached to the dataset |
| `max_dims` | `Option<Vec<u64>>` | Maximum dimensions from the HDF5 dataspace; an axis of `u64::MAX` is `H5S_UNLIMITED`; `None` when the source didn't carry max-dims info |

#### Shape / metadata methods

| Method | Description |
|--------|-------------|
| `len()` | Total element count (`1` for a scalar) |
| `is_empty()` | True when `len() == 0` |
| `attrs()` | All attributes as `&[Attribute]` |
| `attr(name)` | Find one attribute by name → `Option<&Attribute>` |
| `max_dims()` | Maximum dimensions as `Option<&[u64]>`, if the source dataspace carried them |
| `is_unlimited()` | `true` if any axis is `H5S_UNLIMITED` (extendable without bound) |
| `unlimited_axes()` | Indices of axes that are `H5S_UNLIMITED` → `Vec<usize>` (empty if none) |
| `slice(ranges)` | Extract a sub-region (`&[Range<usize>]`, one per dim) → new `Dataset` |
| `reshape(new_shape)` | Validate element count and return a re-shaped `Dataset` |

#### Eager typed accessors → `Result<Vec<T>, OxiH5Error>`

Each checks the dtype (and byte order) and returns `TypeMismatch` on mismatch, `DataTruncated` when the buffer length is not a multiple of the element size.

| Method | Decodes from |
|--------|--------------|
| `as_f16()` | 2-byte float → `Vec<f32>` (software half→single conversion) |
| `as_f32()` | 4-byte float |
| `as_f64()` | 8-byte float |
| `as_i8()` / `as_i16()` / `as_i32()` / `as_i64()` | signed integers |
| `as_u8()` / `as_u16()` / `as_u32()` / `as_u64()` | unsigned integers |
| `as_string()` | fixed-length `String` dataset → `Vec<String>` (NUL-trimmed, UTF-8) |

The half→single decode behind `as_f16`/`iter_f16`/`to_array_f16` is also exposed as a standalone function, `oxih5_core::f16_to_f32(bits: u16) -> f32`.

#### Lazy iterator accessors → `Result<impl Iterator<Item = T>, OxiH5Error>`

Decode values on the fly directly from the byte buffer with no intermediate `Vec`.

| Method | Item |
|--------|------|
| `iter_f16()` | `f32` (half-precision decoded to single) |
| `iter_f32()` / `iter_f64()` | `f32` / `f64` |
| `iter_i8()` / `iter_i16()` / `iter_i32()` / `iter_i64()` | signed integers |
| `iter_u8()` / `iter_u16()` / `iter_u32()` / `iter_u64()` | unsigned integers |

#### `ndarray` bridge (feature `ndarray`)

Full numeric coverage — one `to_array_*` per eager/lazy accessor type above.

| Method | Returns |
|--------|---------|
| `to_array_f16()` | `ndarray::ArrayD<f32>` (widened from half-precision) |
| `to_array_f32()` | `ndarray::ArrayD<f32>` |
| `to_array_f64()` | `ndarray::ArrayD<f64>` |
| `to_array_i8()` | `ndarray::ArrayD<i8>` |
| `to_array_i16()` | `ndarray::ArrayD<i16>` |
| `to_array_i32()` | `ndarray::ArrayD<i32>` |
| `to_array_i64()` | `ndarray::ArrayD<i64>` |
| `to_array_u8()` | `ndarray::ArrayD<u8>` |
| `to_array_u16()` | `ndarray::ArrayD<u16>` |
| `to_array_u32()` | `ndarray::ArrayD<u32>` |
| `to_array_u64()` | `ndarray::ArrayD<u64>` |

### `Dtype` enum — HDF5 datatype model

| Variant | Fields | Notes |
|---------|--------|-------|
| `Int` | `size`, `signed: bool`, `order` | Fixed-width integer |
| `Float` | `size`, `order` | IEEE float (2/4/8 bytes) |
| `String` | `fixed_len: Option<usize>`, `charset` | `None` ⇒ variable-length |
| `Compound` | `fields: Vec<CompoundField>` | Struct-like record |
| `Array` | `base: Box<Dtype>`, `dims` | Fixed-size nested array |
| `Enum` | `base: Box<Dtype>`, `members: Vec<(String, i64)>` | Named integer constants |
| `Opaque` | `size`, `tag: String` | Uninterpreted bytes |
| `Reference` | `ref_type: RefType` | Object / region reference |
| `VarLen` | `base: Box<Dtype>` | Variable-length sequence |
| `Bitfield` | `size`, `order` | Bit-packed field |

`Dtype` implements `Display` (human-readable, e.g. `"Float32 LE"`) and exposes `size() -> Option<usize>` (in-memory element size; `None` for variable-length and unbounded-string types).

### Supporting enums

| Type | Variants | Description |
|------|----------|-------------|
| `ByteOrder` | `Little`, `Big` | Endianness; `Display` ⇒ `"LE"` / `"BE"` |
| `Charset` | `Ascii`, `Utf8` | String encoding |
| `RefType` | `Object`, `Region` | Reference flavour |
| `Dataspace` | `Simple { dims, max_dims }`, `Null`, `Scalar` | Dataset shape model |
| `Link` | `Hard { address }`, `Soft { path }`, `External { file, path }` | Group link target |

### Metadata structs

| Type | Key fields | Description |
|------|-----------|-------------|
| `Attribute` | `name`, `dtype: Dtype`, `dataspace: Dataspace`, `data: Vec<u8>` | A named attribute on a dataset or group |
| `CompoundField` | `name`, `offset: usize`, `dtype: Dtype` | One member of a `Compound` datatype |
| `FilterInfo` | `id: u16`, `name: Option<String>`, `flags: u16`, `client_data: Vec<u32>` | A single pipeline filter |
| `FilterPipeline` | `filters: Vec<FilterInfo>` | Ordered filter chain |
| `PropertyList` | `chunk_dims: Option<Vec<u64>>`, `filters: Option<FilterPipeline>`, `fill_value: Option<Vec<u8>>` | Dataset creation properties |
| `Group` | `name`, `children: Vec<(String, Link)>`, `attributes: Vec<Attribute>` | A decoded group node |

`Attribute` additionally exposes scalar-decode helpers for reading common metadata conventions (e.g. CF/NetCDF attributes) without matching on `dtype` by hand:

| Method | Description |
|--------|-------------|
| `as_i64()` | Decode as a scalar `i64`, widening any 1/2/4/8-byte signed or unsigned integer → `Option<i64>` |
| `as_u64()` | Decode as a scalar `u64` (unsigned integer dtypes only) → `Option<u64>` |
| `as_f64()` | Decode as a scalar `f64` (widens `f32`) → `Option<f64>` |
| `as_str_fixed()` | Decode a fixed-length string attribute, NUL-trimmed → `Option<String>` |
| `is_scalar()` | `true` when the dataspace is `Scalar` or holds exactly one element |
| `shape()` | The attribute's dataspace shape as `Vec<u64>` (`Scalar` ⇒ `[]`, `Null` ⇒ `[0]`) |

### `OxiH5Error` variants

The single error type used across the whole OxiH5 stack.

| Variant | Description |
|---------|-------------|
| `Io(std::io::Error)` | Underlying I/O failure (`#[from]`) |
| `BadSignature` | File does not begin with the HDF5 signature |
| `UnsupportedSuperblock(u8)` | Superblock version not handled |
| `UnsupportedHeader(u8)` | Object-header version not handled |
| `UnsupportedDatatype(u8)` | Datatype class not handled |
| `UnsupportedLayout(u8)` | Data-layout class not handled |
| `NotFound(String)` | Named dataset / group / path not found |
| `TypeMismatch` | Typed accessor called on the wrong dtype |
| `DataTruncated` | Byte buffer length not a multiple of the element size |
| `NotImplemented(String)` | Recognised but unsupported feature |
| `Format(String)` | Malformed / unexpected structure |
| `UnsupportedFilter(String)` | Filter id not implemented in Pure Rust |
| `Corrupted(String)` | Checksum / decompression failure |

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `ndarray` | off | Enables `Dataset::to_array_*` returning `ndarray::ArrayD<T>` |
| `dhat-heap` | off | Links `dhat` for heap profiling (development only) |

## Related crates

- [`oxih5-format`](https://crates.io/crates/oxih5-format) — low-level HDF5 binary-format parsers that produce these types.
- [`oxih5`](https://crates.io/crates/oxih5) — the user-facing facade (`open`, `File`, `Group`, `FileWriter`) and the recommended entry point.

Most users should depend on the `oxih5` facade, which re-exports `Dataset`, `Dtype`, `ByteOrder`, `Attribute`, and `OxiH5Error` from this crate.

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)
