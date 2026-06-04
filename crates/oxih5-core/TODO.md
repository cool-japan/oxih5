# oxih5-core TODO

## Status
Foundation types in place: `Dtype` (Int/Float), `ByteOrder`, `Dataset` with `as_f32`/`as_f64`/`as_i32` conversion, and `OxiH5Error` with 9 error variants. ~120 SLOC production code.

## Core Implementation
- [x] Add `Dtype::String` variant for variable-length and fixed-length string datasets (30 SLOC)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Add `Dtype::Compound { fields: Vec<CompoundField> }` for compound datatypes (60-80 SLOC)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Add `Dtype::Array { base: Box<Dtype>, dims: Vec<usize> }` for array datatypes (30 SLOC)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Add `Dtype::Enum { base: Box<Dtype>, members: Vec<(String, i64)> }` for enum datatypes (40 SLOC)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Add `Dtype::Opaque { size: usize, tag: String }` for opaque datatypes (20 SLOC)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Add `Dtype::Reference { ref_type: RefType }` for object/region references (30 SLOC)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Add `Dtype::VarLen { base: Box<Dtype> }` for variable-length sequences (25 SLOC)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Add `Dtype::Bitfield { size: usize, order: ByteOrder }` for bitfield datatypes (20 SLOC)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Implement `Dataset::as_u8`, `as_u16`, `as_u32`, `as_u64`, `as_i8`, `as_i16`, `as_i64` converters (120 SLOC)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Implement `Dataset::as_f16` converter (half-precision float, manual decode) (40 SLOC)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Implement `Dataset::as_string` for string datasets (60 SLOC)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Add `Dataspace` enum: Simple(dims, max_dims), Null, Scalar (40 SLOC)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Add `Attribute` struct: name + dtype + data (30 SLOC)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Add `FilterPipeline` struct describing compression/filter chain (50 SLOC)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Add `PropertyList` struct for dataset creation properties (40 SLOC)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Add `Link` enum: Hard(addr), Soft(path), External(file, path) (30 SLOC)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Add `Group` struct with children and attributes (40 SLOC)
  - **Plan:** oxih5-core type expansion — 2026-05-25

## API Improvements
- [x] Implement `Dataset::slice(ranges)` for reading sub-regions of data
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Add `Dataset::reshape(new_shape)` validation utility
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Add typed accessors returning `ndarray::ArrayD<T>` behind `ndarray` feature
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Implement `Display` for `Dtype` showing human-readable type descriptions
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Add `OxiH5Error::UnsupportedFilter(String)` variant for unknown filters
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Add `OxiH5Error::Corrupted(String)` variant for checksum failures
  - **Plan:** oxih5-core type expansion — 2026-05-25

## Testing
- [x] Test `as_f32`/`as_f64`/`as_i32` with all valid byte orders
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Test type mismatch error messages for each typed accessor
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Test `Dataset::len()` and `is_empty()` with various shapes (scalar, 1D, multi-D)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Test `Dtype` equality and cloning
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Test `OxiH5Error` Display output for all variants
  - **Plan:** oxih5-core type expansion — 2026-05-25

## Performance
- [x] Benchmark typed conversion (as_f32 etc.) for large datasets (1M+ elements)
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Consider zero-copy slice access instead of `to_vec()` in converters
  - **Plan:** oxih5-core type expansion — 2026-05-25
- [x] Profile memory usage for `Dataset` holding large data vectors
  - **Plan:** oxih5-core type expansion — 2026-05-25

## Integration
- [x] Ensure oxih5-format produces `Dtype` and `Dataspace` compatible with oxih5-core types
  - **Done:** oxih5-format datatype.rs produces all oxih5-core Dtype variants — 2026-05-25
- [x] Ensure oxih5 facade constructs `Dataset` correctly from format-level parsed components
  - **Done:** oxih5 facade constructs Dataset from oxih5-format parsed layout/datatype/dataspace correctly — 2026-05-25
- [x] Coordinate with SciRS2 / NumRS2 for ndarray bridge requirements — implement full numeric `to_array_*` coverage now; external SciRS2 coordination stays open (planned 2026-06-02)
  - **Goal:** `Dataset::to_array_{u8,u16,u32,u64,i8,i16,i64,f16}` behind the `ndarray` feature, returning `ndarray::ArrayD<T>` (f16 widens to `ArrayD<f32>`). Closes the SciRS2/NumRS2 array-surface gap for all numeric dtypes.
  - **Design:** Extend the `#[cfg(feature = "ndarray")] impl Dataset` block (currently lines 937–980 of `src/lib.rs`) with 8 new methods, each mirroring the existing `to_array_{f32,f64,i32}` pattern: call the matching scalar accessor (`as_u8`/…/`as_f16`), normalise empty shape to `vec![1]`, `ArrayD::from_shape_vec(IxDyn(&shape), values)`, map err to `OxiH5Error::Format`. Optionally use a private declarative macro `impl_to_array!` to stay DRY. No `unwrap`, no `#[allow]`.
  - **Files:** `crates/oxih5-core/src/lib.rs` (extend ndarray impl block).
  - **Tests:** `#[cfg(feature = "ndarray")]`-gated tests in `crates/oxih5-core`: round-trip each new accessor on a 2-D `Dataset`, assert `.shape()` and element values; one type-mismatch error case.
  - **Risk:** Low (pure additive). `to_array_f16` naming explicitly documents the widen-to-f32 contract.
