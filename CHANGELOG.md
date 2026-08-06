# Changelog

All notable changes to the OxiH5 workspace are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.4] - Unreleased

## [0.2.3] - 2026-08-06

### Fixed

**Array datatypes were mis-parsed on read**

- **A class-10 (array) datatype's dimensions were read as 8-byte fields.** They
  are 4-byte fields in every version of the class, so an ordinary `(2, 3)`
  array decoded as `[216172782147338240, 72057594037927936]` and its `int32`
  base type as a 67-megabyte unsigned integer — silently, with no error. The
  version-2 form's three reserved bytes and its per-dimension permutation
  indices were not accounted for either, and the version dispatch treated a
  version-1 array as real when the class was *introduced* with version 2 and
  libhdf5 refuses to encode or decode an older one. `datatype::parse_array` now
  implements both real forms (version 2 with reserved bytes and permutation
  indices, version 3 without) and reports a sub-version-2 array rather than
  guessing. Pinned against the datatype message
  `h5t.array_create(NATIVE_INT32, (2, 3))` produces under **h5py 3.16 /
  libhdf5 2.0.0**; before this, oxih5 could not read *any* array-datatype
  dataset written by libhdf5.

**Divide-by-zero / overflow panics and an OOM on crafted or corrupted files**

- **Zero chunk dimension divide-by-zero, closed at every layer.** A crafted
  layout v3/v4 chunked message could claim a chunk dimension of `0`; nothing
  rejected it before a chunk reader divided a dataset coordinate by it,
  panicking. `message::parse_layout` now rejects a zero chunk dimension for
  both layout v3 and v4/v5 chunked encodings at parse time — the single
  validation point every reader relies on — and `chunked_hyperslab::read_chunked_hyperslab`
  carries its own defense-in-depth guard for the same hazard, mirroring the
  fix already in `chunked::read_chunked_slice`/`assemble_chunks_slice`.
- **Zero-stride hyperslab selections now a typed error, not a panic.**
  `DimSelection`'s fields are public for ergonomic construction, so nothing
  stopped a caller building `DimSelection { stride: 0, .. }` — the natural
  mistake, since `stride` is documented `>= 1` but was never enforced. New
  `Hyperslab::validate()` rejects a zero stride in any non-empty dimension and
  is now called from every facade entry point that accepts a caller-supplied
  `Hyperslab` (`scatter_chunk_hyperslab`, `read_chunked_hyperslab`,
  `gather_hyperslab_contiguous`) before the division that used to panic.
- **Overflow-then-undersized-buffer checks in the vlen/global-heap decoders.**
  `decode_vlen_strings`, `decode_vlen_sequences`, `decode_object_refs` and
  `decode_one_value`'s vlen-sequence arm sized their read buffers with plain
  `n_elems * 16` / `seq_len * elem_footprint` multiplication of
  attacker-controlled, on-disk-derived integers. On 32-bit/wasm32 `usize`
  this can wrap to a small value that then passes the length check, letting
  the decode loop slice past the end of the heap object. All four sites now
  use `checked_mul` with a typed `OxiH5Error::Format` on overflow.
- **B-tree v1 chunk index: an out-of-range node/child address bypassed its own
  bounds check via integer overflow.** `btree_v1_chunk::collect`'s guard was
  plain `off + 24 > file_data.len()`; in a release build that addition
  silently wraps for an `off` within 24 of `usize::MAX`, so the wrapped sum
  can come out *smaller* than `file_data.len()` and the check falsely passes
  — the very next line then slices `file_data[off..off + 4]` with the
  still-huge `off` and panics with a "range start index ... out of range"
  message. `off` is a raw on-disk file address (an internal-node child
  pointer, or the index root address itself) with no upstream validation, so
  this is reachable from a corrupted `libver='earliest'` chunked file through
  the public `File::dataset`/`dataset_slice` API. Found by the new
  `fuzz_btree_v1_chunk` target. Every offset computation in `collect` and
  `parse_chunk_key` (node bounds, per-entry key/child offsets, per-dimension
  key offsets) now uses `checked_add`/`checked_mul`, returning a typed error
  on overflow instead of reaching the direct slice.
- **VDS version-1 hyperslab block list: an unbounded block count could OOM the
  process.** `parse_hyperslab_v1_blocks` read `num_blocks` as a raw `u32` off
  disk with no upstream bound and passed it straight to
  `Vec::with_capacity(num_blocks)` — a tiny (39-byte) crafted VDS mapping
  block could request an up-front allocation of tens of gigabytes, aborting
  the process before a single byte was actually read from the (far smaller)
  input buffer. The same "count field used unchecked as `Vec::with_capacity`"
  hazard the Fixed Array chunk-index parser's element count already guards
  against. Found by the new `fuzz_vds` target. `parse_hyperslab_v1_blocks` now
  rejects a block count that could not possibly fit in the reader's remaining
  bytes (each block needs at least `rank * 8` bytes) before allocating for it;
  `Reader` gained a `remaining()` helper to support the check.
- **The one remaining production `.expect()` removed.** `chunked::read_chunked_slice`'s
  parallel (`parallel` feature) code path looked up a chunk's record index
  from an origin coordinate twice — once while filtering to present cells,
  and again inside the parallel `map` closure via
  `chunk_map.get(&origin).expect("origin in map")`. The invariant held today,
  but the redundant lookup meant a future edit to either half could silently
  turn a correctness slip into a worker-thread panic. The resolved index is
  now carried through from the first lookup instead of being re-derived.

### Added

- **Links: soft, external and hard-alias, on write (G011).** The reader has
  resolved all three since 0.1.x — including soft→external chains — with no
  writer counterpart. `FileWriter::create_soft_link`, `create_external_link`
  and `create_hard_link` close that gap, and each is stored the way the format
  actually stores it rather than the way that would be convenient:
  - A **soft link** in an old-style group is a symbol table entry with *cache
    type 2*, the undefined-address sentinel where its object header would be,
    and its target path interned in the group's own local heap beside the link
    names — the layout libhdf5 2.0.0 writes for `f['s'] = h5py.SoftLink(...)`,
    byte for byte. Its target need not exist, and a relative target resolves
    against the group that holds the link.
  - A **hard alias** is an ordinary entry pointing at an object header another
    name already reaches, and it now **raises that header's reference count** —
    libhdf5 writes 2 for an aliased dataset, and leaving it at 1 makes
    `H5Ldelete` on either name free storage the other name still points at. The
    target is resolved against the *finished* layout, so an alias may be
    created before the object it names.
  - An **external link** has no symbol table encoding at all, so creating one
    moves its group to link-message storage — exactly what libhdf5 does when an
    external link is written into a default (`libver='earliest'`) file.
- **New-style (link-message) groups, compact and dense (G013).** A group may
  now store its members as Link messages instead of a symbol table, in both
  forms libhdf5 uses, still inside a superblock-v0 / object-header-v1 file:
  - **Compact**: a Link Info message (0x0002) with both addresses undefined, a
    Group Info message (0x000A), and one Link message (0x0006) per member.
  - **Dense**, past libhdf5's `max_compact` of 8: a **fractal heap writer**
    (`write/fractal_heap.rs`) holding each link message as a managed heap
    object in a root direct block, and a **version-2 B-tree writer**
    (`write/btree_v2.rs`) indexing them by name — a type-5 link name index
    whose records are sorted by the **Jenkins lookup3** hash of the link name,
    because `H5B2__locate_record` binary-searches them and an unsorted index
    does not fail loudly, it just hides links. `write/checksum.rs` implements
    that hash, which is also `H5_checksum_metadata`: the heap header, the heap's
    root direct block, the B-tree header and its leaf node each carry the
    checksum libhdf5 computes, over exactly the byte range libhdf5 covers.
  - **Creation order is preserved.** `FileWriter::set_track_order` gives every
    link an explicit creation-order field and declares the group as tracking
    them, so a reader walking the stored links sees them in the order they were
    made rather than alphabetically. `set_link_storage` selects link messages
    without tracking order.
  - A converted group keeps its symbol table *structures* (an empty B-tree,
    local heap and SNOD) with no symbol table *message*, so the parent's cached
    entry still points at something real — the shape libhdf5 leaves behind when
    it converts a group, verified by reading the result back with libhdf5.
- **Compound (record) datatype datasets (G004).** `create_compound_dataset`
  takes a member list (name, byte offset, type), a record stride and the raw
  row bytes; the class-6 version-1 datatype message it emits is byte-identical
  to what libhdf5 writes for `numpy.dtype([('id', '<i4'), ('value', '<f8')])`.
  Offsets are honoured, never re-packed — a C-padded record and a tightly
  packed one over the same members are different records, and the offsets say
  which one the bytes hold. Overlapping members, a member past the record,
  duplicate or empty names, and unwritable member types are typed errors at the
  call.
- **Variable-length (ragged) sequence datasets (G015).** `create_vlen_sequence_dataset`
  (plus `create_vlen_i32_dataset` / `create_vlen_f64_dataset`) writes a class-9
  subtype-0 sequence over any fixed-size base type, one global-heap object per
  element, sharing the file's one collection set with vlen strings. The
  sequence length in each 16-byte reference counts **elements**, not bytes —
  libhdf5 writes 3 beside a 12-byte heap object for a three-`int32` sequence —
  and an empty element is the all-zero null reference libhdf5 writes. Read back
  with `File::dataset_vlen_sequences`.
- **Array, opaque and bitfield datatype datasets (G016).**
  `create_array_dataset`, `create_opaque_dataset` and `create_bitfield_dataset`.
  The array type is emitted as datatype message **version 2** — the version the
  array class was introduced with, and the only one libhdf5 accepts — with the
  dimensionality byte, three reserved bytes, 4-byte extents and a permutation
  index per dimension. Opaque tags are NUL-padded to eight bytes exactly as
  `h5py.opaque_dtype` produces them.
- **Interop fixture example.** `crates/oxih5/examples/interop_fixtures.rs`
  writes one file per new writer capability so a third-party reader can check
  them; all nine were read back by **h5py 3.16 / libhdf5 2.0.0**, including the
  dense-root-group and `track_order`-root-group cases.

- **Extensible-array chunk indexes are now read, not reported.** The index
  libhdf5 selects for a chunked dataset with exactly one unlimited dimension —
  `create_dataset(..., chunks=..., maxshape=(None, ...))` under
  `libver='latest'`, the ordinary h5py append-able dataset — previously failed
  with a typed `NotImplemented` for every element format libhdf5 actually
  writes. `ea_index` now decodes the whole structure:
  - **Elements are not self-describing**, which is what the old parser assumed.
    The unfiltered client (`H5EA_CLS_CHUNK_ID`) stores a bare 8-byte chunk
    address and takes its size from the uncompressed chunk size; the filtered
    client (`H5EA_CLS_FILT_CHUNK_ID`) stores the address, the stored size in a
    variable-width field whose width is whatever the header's element size
    leaves over, and a 4-byte filter mask.
  - **A chunk's position comes from its element index.** The index is
    `H5VM_array_offset_pre` over *swizzled* coordinates: the unlimited
    dimension is **rotated** to the front (`0,1,2 → 2,0,1` when it is the last
    of three), not swapped with dimension 0 — the two agree only for rank 2 or
    when the unlimited dimension is dimension 1. The chunk grid comes from the
    dataspace's **maximum** dimensions, not its current ones, so that the
    numbering survives the dataset growing. Rank 1 needs no maximum dimensions
    at all; rank ≥ 2 without them is a typed error rather than a guess.
  - **Block traversal is exact.** Data-block element counts now come from the
    `sblk_info` recurrence libhdf5 derives from the header's creation
    parameters (`2^(u/2)` blocks of `2^(ceil(u/2)) × data_blk_min_elmts`
    elements per super block *u*), replacing a heuristic that guessed a
    block's extent from the next recognisable structure in the file. Super
    blocks ("EASB") are located at `14 + ceil(max_nelmts_bits/8)` rather than a
    hard-coded 22, their stated element offset is cross-checked against the
    creation parameters, and their page-init bitmasks
    (`ndblks × ceil(npages/8)` bytes, MSB-first, indexed
    `dblk_idx × npages + page`) are decoded so that the data-block addresses
    after them are found at all. **Paged data blocks** — where one block holds
    more than `2^max_dblk_page_nelmts_bits` elements and is split into pages
    with per-page checksums — are read, and uninitialised pages are skipped
    rather than decoded as allocator leftovers. Every block's back-pointer to
    its own header is verified.
  - **API.** `ea_index::parse_extensible_array` takes an `EaGeometry`
    (chunk shape, current dims, maximum dims, uncompressed chunk bytes) in
    place of a bare rank, and `chunked::{read_chunked, read_chunked_slice}` /
    `chunked_hyperslab::read_chunked_hyperslab` take a
    `chunked::DatasetShape { dims, max_dims }` in place of `dataset_dims: &[u64]`.
    `resolve_chunk_index` now rejects `ExtensibleArray` alongside `BTreeV2` /
    `SingleChunk` / `Implicit` as needing geometry `chunk_records` supplies.
  - **Tests.** 20 unit tests over synthetic arrays encoded the way libhdf5
    encodes them (inline elements, index-block data blocks, secondary blocks,
    paged blocks, uninitialised pages, both clients, narrow stored-size
    fields, the rank-2/3 coordinate rotation, and the structural rejections),
    plus 12 integration tests in `crates/oxih5/tests/ea_index_tests.rs`
    against a new h5py-authored fixture `tests/fixtures/chunked_ea.h5` that
    reaches every level of the structure. `layout_v4_tests.rs`'s
    "reported unsupported" test is now a value assertion.

- **Big-endian datasets and attributes, and half-precision floats — writer
  roadmap items G008 and G012.** Both are read already; neither had a write
  path, so a file oxih5 could parse it could not produce.
  - `FileWriter::write_dataset_numeric(path, values, order, shape)` takes a
    `NumericValues` (`F16`/`F32`/`F64`/`I8`…`U64`) and an explicit
    `ByteOrder`, so eleven element types × two byte orders is one entry point
    rather than twenty-two near-identical methods. The datatype message's
    byte-order bit and the payload bytes are produced from the same `order`
    argument, which is what makes it impossible for them to disagree — the
    failure mode that self-round-trips perfectly and reads back byte-swapped
    everywhere else.  `write_numeric_attr` and `write_numeric_scalar_attr` are
    the attribute counterparts (1-D and scalar dataspace respectively), closing
    the big-endian-attribute half of G006 as well.
  - `create_dataset` and `create_dataset_unlimited` no longer reject a
    big-endian `Dtype`: it now maps to the matching big-endian element type
    instead of erroring.  (0.2.2's guard was the honest behaviour while no
    byte-swap path existed; it is obsolete now.)
  - `oxih5_core::f32_to_f16` is new — round-to-nearest-ties-to-even over the
    24-bit significand, subnormals, saturation to infinity at 65 520, and a NaN
    that never collapses into an infinity.  Unit-tested against IEEE 754
    binary16 bit patterns and by round-tripping all 65 536 of them through
    `f16_to_f32`.
  - Note when reading a half-precision dataset back: `Dataset::as_f32` is for
    binary**32** and returns `TypeMismatch` for a 2-byte float; the accessor is
    `Dataset::as_f16` (or `iter_f16`), which yields `f32` values.  This
    asymmetry predates the writer but is newly reachable now that oxih5 can
    produce such a file.
  - **Verified against h5py 3.16 / libhdf5 2.0.0**, not only against oxih5's
    own reader: `crates/oxih5/tests/be_f16_write_tests.rs` asserts numpy sees
    `>f4`, `>f8`, `>i2`, `>i4`, `>i8`, `>u2`, `>u4`, `>u8`, `>f2` and
    `float16` with exact values, including a 2-D big-endian dataset and
    big-endian array and scalar attributes.

- **`rustfmt.toml` and `clippy.toml`.** Pin `edition = "2021"` (matching
  `workspace.package.edition`; the codebase already matched rustfmt's stable
  defaults, so this changes no formatting) and `msrv = "1.80"` (matching
  `workspace.package.rust-version`, activating `clippy::incompatible_msrv`
  and related MSRV-aware lints).
- **Six new fuzz targets** driving parsers directly with raw bytes, rather
  than only reaching them through a structurally valid whole file (which
  random mutation of `fuzz_file_open` almost never produces): `fuzz_fa_index`,
  `fuzz_ea_index` and `fuzz_btree_v1_chunk` for the three real chunk-index
  parsers; `fuzz_filters` for the read-direction filter pipeline;
  `fuzz_vds` for the VDS mapping-block parser; `fuzz_vlen_values` for the
  vlen/object-reference value decoders. `fuzz_btree_v1_chunk` and `fuzz_vds`
  each found a real crash within seconds of their first run (see Fixed,
  above).
- **Runnable examples.** `crates/oxih5/examples/read_dataset.rs` (open a
  file, then read a dataset whole, via a contiguous slice, and via a strided
  hyperslab) and `write_dataset.rs` (`FileWriter`, chunking, deflate, and
  attributes on both a dataset and a group); `crates/oxinetcdf/examples/write_and_read.rs`
  (`NcFileWriter` → `NcFile`, dimensions, a variable, and global + per-variable
  attributes). All three are self-contained (they write their own fixture to
  a temp path first) and verified with `cargo run --example`.

### Changed

- **`oxih5-format/src/chunked.rs` split into `chunked/{cache,index,read,slice,geometry,tests}.rs`.**
  The file had grown to 1963 lines, 37 under the workspace's 2000-line cap,
  with no natural room left for the next feature. Split along the seams the
  code already implied — the index cache, chunk-index-type resolution,
  whole-dataset reading, hyperslab-range reading, and the low-level I/O/filter/coordinate
  helpers shared by both readers — with every item that used to be reachable
  as `chunked::X` (whether `pub` or crate-visible `pub(crate)`) still
  reachable at that exact path via glob re-exports in `chunked/mod.rs`. Purely
  an internal reorganisation: the public API and every test are unchanged.
- **`oxih5-format/src/ea_index.rs` split into `ea_index/{header,blocks,elements,tests}.rs`.**
  Teaching the extensible-array parser the real libhdf5 structure (see Added)
  more than doubled the file, so it was cut along the same seams the format
  itself has — the array header and its creation parameters, the index/super/
  data block traversal, and the two element clients — with `mod.rs` re-exporting
  every item that was reachable as `ea_index::X` before, so no caller changed a
  `use` path. Both halves of this release's splitting keep every file under the
  workspace's 2000-line-per-file cap.

### Dependencies
- `oxiarc-deflate` / `oxiarc-szip` 0.3.6 → 0.4.1 (upstream releases, two bumps since 0.2.2:
  0.3.6 → 0.4.0 → 0.4.1).

### Documentation

- `crates/oxih5-format/TODO.md`'s status paragraph claimed all four
  chunk-index varieties (B-tree v1, B-tree v2, Fixed Array, Extensible Array)
  were fully supported while the Extensible Array parser still returned a
  typed `NotImplemented`. The claim is now true (see Added), and the paragraph
  says what the Extensible Array reader actually covers. The matching
  "read-side known limitation" entry has been removed from `TODO.md`.
- Removed a stale "hyperslab selections with `block > 1` drop elements on
  chunked datasets" limitation from `TODO.md`'s known-limitations list —
  `test_hyperslab_block2_2d` exercises `block=2`/`stride=2` against a
  chunked+gzip+shuffle 2-D fixture and asserts exact output bytes, so the
  limitation no longer holds (if it ever did).
- Widened `README.md`'s "zero `unwrap()` in production code" claim to also
  cover `.expect()`, and to note the two narrow, by-design assertions that
  remain (a compile-time `assert!` inside a `const fn`, and a
  `debug_assert_eq!` compiled out of release builds).
- Hardened eight fixture-gated integration tests (`vlen_chunked_tests.rs`,
  `vds_tests.rs`, `layout_v4_tests.rs`, `soft_link_old_tests.rs`,
  `read_contig.rs`, `inplace_tests.rs`, `oxinetcdf/fix_ncr_backcompat.rs`,
  `oxinetcdf/src/file.rs::test_read_strings_from_fixture`) that
  silently `return`/skipped when their fixture file was missing, which would
  make the suite pass vacuously (green-but-empty) instead of failing loudly
  if a committed `.h5`/`.nc` fixture were ever lost to a `.gitignore` change
  or a partial checkout. Every fixture they guard is tracked in git today, so
  each now `assert!`s the file exists before proceeding.

## [0.2.2] - 2026-07-22

Every file `FileWriter` and `NcFileWriter` produce is now verified byte-openable
by **h5py 3.16 (libhdf5 2.0.0)** *and* **netCDF4-python 1.7.4** — not merely by
OxiH5's own reader, which had been masking a class of defects where a file this
library accepted was rejected, or silently misread, by libhdf5 or crashed
netCDF-C. The release is the product of a 44-agent differential interop audit:
each agent wrote a permutation of datatype, layout, filter and attribute, opened
the result with both h5py and netCDF4-python, and bisected every divergence to a
minimal reproducer. That audit yielded 33 confirmed conformance defects and 19
capability gaps; this release fixes all 33 defects and closes 9 of the gaps
(the remainder are on the 0.2.3+ roadmap in `TODO.md`).

### Added

- **Shuffle and Fletcher32 filters on write, and true multi-filter pipelines.**
  `FileWriter::set_shuffle(path)` and `set_fletcher32(path)` join `set_deflate`,
  and they compose: a single dataset can carry `shuffle → deflate → fletcher32`
  in one pipeline message, applied in that order on write and inverted in the
  reverse order on read, exactly as h5py's
  `shuffle=True, compression='gzip', fletcher32=True` does. `oxih5_format::filters`
  gained the forward transforms `shuffle` (the exact inverse of the existing
  `unshuffle`) and `append_fletcher32` (the 4-byte little-endian trailer libhdf5
  appends per chunk, byte-for-byte `H5_checksum_fletcher32`). Verified both ways
  against h5py 3.16: any combination of the three filters reads back exactly, and
  a chunk with a corrupted Fletcher-32 trailer makes h5py raise as it should.
  (G001, G007)
- **Fixed-maxshape tiling.** `FileWriter::set_chunking(path, chunk_shape)` tiles a
  dataset of *bounded* shape without promoting dimension 0 to unlimited — the
  earlier `create_dataset_unlimited` path always declared `maxshape[0] = ∞`. A
  `set_chunking` dataset reports its real bounded maxshape to h5py, and a chunk
  extent larger than the corresponding fixed dimension is rejected, leaving the
  writer untouched on error. (G010)
- **Variable-length-reference and compound attributes.** `write_vlen_obj_ref_attr`
  writes an `H5T_VLEN{H5T_REFERENCE}` attribute — the true datatype of a netCDF
  `DIMENSION_LIST` — placing each reference in the global heap;
  `write_ref_index_list_attr` writes the compound
  `{ dataset: object-reference, dimension: u32 }` array a netCDF `REFERENCE_LIST`
  is. h5py reads both back as their named members. (B005, B018)
- **Widened attribute types.** The attribute writers went from four scalar types
  to the full fixed-width set — `write_{i8,i16,i32,i64,u8,u16,u32,u64,f32,f64}_attr`
  — each with a 1-D `_array_attr` sibling (`write_i64_array_attr`,
  `write_f64_array_attr`, `write_string_array_attr`, and the rest), plus
  `write_string_attr_nullterm` for the `H5T_STR_NULLTERM` fixed-string form. A CF
  `valid_range` (i16/u32), a `scale_factor` (f32) or a `flag_values` list now each
  have an encoding where before only i32/i64/f64/string scalars did. (G006)
- **Custom fill values.** `set_fill_value_{f32,f64,i8,i16,i32,i64,u8,u16,u32,u64}(path, value)`
  writes a version-2 fill-value message carrying the caller's value — which h5py
  reports as `dataset.fillvalue` and reads into unwritten/hole elements — where the
  fill-value message had always been the defined-but-zero-length default; a value
  whose type does not match the dataset element type is rejected. (G014)
- **Fixed-length string datasets.** `create_fixed_string_dataset(path, values, shape, width)`
  writes an `H5T_STRING` dataset of fixed width (numpy `S<n>` / netCDF `NC_CHAR`),
  NUL-padded, which h5py reads back at the declared `S`-width; an over-width
  element or a zero width is a typed error. (G003)
- **Boolean datasets.** `write_dataset_bool(path, values, shape)` stores a numpy
  `bool` array the way libhdf5 does — a class-8 enumeration `{ FALSE = 0, TRUE = 1 }`
  over `int8`, byte-pinned against h5py's own `H5Tencode` — so h5py reads it back
  as a `bool` array rather than as `int8`. (G009)
- **Compact-layout datasets.** `set_compact(path)` stores a small dataset's data
  inline in its object header (HDF5 compact layout, class 0) instead of at a
  separate contiguous address; it is idempotent, and rejected for chunked, vlen or
  oversized datasets. (G017)
- **netCDF coordinate variables and conformant dimension linkage.**
  `NcFileWriter` now writes a **coordinate variable** — a dimension and a variable
  sharing one name — as a single dimension-scale dataset carrying its own
  coordinate values, `_Netcdf4Dimid`, `_Netcdf4Coordinates` and the
  `REFERENCE_LIST` of the data variables attached to it, instead of fabricating a
  phantom `[0, 1, …, n−1]` int32 coordinate for every dimension. `DIMENSION_LIST`
  is emitted as `H5T_VLEN{H5T_REFERENCE}` rather than a plain `H5T_REFERENCE`
  array (which segfaulted netCDF-C on open), `REFERENCE_LIST` is written on every
  dimension scale, `_Netcdf4Coordinates` on every multidimensional variable, and
  undefined variable data reads as the netCDF default fill rather than 0. Files
  open cleanly in netCDF4-python 1.7.4 and round-trip through the reader.
  (B005, B010, B011, B018, B019, B020, G002)
- **netCDF reader completeness.** The reader now enumerates every member of an
  old-style group that spans multiple symbol-table nodes, resolves vlen strings
  and sequences across multiple global-heap collections and objects, and sizes
  each decoded array from the dataspace rather than from the raw byte length —
  which had invented a phantom trailing element on padded scalar or odd-length
  integer attributes. Genuine netCDF-4 files, including those written by
  netCDF-C, now round-trip their variables, dimensions and coordinate axes.
  (B012, B013, B014)

### Fixed

**Global heap — libhdf5 conformance**

- **Every small variable-length dataset was unreadable by libhdf5.**
  `GlobalHeapWriter` sized each GCOL collection exactly to its contents (as little
  as 56 bytes), but `H5HG__cache_heap_deserialize` rejects any collection whose
  declared size is below `H5HG_MINSIZE` (4096) *before* it deserialises — so h5py
  raised `OSError: global heap size is too small` on every vlen-string dataset
  under ~4 KB, meaning all `create_vlen_string_dataset` output and every netCDF
  station-name / label array. Collections are now floored at 4096, rounded up to a
  power of two and capped at 65536. (B001)
- **The collection's free space was a size-0 terminator object**, which hangs
  libhdf5's deserializer because it advances its parse cursor by the object's own
  size; the padding is now a real index-0 free-space object whose size spans the
  rest of the collection. (B002)
- **The on-disk object index was truncated with `as u16` / saturating add**,
  corrupting any vlen dataset past a collection's 16-bit slot space; indices are
  32-bit throughout and objects that would overflow a collection start a new one.
  (B008)
- **Vlen-string lengths included the NUL terminator** (`strlen + 1`) where libhdf5
  uses `strlen`, so every string read back one byte long and mis-terminated;
  strings are now stored raw at `strlen`. (B009)

**Attribute encoding**

- **An empty scalar fixed-string attribute emitted a size-0 datatype**, which made
  libhdf5 unable to read *any* attribute on that object; it now carries a
  one-byte width. (B003)
- **Variable-length string datatypes declared ASCII charset** where the data was
  UTF-8 (B017), and **fixed strings used `H5T_STR_NULLTERM` with no room for the
  terminator** where libhdf5 uses `NULLPAD` (B022). Both corrected.
- A duplicate attribute name (B021), an empty attribute name (R004) and a user
  attribute colliding with an auto-generated `DIMENSION_LIST` (R005) are now
  rejected rather than producing an object whose attributes libhdf5 cannot
  iterate; the misleading comments in the float/string dtype encoders were
  corrected (R009).

**Object header**

- **A zero-length contiguous dataset wrote a *defined* data address with size 0**,
  which libhdf5 reports as corruption; an empty dataset now writes the
  undefined-address sentinel, which libhdf5 reads as "empty". (B004)
- Chunked datasets used a fill-value space-allocation time of `Late(2)` where
  libhdf5 uses `Incremental(3)` for chunked storage (B023); `set_deflate` on a
  scalar dataset, which had deferred an error to `build()`, is now rejected up
  front (R008); and `create_dataset` / `create_dataset_unlimited` use checked
  arithmetic instead of panicking (debug) or silently accepting an inconsistent
  shape (release) on a shape / byte-size overflow (R001, R002).

**Chunk index**

- The terminal B-tree v1 chunk key deviated from libhdf5 — a trailing
  element-dimension of 0 instead of the element size, with the multi-dimensional
  real dimensions differing as well (B016); the writer accepted a chunk extent
  larger than a fixed non-dim-0 dimension, producing `chunk[d] > maxdim[d]`
  (B024); the `MAX_CHUNKS` guard ran only *after* every chunk had already been
  materialised and compressed (R003); and `create_dataset_unlimited` silently
  truncated a `chunk_shape` carrying more dimensions than the dataset (R007). All
  corrected.

**Reader**

- The full chunked read ignored the fill-value message, so sparse chunks read back
  as 0 instead of the declared fill value (B007); `parse_attribute_v1` folded
  object-header alignment padding into `Attribute.data` — the 0.2.1 attr-padding
  reader regression — now trimmed (B013); `oxinetcdf`'s scalar accessors returned
  a phantom trailing element on padded scalar / odd-length integer attributes
  (B014); and names with an embedded NUL were accepted, truncating link names in
  the local heap into duplicate or renamed links (B015). All corrected.
- **Variable-length string attributes had no scalar accessor, and multi-object
  global-heap collections at an unaligned file offset dropped every object after
  the first.** A scalar vlen string attribute — the form h5py writes for
  `dset.attrs['units'] = 'm'` and netCDF-C's `setncattr_string` (HDF5 datatype
  class 9) — returned `None` from `AttrView::as_str_fixed`; the new
  `AttrView::as_str` resolves the global-heap reference and returns the string
  for both fixed and vlen scalar forms, and `NcAttribute::as_text` surfaces the
  same. The global-heap reader also aligned each object on an 8-byte boundary
  relative to the *file* rather than to the *collection start*, so a GCOL placed
  at a non-8-aligned address (netCDF-C uses e.g. 4763) reported "object N not
  found" for N ≥ 2 — every multi-byte or multi-attribute vlen string past the
  first. Both corrected. (W3)

**netCDF**

- Fabricated int32 coordinate variables surfaced as phantom coordinate variables
  (B010); multiple variables sharing one unlimited dimension could not be written,
  the shared counter over-growing the dimension (B011); a fixed dimension of size 0
  produced a corrupt, unreadable dataset (R006); and `oxinetcdf`'s build allocated
  coordinate arrays from an unchecked shape product, risking OOM / hang or a debug
  panic (R010). All corrected — see *Added* for the positive-side conventions
  (B005 / B018 / B019 / B020) these repairs enabled.

### Known limitations

- Unchanged from 0.2.1 and still read-side: extensible-array chunk indexes are
  reported, not decoded (typed `NotImplemented`), and hyperslab selections with
  `block > 1` drop elements on chunked datasets.
- Deferred to the 0.2.3+ write-path roadmap (`TODO.md`): compound, big-endian,
  float16, virtual (VDS), region-reference and non-string vlen-sequence datasets;
  array / opaque / bitfield datasets; arbitrary (non-bool) enumerations;
  scaleoffset / nbit / szip filters on write; append / resize and
  modify-existing-file modes; soft / external / hard-alias link creation; and
  new-style (link-message) groups with creation-order preservation.

## [0.2.1] - 2026-07-21

### Added

- **DEFLATE compression on write.** New `FileWriter::set_deflate(path, level)` compresses a dataset with the gzip/DEFLATE filter (HDF5 filter id 1) at any zlib level 0–9. HDF5 can only filter *chunked* data — a filter changes the byte count, and only a chunk index has anywhere to record the new one — so the call converts a contiguous dataset to chunked storage as it sets the filter, exactly as h5py's `compression=` argument does; an already chunked dataset keeps its geometry and its unlimited dimension, so a tiled compressed dataset is `create_dataset_unlimited` with a chunk shape followed by `set_deflate`. Compression runs during the **layout** pass, not the emit pass: a compressed dataset's length is not derivable from anything the caller supplied, and the chunk B-tree key has to record it before any byte is written. A new `write/payload.rs` owns that, with one `data_size()` both passes reserve and write against; a new `write/pipeline.rs` encodes the filter pipeline message (0x000B). Compression uses the COOLJAPAN Pure-Rust `oxiarc-deflate` via the existing `oxih5_format::filters::deflate_compress`, so `oxih5` gained **no new dependency**. Variable-length string datasets are refused with a typed error rather than accepted: their elements are global-heap references, which the read side declines to decode through a filter pipeline, so accepting would produce a file only this library could not read back. Verified against h5py 3.15.1 / libhdf5 1.14.6, which reports `dset.compression == 'gzip'` and `dset.compression_opts == level` and decompresses the values exactly.
- **Chunked datasets are genuinely tiled.** `create_dataset_unlimited`'s `chunk_shape` is now honoured: the dataset is cut into `ceil(shape[d] / chunk_shape[d])` chunks per dimension, each compressed on its own, each keyed by its own B-tree entry. Chunks that hang over an edge are stored **full-size** with the fill value in the overhang, which is what the chunk dimensions in the layout message promise and what libhdf5 itself stores. Chunk origins are in elements, are always multiples of the chunk extent — `H5D__btree_decode_key` divides by it — and are ordered row-major over the chunk grid with dimension 0 most significant, the order confirmed against the B-tree keys of an h5py-authored file rather than inferred. The index grows levels past 64 chunks (a node's capacity), so the layout message points at the tree **root** rather than at the first leaf; trees up to 2²⁴ chunks are supported and refused with a typed error beyond. Verified through h5py at ragged 1-D (`[7]` in chunks of `[3]`), ragged 2-D (`[5,3]` in `[2,2]`, and each dimension ragged alone), ragged 3-D (`[3,4,5]` in `[2,3,2]`), a two-level index (150 chunks) and a three-level one (8500 chunks, all of which `iter_chunks()` enumerates).
- A new `deflate_chunked.h5` fixture in `tests/gen_fixtures.py` and `tests/deflate_chunked_tests.rs` verify the **read** side against gzip-compressed chunked datasets authored by h5py rather than by our own writer — including a ragged 1-D and 2-D tiling and a two-filter `shuffle` + `gzip` pipeline whose inversion order matters.

- **In-place dataset overwrite.** New `oxih5::write_dataset_in_place(path, dataset_path, &[u8])` replaces an existing dataset's data bytes where they already sit, leaving every other byte of the file untouched — the equivalent of the C `hdf5` crate's `Dataset::write_raw`. Rebuilding a file with `FileWriter` is not a substitute: it reconstructs the file from the caller's in-memory model and so destroys `MATLAB_class` attributes, object references, the `#refs#` group, cell arrays and compound structs that a MATLAB v7.3 `.mat` file carries. The supplied buffer must match the dataset's allocated size **exactly**; that invariant is what makes the operation sound, because an unchanged byte count means no address recorded anywhere in the file can shift. Typed wrappers `write_dataset_in_place_{f32,f64,i8,i16,i32,i64,u8,u16,u32,u64}` additionally verify the dataset's element type and encode in the dataset's *own* byte order, so a big-endian file is written back big-endian; `dataset_data_extent()` reports the writable range (`DataExtent { address, size }`) and answers whether a dataset is overwritable at all. Non-contiguous layouts (chunked, compact, virtual), filter pipelines (message 0x000B), variable-length datatypes and group paths are each rejected with a typed error before the file is opened for writing, so a refused call cannot modify anything. Verified end to end against h5py 3.15.1 / libhdf5 1.14.6: h5py writes a MATLAB-shaped file, oxih5 overwrites one dataset in place, h5py confirms the new values and that the attributes, object reference and `#refs#` group survived.

- **Nested groups.** `FileWriter` groups were strictly one level deep: `create_group` rejected any name containing `/`, and so did the four other insertion points, so `a/b/c` was unwritable. Every entry point now takes a **path**. A leading `/` is optional, `"/"` names the root group, and writing to a path **creates the groups above it** — `write_dataset_f64("/a/b/x", …)` creates `a` and `a/b`, matching h5py's `create_intermediate_group=True`. `create_group("a/b/c")` does the same. Malformed paths (`"a//b"`, `"a/"`, `"a/./b"`, `"a/../b"`, `""`) are typed errors rather than names, and a path deeper than 64 components is refused rather than recursed. `write_group_dataset_f64`, `write_group_dataset_i32` and `write_group_string_attr` are retained as thin wrappers, so `oxinetcdf` is unaffected. Verified against libhdf5 1.14.6 / h5py 3.15.1: a three-level hierarchy walked by `f.visit()` and read by full path, and a sub-group whose own links span three symbol table nodes at two different depths.
- **Attributes on groups.** A sub-group's object header was a hard-coded 40 bytes — a 16-byte prefix and one Symbol Table message — so there was nowhere for an attribute to go, and only the root group could carry any. Any group, at any depth, now carries attributes of every supported kind; read back through `Group::attr_views()` or, in h5py, `f['/a/b'].attrs`. The Symbol Table message stays first in the header, which is what keeps `H5G__stab_find` recognising the object as a group at all.
- **Non-string root-group attributes.** `write_root_str_attr` took a `&str` and was the only way to reach the root group, so an integer, float or array global attribute — a CF `geospatial_bounds`, a `valid_range` — could not be written at all. `write_string_attr` / `write_f64_attr` / `write_i64_attr` / `write_i32_attr` / `write_obj_ref_list_attr` now resolve their `path` argument to **either a dataset or a group, at any depth**, with `"/"` meaning the root group; a bare name still finds a root dataset first, so existing callers are unaffected. `write_root_str_attr` is now literally `write_string_attr("/", …)` and keeps its infallible signature.
- **Array-valued attributes.** New `write_i64_array_attr`, `write_f64_array_attr` and `write_string_array_attr` write 1-D attributes; only scalars were representable before, so a CF `valid_range` or `flag_meanings` list had no encoding. Strings are stored at the common width of the longest element, NUL-padded, which is how both oxih5 and libhdf5 recover the individual lengths. Verified through h5py.

### Changed

- **Chunked datasets got bigger — deliberately, and by a fixed amount.** A chunk B-tree node grew from a node sized for its contents to the width libhdf5 reads: **80 → 2096 bytes** for a 1-D dataset, 96 → 2616 for 2-D, 3136 for 3-D, plus `chunk_node_size(rank)` again per extra index level past 64 chunks. That is a real, visible size regression on every chunked or compressed dataset, and it is not optional — the old width is exactly what made those files unopenable (see *Fixed*). The practical consequence is that **compression is a net size loss below roughly 8 KB of data**: a 2096-byte index is more than gzip will save on a small array, so `set_deflate` is worth reaching for on bulk arrays and not on coordinate variables. The multi-feature golden fixture went from 6 968 to 11 504 bytes for this reason and no other; its two chunked datasets account for the whole 4 536-byte difference.
- **The writer's group model is recursive.** `GroupDesc` and the root group's ad-hoc `root_str_attrs: Vec<(String, String)>` collapsed into a single `GroupNode { name, datasets, groups, attrs }`, and the root is planned and emitted by the same recursive code as every other group — it differs only in being pinned to address 96 and mirrored into the superblock's root symbol-table entry. The per-group ordering the symbol-table work established is preserved at every level: sort links by name, build the local heap, chunk into SNODs, plan the B-tree levels, assign addresses, emit. A sub-group's cached B-tree and local heap addresses are read out of the child's finished plan when its parent's symbol table entry is built, so the writer still back-patches nothing. Two consequences worth noting: resolved attributes now live *inside* the plan rather than in a parallel array, so an attribute that reserved space but was not emitted is no longer representable; and `resolve_str_attrs`, the second, string-only copy of the raw → resolved attribute conversion, is gone. The restructure was verified byte-for-byte neutral before the new features were added — the `w1x_golden_bytes_multi_feature` fixture hashed identically across it. Internally the layout engine split into `write/tree.rs` (object model and path resolution), `write/plan.rs` (pass one) and `write/build.rs` (pass two).

- **`crates/oxih5/src/lib.rs` split (internal only).** lib.rs had reached 1999 of the project's 2000-line cap. Its contents moved to `file.rs` (the `File` handle), `group_handle.rs` (the `Group` handle), `reader.rs` (path navigation and object-header dataset assembly) and `slicing.rs` (lazy range and hyperslab reads), leaving lib.rs at 388 lines; `links.rs` now uses explicit imports instead of a `use super::*` glob. The public API is unchanged — `oxih5::File` and `oxih5::Group` keep their paths via re-export, and the full test suite passes unaltered before and after.
- **The writer's root and sub-group item limits are gone.** `FileWriter` previously rejected the 65th root item ("maximum 64 items at root") and the 33rd dataset in a sub-group; both were artefacts of emitting a single symbol table node per group, and both were documented public constraints. Groups now grow without a fixed ceiling — a resource guard replaces the format limit and refuses a single group of more than 2^24 links with a typed `OxiH5Error::Format` rather than attempting the allocation. `test_write_capacity_exceeded` is replaced by `test_write_three_hundred_datasets_roundtrip`.
- **Listing order for files written by `FileWriter` is now name order, not declaration order.** `File::dataset_names()`, `Group::datasets()` and `Group::groups()` return links in the order the symbol table stores them, which is now sorted — matching what libhdf5-written files already produced. `oxinetcdf` is unaffected: it resolves variables and dimensions by name and by the `_Netcdf4Dimid` attribute, never by list position.

### Fixed

- **Every chunked dataset this writer had ever produced was unreadable by libhdf5.** A chunk B-tree node was sized for the one chunk it held — `24 + 2*key + 8`, so 80 bytes for a 1-D dataset — but libhdf5 computes the node image size it reads *before* it has seen the node, from `24 + two_k*8 + (two_k+1)*key` where `two_k` is `2 × btree_k[H5B_CHUNK]`. For a superblock v0 or v1 file that `K` is never read from the file: it is `HDF5_BTREE_CHUNK_IK_DEF`, a compile-time **32**. libhdf5 therefore asked for 2096 bytes, got 80, and refused the file with `addr overflow, addr = 3000, size = 2096, eoa = 3112`. Nodes are now emitted at that full width with the slots past `entries_used` zero-filled. Two further defects in the same node: the terminal key `key[K]` was left entirely zero, so `H5D__btree_cmp3` saw an empty search range and reported the chunk as unallocated even once the node was the right size — it now carries the dataset extent rounded up to a chunk boundary, with `nbytes = 0` — and the node's level byte is now set, so a multi-level index is possible at all.
- **A 2-D unlimited variable read back mostly zeroes.** `create_dataset_unlimited` stored the whole dataset as one chunk regardless of `chunk_shape`, which only affected the *declared* chunk dimensions, and a short `chunk_shape` was completed with 1s. `oxinetcdf` passes `&[shape[0].max(1)]` for a variable of any rank (`crates/oxinetcdf/src/write.rs:597`), so a 2-D variable of shape `[5, 3]` declared chunk dimensions `[5, 1]` — five elements — while the single chunk on disk held all fifteen. Every reader clamps a chunk to its declared volume, so twelve of the fifteen values silently read back as zero from a file that opened without complaint. A short chunk shape is now completed from the *dataset shape*, and the tiling is actually performed, so declaration and storage cannot disagree. `oxinetcdf` is fixed without any change to `oxinetcdf`; its existing test asserted only `shape == [5, 3]` and never read the values.
- **Written files with 9 or more links in a group were unreadable by libhdf5.** `FileWriter` emitted one oversized symbol table node per group (sized for 64 root entries, 32 per sub-group) while the superblock declared libhdf5's default `leaf_node_K = 4`. libhdf5 sizes a SNOD's on-disk image from that field alone — `8 + 2*K*40` = 328 bytes — and only then reads the node's own `nsyms`, so the 9th entry made `H5G__ent_decode_vec` run off the end of the image buffer (`H5Gcache.c line 188 in H5G__cache_node_deserialize(): unable to decode symbol table entries`). Files with 1, 2, 4 or 8 root items opened; 9, 10, 12 and 16 did not. Links are now chunked into 328-byte nodes and indexed by a group B-tree that grows a level once 32 of them fill one node, with every node written at its full declared width and the slots past `entries_used` zero-filled. Verified against libhdf5 1.14.6 / h5py 3.15.1 at 0, 1, 8, 9, 32, 255, 256, 257, 300 and 1000 root links, and for sub-groups.
- **Datasets declared out of name order were silently lost.** `H5G__node_found` binary-searches a symbol table node with the entry names resolved through the local heap, so an unsorted node does not fail loudly — it hides links. Two files differing only in declaration order behaved differently: `["aaa","bbb"]` opened fine, `["bbb","aaa"]` raised `KeyError: object 'bbb' doesn't exist`, while `list(f.keys())` (a linear walk) still listed both. A realistic NetCDF-style file that declared `values` before `lat` lost `values` entirely. Group links are now sorted by raw name bytes before the local heap is built, so heap offsets and B-tree keys come out monotonic.
- Group B-tree keys are now correct for multi-node trees: `key[i+1]` is the greatest name under child `i` (`H5G__node_cmp3` treats `key[i]` as an exclusive lower bound), and sibling addresses are linked within each level.
- **A dataset could shadow a group of the same name.** `create_group` checked for both an existing group *and* an existing dataset, but `write_dataset_*`, `create_vlen_string_dataset`, `create_dataset_unlimited` and `create_dataset` checked only against other datasets — so `create_group("x")` followed by `write_dataset_f64("x", …)` produced two links named `x` in one symbol table, which the format cannot represent and which each reader resolves by whichever entry it happens to meet first. One `name_taken()` now answers the question for both kinds at every insertion point, in every group rather than only at the root.
- **Object references silently became `0xffff_ffff_ffff_ffff`.** `write_obj_ref_list_attr` resolved its targets against root datasets only, and an unresolvable name fell back to the undefined-address sentinel via `.unwrap_or(u64::MAX)` — so a misspelt `DIMENSION_LIST` entry, or a target inside a group, produced a file that opened cleanly and carried a dangling reference, failing (if ever) at dereference time and naming nothing useful. The name → address map is now built over the whole plan tree, keyed by path from the root — with or without a leading `/`, and by bare name for root-level objects, since those are the same key — and covers groups as well as datasets. A target that names nothing is an `OxiH5Error::Format` from `build()` naming both the attribute and the target.

- **Soft links inside old-style (symbol-table) groups did not resolve.** `File::dataset("alias")` on a file written by h5py with its *default* `libver` failed with `Corrupted("object header offset 18446744073709551615 too large")` — the HDF5 undefined-address sentinel (`u64::MAX`) being passed on as if it were a real file offset. In an old-style group a soft link is not a link message with a type field: it is a symbol-table entry whose **cache type is 2**, whose object-header address is that sentinel, and whose actual value is a path stored in the group's local heap at the offset held in the first four bytes of the entry's scratch pad. The SNOD parser read only the first 16 bytes of each 40-byte entry, so cache type 2 was invisible to it and every soft link looked like a hard link to an impossible address. This affected ordinary files rather than a corner case, because h5py's default `libver` produces old-style groups; new-style (Link-message) groups already resolved soft links correctly.

  `snod::SymTabEntry` now additionally carries a decoded `SymTabCache` (`None` / `Group { btree, heap }` / `SoftLink { link_value_offset }` / `Unknown`), added alongside the existing fields so the other call sites are unaffected, and `group::find_entry` / `group::list_entries` return a `SymTabLink` that distinguishes hard from soft. `group::find_dataset` keeps its signature and resolves hard links only, but now reports a soft link as a typed `NotImplemented` naming its target instead of yielding the sentinel.

  Rather than teach the old-style path about soft links separately, the two group styles were unified behind one resolver: `GroupRef` identifies a group by object header plus how its members are indexed, `lookup_child` is the single per-segment step, and `resolve_path_target` walks a path through groups of *either* style. `File::dataset`, `File::group`, `File::header_addr_of`, `Group::dataset`, `Group::group`, `Group::datasets`/`groups`, `File::walk`, `dataset_slice` and `dataset_hyperslab` all go through it, so a soft link resolves identically however it is reached, and the existing soft → external chain handling is preserved. Three consequences beyond the reported bug: a soft link whose value is **relative** is now resolved against the group that holds it, as libhdf5 does, instead of against the file root; soft links are listed by `Group::datasets()` / `Group::groups()` and classified by what they point at, matching the behaviour new-style groups already had; and the cycle guard is keyed by `(group, path)` rather than by path alone, since the same relative value in two groups names two different objects. A dangling soft link is a typed `NotFound` naming the missing target and a cycle is a typed error — neither panics. New fixture `tests/fixtures/soft_links_old.h5` (h5py, default `libver`) and 11 tests in `tests/soft_link_old_tests.rs` cover link-to-dataset, link-to-group, link-to-nested-dataset, soft → soft chains, relative values, dangling, cyclic, listing, slicing and `walk`.

- **Data layout message version 4 was unsupported.** `parse_layout` handled versions 1 and 3 only, so every dataset in a file written with `libver='latest'` failed with `Format("unsupported layout version=4 class=1")` (contiguous) or `class=0` (compact) and could not be read at all. Versions 3 and 4 encode compact and contiguous data identically, so both are now accepted by the same arms. This also repaired three pre-existing failures in `tests/vds_tests.rs`, whose committed source fixtures were regenerated with libhdf5 1.14.6 and therefore carried v4 contiguous layouts.

  The version-4 **chunked** decoder was rewritten against the specification. It had been derived empirically from fixed-array files and assumed 1-byte chunk dimensions and a single index parameter byte for every index type; in fact the dimensions are `dimensionality` little-endian values of a width given by the message's "dimension size encoded length" byte, and the indexing-type information block that precedes the index address is 0 bytes for implicit and unfiltered single-chunk, 12 for a filtered single chunk, 1 for a fixed array, 5 for an extensible array and 6 for a version-2 B-tree. Reading the index address from the fixed-array offset in every case is why extensible-array and B-tree-v2 datasets failed with nonsense addresses such as `EA: header at 0x7530a100404 exceeds file length`, and why a chunk dimension above 255 (which widens the encoding to 2 bytes) failed the same way. Two indexing types are now supported that previously reported "unknown HDF5 index type": **single chunk** (type 1), including the filtered form whose stored size and filter mask are carried inline in the layout message because there is no index structure to hold them, and **implicit** (type 2), whose chunks are laid out back to back in row-major chunk order at the message's address. An index address of `u64::MAX` — a dataset created but never written — now reads as fill value rather than erroring, matching libhdf5.

  Two chunk index readers were fixed at the same time, both of which had been validated only against synthetic fixtures that encoded the wrong layout. **Version-2 B-tree** records were decoded as `address + size + filter_mask + offsets` for both record types; an *unfiltered* record (type 10) actually stores only `address` followed by the scaled coordinates, and the coordinates in both types are **chunk-grid indices**, not element offsets. The result was that every unfiltered v2-B-tree chunked dataset decoded as zeros. **Extensible array** headers had their index-block address read at byte 28, omitting the six 8-byte counters that precede it, so the address was garbage for every real file; it is at byte 60.

  New fixture `tests/fixtures/layout_v4.h5` (h5py, `libver='latest'`) carries one dataset per layout class and per chunk indexing type, and `tests/layout_v4_tests.rs` asserts the decoded values.

### Known limitations

- **Extensible-array chunk indexes are reported, not decoded.** With the layout message and the array header now read correctly, decoding an extensible array still requires interpreting its *elements*, and this crate's parser expects each element to be a self-describing chunk record (`address + size + filter mask + one offset per dimension`). A real libhdf5 extensible array stores a bare 8-byte chunk address when the dataset is unfiltered, or `address + a variable-width stored size + filter mask` when it is filtered, and derives each chunk's position from its linear index in the array combined with the dataset's chunk shape — which is not currently plumbed into `ea_index`. Reading such a dataset (a chunked dataset with exactly one unlimited dimension) now fails with a typed `NotImplemented` naming the extensible-array index and the element size, rather than the previous nonsense-address error or a silent buffer of zeros.
- **Hyperslab selections with `block > 1` drop elements on chunked datasets.** Pre-existing and independent of the above: `scatter_chunk_hyperslab` returns zeros for the elements a block covers beyond its first, for every chunk index type including layout-v3 version-1 B-trees. `DimSelection { start: 1, stride: 1, count: 1, block: 2 }` yields `[1, 0]` where the equivalent `count: 2, block: 1` correctly yields `[1, 2]`.

## [0.2.0] - 2026-07-18

### Added

- **Superblock version 1 support**: files written with HDF5 superblock v1 (the "transitional" format that inserts 4 extra bytes after the File Consistency Flags, shifting the Base Address and Root Group Symbol Table Entry by +4 bytes) now parse correctly instead of failing with `OxiH5Error::UnsupportedSuperblock(1)`. New `oxih5_format::superblock::parse_v1`; covered by 4 new unit tests (`test_superblock_v1_parse`, `test_superblock_v1_nonzero_base_and_root`, `test_superblock_v1_too_short`, `test_superblock_v1_bad_offset_width`).
- **Superblock extension parsing** (versions 2/3): new `oxih5_format::superblock::{SuperblockExtension, BtreeKValues}` and `read_superblock_extension()` decode the superblock extension object header's B-tree 'K' Values (0x0013), Shared Message Table (0x000F), File Space Info (0x0018), and Driver Info (0x0014) messages. Exposed at the facade level via `oxih5::File::superblock_extension()` and re-exported as `oxih5::{SuperblockExtension, BtreeKValues}`. 9 new unit tests plus a new integration test `test_superblock_extension_none_on_v0_file`.
- `Superblock::version` and `FileInfo::superblock_extension_address` fields expose the actual parsed superblock version and extension address: `File::info().superblock_version` now reports the real on-disk version (0/1/2/3) instead of being hardcoded to `0`.

### Fixed

- **Object header v2 continuation blocks (OCHK) with creation-order tracking**: `parse_v2_ochk_block` always assumed `track_creation_order = false` for messages inside continuation blocks, so an object header that tracks creation order (flags bit 2 set) and spills messages into an OCHK continuation block had those messages misdecoded — the parser read the 2-byte creation-order field as message body data instead of skipping it. The owning object header's `track_creation_order` flag is now threaded through `parse_v2_block` → `parse_v2_ochk_block` (including nested continuations). New regression test `test_parse_messages_v2_ochk_continuation_creation_order`.

---

## [0.1.4] - 2026-07-17

### Added

- **Virtual Dataset (VDS) reading**: datasets with HDF5 layout class 3 ("virtual") now resolve end-to-end instead of always failing with `NotImplemented`. New `oxih5_format::vds` module — `VdsMapping` / `VdsEntry` / `VdsSelection` (`None`/`All`/`Hyperslab`), `parse_vds_mapping` / `parse_vds_block` (decodes the global-heap mapping block: version 0 and version-1 filename-deduplicated encodings, hyperslab selection versions 1 and 3), `selection_element_offsets`. `oxih5::File` opens each source dataset referenced by a mapping (same file, or an external file resolved relative to the containing file's directory) and scatters the selected regions into the virtual dataset's buffer. 9 new unit tests in `vds.rs` plus 4 new integration tests in `tests/vds_tests.rs`.
- **Chunked variable-length dataset reads**: chunked datasets of `Dtype::VarLen` or variable-length string elements can now be read in full and via `dataset_slice`/hyperslab selections; every chunked vlen dataset previously failed with "variable-length element size not supported". A new `on_disk_elem_footprint` helper accounts for the fixed 16-byte global-heap-reference footprint of vlen elements, and filters that are meaningless on vlen data (shuffle, fletcher32, nbit, scaleoffset) are now explicitly rejected. 2 new integration tests in `tests/vlen_chunked_tests.rs`, including a hyperslab selection that straddles a chunk boundary.
- **`File::dataset_vlen_sequences(path)`**: new public method decoding a variable-length *sequence* dataset (datatype class 9) into `Vec<oxih5_format::values::Value>`; works for both contiguous and chunked layouts.
- **szip RAW-mode chunk decoding**: new public `oxih5_format::filters::apply_pipeline_sized` accepts the caller's known decoded chunk size, letting the szip filter (id 4) decode "RAW" streams that carry no HDF5 framing header — previously RAW-mode szip chunks always failed with `UnsupportedFilter`; `apply_pipeline` is now a thin length-agnostic wrapper around it. New `szip` Cargo feature on the `oxih5` crate (`szip = ["oxih5-format/szip"]`) exposes this at the top level. 2 new unit tests.
- **Fractal-heap I/O-filtered root direct blocks**: fractal heaps whose root direct block is stored filtered on disk (e.g. deflate/shuffle) can now be decoded. The heap header's I/O-Filters-Encoded-Length field is now read at its correct offset and width (previously read as the wrong single byte, so a filtered root block was never detected and any I/O-filtered heap was rejected outright); indirect-block roots with filters remain `NotImplemented`. New roundtrip test `test_io_filter_root_block_roundtrip`.
- **Soft link → external link chains**: a soft link whose target is itself an external link (e.g. `/soft` → `/ext` → `other.h5:/payload`) now resolves through `File::dataset` instead of failing with `NotImplemented`. Covered by the new `soft_link_through_external_link` integration test.
- Expanded top-level `oxih5` crate documentation with two runnable doctests (write/read round-trip; hyperslab slice read).
- 20 new tests this release; all 486 tests pass (`cargo test --workspace --all-features`).

### Changed

- **`oxih5_format::message::LayoutInfo::VirtualDataset`** (breaking): the `entry_count: u32` field is renamed to `heap_index: u32` and reinterpreted as the global-heap *object index* of the serialized VDS mapping block — the old field never actually held an entry count (the count lives inside the heap block itself), which is why VDS layouts could not be resolved before this release. Code matching this enum variant must update the field name.
- `oxiarc-deflate` updated from `0.3.3` to `0.3.6`; `oxiarc-szip` updated from `0.3.3` to `0.3.6` (workspace dependencies).
- Link-resolution helpers (soft-link and external-link navigation) extracted from `oxih5/src/lib.rs` into a new internal `links.rs` module to keep individual source files under the project's 2000-line limit; no behavior change beyond the new soft→external resolution above.

### Fixed

- **On-disk vlen (global-heap) reference layout**: `oxih5_format::values::parse_vlen_ref` decoded the 16-byte vlen reference with the wrong byte layout (object index read as `u16` from bytes 4–6, heap address from bytes 8–16, bytes 6–8 treated as reserved padding). The correct HDF5 encoding (matching libhdf5/h5py) is `length(4) + heap_address(8) + object_index(4)`; the bug silently misdecoded vlen string/sequence data in real HDF5 files whenever the referenced global-heap collection had a nonzero address — existing unit tests only ever exercised heap address 0, which masked it. The writer's counterpart (`write_vlen_ref` in `oxih5/src/write/mod.rs`) is corrected to match, so files written by `FileWriter` are now byte-layout-compatible with other HDF5 implementations; `GlobalHeapWriter`'s module docs updated accordingly.
- `FileWriter::write_dataset_f32/f64/i32/i64/u8` now validate that the supplied data length matches `shape.iter().product() × element_size`, returning `OxiH5Error::Format` on mismatch instead of risking a corrupt on-disk file or a later out-of-bounds panic during `build()`. New test `test_write_shape_data_mismatch_rejected`.

### Security

- A crafted/corrupted chunked-dataset layout claiming a zero chunk dimension no longer panics with a divide-by-zero when computing overlapping chunk-grid cells; `read_chunked_slice` and `assemble_chunks_slice` now return a typed `OxiH5Error::Format` instead. New regression test `test_chunked_slice_zero_chunk_dim_errors`.
- A crafted/corrupted Fixed Array header claiming an implausible "Number of Elements" count (e.g. `u64::MAX`) is now rejected — bounded to 16Mi (`1 << 24`) elements and cross-checked against the bytes actually remaining in the file — before being used as a `Vec::with_capacity` argument, which would otherwise panic (capacity overflow) or attempt a huge allocation. New test `test_fa_oversized_element_count_rejected`.

---

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

[0.2.3]: https://github.com/cool-japan/oxih5/releases/tag/v0.2.3
[0.2.2]: https://github.com/cool-japan/oxih5/releases/tag/v0.2.2
[0.2.1]: https://github.com/cool-japan/oxih5/releases/tag/v0.2.1
[0.2.0]: https://github.com/cool-japan/oxih5/releases/tag/v0.2.0
[0.1.4]: https://github.com/cool-japan/oxih5/releases/tag/v0.1.4
[0.1.3]: https://github.com/cool-japan/oxih5/releases/tag/v0.1.3
[0.1.2]: https://github.com/cool-japan/oxih5/releases/tag/v0.1.2
[0.1.1]: https://github.com/cool-japan/oxih5/releases/tag/v0.1.1
[0.1.0]: https://github.com/cool-japan/oxih5/releases/tag/v0.1.0
