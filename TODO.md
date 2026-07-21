# OxiH5 Project TODO

## Status — 0.2.1 (2026-07-21)

Functional read/write HDF5 library (~27.9 k SLOC Rust, 682 tests with
`--all-features` / 661 with default features, all pass).
Supports superblock v0/v1/v2/v3 (+ v2/v3 superblock extension parsing:
B-tree K values, shared message table, file space info, driver info via
`File::superblock_extension()`), object header v1/v2 (+continuation, with
creation-order now correctly threaded through OCHK continuation blocks),
B-tree v1/v2, local heap, SNOD, fractal heap (FRHP+FHDB+FHIB), extensible/fixed array
chunk indices, all 11 datatype classes, dataspace v1/v2, attributes (0x000C
v1/v2/v3), filter pipeline (deflate/shuffle/fletcher32/nbit/scaleoffset), fill
value, global heap, and **contiguous + compact + chunked + virtual** layouts.  New-style
groups (libver='latest'): Link Info (0x0002), Link messages (0x0006), fractal
heap traversal for large groups (>8 links), B-tree v2 type-5 name index.
Verified against real h5py fixtures including superblock v3, 20-dataset large
groups, 2-D partial-edge chunks.  Write support via `FileWriter`.

---

## Milestones

### M0 — Skeleton (DONE)
- [x] Workspace compiles clean
- [x] `oxih5-core` types: Dataset, Dtype, ByteOrder, OxiH5Error
- [x] `oxih5-format` module scaffold
- [x] `oxih5` facade stubs: open, read_dataset, File::dataset, File::dataset_names
- [x] `deny.toml` bans all HDF5 FFI crates (tree-wide, no exceptions)

### M1 — Full read chain (DONE)
- [x] Superblock v0 parsing
- [x] Object header v1 message parsing with continuation support
- [x] Dataspace, datatype (int/float), contiguous layout message parsing
- [x] B-tree v1 group traversal
- [x] Local heap and SNOD symbol table parsing
- [x] Group listing and dataset lookup
- [x] End-to-end file open and dataset read

### M2 — Chunked + ndarray (DONE, 2026-05-25)
- [x] B-tree v1 (node type 1) + v2 traversal, extensible/fixed array chunk indices
- [x] Chunked data layout assembly
- [x] Shuffle + fletcher32 + nbit + scaleoffset filters
- [x] Gzip/deflate decompression via oxiarc-deflate (NEVER flate2/miniz)
- [x] SZIP/AEC via oxiarc-szip (szip feature)
- [x] ndarray 0.17 feature gate on oxih5 facade crate

### M3 — Extended datatypes + new-style groups (DONE, 2026-05-25)
- [x] All 11 Dtype variants in oxih5-core + format-level parsers
- [x] Superblock v2/v3 parsing
- [x] Object header v2 + OCHK continuation
- [x] Link Info (0x0002) + Link messages (0x0006) (hard/soft/external)
- [x] Fractal heap (FRHP + FHDB + FHIB) for large groups
- [x] B-tree v2 type-5 name index
- [x] Attribute message (0x000C) v1/v2/v3 parsing
- [x] Attribute struct + Dataset::attrs()/attr() facade methods

### M4 — mmap + lazy + fuzz (DONE, 2026-05-25)
- [x] Memory-mapped I/O (open_mmap)
- [x] Lazy chunk decompression
- [x] Parallel chunk reading (`parallel` feature via rayon)
- [x] Fuzz corpus (4 cargo-fuzz targets: fuzz_superblock, fuzz_header,
      fuzz_message, fuzz_file_open)
- [x] Dataset::slice — multi-dimensional sub-region extraction
- [x] Dataset::reshape — zero-copy shape reinterpretation
- [x] ChunkIndexCache — shared cache across multiple reads
- [x] Hierarchical path navigation (/group/subgroup/dataset)
- [x] External link resolution (opens referenced file)
- [x] Group handle API: File::root, File::group, Group::datasets, Group::groups,
      Group::dataset, Group::attrs, Group::dataset_slice
- [x] File::walk, File::contains, File::info

### M5 — Write support + full release (DONE, 2026-06-01)
- [x] FileWriter: flat HDF5 creation, contiguous layout, h5py-verified
- [x] Version bump to 0.1.0; CHANGELOG.md created
- [x] README.md updated to reflect all completed milestones
- [x] cargo check/clippy/nextest all clean (0 errors, 0 warnings, 273 tests pass)
- [x] oxih5-core dry-run publish passes

### M9 — Superblock v1 + extension parsing (DONE, 0.2.0, 2026-07-18)
- [x] Superblock v1 ("transitional" format) parsing: new
      `oxih5_format::superblock::parse_v1`; 4 new unit tests
      (`test_superblock_v1_parse`, `test_superblock_v1_nonzero_base_and_root`,
      `test_superblock_v1_too_short`, `test_superblock_v1_bad_offset_width`)
- [x] Superblock extension parsing (v2/v3): new
      `oxih5_format::superblock::{SuperblockExtension, BtreeKValues}` +
      `read_superblock_extension()`, decoding B-tree 'K' Values (0x0013),
      Shared Message Table (0x000F), File Space Info (0x0018), and Driver
      Info (0x0014) messages; exposed via `oxih5::File::superblock_extension()`
      (re-exported as `oxih5::{SuperblockExtension, BtreeKValues}`); 6 new
      unit tests + `test_superblock_extension_none_on_v0_file` integration test
- [x] `Superblock::version` / `FileInfo::superblock_extension_address` expose
      the real on-disk superblock version — `File::info().superblock_version`
      no longer hardcoded to `0`
- [x] Fixed object-header v2 continuation block (OCHK) creation-order bug:
      `track_creation_order` was always assumed `false` inside OCHK blocks,
      misdecoding messages when the owning object header tracks creation
      order; now threaded through `parse_v2_block` → `parse_v2_ochk_block`
      (including nested continuations); regression test
      `test_parse_messages_v2_ochk_continuation_creation_order`
- [x] cargo fmt clean; clippy --all-features --all-targets -D warnings clean;
      rustdoc -D warnings clean; 494 tests pass (`--all-features`) / 473 with
      default features; cargo audit 0 vulnerabilities; cargo +nightly udeps
      0 unused dependencies

---

## Open Items (post-0.1.0)

### Testing
- [ ] Benchmark against hdf5-rust (FFI) for read throughput — blocked on
      COOLJAPAN Pure Rust policy (hdf5-rust requires C FFI)
  - **Refinement (2026-06-03):** FFI baseline impossible under deny.toml; buildable substitute = absolute read-throughput bench (item A5, 0.1.1).
- [x] Profile and optimize hot paths (superblock + header parsing) (done 2026-06-02)
  - **Goal:** Committed criterion micro-benchmarks for superblock + object-header parsing, plus allocation reductions on those hot paths.
  - **Design:** New `oxih5-format/benches/parse_bench.rs` with in-memory fixtures from existing test builders. Bench groups: `parse_superblock_{v0,v2,v3}`, `parse_header_{v1,v2}_{1msg,64msg}`. Optimizations: T1 pre-size messages Vec (`with_capacity`); T3 defer v2 continuation HashSet until a 2nd OCHK block appears; T4 direct slice reads in superblock.
  - **Files:** `oxih5-format/Cargo.toml`, `oxih5-format/benches/parse_bench.rs`, `oxih5-format/src/header.rs`, `oxih5-format/src/superblock.rs`
  - **Tests:** Existing parser unit tests stay green; bench compiles with `cargo bench --no-run -p oxih5-format`.
  - **Risk:** Capacity hints and deferred allocation cannot change parse results — existing tests guard correctness.

### 0.1.1 — Reader value-decoding completeness

- [x] Pre-split `oxih5-core/src/lib.rs` into sibling modules via `splitrs` (done 2026-06-03)
  - **Goal:** lib.rs (1734 lines) split to provide headroom for A3/A4 additions, behavior unchanged.
  - **Design:** Use `splitrs` to extract Dataset conversion impls + Attribute impls into `dataset_convert.rs`, `attribute.rs`, mod-declared from lib.rs. No logic change.
  - **Files:** `crates/oxih5-core/src/lib.rs` (+ new sibling modules)
  - **Tests:** existing core tests stay green; `cargo nextest run -p oxih5-core --all-features` + clippy clean.
  - **Risk:** split must be behavior-preserving; guarded by existing suite.

- [x] `oxih5-format/src/values.rs`: vlen value decoding via the global heap (done 2026-06-03)
  - **Goal:** Decode vlen STRING and vlen-of-base SEQUENCE values by resolving 16-byte on-disk pointers through `GlobalHeap` (first real use of dead-code heap).
  - **Design:** New `values.rs` with `Value` enum, `parse_vlen_ref`, `heap_object_bytes` (base_address-adjusted, u16-narrowed index, per-collection parse cache), `decode_vlen_strings`, `decode_vlen_sequences`. Empty vlen → empty slice, not error.
  - **Files:** `crates/oxih5-format/src/lib.rs`, `crates/oxih5-format/src/values.rs` (NEW)
  - **Tests:** `values.rs` unit tests with in-memory GCOL; upgrade `read_contig.rs` vlen tests to exact-value assertions.
  - **Risk:** u32→u16 narrowing, base_address overflow, collection cache correctness.

- [x] Object-reference value decode + `File::object_at`/`dataset_at` public resolver (done 2026-06-03)
  - **Goal:** Decode 8-byte object references into target addresses and expose a public address→object API.
  - **Design:** `decode_object_refs` in `values.rs`; `pub enum ObjectKind { Dataset(Dataset), Group(Group) }`, `File::object_at(addr)`, `File::dataset_at(addr)` in facade — wrapping existing private `read_dataset_from_object_header`/`read_attributes_from_header`.
  - **Files:** `crates/oxih5-format/src/values.rs` (append), `crates/oxih5/src/lib.rs`
  - **Tests:** synthetic `decode_object_refs` unit test; `File::object_at` round-trip via `FileWriter`.
  - **Risk:** group-vs-dataset discrimination; undefined refs (u64::MAX); region refs partial.

- [x] Compound value decoding + `decode_compound`/`decode_one_value` central dispatcher (done 2026-06-03)
  - **Goal:** Split `Dtype::Compound` raw element bytes into per-field typed `Value`s, incl. vlen/ref members.
  - **Design:** `decode_one_value` central dispatcher + `decode_compound_element` + `decode_compound` in `values.rs`; depth-guarded recursion; element stride from `data.len()/nelem`. Core pure fast-path: `Dataset::compound_fields`, `Dataset::field_bytes`.
  - **Files:** `crates/oxih5-format/src/values.rs` (append), `crates/oxih5-core/src/` (split modules)
  - **Tests:** synthetic compound-byte unit test; upgrade compound fixture tests to value assertions.
  - **Risk:** trailing element padding; nested vlen/ref in compound.

- [x] Typed `Attribute` accessors + `AttrView` + `dataset_strings` facade (done 2026-06-03)
  - **Goal:** Mirror `Dataset::as_*` for attributes; resolve layering: core has no file bytes → heap-dependent accessors on `AttrView` facade wrapper.
  - **Design:** Core `Attribute`: `as_i64/as_u64/as_f64/as_str_fixed/is_scalar/shape` (file-independent). Facade `AttrView<'a>`: `as_strings` (fixed+vlen), `as_object_refs`, `as_compound`; via `Group::attr_views`/`attr_view`, `File::attr_views`. `File`/`Group::dataset_strings` for vlen-string datasets.
  - **Files:** `crates/oxih5-core/src/` (Attribute impl), `crates/oxih5/src/lib.rs` (AttrView, dataset_strings, re-exports)
  - **Tests:** upgrade `with_attrs`/`multi_attr` fixture tests to exact decoded values; assert vlen string attrs and fixed-width attrs.
  - **Risk:** `AttrView<'a>` lifetime threading; facade file-size watch.

- [x] Absolute read-throughput benchmarks (done 2026-06-03)
  - **Goal:** Report OxiH5 read throughput in MB/s for contiguous + chunked layouts (buildable substitute for policy-blocked hdf5-rust FFI baseline).
  - **Design:** Extend `crates/oxih5/benches/read_bench.rs` with `Throughput::Bytes` criterion groups: `throughput_contiguous_f64`, `throughput_chunked_gzip`, full-read vs mmap. No FFI.
  - **Files:** `crates/oxih5/benches/read_bench.rs`
  - **Tests:** `cargo bench --no-run -p oxih5` compiles; existing read tests green.
  - **Risk:** benches excluded from nextest; clippy `-D warnings` on bench code.

### Integration
- [ ] Coordinate with SciRS2 for ML model weight reading — blocked on SciRS2
      API stabilization; oxih5 already supports float32/float64/int32 arrays
  - **Refinement (2026-06-03):** oxih5-side prerequisite (typed Attribute accessors + vlen-string decode, items A1/A4) lands 0.1.1; SciRS2 API coordination remains upstream-blocked.
- [x] Scope `oxinetcdf` conventions layer atop OxiH5 — separate subcrate (done 2026-06-03)
  - **Goal:** `oxinetcdf` Slice 1: read a NetCDF-4 file and resolve dims/vars/axis-linkage from HDF5 conventions (DIMENSION_SCALE, DIMENSION_LIST, _Netcdf4Dimid, REFERENCE_LIST).
  - **Design:** New `crates/oxinetcdf/` workspace member. `NcFile`/`NcGroup`/`NcDimension`/`NcVariable`/`NcAxis`/`NcAttribute`/`NcError`. Resolver consumes `File::object_at`/`dataset_at` (A2) and `AttrView::as_object_refs`/`as_text`/`as_i64` (A4). Tests skip-guarded (python/netCDF4 optional).
  - **Files:** `crates/oxinetcdf/` (new subcrate), workspace `Cargo.toml`
  - **Prerequisites:** A2, A4.
  - **Tests:** pure unit tests always run; skip-guarded E2E tests generate fixtures at runtime via python3+netCDF4.
  - **Risk:** resolver E2E unverified in envs without netCDF4 (accepted, documented).

  #### oxinetcdf — deferred follow-ups (post-Slice-1)
  - [x] Deep group hierarchy: recursively resolve subgroups into `NcGroup` trees
        (done 2026-06-10: `resolver.rs` `resolve_group_deep` with MAX_GROUP_DEPTH=64, cycle
        detection via visited-path set, `NcGroup.children: Vec<NcGroup>`; 4 new unit tests)
  - [x] Cross-group shared dimensions: DIMENSION_LIST refs across group boundaries
        (done 2026-06-10: two-phase scan — `collect_global_dims` (Phase 1) walks all groups and
        builds addr→`GlobalDim` registry via new `File::header_addr_of`; `resolve_dim_list` (Phase 2)
        resolves refs via local cache → global registry → lazy `attrs_of` → phony; `NcAxis` gains
        `group_path` + `is_unlimited` fields)
  - [x] Dataset `max_dims` exposure on `oxih5::Dataset` for exact `is_unlimited` detection
        (done 2026-06-10: `DataspaceInfo::max_dims`, `Dataset::max_dims/is_unlimited/unlimited_axes`;
        8 new tests; parse_dataspace updated to read flags+max-dims block; all 350 tests pass)
  - [x] Attrs-only metadata accessor on oxih5 (`File::attrs_of`) to avoid loading variable data during resolution
        (done 2026-06-10: `File::attrs_of(addr: u64)` delegates to existing
        `read_attributes_from_header`; returns `NotFound` for `u64::MAX`; 2 new unit tests)
  - [x] User-defined types: enum, vlen, opaque, compound variables
        (done 2026-06-10: `NcType` enum in `types.rs`; `From<&Dtype>` covers all 11 Dtype variants;
        `NcVariable::nc_type()` convenience method; 12 unit tests in types.rs)
  - [x] NC_STRING variable data decode (vlen UTF-8) via oxih5 vlen dataset path
        (done 2026-06-10: `NcVariable::read_strings(nc)` delegates to `File::dataset_strings`;
        `NcAttribute::new_with_view` eagerly decodes vlen strings at open time so `as_text()` works;
        `NcFile::h5()` public accessor added)
  - [x] `_FillValue`-aware masked reads (apply fill value → `Option`/NaN)
        (done 2026-06-10: `apply_fill_mask<T>`, `apply_fill_mask_f32/f64` (bit-exact NaN safe);
        `NcVariable::read_f64_masked` (NaN for fill), `read_i64_masked` (Option for fill);
        priority: `_FillValue` attr first; 5 unit tests)
  - [x] CF conventions: `coordinates`, `bounds`, `grid_mapping` semantic linking
        (done 2026-06-10: `NcGroup::coordinates_of/bounds_of/grid_mapping_of`; `cf.rs` module with
        `parse_cf_name_list/cf_group_prefix/cf_var_name` helpers; supports CF-1.7 `group:var` form;
        9 unit tests in cf.rs + 7 in model.rs)
  - [x] NetCDF-4 writing: `NcFileWriter` emitting DIMENSION_SCALE/CLASS/NAME/_Netcdf4Dimid/DIMENSION_LIST
        (done 2026-06-10: W0a attribute writing on `FileWriter` — `write_string_attr`, `write_f64_attr`,
        `write_i64_attr`, `write_i32_attr`, `write_obj_ref_list_attr`; SNOD capacity increased to 64;
        C9 `NcFileWriter` with `def_dim/def_var/put_var_f64/put_var_i32/put_att_str/close`; 7 round-trip
        tests; 420 total tests, zero warnings; all changes uncommitted in working tree)
  - [x] Sub-group creation: `FileWriter::create_group` + `write_group_dataset_f64/i32` + `write_group_string_attr`
        (done 2026-06-10: W0b — SNOD cache_type=1 scratch-pad for group OH/B-tree/heap; group SNOD 1288 bytes;
        round-trip tests `w0b_create_group_and_dataset_roundtrip` + `w0b_group_groups_listing`)
  - [x] Unlimited/chunked dataset layout: `FileWriter::create_dataset_unlimited`
        (done 2026-06-10: W0c — B-tree v1 type-1 single-chunk node; chunked layout v3 with max_dim[0]=u64::MAX;
        1-D and 2-D round-trip tests `w0c_unlimited_dataset_roundtrip` + `w0c_2d_unlimited_roundtrip`)
  - [x] Root group string attributes: `FileWriter::write_root_str_attr`
        (done 2026-06-10: C11 infrastructure — dynamic OH size via `compute_root_oh_size`; round-trip tests
        `root_str_attr_roundtrip` + `root_str_attr_does_not_break_dataset_reads`)
  - [x] Unlimited-dimension append in `NcFileWriter`: `def_dim_unlimited` + `put_vara_f64/i32`
        (done 2026-06-10: C10 — rewrite-on-append strategy; `var_trailing_stride` helper; unlimited coord vars
        use `create_dataset_unlimited`; tests `c10_unlimited_dim_append_roundtrip` + `c10_unlimited_2d_append_roundtrip`)
  - [x] NETCDF4_CLASSIC strict-mode: `NcFileWriter::set_classic_mode`
        (done 2026-06-10: C11 — writes `_nc3_strict = ""` on root group; removed from reserved-attr filter so
        it appears in `NcGroup::attrs`; tests `c11_classic_mode_nc3_strict_attribute` + `c11_non_classic_no_nc3_strict`)
  - [x] GlobalHeap (GCOL) writer + NC_STRING variable support
        (done 2026-06-10: W0d — `GlobalHeapWriter` in `oxih5-format/src/global_heap_writer.rs` (GCOL
        serialiser; re-exported from `oxih5_format::GlobalHeapWriter`); `ElemType::VlenStr` + `DatasetDesc::vlen_strings`
        + `DatasetDesc::data_len()` in write/mod.rs; GCOL appended at EOF; 16-byte vlen refs in dataset data area;
        `FileWriter::create_vlen_string_dataset`; `NcFileWriter::def_var_strings` + `put_var_strings` + NcType::String
        arm in build_bytes; 12 new tests (`w0d_gcol_round_trip`, `w0d_gcol_empty_string`,
        `w0d_gcol_with_coexisting_numeric_dataset`, 6 GlobalHeapWriter unit tests,
        `nc_string_variable_round_trip`, `nc_string_var_with_empty_strings`); 440 total tests, zero warnings)
  - [x] Nested groups: paths at every writer entry point, with intermediate groups created on demand
        (done 2026-07-21: W1b — `GroupDesc` and `root_str_attrs` collapsed into one recursive
        `tree::GroupNode`; new `write/tree.rs` (object model, `split_path`, `group_mut`, `name_taken`,
        `insertion_point`, `attrs_mut`) and `write/plan.rs` (recursive pass one), `write/build.rs` now
        pass two only; the root group is planned and emitted like any other, pinned to address 96;
        `write_dataset_f64("/a/b/x", …)` creates `a` and `a/b` (h5py `create_intermediate_group=True`);
        one `name_taken` at every insertion point, so a dataset can no longer shadow a group; object
        references resolved over the whole plan tree by path and an unresolvable target is now an error
        instead of `u64::MAX`; tests `w1b_three_level_nesting_roundtrip`,
        `w1b_intermediate_groups_auto_created`, `w1b_dataset_cannot_shadow_group_name`,
        `w1b_group_cannot_shadow_dataset_name`, `w1b_objref_across_groups_resolves`,
        `w1b_unresolvable_objref_is_an_error`, `w1b_deep_nesting_many_groups`, plus h5py interop
        `test_write_h5py_three_level_nesting` and `test_write_h5py_subgroup_spanning_multiple_snods`)
  - [x] Group attributes, non-string root-group attributes, and array-valued attributes
        (done 2026-07-21: W1c — `GroupNode.attrs` → `OhMsg::Attr` after `OhMsg::SymbolTable`, so any group
        at any depth carries attributes; `write_string_attr`/`write_f64_attr`/`write_i64_attr`/
        `write_i32_attr`/`write_obj_ref_list_attr` resolve their path to a dataset *or* a group, `"/"`
        being the root group, and `write_root_str_attr` is now `write_string_attr("/", …)`;
        new `write_i64_array_attr`/`write_f64_array_attr`/`write_string_array_attr` with a
        `ResolvedAttrKind::vector_len()` that decides scalar-versus-1-D dataspace once; tests
        `w1c_subgroup_attrs_roundtrip`, `w1c_root_group_non_string_attrs`,
        `w1c_root_str_attr_is_a_root_group_attr`, `w1c_array_valued_attrs`, plus h5py interop
        `test_write_h5py_group_attributes` and `test_write_h5py_root_group_attributes`;
        golden hash moved `0x3a9a_d435_7143_2277` → `0x199c_ba54_0314_ee2a` because the W1x fixture was
        extended to cover the new features — the restructure itself was byte-neutral)

  - [x] Chunk B-tree conformance + DEFLATE compression on write
        (done 2026-07-21: W1e — `write/chunked.rs` rewritten; a node is now `chunk_node_size(rank)`
        bytes wide (2096 for 1-D) because libhdf5 sizes the image from a compile-time
        `HDF5_BTREE_CHUNK_IK_DEF = 32` that superblock v0 never records, and the old 80-byte node
        made *every* chunked file we had ever written fail with `addr overflow, addr = 3000,
        size = 2096, eoa = 3112`; terminal key `key[K]` now carries the extent rounded up to a
        chunk boundary with `nbytes = 0`, where it used to be all-zero and gave `H5D__btree_cmp3`
        an empty search range; `DatasetDesc`'s `unlimited: bool` + `chunk_shape` collapsed into
        `Storage::{Contiguous, Chunked}` plus `Option<Filter>`; new `write/payload.rs` compresses
        during the *layout* pass — a compressed length is not derivable, and the B-tree key needs
        it before emission — with one `data_size()` both passes agree through, and new
        `write/pipeline.rs` encodes the 0x000B message (v1 emitted, v2 implemented behind
        `PIPELINE_VERSION`, both round-tripped through the real `parse_filter_pipeline`, which
        silently returns an *empty* pipeline with `Ok` for any other version); public API is one
        method, `FileWriter::set_deflate(path, level)`, which flips storage to chunked and rejects
        vlen-string datasets because the read side refuses to decode them through a pipeline;
        a short `chunk_shape` is now completed from the *dataset shape* rather than from 1s, which
        fixes `oxinetcdf`'s 2-D unlimited variables reading back mostly zeroes without touching
        `oxinetcdf`; tests `w1e_deflate_roundtrip_f64`, `w1e_deflate_every_level_roundtrips`,
        `w1e_deflate_pipeline_msg_parses`, `w1e_chunk_btree_terminal_key_is_extent`,
        `w1e_zero_length_chunked_dataset`, `w1e_zero_length_dataset_indexes_nothing`,
        `w1e_vlen_string_deflate_rejected`, `w1e_set_deflate_rejects_non_datasets`,
        `w1e_deflate_preserves_an_unlimited_dimension`, h5py interop
        `test_write_h5py_deflate_roundtrip` + `test_write_h5py_chunked_unlimited`, and read-side
        `w1e_read_h5py_deflate_*` against a new h5py-authored `fixtures/deflate_chunked.h5`;
        golden hash moved `0x199c_ba54_0314_ee2a` → `0xa273_56da_84d2_67f6` and the fixture grew
        6 968 → 11 504 bytes, entirely from the two chunk B-tree nodes)
  - [x] Real chunk tiling: N chunks, N-entry multi-level index, per-chunk compression
        (done 2026-07-21: W1e2 — `chunk_origins` cuts the dataset into `ceil(shape/chunk)` tiles
        row-major with dimension 0 most significant, the order confirmed by dumping the B-tree keys
        of an h5py-authored 2-D file rather than inferred; `payload::cut_tile` stores an edge chunk
        **full-size** with fill in the overhang, because a short tile would contradict the chunk
        dimensions in the layout message; new `chunked::ChunkTree` plans, places and emits the whole
        index, growing levels past 64 chunks so the layout message points at the tree *root* and not
        at the first leaf — `key[i]` is a child's inclusive lower bound and `key[i+1]` its exclusive
        upper bound, the opposite of the group B-tree's convention in `btree_v1.rs`; capped at 2^24
        chunks with a typed error; tests `w1e2_multi_chunk_roundtrip_ragged_1d` (`[7]` in `[3]`),
        `w1e2_multi_chunk_roundtrip_ragged_2d` (`[5,3]` in `[2,2]`, plus each dimension ragged
        alone), `w1e2_multi_level_chunk_index_roundtrip`, `tree_depth_follows_the_chunk_count`,
        `a_two_level_tree_brackets_every_child`, and h5py interop
        `test_write_h5py_multi_chunk_tiling`, which additionally checks the chunk origins
        `iter_chunks()` decodes and enumerates all 8500 chunks of a three-level index.
        **Mutation-tested**: transposing the chunk grid to column-major leaves every oxih5
        round-trip test passing and fails only the h5py test — the same trap four agents have now
        hit, so the h5py interop tests are the load-bearing ones for chunk conformance)

### Publish prerequisites
- [x] oxiarc-szip must be published to crates.io before oxih5-format can be
      published (it is an optional dependency in the `szip` feature)
  - **UNBLOCKED (2026-06-03): oxiarc-szip is now published on crates.io (v0.3.2)**
  - Workspace Cargo.toml updated to registry dep `oxiarc-szip = "0.3.2"` (no path dep needed).
- [ ] Publish oxih5-core → oxih5-format → oxih5 → oxinetcdf to crates.io
  - **BLOCKED: requires explicit cargo publish approval from User per COOLJAPAN policy**
- Publish order: oxih5-core → oxih5-format → oxih5 → oxinetcdf


---

<!-- production-readiness-backlog 2026-07-16 -->
## Production-Readiness Backlog — 2026-07-16

_Consolidated from static audit + Opus adversarial bug-hunt (48 verified defects across noffi) + baseline nextest/clippy + design investigation. See `../NOFFI_PRODUCTION_BACKLOG.md` for the full cross-project list and severity/model legend. Confirmed bugs and H1–H4 shipped in 0.1.4 (see checkboxes below); H5/H6 remain open._

**Confirmed bugs — Opus-verified:**
- [x] **S · high** `oxih5-format/src/fa_index.rs:221` — fixed-array "Number of Elements" u64 header used directly as `Vec` capacity → capacity-overflow panic / huge alloc pre-validation. R2/N0 (done 0.1.4: bounded to 16Mi elements (`FA_MAX_ELEMENTS = 1 << 24`) and cross-checked against bytes remaining in the file before use as `Vec::with_capacity`; `test_fa_oversized_element_count_rejected`.)
- [x] **S · high** `oxih5-format/src/chunked.rs:567` — hyperslab/sliced chunk read divides range by per-dim chunk size without zero-check → divide-by-zero panic on crafted file. R2/N0 (done 0.1.4: zero chunk-dim now returns `OxiH5Error::Format` instead of panicking; `test_chunked_slice_zero_chunk_dim_errors`.)
**Designed / audit gaps:**
- [x] **B/easy · H1** workspace.dependencies 0.1.3→0.1.4 drift. (done 0.1.4: all four internal crates pinned to `version = "0.1.4"` in root `Cargo.toml` workspace.dependencies.)
- [x] **A/hard/Opus · H2** virtual dataset layout. (done 0.1.4: new `oxih5-format/src/vds.rs`, 728 lines — `VdsMapping`/`VdsEntry`/`VdsSelection`, `parse_vds_mapping`/`parse_vds_block`, `selection_element_offsets`; 9 unit tests + 4 integration tests in `tests/vds_tests.rs`.)
- [x] **A/hard/Opus · H3** variable-length elements in chunked/hyperslab. (done 0.1.4: new `on_disk_elem_footprint` helper accounts for the 16-byte vlen global-heap-reference footprint; incompatible filters now explicitly rejected; `crates/oxih5/tests/vlen_chunked_tests.rs`.)
- [x] **A/med/Opus · H4** szip RAW mode + FractalHeap I/O filters + soft→external link. (done 0.1.4: public `apply_pipeline_sized` in `filters.rs` for RAW-mode szip; fractal-heap I/O-Filters-Encoded-Length field now read at the correct offset/width; `soft_link_through_external_link` integration test in `crates/oxih5/tests/vds_tests.rs`.)
- [ ] **B/med · H5** write-path unwrap reduction (232 non-test, up from 225) + panic!17 triage (overlaps confirmed bugs). (clarified 2026-07-18: a full workspace audit found 0 unwrap() call sites in actual production logic — the historical "232 non-test" count conflated test-module and doctest unwraps with production code; see README Policy Compliance. The panic! triage portion remains open.)
- [ ] **B/easy · H6** preventive split lib.rs(**done**)/chunked.rs(1963, was 1730); examples/doctests. lib.rs reached 1999 of the 2000-line cap and has now been split (0.2.1) into `file.rs` (`File`), `group_handle.rs` (`Group`), `reader.rs` (navigation + object-header dataset assembly) and `slicing.rs` (lazy range/hyperslab reads), leaving lib.rs at 388 lines; `links.rs` swapped its `use super::*` glob for explicit imports. Behaviour-preserving — the public API surface is unchanged apart from the two `pub use` re-exports that keep `oxih5::File`/`oxih5::Group` at their original paths. `oxih5-format/src/chunked.rs` (1963) remains the outstanding half.
