# oxih5-format TODO

## Status
Functional low-level HDF5 parser: superblock v0, object header v1 with continuation, B-tree v1 group traversal, local heap, SNOD symbol table, dataspace v1, datatype (int/float classes), contiguous layout v3, symbol table message. Group listing and dataset lookup work end-to-end. ~400 SLOC production code across 7 modules.

## Core Implementation
- [x] Implement superblock v2 parsing (different layout: file consistency flags, superblock extension, root group object header addr at different offset) (150-200 SLOC)
  - **Done:** superblock.rs `parse_v2()` — soo-variable field reads; tested against real h5py libver='latest' output — 2026-05-25
- [x] Implement superblock v3 parsing (similar to v2 with checksum) (50 SLOC delta from v2)
  - **Done:** superblock.rs `parse_v2()` handles version 2 and 3 identically (same on-disk layout) — 2026-05-25
- [x] Implement object header v2 parsing (variable-size prefix, header continuation messages, creation order tracking) (250-300 SLOC)
  - **Done:** header.rs `parse_messages_v2()` — OHDR + OCHK block parsing; 1-byte message type; chunk_size_size from flags bits 0-1; optional timestamps (bit5) + attr phase-change (bit4) prefix blocks; creation-order per-message header (bit2); bounded by chunk0_size; OCHK continuation with cont_length; cycle/depth detection — 2026-05-25
- [x] Implement B-tree v1 raw-data-chunk index (node type 1) for layout-v3 chunked datasets (libver='earliest' default) (150-200 SLOC)
  - **Done:** btree_v1_chunk.rs — parses interleaved chunk keys (size + filter mask + (ndims+1) offsets) and child pointers, recursing through internal nodes; verified against h5py fixtures — 2026-05-25
- [x] Implement B-tree v2 traversal for chunked datasets (different node format: signature BTHD, record types, internal/leaf structure) (300-400 SLOC)
  - **Done:** btree_v2.rs — 2026-05-25
- [x] Wire chunk index → raw read → filter pipeline → scatter into `chunked::read_chunked()` + `resolve_chunk_index()` (ChunkIndex enum: BTreeV1/V2/FA/EA)
  - **Done:** chunked.rs high-level read path consumed by the oxih5 facade — 2026-05-25
- [x] Implement extensible array index for chunked datasets (EA header, index block, data block, secondary blocks) (200-300 SLOC)
  - **Done:** ea_index.rs — header + index block inline elements + data blocks (EADB) + secondary blocks (EASB) parsed; `parse_secondary_block()` resolves EASB→EADB indirection; paged data blocks not yet implemented — 2026-05-25
- [x] Implement fixed array index for fixed-size chunked datasets (FA header, data block, page bitmap) (150-200 SLOC)
  - **Done:** fa_index.rs (non-paged data blocks; paged data blocks not yet supported) — 2026-05-25
- [x] Implement chunked data layout (class 2): read chunk index, assemble chunks into contiguous buffer (200-250 SLOC)
  - **Done:** chunked.rs — 2026-05-25
- [x] Implement compact data layout (class 0): inline data stored in the object header message body (30-40 SLOC)
  - **Done:** message.rs LayoutInfo::Compact variant — 2026-05-25
- [x] Implement global heap parsing (collection + object access for VL data) (100-150 SLOC)
  - **Done:** global_heap.rs — 2026-05-25
- [x] Implement fractal heap parsing (v2 B-tree backed heap for large groups) (300-400 SLOC)
  - **Done:** fractal_heap.rs — FRHP header + FHDB direct-block read + FHIB indirect-block traversal (managed heap IDs, multi-level traversal with depth guard); indirect block entries for direct and sub-indirect blocks — 2026-05-25
- [x] Implement new-style group links (link info message 0x002, link message 0x0006: hard/soft/external) (150-200 SLOC)
  - **Done:** link_msg.rs `parse_link_info()` + `parse_link()` (hard/soft/external); group.rs `list_new_style_links()` + `is_new_style_group()`; facade handles new-style root/nested groups — 2026-05-25
- [x] Implement filter pipeline message (0x000B): parse filter IDs, flags, client data (80-100 SLOC)
  - **Done:** message.rs parse_filter_pipeline() v1+v2 — 2026-05-25
- [x] Implement deflate filter decompression via oxiarc-deflate (zlib_decompress) -- NEVER flate2/miniz_oxide (60-80 SLOC)
  - **Done:** filters.rs `inflate_deflate()` + `apply_pipeline()` (reverse-order filter chain w/ per-chunk filter mask) — 2026-05-25
- [x] Implement shuffle filter (byte-reordering for better compression ratios) (40-60 SLOC)
  - **Done:** filters.rs unshuffle() — 2026-05-25
- [x] Implement fletcher32 checksum filter (verification) (40-50 SLOC)
  - **Done:** filters.rs verify_fletcher32() — 2026-05-25
- [x] Implement szip decompression filter (if feasible in Pure Rust, else feature-gate) (200-300 SLOC)
  - **Done:** filters.rs `decode_szip()` (feature-gated: `--features szip`); wires `oxiarc-szip` AEC/CCSDS-121 decoder; parses HDF5 szip framing (4-byte LE uncompressed-byte-count header + AEC bitstream); extracts options_mask/bpp/ppb/pps from `client_data`; handles MSB/NN/RAW flags; 3 unit tests (round-trip all-zeros, malformed framing, missing client_data) — 2026-05-30
- [x] Implement nbit filter (precision reduction) (60-80 SLOC)
  - **Done:** filters.rs `unpack_nbit()` helper + wired into `apply_one_inverse` via `client_data` descriptor (integer class only: client_data[1]==0, extracts sizeof/precision/bit_offset); compound/array/float nbit still returns UnsupportedFilter — 2026-05-25
- [x] Implement scaleoffset filter (integer/float scaling) (80-100 SLOC)
  - **Done (integer only):** filters.rs `decode_scaleoffset_int()` helper + wired into `apply_one_inverse` via per-chunk header (byte 0 = min_bits, bytes 1..1+D = min_val LE, remainder = nbit-packed data); float scaleoffset not yet supported — 2026-05-25
- [x] Implement attribute message (0x000C) parsing: name + datatype + dataspace + value (100-150 SLOC)
  - **Done:** message.rs parse_attribute() v1/v2/v3 — 2026-05-25
- [x] Implement fill value message (0x0005) parsing (40-50 SLOC)
  - **Done:** message.rs parse_fill_value() v1/v2/v3 — 2026-05-25
- [x] Implement modification time message (0x0012) parsing (20-30 SLOC)
  - **Done:** message.rs parse_modification_time() v1 — returns Unix u32 timestamp — 2026-05-25
- [x] Implement compound datatype parsing (member offsets, sizes, names, nested types) (100-150 SLOC)
  - **Done:** datatype.rs class 6 — v1+v2, recursive via parse_datatype_consuming — 2026-05-25
- [x] Implement string datatype parsing (fixed-length and variable-length, charset: ASCII/UTF-8) (60-80 SLOC)
  - **Done:** datatype.rs class 3 (fixed-length) + class 9 with is_string=1 (VLen) — 2026-05-25
- [x] Implement enum datatype parsing (base type + name-value pairs) (50-60 SLOC)
  - **Done:** datatype.rs class 8 — v1+v2 — 2026-05-25
- [x] Implement array datatype parsing (base type + fixed dimensions) (30-40 SLOC)
  - **Done:** datatype.rs class 10 — v1+v2 — 2026-05-25
- [x] Implement variable-length datatype parsing (global heap reference sequences) (80-100 SLOC)
  - **Done:** datatype.rs class 9 (sequence + string variants) — 2026-05-25
- [x] Implement reference datatype parsing (object reference, region reference) (60-80 SLOC)
  - **Done:** datatype.rs class 7 — 2026-05-25
- [x] Implement virtual dataset mapping (VDS) parsing for virtual layout (150-200 SLOC)
  - **Done:** VirtualDataset variant added to LayoutInfo; layout v4/class-3 body parsed (heap_address + data_size); NotImplemented error returned in facade; vds_main.h5 fixture + test added — 2026-05-25 (full virtual reading not yet supported)
- [x] Implement external file link resolution (file + object path) (60-80 SLOC)
  - **Done:** oxih5/src/lib.rs `resolve_new_style_dataset()` + `resolve_external_link()` helper; `source_dir: PathBuf` added to `File` and `Group`; relative and absolute ext-file paths resolved; `parse_link()` in link_msg.rs fixed to handle libhdf5/h5py `libver='earliest'` encoding where link type 64 is stored in the charset byte; 3 integration tests added (root external link, nested group external link, local dataset still works) — 2026-05-25

## API Improvements
- [x] Add `Size` enum abstracting 2/4/8-byte offset/length sizes instead of hardcoding soo=8/sol=8
  - **Plan:** oxih5-format format parsers + chunked infrastructure — 2026-05-25
- [x] Add cursor-based reader abstraction instead of raw `&[u8]` + offset everywhere
  - **Plan:** oxih5-format format parsers + chunked infrastructure — 2026-05-25
- [x] Make `read_u*_le` functions generic over offset size
  - **Plan:** oxih5-format format parsers + chunked infrastructure — 2026-05-25
- [x] Add `ParseContext` struct carrying superblock info (offset sizes, base address) through parse calls
  - **Done:** context.rs — 2026-05-25
- [x] Return parsed filter pipeline info alongside layout info so facade can apply decompression
  - **Done:** read_dataset_from_group extracts 0x000B filter pipeline and passes FilterPipeline to chunked::read_chunked() — 2026-05-25

## Testing
- [x] Generate HDF5 test fixtures with h5py covering: chunked layout, compact layout, compressed (gzip), multi-dimensional, compound types, string datasets, big-endian, attributes
  - **Done:** compound_1d.h5 (compound dtype [('x',f4),('y',f4),('z',f4)] shape [2]), string_fixed_1d.h5 (S10 fixed-length strings shape [2]), chunked_btree_v1.h5 (float64 arange(100) chunks=(10,) gzip level 6 B-tree v1), multi_attr.h5 (float32 linspace(0,1,20) with units/count/scale/calibrated attrs) — 2026-05-25
- [x] Test superblock v2/v3 parsing against h5py-generated files with `libver='latest'`
  - **Done:** libver_latest_small.h5 (superblock v3, new-style group with direct Link messages) and libver_latest_large.h5 (superblock v3, fractal heap + B-tree v2 type-5 name index, 20-dataset group); 5 integration tests added to read_contig.rs — 2026-05-25
- [x] Test B-tree v2 traversal with deeply nested chunked datasets
  - **Plan:** oxih5-format format parsers + chunked infrastructure — 2026-05-25
- [x] Test filter pipeline round-trip: write compressed with h5py, read + decompress with oxih5-format
  - **Done:** libver_latest_chunked.h5 fixture (superblock v3, extensible array chunk index, gzip/shuffle/plain); tests 40-42 in read_contig.rs verify full path: superblock v3 + object header v2 + new-style group + B-tree v2/EA chunk index + gzip/shuffle/plain data — 2026-05-25
- [x] Fuzz the superblock/header/message parsers with arbitrary byte sequences
  - **Done:** 5 fuzz tests pass; fixed overflow panics in header.rs, message.rs, heap.rs, global_heap.rs, snod.rs, link_msg.rs, fractal_heap.rs — all `as usize` casts replaced with `usize::try_from` + `checked_add`/`checked_mul`; filter pipeline length guard added — 2026-05-30
- [x] Test continuation message chains (object headers spanning multiple blocks)
  - **Done:** multi_attr fixture exercises OH v1 with continuation message (0x0010); fixed NIL-as-terminator bug in header.rs (NIL now skipped per HDF5 spec) — 2026-05-25

## Performance
- [x] Profile parse overhead for large files (>1GB) with many groups/datasets (2026-06-02)
  - **Done:** T7 (LocalHeap borrows &'a [u8] — no segment copy), T8 (FractalHeap takes &[u8] — no Arc::new(to_vec()) per group), T6 (Vec::with_capacity pre-sizing). traverse_bench.rs added (heap_parse_name_at, snod_parse, group_traverse_btree over 8/32/64/128 entries). All 179 oxih5-format + 96 oxih5 tests pass; zero new warnings; bench compiles.
  - **Files:** `oxih5-format/src/heap.rs`, `oxih5-format/src/fractal_heap.rs`, `oxih5-format/src/group.rs`, `oxih5/src/lib.rs`, `oxih5-format/tests/fuzz_parsers.rs`, `oxih5-format/Cargo.toml`, `oxih5-format/benches/traverse_bench.rs`
- [x] Benchmark contiguous vs chunked read paths
- [x] Consider memory-mapped I/O (`mmap`) as alternative to `read_to_vec` for large files — implemented at the facade layer (`oxih5::open_mmap`, `FileData::Mapped`, `memmap2`; see oxih5/src/lib.rs).
- [x] Lazy chunk loading: only decompress chunks needed for requested slice (planned 2026-06-02)
  - **Goal:** `File::dataset_slice`/`Group::dataset_slice` decompress only chunks overlapping the requested slice. Full strided hyperslab support underneath. Sparse chunks honor declared fill value. Multi-dim Fixed-Array datasets read correctly.
  - **Design:** New `hyperslab.rs` module with `DimSelection`/`Hyperslab` + `scatter_chunk_hyperslab`. Wire-up: new internal `read_dataset_slice_from_messages` that calls `chunked::read_chunked_slice` for chunked layouts, falls back to full-read+slice for contiguous/compact. Fill value: `message::parse_fill_value` (msg 0x0005), optional fill param to `read_chunked_slice`. FA-grid fix: `compute_grid_dims` uses `ceil(dataset_dims[d]/chunk_dims[d])` with true dataset dims.
  - **Files:** create `oxih5-format/src/hyperslab.rs`; edit `oxih5-format/src/lib.rs`, `chunked.rs`, `fa_index.rs`, `message.rs`, `oxih5/src/lib.rs`, `oxih5/tests/read_contig.rs`
  - **Tests:** Hyperslab unit tests (3-D crossing chunk borders, strided, block>1, edge chunks, sparse fill). Integration: `dataset_slice == full.slice` for all chunked fixtures. Failing-first regressions for FA-grid and non-zero fill.
  - **Risk:** chunked.rs 1492 lines — run rslines 50, splitrs if needed. Fill param changes two signatures but is facade-internal. FA fixture hand-built from fa_index.rs test helpers if h5py unavailable.
- [x] Fix chunk fill value: honor HDF5 fill value message (0x0005) for sparse chunks (planned 2026-06-02)
  - **Goal:** Sparse/missing chunks and output buffers use the dataset's declared fill value instead of literal zero.
  - **Design:** `message::parse_fill_value` for msg type 0x0005. Tile fill bytes for output init and sparse-chunk regions in `read_chunked_slice`/`read_chunked`. `None` → zero-fill preserved.
  - **Files:** `oxih5-format/src/message.rs`, `oxih5-format/src/chunked.rs`, `oxih5/src/lib.rs`
  - **Tests:** Failing-first: a sparse chunked dataset with non-zero fill; assert correct fill after fix.
  - **Risk:** Changes two function signatures (confined to format internals + facade internals, no public API break).
- [x] Fix Fixed-Array chunk index grid approximation (planned 2026-06-02)
  - **Goal:** `fa_index::compute_grid_dims` uses the true chunk grid `ceil(dataset_dims[d]/chunk_dims[d])` instead of `n^(1/ndims)` approximation that mis-places chunks in non-cube multi-dim datasets.
  - **Design:** Thread `dataset_dims` into `compute_grid_dims`; compute `grid[d] = (dataset_dims[d]+chunk_dims[d]-1)/chunk_dims[d]`. Fix call site in `parse_fixed_array_v4`.
  - **Files:** `oxih5-format/src/fa_index.rs`, call sites in `chunked.rs`
  - **Tests:** Failing-first: 2-D FA fixture (hand-built bytes if needed) with non-cube grid; assert correct element placement after fix.
  - **Risk:** Low — the fix is the correct formula; existing tests plus the new regression guard correctness.

## Integration
- [x] Ensure all parsed types (Dtype, Dataspace, Layout) map cleanly to oxih5-core types
  **Verified:** datatype.rs produces all Dtype variants; Dataset::slice/reshape/as_* type-check cleanly — 2026-05-25; confirmed via compound/string/chunked integration tests — 2026-05-25
- [x] Provide enough parsed metadata for oxih5 facade to construct `Dataset` with full fidelity
  **Verified:** read_dataset_from_group extracts dataspace + dtype + layout + filter pipeline; Dataset fully populated including .attributes — 2026-05-25; confirmed via multi_attr fixture (4 attrs: units/count/scale/calibrated) — 2026-05-25
- [x] Coordinate filter decompression with OxiARC crates (oxiarc-deflate for gzip) — done: filters.rs calls oxiarc_deflate::zlib_decompress; dep wired in Cargo.toml.
