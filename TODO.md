# OxiH5 Project TODO

## Status — 0.1.0 (2026-06-01)

Functional read/write HDF5 library (~11.7 k SLOC Rust, 273 tests, all pass).
Supports superblock v0/v2/v3, object header v1/v2 (+continuation), B-tree
v1/v2, local heap, SNOD, fractal heap (FRHP+FHDB+FHIB), extensible/fixed array
chunk indices, all 11 datatype classes, dataspace v1/v2, attributes (0x000C
v1/v2/v3), filter pipeline (deflate/shuffle/fletcher32/nbit/scaleoffset), fill
value, global heap, and **contiguous + compact + chunked** layouts.  New-style
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

---

## Open Items (post-0.1.0)

### Testing
- [ ] Benchmark against hdf5-rust (FFI) for read throughput — blocked on
      COOLJAPAN Pure Rust policy (hdf5-rust requires C FFI)
- [ ] Profile and optimize hot paths (superblock + header parsing)

### Integration
- [ ] Coordinate with SciRS2 for ML model weight reading — blocked on SciRS2
      API stabilization; oxih5 already supports float32/float64/int32 arrays
- [ ] Scope oxinetcdf conventions layer atop OxiH5 — separate subcrate

### Publish prerequisites
- [ ] oxiarc-szip must be published to crates.io before oxih5-format can be
      published (it is an optional dependency in the `szip` feature)
- Publish order: oxih5-core → oxih5-format → oxih5
