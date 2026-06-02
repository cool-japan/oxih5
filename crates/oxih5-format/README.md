# oxih5-format — Low-level HDF5 binary-format parsers for OxiH5

[![Crates.io](https://img.shields.io/crates/v/oxih5-format.svg)](https://crates.io/crates/oxih5-format)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`oxih5-format` is the binary-parsing layer of **OxiH5**, the COOLJAPAN Pure-Rust HDF5 reader/writer. It turns raw HDF5 file bytes — exactly as produced by h5py / libhdf5 — into the typed data model from [`oxih5-core`]. Every standard structure of the HDF5 file format is decoded here: the superblock, object headers (v1 and v2), all standard header messages, local/global/fractal heaps, B-tree v1 and v2 nodes, the extensible- and fixed-array chunk indices, the filter pipeline, and full chunked-dataset assembly.

This crate sits between [`oxih5-core`] (the data model) and the [`oxih5`] facade (the file API). It is 100% Pure Rust with `#![forbid(unsafe_code)]`; DEFLATE/zlib decompression is delegated to the COOLJAPAN [`oxiarc-deflate`] crate (never flate2/miniz). It exposes a flat, function-oriented API — there is no `File` handle here; that abstraction lives in `oxih5`. Most users should depend on `oxih5` instead and reach for `oxih5-format` only when building custom HDF5 tooling.

## Installation

```toml
[dependencies]
oxih5-format = "0.1.0"

# Optional: rayon-parallel chunk assembly
oxih5-format = { version = "0.1.0", features = ["parallel"] }
```

## Quick Start

```rust,no_run
use oxih5_format::{superblock, header, message};

let bytes: Vec<u8> = std::fs::read("data.h5")?;

// 1. Parse the superblock to find the root object header.
let sb = superblock::parse(&bytes)?;

// 2. Decode the object header's message list.
let messages = header::parse_messages(&bytes, sb.root_object_header_address)?;

// 3. Inspect a message — e.g. find the symbol-table message (type 0x0011).
for msg in &messages {
    if msg.msg_type == 0x0011 {
        let st = message::parse_symbol_table(&msg.data)?;
        println!("b-tree @ {}, heap @ {}", st.btree_address, st.heap_address);
    }
}
# Ok::<(), oxih5_core::OxiH5Error>(())
```

## API Overview

The crate re-exports [`ChunkIndexCache`] from `chunked` at the crate root; everything else is reached through its module path.

### `superblock` — file superblock

| Item | Description |
|------|-------------|
| `struct Superblock` | `size_of_offsets`, `size_of_lengths`, `base_address`, `root_object_header_address` |
| `fn parse(data) -> Superblock` | Parse v0 / v2 / v3 superblocks from the file start |
| `fn read_u16_le` / `read_u32_le` / `read_u64_le` | Bounds-checked little-endian integer readers |

### `header` — object headers

| Item | Description |
|------|-------------|
| `struct Message` | `msg_type: u16`, `data: Vec<u8>` (body bytes preserved verbatim) |
| `fn parse_messages(file_data, offset) -> Vec<Message>` | Decode a v1 or v2 object header (handles continuation blocks) |

### `message` — header-message decoders

| Item | Description |
|------|-------------|
| `struct DataspaceInfo` | Decoded dataspace dimensions |
| `struct DatatypeInfo` | Wraps an `oxih5_core::Dtype` |
| `struct SymbolTableInfo` | `btree_address`, `heap_address` (old-style groups) |
| `enum LayoutInfo` | `Contiguous`, `Compact`, `Chunked`, `VirtualDataset` (see below) |
| `fn parse_dataspace` | Dataspace message (0x0001) → `DataspaceInfo` |
| `fn parse_dataspace_rich` | Dataspace message → `oxih5_core::Dataspace` |
| `fn parse_datatype` | Datatype message (0x0003) → `DatatypeInfo` |
| `fn parse_layout` | Data-layout message (0x0008) → `LayoutInfo` |
| `fn parse_fill_value` | Fill-value message (0x0005) → `Option<Vec<u8>>` |
| `fn parse_filter_pipeline` | Filter-pipeline message (0x000B) → `FilterPipeline` |
| `fn parse_attribute` | Attribute message (0x000C) → `oxih5_core::Attribute` |
| `fn parse_symbol_table` | Symbol-table message (0x0011) → `SymbolTableInfo` |
| `fn parse_modification_time` | Object-modification-time message → `u32` |

`LayoutInfo` variants: `Contiguous { data_address, data_size }`, `Compact { data }`, `Chunked { data_address, dimensionality, chunk_dims, index_type }`, `VirtualDataset { heap_address, entry_count }`.

### `datatype` — datatype class parsing

| Item | Description |
|------|-------------|
| `fn parse_datatype(body) -> Dtype` | Parse any of the 11 HDF5 datatype classes |
| `fn parse_datatype_consuming(body, depth) -> (Dtype, usize)` | Parse a (possibly nested) datatype, returning bytes consumed |

### `group` — name → object-header resolution

| Item | Description |
|------|-------------|
| `fn list_datasets(file_data, btree, heap) -> Vec<String>` | Names of root-group entries (old-style) |
| `fn find_dataset(file_data, btree, heap, name) -> u64` | Object-header address of a named entry |
| `fn list_new_style_links(file_data, oh_addr, ctx) -> Vec<ParsedLink>` | Links of a new-style group (direct + fractal-heap) |
| `fn is_new_style_group(file_data, oh_addr) -> bool` | Whether an object uses Link-message groups |

### Heaps

| Module / Item | Description |
|---------------|-------------|
| `heap::LocalHeap` | Old-style group name storage; `parse(...)`, `name_at(offset) -> &str` |
| `global_heap::GlobalHeap` | Variable-length / VLen data; `parse(...)`, `object(index) -> &[u8]` |
| `fractal_heap::FractalHeap` | New-style group object storage; `parse(...)`, `parse_heap_id`, `read_object`, plus `header_address` / `heap_id_len` / `table_width` / `root_indirect_rows` / `block_size_for_row` accessors |

### B-trees, indices, and SNOD

| Module / Item | Description |
|---------------|-------------|
| `btree::BTreeV1` | `leaf_addresses: Vec<u64>`; `parse(file_data, addr)` collects all SNOD leaves |
| `btree_v2::BTreeV2` | New-style B-tree chunk index; `parse(file_data, addr, ndims)`, `records() -> &[ChunkRecord]` |
| `btree_v2::ChunkRecord` | `address`, `size: u32`, `filter_mask: u32`, `offsets: Vec<u64>` |
| `btree_v2::parse_name_index` | Resolve a B-tree v2 name index |
| `btree_v1_chunk::parse` | B-tree v1 chunk index (`libver='earliest'`) |
| `ea_index::parse_extensible_array` | Extensible-array chunk index (one unlimited dimension) |
| `fa_index::parse_fixed_array` / `parse_fixed_array_v4` | Fixed-array chunk index (no unlimited dimensions) |
| `snod::SymTabEntry` | `name_offset: u64`, `object_header_address: u64` |
| `snod::parse(file_data, addr) -> Vec<SymTabEntry>` | Decode a symbol-table node |

### `chunked` — chunked-dataset assembly

| Item | Description |
|------|-------------|
| `struct ChunkIndexCache` (re-exported at root) | Thread-safe `(index_addr, ndims) → records` cache; `new()`, `get_or_insert(...)` |
| `enum ChunkIndex` | `BTreeV1`, `BTreeV2`, `FixedArray`, `ExtensibleArray` |
| `fn resolve_chunk_index(file_data, index, addr, ndims) -> Vec<ChunkRecord>` | Resolve any index variety into chunk records |
| `fn assemble_chunks(...)` | Scatter chunk records into a contiguous row-major buffer |
| `fn read_chunked(file_data, layout, pipeline, dataset_dims, elem_size, cache) -> Vec<u8>` | High-level: resolve + read + unfilter + scatter a whole chunked dataset |
| `fn read_chunked_slice(..., ranges, cache) -> Vec<u8>` | As above, but only the requested N-dimensional sub-region |

### `filters` — filter pipeline (inverse / read direction)

| Item | Description |
|------|-------------|
| `mod filter_id` | Standard filter ids: `DEFLATE=1`, `SHUFFLE=2`, `FLETCHER32=3`, `SZIP=4`, `NBIT=5`, `SCALEOFFSET=6` |
| `fn apply_pipeline(raw, pipeline, filter_mask, elem_size) -> Vec<u8>` | Apply the inverse of every active filter in reverse order |
| `fn inflate_deflate(data) -> Vec<u8>` | zlib inflate via `oxiarc-deflate` (Pure Rust) |
| `fn unshuffle(data, elem_size) -> Vec<u8>` | Inverse byte-shuffle |
| `fn verify_fletcher32(data) -> Vec<u8>` | Verify and strip the Fletcher-32 checksum |
| `fn unpack_nbit(...)` | Inverse N-bit packing |
| `fn decode_scaleoffset_int(...)` | Inverse integer scale+offset |

### `link_msg` — link messages

| Item | Description |
|------|-------------|
| `struct ParsedLinkInfo` | `creation_order_tracked`, `creation_order_index_address`, `name_index_address`, `fractal_heap_address` (Link Info message 0x0002) |
| `struct ParsedLink` | `name: String`, `link: oxih5_core::Link` |
| `fn parse_link_info(body, ctx) -> ParsedLinkInfo` | Decode a Link Info message |
| `fn parse_link(body, ctx) -> ParsedLink` | Decode a Link message (0x0006): hard / soft / external |

### `context` — parse context

| Item | Description |
|------|-------------|
| `struct ParseContext` | `size_of_offsets`, `size_of_lengths`, `base_address` |
| `ParseContext::new(...)` / `default_v0()` | Constructors (`default_v0` ⇒ 8-byte offsets/lengths) |
| `read_offset` / `read_length` / `read_int_generic` | Width-aware little-endian integer readers |

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `parallel` | off | Use `rayon` for parallel chunk assembly |
| `szip` | off | Enable SZIP decode via `oxiarc-szip` |

## Errors

All functions return `Result<_, oxih5_core::OxiH5Error>`. See the [`oxih5-core`] documentation for the full variant list (`BadSignature`, `Format`, `Corrupted`, `NotImplemented`, `UnsupportedFilter`, …).

## Related crates

- [`oxih5-core`](https://crates.io/crates/oxih5-core) — the data-model types these parsers produce.
- [`oxih5`](https://crates.io/crates/oxih5) — the user-facing facade; the recommended dependency for reading files.

## License

Apache-2.0 — COOLJAPAN OU (Team Kitasan)
