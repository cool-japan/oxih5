//! HDF5 file writer — Phase 8 write support.
//!
//! Produces minimal, valid HDF5 files using superblock v0, old-style group
//! (B-tree v1 + SNOD + local heap), and contiguous data layout.
//!
//! Constraints:
//! - Flat files only (no nested groups)
//! - Up to 8 datasets per file
//! - No compression
//! - Supported element types: f32, f64, i32, i64, u8

use oxih5_core::OxiH5Error;
use std::path::Path;

// ---------------------------------------------------------------------------
// Element-type enum
// ---------------------------------------------------------------------------

/// Supported dataset element types for writing.
#[derive(Debug, Clone, Copy)]
enum ElemType {
    F32,
    F64,
    I32,
    I64,
    U8,
}

// ---------------------------------------------------------------------------
// Internal dataset descriptor
// ---------------------------------------------------------------------------

struct DatasetDesc {
    name: String,
    raw: Vec<u8>,
    shape: Vec<usize>,
    elem_type: ElemType,
}

// ---------------------------------------------------------------------------
// FileWriter — public API
// ---------------------------------------------------------------------------

/// Flat HDF5 file writer (no nested groups, contiguous layout, no compression).
///
/// Supports up to 8 datasets per file.
///
/// # Example
/// ```no_run
/// use oxih5::FileWriter;
/// let path = std::env::temp_dir().join("example.h5");
/// FileWriter::new()
///     .write_dataset_f32("data", &[1.0f32, 2.0, 3.0], &[3]).unwrap()
///     .build(&path)
///     .unwrap();
/// ```
pub struct FileWriter {
    datasets: Vec<DatasetDesc>,
}

impl Default for FileWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl FileWriter {
    /// Create a new, empty file writer.
    pub fn new() -> Self {
        Self {
            datasets: Vec::new(),
        }
    }

    /// Add a float32 dataset.
    pub fn write_dataset_f32(
        &mut self,
        name: &str,
        data: &[f32],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_dataset(name, raw, shape, ElemType::F32)
    }

    /// Add a float64 dataset.
    pub fn write_dataset_f64(
        &mut self,
        name: &str,
        data: &[f64],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_dataset(name, raw, shape, ElemType::F64)
    }

    /// Add a signed int32 dataset.
    pub fn write_dataset_i32(
        &mut self,
        name: &str,
        data: &[i32],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_dataset(name, raw, shape, ElemType::I32)
    }

    /// Add a signed int64 dataset.
    pub fn write_dataset_i64(
        &mut self,
        name: &str,
        data: &[i64],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_dataset(name, raw, shape, ElemType::I64)
    }

    /// Add a uint8 dataset.
    pub fn write_dataset_u8(
        &mut self,
        name: &str,
        data: &[u8],
        shape: &[usize],
    ) -> Result<&mut Self, OxiH5Error> {
        let raw = data.to_vec();
        self.add_dataset(name, raw, shape, ElemType::U8)
    }

    /// Write the HDF5 file to disk.
    ///
    /// This method takes `&mut self` so it can be called after chained
    /// `write_dataset_*` calls.
    pub fn build(&mut self, path: impl AsRef<Path>) -> Result<(), OxiH5Error> {
        let n = self.datasets.len();

        // ------------------------------------------------------------------
        // 1. Build heap data following h5py's 8-byte alignment convention.
        //
        // h5py aligns each heap entry to an 8-byte boundary:
        //   - Offset 0: 8 zero bytes (root group name placeholder)
        //   - Offset 8: name0 + NUL, padded to next 8-byte boundary
        //   - Offset 8+aligned(name0): name1 + NUL, padded, …
        //
        // After all names, we write a free block:
        //   [next_free=1(u64), size(u64)]
        // and set free_list_head = offset of this free block.
        //
        // The segment is at least 88 bytes (h5py minimum) to ensure
        // there's always room for the free block.
        // ------------------------------------------------------------------
        // 8 null bytes for root name (offset 0)
        let mut heap_used: Vec<u8> = vec![0u8; 8];
        let mut name_offsets: Vec<u64> = Vec::with_capacity(n);
        for ds in &self.datasets {
            name_offsets.push(heap_used.len() as u64);
            heap_used.extend_from_slice(ds.name.as_bytes());
            heap_used.push(0); // NUL terminator
                               // Pad to next 8-byte boundary.
            let cur_len = heap_used.len();
            let aligned = (cur_len + 7) & !7;
            heap_used.resize(aligned, 0);
        }
        let used_size = heap_used.len();

        // Heap segment size: used + 16 (free block), aligned to 8 bytes,
        // minimum 88 bytes (matches h5py's default allocation).
        let heap_data_size_aligned = ((used_size + 16 + 7) & !7).max(88);
        let mut heap_data = vec![0u8; heap_data_size_aligned];
        heap_data[..used_size].copy_from_slice(&heap_used);
        // Write free block at used_size: [next_free=1(u64), size(u64)]
        let free_size = heap_data_size_aligned - used_size;
        heap_data[used_size..used_size + 8].copy_from_slice(&1u64.to_le_bytes());
        heap_data[used_size + 8..used_size + 16].copy_from_slice(&(free_size as u64).to_le_bytes());

        // ------------------------------------------------------------------
        // 2. Pre-compute addresses.
        // Fixed layout (all constants):
        //   0..8    Signature
        //   8..96   Superblock v0 (remaining 88 bytes)
        //   96      Root group OH v1 (40 bytes)
        //   136     B-tree v1 leaf (48 bytes)
        //   184     Local heap header (32 bytes)
        //   216     Local heap data  (heap_data_size_aligned bytes)
        //   snod    SNOD (328 bytes)
        //   …       Dataset OHs + raw data
        // ------------------------------------------------------------------
        const SNOD_SIZE: usize = 8 + 8 * 40; // 328

        let snod_addr: usize = 216 + heap_data_size_aligned;

        let mut oh_addrs: Vec<usize> = Vec::with_capacity(n);
        let mut data_addrs: Vec<usize> = Vec::with_capacity(n);
        let mut current: usize = snod_addr + SNOD_SIZE;

        for ds in &self.datasets {
            oh_addrs.push(current);
            let oh_sz = compute_oh_size(ds.shape.len(), &ds.elem_type);
            current += oh_sz;
            data_addrs.push(current);
            current += ds.raw.len();
            // Align raw data to 8-byte boundary before next OH.
            current = (current + 7) & !7;
        }

        let eof_addr: usize = current;

        // ------------------------------------------------------------------
        // 3. Allocate zero-filled buffer and write all structures.
        // ------------------------------------------------------------------
        let mut buf = vec![0u8; eof_addr];

        // B-tree key[1]: for a leaf node with one SNOD child, key[1] must
        // be the heap offset of the last (lexicographically greatest) name
        // entry so that h5py can correctly bound its binary search.
        // For an empty file there are no entries so key[1] = 0.
        let btree_key1: u64 = name_offsets.last().copied().unwrap_or(0);

        write_signature(&mut buf);
        write_superblock(&mut buf, eof_addr as u64);
        write_root_oh(&mut buf);
        write_btree_leaf(&mut buf, snod_addr as u64, btree_key1);
        write_local_heap(&mut buf, &heap_data, heap_data_size_aligned, used_size);
        write_snod(&mut buf, snod_addr, n, &name_offsets, &oh_addrs);

        for (i, ds) in self.datasets.iter().enumerate() {
            write_dataset_oh(&mut buf, oh_addrs[i], ds, data_addrs[i] as u64);
            let raw_end = data_addrs[i] + ds.raw.len();
            buf[data_addrs[i]..raw_end].copy_from_slice(&ds.raw);
        }

        // ------------------------------------------------------------------
        // 4. Write buffer to disk.
        // ------------------------------------------------------------------
        std::fs::write(path, &buf).map_err(OxiH5Error::Io)
    }

    // ------------------------------------------------------------------
    // Internal helper — validates and enqueues a dataset descriptor.
    // ------------------------------------------------------------------
    fn add_dataset(
        &mut self,
        name: &str,
        raw: Vec<u8>,
        shape: &[usize],
        elem_type: ElemType,
    ) -> Result<&mut Self, OxiH5Error> {
        if self.datasets.len() >= 8 {
            return Err(OxiH5Error::Format(
                "FileWriter capacity exceeded: maximum 8 datasets per file".to_string(),
            ));
        }
        if name.is_empty() {
            return Err(OxiH5Error::Format(
                "dataset name must not be empty".to_string(),
            ));
        }
        if name.contains('/') {
            return Err(OxiH5Error::Format(format!(
                "dataset name '{name}' must not contain '/'"
            )));
        }
        if self.datasets.iter().any(|d| d.name == name) {
            return Err(OxiH5Error::Format(format!(
                "duplicate dataset name '{name}'"
            )));
        }
        self.datasets.push(DatasetDesc {
            name: name.to_string(),
            raw,
            shape: shape.to_vec(),
            elem_type,
        });
        Ok(self)
    }
}

// ---------------------------------------------------------------------------
// Size helpers
// ---------------------------------------------------------------------------

/// Total bytes consumed by one message in the object header:
/// 8-byte message header + body + padding to 8-byte boundary.
///
/// The `size` field in the message header stores the **unpadded** body_size.
/// The parser advances by `8 + aligned_size`; we track the same quantity.
fn msg_total(body_size: usize) -> usize {
    (8 + body_size + 7) & !7
}

/// Total size of a dataset object header (prefix + all message bodies).
fn compute_oh_size(ndims: usize, elem_type: &ElemType) -> usize {
    // Dataspace v1 with max dims: 8-byte header + ndims*8 dims + ndims*8 max_dims
    let ds_msg = msg_total(8 + ndims * 8 * 2);
    // Datatype: float class has 24-byte body, int class has 16-byte body
    let dt_msg = match elem_type {
        ElemType::F32 | ElemType::F64 => msg_total(24),
        ElemType::I32 | ElemType::I64 | ElemType::U8 => msg_total(16),
    };
    let fv_msg = msg_total(8); // Fill Value v2 (body=8) → total=16
                               // Layout v3 contiguous: 1+1+8+8=18 bytes but h5py requires msg body
                               // to end on an 8-byte boundary; 18 mod 8 = 2, so declare 24 bytes
                               // (6 trailing zeros are already present from zero-initialised buffer).
    let lo_msg = msg_total(24); // Layout v3 contiguous (body=24) → total=32
    16 + ds_msg + dt_msg + fv_msg + lo_msg
}

// ---------------------------------------------------------------------------
// Byte-write helpers
// ---------------------------------------------------------------------------

#[inline]
fn write_u16_le(buf: &mut [u8], offset: usize, val: u16) {
    buf[offset..offset + 2].copy_from_slice(&val.to_le_bytes());
}

#[inline]
fn write_u32_le(buf: &mut [u8], offset: usize, val: u32) {
    buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
}

#[inline]
fn write_u64_le(buf: &mut [u8], offset: usize, val: u64) {
    buf[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
}

/// Write an 8-byte object header message header.
///
/// Format: type(2) + size(2, unpadded body) + flags(1) + reserved(3).
/// Buffer was zero-initialised, so flags and reserved are already 0.
fn write_msg_header(buf: &mut [u8], offset: usize, msg_type: u16, body_size: u16) {
    write_u16_le(buf, offset, msg_type);
    write_u16_le(buf, offset + 2, body_size);
    // offset+4 = flags = 0 (already)
    // offset+5..8 = reserved = 0 (already)
}

/// Write an 8-byte object header message header with explicit flags byte.
fn write_msg_header_flags(buf: &mut [u8], offset: usize, msg_type: u16, body_size: u16, flags: u8) {
    write_u16_le(buf, offset, msg_type);
    write_u16_le(buf, offset + 2, body_size);
    buf[offset + 4] = flags;
    // offset+5..8 = reserved = 0 (already)
}

// ---------------------------------------------------------------------------
// HDF5 signature (bytes 0..8)
// ---------------------------------------------------------------------------

fn write_signature(buf: &mut [u8]) {
    buf[0..8].copy_from_slice(&[0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a]);
}

// ---------------------------------------------------------------------------
// Superblock v0 (bytes 8..96)
// ---------------------------------------------------------------------------
//
// Byte layout (relative to start of file):
//   0..8   Signature (written by write_signature)
//   8      superblock version = 0
//   9      free-space version = 0
//  10      root group STE version = 0
//  11      reserved = 0
//  12      shared header msg version = 0
//  13      size_of_offsets = 8
//  14      size_of_lengths = 8
//  15      reserved = 0
//  16..18  leaf_node_K = 4 (u16 LE)
//  18..20  internal_node_K = 16 (u16 LE)
//  20..24  file consistency flags = 0 (u32 LE)
//  24..32  base address = 0 (u64 LE)
//  32..40  free space address = u64::MAX (u64 LE)
//  40..48  eof address (u64 LE)
//  48..56  driver info block = u64::MAX (u64 LE)
//  56..96  Root Group Symbol Table Entry (40 bytes):
//    56..64  link_name_offset = 0 (u64 LE)
//    64..72  object_header_address = 96 (u64 LE)  ← root OH is always at 96
//    72..76  cache_type = 1 (u32 LE)              ← 1 = root group
//    76..80  reserved = 0 (u32 LE)
//    80..88  B-tree address = 136 (u64 LE)        ← B-tree always at 136
//    88..96  local heap address = 184 (u64 LE)    ← heap always at 184

fn write_superblock(buf: &mut [u8], eof_addr: u64) {
    // Version fields
    buf[8] = 0x00; // superblock version 0
    buf[9] = 0x00; // free-space version
    buf[10] = 0x00; // root group STE version
    buf[11] = 0x00; // reserved
    buf[12] = 0x00; // shared header msg version
    buf[13] = 0x08; // size_of_offsets = 8
    buf[14] = 0x08; // size_of_lengths = 8
    buf[15] = 0x00; // reserved
    write_u16_le(buf, 16, 4); // leaf_node_K
    write_u16_le(buf, 18, 16); // internal_node_K
    write_u32_le(buf, 20, 0); // file consistency flags

    // Addresses
    write_u64_le(buf, 24, 0); // base address
    write_u64_le(buf, 32, u64::MAX); // free space address (undefined)
    write_u64_le(buf, 40, eof_addr); // end of file
    write_u64_le(buf, 48, u64::MAX); // driver info block (undefined)

    // Root Group Symbol Table Entry
    write_u64_le(buf, 56, 0); // link_name_offset = 0
    write_u64_le(buf, 64, 96); // root group OH address
    write_u32_le(buf, 72, 1); // cache_type = 1 (root group)
    write_u32_le(buf, 76, 0); // reserved
    write_u64_le(buf, 80, 136); // B-tree address
    write_u64_le(buf, 88, 184); // local heap address
}

// ---------------------------------------------------------------------------
// Root group object header v1 (bytes 96..136, 40 bytes)
// ---------------------------------------------------------------------------
//
// OH v1 prefix (16 bytes at 96..112):
//   96      version = 1
//   97      reserved = 0
//   98..100 num_messages = 1 (u16 LE)
//  100..104 reference count = 1 (u32 LE)
//  104..108 header_data_size = 24 (u32 LE)  ← one 24-byte Symbol Table message
//  108..112 reserved = 0 (u32 LE)
//
// Symbol Table message (0x0011) at 112..136 (8-byte header + 16-byte body):
//  112..114  msg type = 0x0011 (u16 LE)
//  114..116  body size = 16 (u16 LE)      ← unpadded
//  116       flags = 0
//  117..120  reserved = 0,0,0
//  120..128  B-tree address = 136 (u64 LE)
//  128..136  local heap address = 184 (u64 LE)

fn write_root_oh(buf: &mut [u8]) {
    const BASE: usize = 96;

    // OH v1 prefix
    buf[BASE] = 0x01; // version
    buf[BASE + 1] = 0x00; // reserved
    write_u16_le(buf, BASE + 2, 1); // num_messages = 1
    write_u32_le(buf, BASE + 4, 1); // reference count = 1
                                    // header_data_size: one message = 8-byte header + 16-byte body = 24 bytes
                                    // (body 16 is already a multiple of 8, so no extra padding needed)
    write_u32_le(buf, BASE + 8, 24); // header_data_size
    write_u32_le(buf, BASE + 12, 0); // reserved

    // Symbol Table message at 112
    const MSG_BASE: usize = 112;
    write_msg_header(buf, MSG_BASE, 0x0011, 16); // body_size=16 (unpadded)
    write_u64_le(buf, MSG_BASE + 8, 136); // B-tree address
    write_u64_le(buf, MSG_BASE + 16, 184); // local heap address
}

// ---------------------------------------------------------------------------
// B-tree v1 leaf node (bytes 136..184, 48 bytes)
// ---------------------------------------------------------------------------
//
//  136..140  "TREE"
//  140       node_type = 0 (group B-tree)
//  141       level = 0 (leaf)
//  142..144  entries_used = 1 (u16 LE)
//  144..152  left sibling = u64::MAX (u64 LE)
//  152..160  right sibling = u64::MAX (u64 LE)
//  160..168  key[0] = 0 (u64 LE)
//  168..176  child[0] = snod_addr (u64 LE)
//  176..184  key[1] = heap offset of last entry name (u64 LE)

fn write_btree_leaf(buf: &mut [u8], snod_addr: u64, key1: u64) {
    const BASE: usize = 136;
    buf[BASE..BASE + 4].copy_from_slice(b"TREE");
    buf[BASE + 4] = 0x00; // node_type = 0 (group)
    buf[BASE + 5] = 0x00; // level = 0 (leaf)
    write_u16_le(buf, BASE + 6, 1); // entries_used = 1
    write_u64_le(buf, BASE + 8, u64::MAX); // left sibling (undefined)
    write_u64_le(buf, BASE + 16, u64::MAX); // right sibling (undefined)
    write_u64_le(buf, BASE + 24, 0); // key[0]
    write_u64_le(buf, BASE + 32, snod_addr); // child[0]
                                             // key[1]: heap offset of the last (greatest) name entry; bounds the child.
                                             // h5py requires this to be non-zero for non-empty files so it can
                                             // locate dataset names via binary search on the B-tree.
    write_u64_le(buf, BASE + 40, key1); // key[1]
}

// ---------------------------------------------------------------------------
// Local heap header (bytes 184..216) + data segment (bytes 216..)
// ---------------------------------------------------------------------------
//
//  184..188  "HEAP"
//  188       version = 0
//  189..192  reserved = 0,0,0
//  192..200  data_segment_size = heap_data_size_aligned (u64 LE)
//  200..208  free_list_head = u64::MAX (u64 LE)  "no free space"
//  208..216  data_segment_address = 216 (u64 LE)
//  216…      heap data (already written into buf via copy)

fn write_local_heap(
    buf: &mut [u8],
    heap_data: &[u8],
    heap_data_size_aligned: usize,
    free_list_head: usize,
) {
    const BASE: usize = 184;
    buf[BASE..BASE + 4].copy_from_slice(b"HEAP");
    buf[BASE + 4] = 0x00; // version = 0
                          // BASE+5..8 = reserved = 0 (already)
    write_u64_le(buf, BASE + 8, heap_data_size_aligned as u64); // data segment size
    write_u64_le(buf, BASE + 16, free_list_head as u64); // offset of first free block
    write_u64_le(buf, BASE + 24, 216); // data segment address (constant)

    // Write heap data into the data segment.
    buf[216..216 + heap_data.len()].copy_from_slice(heap_data);
}

// ---------------------------------------------------------------------------
// SNOD (Symbol Table Node) at snod_addr
// ---------------------------------------------------------------------------
//
// K=4, capacity = 2K = 8 entries. Total = 8 + 8×40 = 328 bytes.
//
//  0..4    "SNOD"
//  4       version = 1
//  5       reserved = 0
//  6..8    num_symbols = n (u16 LE)
//  8..8+40*n  Symbol Table Entries (STE) for each dataset
//  8+40*n..328  zero padding for unused entries
//
// Each STE (40 bytes):
//   0..8    name_offset_in_heap (u64 LE)
//   8..16   object_header_address (u64 LE)
//  16..20   cache_type = 0 (u32 LE)
//  20..24   reserved = 0 (u32 LE)
//  24..40   scratch-pad = 16 zero bytes

fn write_snod(
    buf: &mut [u8],
    snod_addr: usize,
    n: usize,
    name_offsets: &[u64],
    oh_addrs: &[usize],
) {
    let base = snod_addr;
    buf[base..base + 4].copy_from_slice(b"SNOD");
    buf[base + 4] = 0x01; // version = 1
    buf[base + 5] = 0x00; // reserved
    write_u16_le(buf, base + 6, n as u16); // num_symbols

    for i in 0..n {
        let ste_base = base + 8 + i * 40;
        write_u64_le(buf, ste_base, name_offsets[i]); // name offset in heap
        write_u64_le(buf, ste_base + 8, oh_addrs[i] as u64); // OH address
        write_u32_le(buf, ste_base + 16, 0); // cache_type = 0
        write_u32_le(buf, ste_base + 20, 0); // reserved
                                             // scratch-pad: 16 zero bytes at ste_base+24 — already zero
    }
    // Remaining entries (n..7) are already zero from the vec initialisation.
}

// ---------------------------------------------------------------------------
// Dataset object header v1
// ---------------------------------------------------------------------------
//
// OH v1 prefix (16 bytes):
//   0       version = 1
//   1       reserved = 0
//   2..4    num_messages = 4 (u16 LE)
//   4..8    reference_count = 1 (u32 LE)
//   8..12   header_data_size (u32 LE) = sum of msg_total for all 4 messages
//  12..16   reserved = 0 (u32 LE)
//
// Followed by 4 messages:
//   1. Dataspace  (0x0001)
//   2. Datatype   (0x0003)
//   3. Fill Value (0x0005)
//   4. Layout     (0x0008)

fn write_dataset_oh(buf: &mut [u8], oh_addr: usize, ds: &DatasetDesc, data_addr: u64) {
    let ndims = ds.shape.len();

    // ---- Body sizes (unpadded) ----
    // Dataspace v1 with max dims: 8-byte fixed header + dims + max_dims
    let ds_body_size = 8 + ndims * 8 * 2;
    // Datatype: float uses 24-byte body, int uses 16-byte body
    let dt_body_size: usize = match ds.elem_type {
        ElemType::F32 | ElemType::F64 => 24,
        ElemType::I32 | ElemType::I64 | ElemType::U8 => 16,
    };
    let fv_body_size: usize = 8; // Fill Value v2 (8 bytes, allocation_time=2)
                                 // Layout v3 contiguous has 18 meaningful bytes (ver+class+addr+size),
                                 // but the body_size field must be declared as 24 so that
                                 // (msg_start + 8 + body_size) % 8 == 0 (required by HDF5 library / h5py).
                                 // The 6 trailing bytes are zero from the pre-zeroed buffer.
    let lo_body_size: usize = 24; // Layout v3 contiguous (padded to 8-byte boundary)

    let header_data_size = msg_total(ds_body_size)
        + msg_total(dt_body_size)
        + msg_total(fv_body_size)
        + msg_total(lo_body_size);

    // ---- OH v1 prefix (16 bytes) ----
    buf[oh_addr] = 0x01; // version = 1
    buf[oh_addr + 1] = 0x00; // reserved
    write_u16_le(buf, oh_addr + 2, 4); // num_messages = 4
    write_u32_le(buf, oh_addr + 4, 1); // reference count = 1
    write_u32_le(buf, oh_addr + 8, header_data_size as u32);
    write_u32_le(buf, oh_addr + 12, 0); // reserved

    let mut pos = oh_addr + 16;

    // ---- Message 1: Dataspace (0x0001), flags=0x00 ----
    // Body: version(1) + ndims(1) + flags=0x01(max dims)(1) + reserved(5) +
    //       dims(ndims*8) + max_dims(ndims*8)
    write_msg_header(buf, pos, 0x0001, ds_body_size as u16);
    let body_start = pos + 8;
    buf[body_start] = 0x01; // version = 1
    buf[body_start + 1] = ndims as u8; // dimensionality
    buf[body_start + 2] = 0x01; // flags = 0x01 (max dims present)
                                // bytes 3..8 = reserved = 0 (already)
    for (i, &dim) in ds.shape.iter().enumerate() {
        write_u64_le(buf, body_start + 8 + i * 8, dim as u64); // dim sizes
        write_u64_le(buf, body_start + 8 + ndims * 8 + i * 8, dim as u64); // max dim sizes
    }
    pos += msg_total(ds_body_size);

    // ---- Message 2: Datatype (0x0003), flags=0x01 (constant) ----
    write_msg_header_flags(buf, pos, 0x0003, dt_body_size as u16, 0x01);
    let dt_start = pos + 8;
    write_datatype_body(buf, dt_start, &ds.elem_type);
    pos += msg_total(dt_body_size);

    // ---- Message 3: Fill Value (0x0005), flags=0x01 (constant) ----
    // Body v2: version(1)=2 + alloc_time(1)=2 + write_time(1)=2 + defined(1)=1 + 4 zeros
    write_msg_header_flags(buf, pos, 0x0005, fv_body_size as u16, 0x01);
    let fv_start = pos + 8;
    buf[fv_start] = 0x02; // version = 2
    buf[fv_start + 1] = 0x02; // space_allocation_time = 2 (early)
    buf[fv_start + 2] = 0x02; // fill_write_time = 2 (if defined)
    buf[fv_start + 3] = 0x01; // fill_value_defined = 1 (not defined, but h5py sets 1)
                              // bytes 4..8 = 0x00000000 (already zero)
    pos += msg_total(fv_body_size);

    // ---- Message 4: Data Layout (0x0008), flags=0x00 ----
    // body_size=24 (declared); only 18 bytes are meaningful data.
    // The 6 trailing bytes remain zero (from zero-initialised buffer).
    // Declaring 24 ensures (msg_start+8+body_size) % 8 == 0 as required by h5py.
    write_msg_header(buf, pos, 0x0008, lo_body_size as u16);
    let lo_start = pos + 8;
    // Layout v3 contiguous: version(1)=3 + class(1)=1 + data_addr(8) + data_size(8)
    buf[lo_start] = 0x03; // version = 3
    buf[lo_start + 1] = 0x01; // class = 1 (contiguous)
    write_u64_le(buf, lo_start + 2, data_addr);
    write_u64_le(buf, lo_start + 10, ds.raw.len() as u64);
    // bytes lo_start+18..lo_start+24 are 0x00 (already zero)
}

// ---------------------------------------------------------------------------
// Datatype body encodings
// ---------------------------------------------------------------------------
//
// Float class (class=1, ver=1):  20-byte body
// Int class   (class=0, ver=1):  14-byte body

/// Write the datatype class property block.
///
/// Byte layouts are derived from h5py reference files (libver='earliest'):
/// - Float class (class=1, ver=1): 24-byte body
/// - Integer class (class=0, ver=1): 16-byte body
fn write_datatype_body(buf: &mut [u8], start: usize, elem_type: &ElemType) {
    match elem_type {
        ElemType::F32 => {
            // IEEE 754 single precision, little-endian.
            // Exact bytes from h5py reference file.
            let body: [u8; 24] = [
                0x11, 0x20, 0x1f, 0x00, // class_ver + bit_fields[0..2]
                0x04, 0x00, 0x00, 0x00, // elem_size = 4
                0x00, 0x00, // bit_offset = 0
                0x20, 0x00, // bit_precision = 32
                0x17, // exponent_location = 23
                0x08, // exponent_size = 8
                0x00, // mantissa_location = 0
                0x17, // mantissa_size = 23
                0x7f, 0x00, 0x00, 0x00, // exponent_bias = 127
                0x00, 0x00, 0x00, 0x00, // padding / reserved
            ];
            buf[start..start + 24].copy_from_slice(&body);
        }
        ElemType::F64 => {
            // IEEE 754 double precision, little-endian.
            let body: [u8; 24] = [
                0x11, 0x20, 0x3f, 0x00, // class_ver + bit_fields[0..2]
                0x08, 0x00, 0x00, 0x00, // elem_size = 8
                0x00, 0x00, // bit_offset = 0
                0x40, 0x00, // bit_precision = 64
                0x34, // exponent_location = 52
                0x0b, // exponent_size = 11
                0x00, // mantissa_location = 0
                0x34, // mantissa_size = 52
                0xff, 0x03, 0x00, 0x00, // exponent_bias = 1023
                0x00, 0x00, 0x00, 0x00, // padding / reserved
            ];
            buf[start..start + 24].copy_from_slice(&body);
        }
        ElemType::I32 => {
            // Signed 32-bit integer, little-endian.
            // bit_fields[0]=0x08: signed (bit 3), LE (bit 0 = 0).
            let body: [u8; 16] = [
                0x10, 0x08, 0x00, 0x00, // class_ver + bit_fields[0..2]
                0x04, 0x00, 0x00, 0x00, // elem_size = 4
                0x00, 0x00, // bit_offset = 0
                0x20, 0x00, // bit_precision = 32
                0x00, 0x00, // pad_type_low, pad_type_high
                0x00, 0x00, // reserved
            ];
            buf[start..start + 16].copy_from_slice(&body);
        }
        ElemType::I64 => {
            // Signed 64-bit integer, little-endian.
            let body: [u8; 16] = [
                0x10, 0x08, 0x00, 0x00, // class_ver + bit_fields[0..2]
                0x08, 0x00, 0x00, 0x00, // elem_size = 8
                0x00, 0x00, // bit_offset = 0
                0x40, 0x00, // bit_precision = 64
                0x00, 0x00, // pad_type
                0x00, 0x00, // reserved
            ];
            buf[start..start + 16].copy_from_slice(&body);
        }
        ElemType::U8 => {
            // Unsigned 8-bit integer.
            // bit_fields[0]=0x00: unsigned (bit 3 clear), LE (bit 0 = 0).
            let body: [u8; 16] = [
                0x10, 0x00, 0x00, 0x00, // class_ver + bit_fields[0..2]
                0x01, 0x00, 0x00, 0x00, // elem_size = 1
                0x00, 0x00, // bit_offset = 0
                0x08, 0x00, // bit_precision = 8
                0x00, 0x00, // pad_type
                0x00, 0x00, // reserved
            ];
            buf[start..start + 16].copy_from_slice(&body);
        }
    }
}
