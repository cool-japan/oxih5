//! Virtual Dataset (VDS) global-heap mapping block parsing.
//!
//! A dataset stored with layout class 3 ("virtual") does not hold any raw data
//! of its own.  Instead its data layout message carries a global-heap object ID
//! (`collection address` + `object index`) that points at a serialized *mapping
//! block* in the file's global heap.  That block describes, for each source
//! dataset, which region of the source maps onto which region of the virtual
//! dataset's dataspace.
//!
//! On-disk layout of the mapping block (HDF5 file-format spec §IV.A.2.q,
//! "Virtual Dataset Global Heap Block"):
//!
//! ```text
//!   Version              : 1 byte  (== 0)
//!   Number of Entries    : `size_of_lengths` bytes (usually 8, LE)
//!   Entry 0 .. N-1:
//!       Source Filename  : NUL-terminated string
//!       Source Dataset   : NUL-terminated string
//!       Source Selection : dataspace selection (see below)
//!       Virtual Selection: dataspace selection
//!   Checksum             : 4 bytes (Jenkins lookup3 over the block; not verified here)
//! ```
//!
//! Dataspace selection encoding (the `Selection Type` field, first 4 bytes LE):
//!
//! * `0` = none        (empty selection)
//! * `1` = points      (explicit point list — not yet supported)
//! * `2` = hyperslab   (regular hyperslab; version 1 and version 2 handled)
//! * `3` = all         (the entire dataspace)

use crate::hyperslab::{DimSelection, Hyperslab};
use oxih5_core::OxiH5Error;

/// A parsed dataspace selection as it appears inside a VDS mapping entry.
#[derive(Debug, Clone, PartialEq)]
pub enum VdsSelection {
    /// `H5S_SEL_NONE` — selects nothing.
    None,
    /// `H5S_SEL_ALL` — selects the entire dataspace.  The concrete extent is
    /// only known once the corresponding dataset's dataspace is available.
    All,
    /// `H5S_SEL_HYPERSLABS` — a regular hyperslab, one [`DimSelection`] per rank.
    Hyperslab(Hyperslab),
}

/// A single VDS mapping entry: one source region → one virtual region.
#[derive(Debug, Clone)]
pub struct VdsEntry {
    /// Source file path.  `"."` (or empty) means "this same file".
    pub source_file: String,
    /// Path of the source dataset within `source_file`.
    pub source_dataset: String,
    /// Region selected within the source dataset.
    pub source_selection: VdsSelection,
    /// Region of the virtual dataset that the source region maps onto.
    pub virtual_selection: VdsSelection,
}

/// The complete set of VDS mapping entries decoded from a global-heap block.
#[derive(Debug, Clone)]
pub struct VdsMapping {
    /// Mapping entries, in file order.
    pub entries: Vec<VdsEntry>,
}

/// Parse the VDS mapping block stored as global-heap object `heap_index` inside
/// the collection at `heap_address`.
///
/// `size_of_lengths` is the superblock's "Size of Lengths" field (typically 8);
/// it controls the width of the "Number of Entries" field.
pub fn parse_vds_mapping(
    file_data: &[u8],
    heap_address: u64,
    heap_index: u32,
    size_of_lengths: usize,
) -> Result<VdsMapping, OxiH5Error> {
    let heap = crate::global_heap::GlobalHeap::parse(file_data, heap_address)?;
    let index = u16::try_from(heap_index).map_err(|_| {
        OxiH5Error::Format(format!("VDS: heap object index {heap_index} out of range"))
    })?;
    let block = heap.object(index)?;
    parse_vds_block(block, size_of_lengths)
}

/// Parse a raw VDS mapping block (the contents of the global-heap object).
pub fn parse_vds_block(block: &[u8], size_of_lengths: usize) -> Result<VdsMapping, OxiH5Error> {
    let mut cur = Reader::new(block);
    let version = cur.u8()?;
    if version > 1 {
        return Err(OxiH5Error::Format(format!(
            "VDS block: unsupported version {version}"
        )));
    }
    let num_entries = cur.uint(size_of_lengths)?;
    let num_entries = usize::try_from(num_entries)
        .map_err(|_| OxiH5Error::Format("VDS block: entry count out of range".into()))?;
    // Guard against absurd counts that would otherwise pre-allocate huge vectors.
    if num_entries > block.len() {
        return Err(OxiH5Error::Format(format!(
            "VDS block: entry count {num_entries} exceeds block size {}",
            block.len()
        )));
    }

    // Version 1 deduplicates repeated source filenames: each entry is prefixed
    // by a 1-byte flag — `0` means an inline NUL-terminated filename follows,
    // `1` means an 8-byte back-reference to an earlier entry's filename.  We
    // keep the resolved filename of every entry so references can be resolved.
    let mut entries = Vec::with_capacity(num_entries);
    let mut file_names: Vec<String> = Vec::with_capacity(num_entries);
    for _ in 0..num_entries {
        let source_file = if version >= 1 {
            let flag = cur.u8()?;
            match flag {
                0 => cur.cstr()?,
                1 => {
                    let idx = usize::try_from(cur.uint(size_of_lengths)?).map_err(|_| {
                        OxiH5Error::Format("VDS block: filename reference out of range".into())
                    })?;
                    file_names.get(idx).cloned().ok_or_else(|| {
                        OxiH5Error::Format(format!(
                            "VDS block: filename reference {idx} has no prior entry"
                        ))
                    })?
                }
                other => {
                    return Err(OxiH5Error::Format(format!(
                        "VDS block: unknown source-file flag {other}"
                    )))
                }
            }
        } else {
            cur.cstr()?
        };
        file_names.push(source_file.clone());
        let source_dataset = cur.cstr()?;
        let source_selection = parse_selection(&mut cur)?;
        let virtual_selection = parse_selection(&mut cur)?;
        entries.push(VdsEntry {
            source_file,
            source_dataset,
            source_selection,
            virtual_selection,
        });
    }
    // A trailing 4-byte checksum follows; it is not verified here.
    Ok(VdsMapping { entries })
}

/// Enumerate the row-major *element* indices selected by `sel` within a
/// dataspace of the given `shape`.
///
/// * [`VdsSelection::All`] selects every element (`0..product(shape)`).
/// * [`VdsSelection::None`] selects nothing.
/// * [`VdsSelection::Hyperslab`] enumerates its selected coordinates in
///   row-major order (last dimension varying fastest), matching the order HDF5
///   uses when pairing a source selection with a virtual selection.
///
/// The returned indices are element offsets (multiply by element size for bytes).
pub fn selection_element_offsets(
    sel: &VdsSelection,
    shape: &[u64],
) -> Result<Vec<usize>, OxiH5Error> {
    let total: u64 = shape.iter().product();
    match sel {
        VdsSelection::None => Ok(Vec::new()),
        VdsSelection::All => {
            let n = usize::try_from(total)
                .map_err(|_| OxiH5Error::Format("VDS: dataspace too large".into()))?;
            Ok((0..n).collect())
        }
        VdsSelection::Hyperslab(hs) => {
            if hs.dims.len() != shape.len() {
                return Err(OxiH5Error::Format(format!(
                    "VDS: selection rank {} does not match dataspace rank {}",
                    hs.dims.len(),
                    shape.len()
                )));
            }
            // Row-major strides for the dataspace.
            let rank = shape.len();
            let mut strides = vec![1u64; rank];
            for d in (0..rank.saturating_sub(1)).rev() {
                strides[d] = strides[d + 1] * shape[d + 1];
            }
            // Selected coordinate list per dimension.
            let per_dim: Vec<Vec<u64>> = hs
                .dims
                .iter()
                .map(|ds| {
                    let mut coords = Vec::new();
                    for k in 0..ds.count {
                        for j in 0..ds.block {
                            coords.push(ds.start + k * ds.stride + j);
                        }
                    }
                    coords
                })
                .collect();
            // Validate bounds.
            for (d, coords) in per_dim.iter().enumerate() {
                if let Some(&max) = coords.iter().max() {
                    if max >= shape[d] {
                        return Err(OxiH5Error::Format(format!(
                            "VDS: selection index {max} exceeds dimension {d} extent {}",
                            shape[d]
                        )));
                    }
                }
            }
            // Cartesian product in row-major order.
            let count: usize = per_dim.iter().map(|c| c.len()).product();
            let mut offsets = Vec::with_capacity(count);
            let mut idx = vec![0usize; rank];
            if per_dim.iter().any(|c| c.is_empty()) {
                return Ok(offsets);
            }
            loop {
                let mut lin = 0u64;
                for d in 0..rank {
                    lin += per_dim[d][idx[d]] * strides[d];
                }
                offsets.push(
                    usize::try_from(lin)
                        .map_err(|_| OxiH5Error::Format("VDS: offset overflow".into()))?,
                );
                // Increment the multi-index (last dim fastest).
                let mut d = rank;
                loop {
                    if d == 0 {
                        return Ok(offsets);
                    }
                    d -= 1;
                    idx[d] += 1;
                    if idx[d] < per_dim[d].len() {
                        break;
                    }
                    idx[d] = 0;
                }
            }
        }
    }
}

/// Parse one serialized dataspace selection starting at the reader's cursor.
///
/// Every selection begins with `type: u32` then `version: u32` (HDF5's
/// `UINT32ENCODE(type); UINT32ENCODE(version)` preamble).
fn parse_selection(cur: &mut Reader<'_>) -> Result<VdsSelection, OxiH5Error> {
    let sel_type = cur.u32()?;
    let version = cur.u32()?;
    match sel_type {
        0 => {
            // None: reserved(4) + length(4, == 0).
            let _reserved = cur.u32()?;
            let _length = cur.u32()?;
            Ok(VdsSelection::None)
        }
        3 => {
            // All: reserved(4) + length(4, == 0).
            let _reserved = cur.u32()?;
            let _length = cur.u32()?;
            Ok(VdsSelection::All)
        }
        2 => parse_hyperslab_selection(cur, version),
        1 => Err(OxiH5Error::NotImplemented(
            "VDS: point selections are not supported".into(),
        )),
        other => Err(OxiH5Error::Format(format!(
            "VDS: unknown selection type {other}"
        ))),
    }
}

/// Parse a hyperslab dataspace selection (type 2) whose `type` and `version`
/// words have already been consumed.
///
/// * Version 1 — classic: `reserved(4), length(4), rank(4), num_blocks(4)`,
///   then a block list of `start[rank], end[rank]` (each value `u32`).
/// * Version 3 — modern (what HDF5 ≥ 1.10 writes for VDS): `flags(1),
///   enc_size(1), rank(4)`, then, for a *regular* hyperslab (flags bit 0 set),
///   per-dimension `start, stride, count, block`, each `enc_size` bytes LE.
fn parse_hyperslab_selection(
    cur: &mut Reader<'_>,
    version: u32,
) -> Result<VdsSelection, OxiH5Error> {
    match version {
        1 => {
            let _reserved = cur.u32()?;
            let _length = cur.u32()?;
            let rank = cur.u32()? as usize;
            let num_blocks = cur.u32()? as usize;
            parse_hyperslab_v1_blocks(cur, rank, num_blocks)
        }
        3 => {
            let flags = cur.u8()?;
            let enc_size = cur.u8()? as usize;
            if !matches!(enc_size, 1 | 2 | 4 | 8) {
                return Err(OxiH5Error::Format(format!(
                    "VDS hyperslab v3: invalid encode size {enc_size}"
                )));
            }
            let rank = cur.u32()? as usize;
            if rank == 0 || rank > 32 {
                return Err(OxiH5Error::Format(format!(
                    "VDS hyperslab v3: implausible rank {rank}"
                )));
            }
            // Bit 0 of `flags` marks a regular hyperslab; irregular selections
            // store an explicit span tree we do not decode.
            if flags & 0x01 == 0 {
                return Err(OxiH5Error::NotImplemented(
                    "VDS: irregular (non-regular) hyperslab selections are not supported".into(),
                ));
            }
            let mut dims = Vec::with_capacity(rank);
            for _ in 0..rank {
                let start = cur.uint(enc_size)?;
                let stride = cur.uint(enc_size)?;
                let count = cur.uint(enc_size)?;
                let block = cur.uint(enc_size)?;
                dims.push(DimSelection {
                    start,
                    stride: stride.max(1),
                    count,
                    block,
                });
            }
            Ok(VdsSelection::Hyperslab(Hyperslab { dims }))
        }
        other => Err(OxiH5Error::NotImplemented(format!(
            "VDS hyperslab: unsupported selection info version {other}"
        ))),
    }
}

/// Parse version-1 hyperslab block descriptors and fold them into a regular
/// [`Hyperslab`].  Version 1 stores an explicit list of `(start, end)` blocks.
///
/// A single block collapses to a stride-1 selection.  Multiple *regular* blocks
/// (equal spacing, equal size) collapse into a strided selection.  Irregular
/// unions cannot be represented by a single regular [`Hyperslab`] and yield a
/// typed `NotImplemented` error.
fn parse_hyperslab_v1_blocks(
    cur: &mut Reader<'_>,
    rank: usize,
    num_blocks: usize,
) -> Result<VdsSelection, OxiH5Error> {
    if rank == 0 || rank > 32 {
        return Err(OxiH5Error::Format(format!(
            "VDS hyperslab v1: implausible rank {rank}"
        )));
    }
    if num_blocks == 0 {
        return Ok(VdsSelection::None);
    }
    // `num_blocks` is a raw `u32` read straight off disk
    // (`parse_hyperslab_selection`'s version-1 arm) with no upstream bound,
    // so `Vec::with_capacity(num_blocks)` below would otherwise let a tiny
    // attacker-controlled input request an up-front allocation of tens of
    // gigabytes — an OOM found by fuzzing (`fuzz/fuzz_targets/fuzz_vds.rs`,
    // on a 39-byte input) — the same "count field used unchecked as
    // `Vec::with_capacity`" hazard already fixed for the Fixed Array
    // chunk-index parser's element count. Each block needs at least
    // `rank * 8` bytes (`start[rank]` + `end[rank]`, each a `u32`), so reject
    // any count that could not possibly fit in what's left of the buffer
    // before allocating for it.
    let min_bytes_per_block = rank * 8;
    if num_blocks > cur.remaining() / min_bytes_per_block {
        return Err(OxiH5Error::Format(format!(
            "VDS hyperslab v1: block count {num_blocks} exceeds what fits in the remaining {} bytes",
            cur.remaining()
        )));
    }
    // Read the blocks: each block is start[rank] then end[rank] (inclusive), all u32.
    let mut blocks = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        let mut start = Vec::with_capacity(rank);
        for _ in 0..rank {
            start.push(cur.u32()? as u64);
        }
        let mut end = Vec::with_capacity(rank);
        for _ in 0..rank {
            end.push(cur.u32()? as u64);
        }
        blocks.push((start, end));
    }

    if num_blocks == 1 {
        let (start, end) = &blocks[0];
        let dims = start
            .iter()
            .zip(end.iter())
            .map(|(&s, &e)| DimSelection {
                start: s,
                stride: 1,
                count: e.saturating_sub(s) + 1,
                block: 1,
            })
            .collect();
        return Ok(VdsSelection::Hyperslab(Hyperslab { dims }));
    }

    // Multiple blocks that vary along exactly one dimension with constant
    // spacing and constant block size collapse to a strided selection.
    if let Some(hs) = fold_regular_blocks(&blocks, rank) {
        return Ok(VdsSelection::Hyperslab(hs));
    }

    Err(OxiH5Error::NotImplemented(
        "VDS: irregular multi-block hyperslab unions are not supported".into(),
    ))
}

/// Attempt to fold an explicit block list into a single regular hyperslab.
/// Returns `None` when the blocks do not form a regular grid varying along a
/// single dimension.
fn fold_regular_blocks(blocks: &[(Vec<u64>, Vec<u64>)], rank: usize) -> Option<Hyperslab> {
    let (first_start, first_end) = &blocks[0];
    let block_size: Vec<u64> = first_start
        .iter()
        .zip(first_end.iter())
        .map(|(&s, &e)| e.saturating_sub(s) + 1)
        .collect();

    // Identify the single dimension along which the start offset varies.
    let mut varying_dim = None;
    for (b_start, b_end) in &blocks[1..] {
        for d in 0..rank {
            let bsz = b_end[d].saturating_sub(b_start[d]) + 1;
            if bsz != block_size[d] {
                return None; // block size must be constant across blocks
            }
            if b_start[d] != first_start[d] {
                match varying_dim {
                    None => varying_dim = Some(d),
                    Some(vd) if vd == d => {}
                    Some(_) => return None, // varies along more than one dim
                }
            }
        }
    }
    let vd = varying_dim?;

    // Collect the sorted, unique start offsets along the varying dimension and
    // check they are equally spaced.
    let mut starts: Vec<u64> = blocks.iter().map(|(s, _)| s[vd]).collect();
    starts.sort_unstable();
    starts.dedup();
    if starts.len() != blocks.len() {
        return None; // duplicate / overlapping blocks
    }
    let stride = starts[1] - starts[0];
    if stride == 0 {
        return None;
    }
    for w in starts.windows(2) {
        if w[1] - w[0] != stride {
            return None;
        }
    }

    let mut dims = Vec::with_capacity(rank);
    for d in 0..rank {
        if d == vd {
            dims.push(DimSelection {
                start: starts[0],
                stride,
                count: starts.len() as u64,
                block: block_size[d],
            });
        } else {
            dims.push(DimSelection {
                start: first_start[d],
                stride: 1,
                count: 1,
                block: block_size[d],
            });
        }
    }
    Some(Hyperslab { dims })
}

/// A tiny bounds-checked little-endian cursor over the mapping block bytes.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Number of bytes not yet consumed.
    ///
    /// `pos` never exceeds `data.len()` (every advance goes through
    /// [`Reader::take`], which bounds-checks before updating `pos`), so this
    /// subtraction cannot underflow.
    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], OxiH5Error> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| OxiH5Error::Format("VDS block: length overflow".into()))?;
        let slice = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| OxiH5Error::Format("VDS block: unexpected end of data".into()))?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, OxiH5Error> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, OxiH5Error> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Read an unsigned integer of `width` bytes (little-endian).
    fn uint(&mut self, width: usize) -> Result<u64, OxiH5Error> {
        if width == 0 || width > 8 {
            return Err(OxiH5Error::Format(format!(
                "VDS block: unsupported integer width {width}"
            )));
        }
        let b = self.take(width)?;
        let mut buf = [0u8; 8];
        buf[..width].copy_from_slice(b);
        Ok(u64::from_le_bytes(buf))
    }

    /// Read a NUL-terminated UTF-8 string, consuming the terminator.
    fn cstr(&mut self) -> Result<String, OxiH5Error> {
        let start = self.pos;
        let nul = self.data[start..]
            .iter()
            .position(|&b| b == 0)
            .ok_or_else(|| OxiH5Error::Format("VDS block: unterminated string".into()))?;
        let bytes = &self.data[start..start + nul];
        let s = std::str::from_utf8(bytes)
            .map_err(|_| OxiH5Error::Format("VDS block: non-UTF-8 string".into()))?
            .to_string();
        self.pos = start + nul + 1; // skip the NUL
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode an "all" selection: type(4)=3, version(4)=1, reserved(4), length(4).
    fn enc_all() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&3u32.to_le_bytes());
        v.extend_from_slice(&1u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v
    }

    /// Encode a version-3 regular hyperslab selection with 2-byte value encoding,
    /// matching what HDF5 writes for VDS mappings.
    fn enc_hyper_v3(dims: &[(u16, u16, u16, u16)]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&2u32.to_le_bytes()); // type = hyperslab
        v.extend_from_slice(&3u32.to_le_bytes()); // version 3
        v.push(0x01); // flags: regular
        v.push(2); // enc_size = 2 bytes per value
        v.extend_from_slice(&(dims.len() as u32).to_le_bytes()); // rank
        for &(s, st, c, b) in dims {
            v.extend_from_slice(&s.to_le_bytes());
            v.extend_from_slice(&st.to_le_bytes());
            v.extend_from_slice(&c.to_le_bytes());
            v.extend_from_slice(&b.to_le_bytes());
        }
        v
    }

    fn build_block(entries: &[(&str, &str, Vec<u8>, Vec<u8>)]) -> Vec<u8> {
        let mut v = Vec::new();
        v.push(0); // version
        v.extend_from_slice(&(entries.len() as u64).to_le_bytes()); // num entries (8 bytes)
        for (file, dset, src_sel, virt_sel) in entries {
            v.extend_from_slice(file.as_bytes());
            v.push(0);
            v.extend_from_slice(dset.as_bytes());
            v.push(0);
            v.extend_from_slice(src_sel);
            v.extend_from_slice(virt_sel);
        }
        v.extend_from_slice(&0u32.to_le_bytes()); // checksum placeholder
        v
    }

    #[test]
    fn parse_all_to_all_entry() {
        let block = build_block(&[(".", "src", enc_all(), enc_all())]);
        let mapping = parse_vds_block(&block, 8).unwrap();
        assert_eq!(mapping.entries.len(), 1);
        let e = &mapping.entries[0];
        assert_eq!(e.source_file, ".");
        assert_eq!(e.source_dataset, "src");
        assert_eq!(e.source_selection, VdsSelection::All);
        assert_eq!(e.virtual_selection, VdsSelection::All);
    }

    #[test]
    fn parse_hyperslab_v2_entry() {
        let src = enc_hyper_v3(&[(0, 1, 4, 1)]);
        let virt = enc_hyper_v3(&[(2, 1, 4, 1)]);
        let block = build_block(&[("ext.h5", "data", src, virt)]);
        let mapping = parse_vds_block(&block, 8).unwrap();
        let e = &mapping.entries[0];
        assert_eq!(e.source_file, "ext.h5");
        match &e.virtual_selection {
            VdsSelection::Hyperslab(hs) => {
                assert_eq!(hs.dims.len(), 1);
                assert_eq!(hs.dims[0].start, 2);
                assert_eq!(hs.dims[0].count, 4);
            }
            other => panic!("expected hyperslab, got {other:?}"),
        }
    }

    #[test]
    fn parse_via_global_heap() {
        let block = build_block(&[(".", "s", enc_all(), enc_all())]);
        let gcol = crate::global_heap::build_gcol_for_test(&[(1, &block)]);
        let mapping = parse_vds_mapping(&gcol, 0, 1, 8).unwrap();
        assert_eq!(mapping.entries.len(), 1);
    }

    #[test]
    fn reject_bad_version() {
        let mut block = build_block(&[(".", "s", enc_all(), enc_all())]);
        block[0] = 9; // bad version
        assert!(parse_vds_block(&block, 8).is_err());
    }

    #[test]
    fn truncated_block_errors() {
        let block = build_block(&[(".", "s", enc_all(), enc_all())]);
        let truncated = &block[..block.len() - 12];
        assert!(parse_vds_block(truncated, 8).is_err());
    }

    /// Build a version-1 block (filename deduplication): entry 0 stores the
    /// filename inline, entry 1 back-references it.
    fn build_block_v1(file: &str, entries: &[(&str, Vec<u8>, Vec<u8>)]) -> Vec<u8> {
        let mut v = Vec::new();
        v.push(1); // version 1
        v.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (i, (dset, src_sel, virt_sel)) in entries.iter().enumerate() {
            if i == 0 {
                v.push(0); // flag: inline
                v.extend_from_slice(file.as_bytes());
                v.push(0);
            } else {
                v.push(1); // flag: reference
                v.extend_from_slice(&0u64.to_le_bytes()); // ref entry 0
            }
            v.extend_from_slice(dset.as_bytes());
            v.push(0);
            v.extend_from_slice(src_sel);
            v.extend_from_slice(virt_sel);
        }
        v.extend_from_slice(&0u32.to_le_bytes()); // checksum
        v
    }

    #[test]
    fn parse_v1_dedup_filenames() {
        let block = build_block_v1(
            "shared.h5",
            &[
                ("a", enc_all(), enc_hyper_v3(&[(0, 1, 4, 1)])),
                ("b", enc_all(), enc_hyper_v3(&[(4, 1, 4, 1)])),
            ],
        );
        let mapping = parse_vds_block(&block, 8).unwrap();
        assert_eq!(mapping.entries.len(), 2);
        assert_eq!(mapping.entries[0].source_file, "shared.h5");
        assert_eq!(mapping.entries[1].source_file, "shared.h5"); // resolved via reference
        assert_eq!(mapping.entries[0].source_dataset, "a");
        assert_eq!(mapping.entries[1].source_dataset, "b");
    }

    // -----------------------------------------------------------------------
    // Regression: an implausible version-1 hyperslab block count must error,
    // not attempt a huge up-front allocation.
    //
    // `num_blocks` is a raw `u32` read straight off disk with no upstream
    // bound; before the fix, `Vec::with_capacity(num_blocks)` in
    // `parse_hyperslab_v1_blocks` would try to reserve capacity for up to
    // `u32::MAX` `(Vec<u64>, Vec<u64>)` tuples, aborting the process with an
    // OOM well before any of those blocks were actually read from the
    // (tiny) input buffer. Found by fuzzing
    // (`fuzz/fuzz_targets/fuzz_vds.rs`, OOM on a 39-byte input).
    // -----------------------------------------------------------------------

    #[test]
    fn test_hyperslab_v1_num_blocks_implausible_rejected_not_oom() {
        let data: &[u8] = &[0u8; 8]; // far too small to hold even one block
        let mut cur = Reader::new(data);
        let result = parse_hyperslab_v1_blocks(&mut cur, 2, u32::MAX as usize);
        assert!(
            result.is_err(),
            "an implausible block count must return an error, not attempt a huge allocation"
        );
    }

    /// Same hazard, exercised through the public parsing entry point
    /// (`parse_selection`, as reached from `parse_vds_block`) rather than
    /// calling the private helper directly.
    #[test]
    fn test_parse_selection_hyperslab_v1_huge_num_blocks_rejected_not_oom() {
        let mut v = Vec::new();
        v.extend_from_slice(&2u32.to_le_bytes()); // selection type = hyperslab
        v.extend_from_slice(&1u32.to_le_bytes()); // version 1
        v.extend_from_slice(&0u32.to_le_bytes()); // reserved
        v.extend_from_slice(&0u32.to_le_bytes()); // length
        v.extend_from_slice(&2u32.to_le_bytes()); // rank = 2
        v.extend_from_slice(&u32::MAX.to_le_bytes()); // num_blocks = implausible
        let mut cur = Reader::new(&v);
        let result = parse_selection(&mut cur);
        assert!(
            result.is_err(),
            "an implausible v1 block count must return an error, not attempt a huge allocation"
        );
    }

    #[test]
    fn selection_offsets_all_and_hyperslab() {
        // "all" over shape [2,3] → 0..6 in row-major order.
        assert_eq!(
            selection_element_offsets(&VdsSelection::All, &[2, 3]).unwrap(),
            vec![0, 1, 2, 3, 4, 5]
        );
        // hyperslab start=1,count=2,block=1 on a length-6 axis → indices 1,2.
        let hs = Hyperslab {
            dims: vec![DimSelection {
                start: 1,
                stride: 1,
                count: 2,
                block: 1,
            }],
        };
        assert_eq!(
            selection_element_offsets(&VdsSelection::Hyperslab(hs), &[6]).unwrap(),
            vec![1, 2]
        );
    }

    #[test]
    fn selection_offsets_2d_hyperslab_row_major() {
        // Select column 1 of a [2,3] dataspace: rows 0..2, col 1 → offsets 1, 4.
        let hs = Hyperslab {
            dims: vec![
                DimSelection {
                    start: 0,
                    stride: 1,
                    count: 2,
                    block: 1,
                },
                DimSelection {
                    start: 1,
                    stride: 1,
                    count: 1,
                    block: 1,
                },
            ],
        };
        assert_eq!(
            selection_element_offsets(&VdsSelection::Hyperslab(hs), &[2, 3]).unwrap(),
            vec![1, 4]
        );
    }

    #[test]
    fn selection_offsets_out_of_bounds_errors() {
        let hs = Hyperslab {
            dims: vec![DimSelection {
                start: 5,
                stride: 1,
                count: 2,
                block: 1,
            }],
        };
        assert!(selection_element_offsets(&VdsSelection::Hyperslab(hs), &[6]).is_err());
    }
}
