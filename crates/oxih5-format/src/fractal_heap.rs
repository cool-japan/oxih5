use oxih5_core::OxiH5Error;

// ---------------------------------------------------------------------------
// Fractal Heap reader
// ---------------------------------------------------------------------------

/// Fractal heap reader for HDF5 new-style groups (signature "FRHP").
///
/// The fractal heap stores variable-length objects such as group names and
/// links, and is used by new-style HDF5 groups (v2 B-tree + fractal heap
/// instead of old-style SNOD + local heap).
///
/// Only "managed" (type-0) heap IDs are supported; huge and tiny objects
/// return `NotImplemented`.
///
/// The lifetime `'a` is tied to the `file_data` slice from which this heap was
/// parsed.  No copy of the file bytes is made (T8 optimization: eliminates the
/// previous `Arc::new(file_data.to_vec())` per new-style group traversal).
#[derive(Debug)]
pub struct FractalHeap<'a> {
    /// Borrow of the raw file bytes — no allocation on construction.
    file_data: &'a [u8],
    /// Byte offset of the "FRHP" header within `file_data`.
    header_address: u64,
    /// Number of bytes in every heap ID (2..=255).
    heap_id_len: u8,
    /// Maximum size (bytes) of a managed (inline) object.
    ///
    /// Stored from the FRHP header; currently unused in read paths but retained
    /// for completeness / debug output.
    #[allow(dead_code)]
    max_managed_obj_size: u32,
    /// Address of the root direct/indirect block (u64::MAX = empty).
    root_block_address: u64,
    /// Starting block size of the doubling table.
    starting_block_size: u64,
    /// Maximum size of a direct block.
    max_direct_block_size: u64,
    /// Width of the doubling table (columns per row).
    table_width: u16,
    /// Number of bits needed to express the maximum heap virtual address.
    max_heap_size_bits: u16,
    /// Starting rows count stored in the root indirect block header.
    root_indirect_rows: u16,
    /// Current number of rows in the root block (0 = direct root).
    current_rows: u16,
    /// File size_of_offsets (from the superblock).
    ///
    /// The FHDB and FHIB "Block Offset" field is encoded as this many bytes
    /// (empirically: soo, not `ceil(max_heap_size_bits/8)`).
    size_of_offsets: u8,
}

impl<'a> FractalHeap<'a> {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Parse a fractal heap starting at `header_address` in `file_data`.
    ///
    /// `size_of_offsets` is the file-level `soo` field from the superblock (typically 8).
    /// It is needed to decode the "Block Offset" field inside FHDB and FHIB nodes
    /// which is stored in `soo` bytes regardless of `max_heap_size_bits`.
    ///
    /// `file_data` is borrowed for the lifetime of the returned `FractalHeap`.
    /// No copy of the file bytes is made.
    pub fn parse(
        file_data: &'a [u8],
        header_address: u64,
        size_of_offsets: u8,
    ) -> Result<Self, OxiH5Error> {
        let base = usize::try_from(header_address).map_err(|_| {
            OxiH5Error::Corrupted(format!(
                "FractalHeap: header address {header_address} exceeds addressable range"
            ))
        })?;
        let base4 = base.checked_add(4).ok_or_else(|| {
            OxiH5Error::Corrupted(format!(
                "FractalHeap: header address {header_address} too large"
            ))
        })?;
        let d = file_data;

        // Signature "FRHP"
        let sig = d
            .get(base..base4)
            .ok_or_else(|| OxiH5Error::Format("FractalHeap: truncated at signature".into()))?;
        if sig != b"FRHP" {
            return Err(OxiH5Error::Format(format!(
                "FractalHeap: bad signature {:?} at {:#x}",
                sig, base
            )));
        }

        // Version
        let version = *d
            .get(base + 4)
            .ok_or_else(|| OxiH5Error::Format("FractalHeap: missing version byte".into()))?;
        if version != 0 {
            return Err(OxiH5Error::Format(format!(
                "FractalHeap: unsupported version {version}"
            )));
        }

        // Byte 5: heap ID length
        let heap_id_len = *d
            .get(base + 5)
            .ok_or_else(|| OxiH5Error::Format("FractalHeap: missing heap_id_len".into()))?;

        // Byte 6: I/O filters encoded length
        let io_filter_len = *d
            .get(base + 6)
            .ok_or_else(|| OxiH5Error::Format("FractalHeap: missing io_filter_len".into()))?;
        if io_filter_len != 0 {
            return Err(OxiH5Error::NotImplemented(
                "FractalHeap: I/O filters on the heap are not yet supported".into(),
            ));
        }

        // Byte 7: flags (we read but don't act on them yet)
        let _flags = *d
            .get(base + 7)
            .ok_or_else(|| OxiH5Error::Format("FractalHeap: missing flags byte".into()))?;

        // Bytes 8–11: Maximum Managed Object Size
        let max_managed_obj_size = read_u32(d, base + 8)?;

        // The fixed-layout fields (when io_filter_len == 0, no optional filter block,
        // soo = size_of_offsets, sol = size_of_lengths — both 8 for our files):
        //
        //   8–11    Max Managed Object Size          (4 bytes)
        //  12–19    Next Huge Object ID              (8 bytes)
        //  20–27    v2 BTree Address for Huge Objects (soo=8)
        //  28–35    Free space in managed blocks      (sol=8)
        //  36–43    Address of managed block FSM      (soo=8)
        //  44–51    Amount of managed space in heap   (sol=8)
        //  52–59    Amount of allocated managed space (sol=8)
        //  60–67    Offset of direct block alloc iter (sol=8)
        //  68–75    Number of managed objects         (sol=8)
        //  76–83    Size of huge objects              (sol=8)
        //  84–91    Number of huge objects            (sol=8)
        //  92–99    Size of tiny objects              (sol=8)
        // 100–107   Number of tiny objects            (sol=8)
        // 108–109   (reserved/unknown 2-byte field)   (2 bytes, always 0)
        // 110–111   Table Width                        (2 bytes)
        // 112–119   Starting Block Size                (sol=8)
        // 120–127   Maximum Direct Block Size          (sol=8)
        // 128–129   Maximum Heap Size (Bits)           (2 bytes)
        // 130–131   Starting # Rows in Root Indirect Block (2 bytes)
        // 132–139   Address of Root Block              (soo=8)
        // 140–141   Current # Rows in Root Indirect Block (2 bytes)
        // 142–145   Checksum                           (4 bytes)
        //
        // NOTE: The 2-byte field at 108-109 (value=0) is present in HDF5 files
        // written by h5py / libhdf5 1.10+ with libver='latest'.  It is not
        // described in the publicly-available spec excerpt but is confirmed
        // empirically by binary analysis of h5py-generated files.

        let table_width = read_u16(d, base + 110)?;
        let starting_block_size = read_u64(d, base + 112)?;
        let max_direct_block_size = read_u64(d, base + 120)?;
        let max_heap_size_bits = read_u16(d, base + 128)?;
        let root_indirect_rows = read_u16(d, base + 130)?;
        let root_block_address = read_u64(d, base + 132)?;
        let current_rows = read_u16(d, base + 140)?;

        Ok(Self {
            file_data,
            header_address,
            heap_id_len,
            max_managed_obj_size,
            root_block_address,
            starting_block_size,
            max_direct_block_size,
            table_width,
            max_heap_size_bits,
            root_indirect_rows,
            current_rows,
            size_of_offsets,
        })
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// The address this heap header was parsed from.
    pub fn header_address(&self) -> u64 {
        self.header_address
    }

    /// Byte length of every heap ID produced by this heap.
    pub fn heap_id_len(&self) -> u8 {
        self.heap_id_len
    }

    /// Width of the doubling table (number of columns per row).
    pub fn table_width(&self) -> u16 {
        self.table_width
    }

    /// Starting number of rows declared in the root indirect block header.
    pub fn root_indirect_rows(&self) -> u16 {
        self.root_indirect_rows
    }

    /// Parse a managed-type heap ID into `(heap_offset, object_size)`.
    ///
    /// The heap ID layout (when `io_filter_len == 0`) is:
    /// ```text
    /// Byte 0:                  version (upper 6 bits, currently 0) + type (lower 2 bits)
    ///                          type 0 = managed, 1 = huge-indirect, 2 = huge-direct, 3 = tiny
    /// Bytes 1..1+offset_size:  virtual heap offset (LE)
    /// Bytes 1+offset_size..heap_id_len:  object length (LE)
    /// ```
    /// where:
    /// - `offset_size = ceil(max_heap_size_bits / 8)`, minimum 1
    /// - `length_size = heap_id_len - 1 - offset_size` (the remaining bytes)
    ///
    /// The `max_managed_obj_size` field from the header is NOT used to compute
    /// `length_size`; instead, `length_size` is derived from the known `heap_id_len`
    /// and `offset_size`.  This matches empirical analysis of h5py-generated files.
    pub fn parse_heap_id(&self, heap_id: &[u8]) -> Result<(u64, usize), OxiH5Error> {
        if heap_id.len() < self.heap_id_len as usize {
            return Err(OxiH5Error::Format(format!(
                "FractalHeap: heap ID too short ({} < {})",
                heap_id.len(),
                self.heap_id_len
            )));
        }

        let id_type = heap_id[0] & 0x03;
        if id_type != 0 {
            return Err(OxiH5Error::NotImplemented(format!(
                "FractalHeap: heap ID type {id_type} (only managed=0 is supported)"
            )));
        }

        // offset_size: bytes needed to encode any virtual address within the heap
        let offset_size = (self.max_heap_size_bits as usize).div_ceil(8).max(1);

        // length_size: the remainder of the ID after the type byte and offset bytes.
        // This is the approach used by libhdf5: the heap ID is exactly heap_id_len bytes
        // and the object-size field occupies whatever bytes remain.
        let id_body = self.heap_id_len as usize - 1; // bytes after the type byte
        let length_size = id_body.saturating_sub(offset_size).max(1);

        if 1 + offset_size + length_size > heap_id.len() {
            return Err(OxiH5Error::Format(format!(
                "FractalHeap: heap ID too short for offset_size={offset_size} length_size={length_size}"
            )));
        }

        let offset = le_bytes_to_u64(&heap_id[1..1 + offset_size]);
        let length = le_bytes_to_usize(&heap_id[1 + offset_size..1 + offset_size + length_size]);

        Ok((offset, length))
    }

    /// Retrieve a managed object from the heap by its virtual heap offset and
    /// known byte length.
    ///
    /// For objects stored in a root direct block (the most common case for
    /// small groups), this resolves immediately.  Root-is-indirect traversal
    /// returns `NotImplemented`.
    pub fn read_object(&self, heap_offset: u64, object_size: usize) -> Result<Vec<u8>, OxiH5Error> {
        if self.root_block_address == u64::MAX {
            return Err(OxiH5Error::NotFound("fractal heap: heap is empty".into()));
        }

        let num_direct_rows = self.num_direct_rows();

        if self.current_rows == 0 || self.current_rows <= num_direct_rows {
            // Root is a single direct block.
            self.read_from_direct_block(self.root_block_address, heap_offset, object_size)
        } else {
            // Root is an indirect block.
            self.read_from_indirect_block(
                self.root_block_address,
                self.current_rows,
                heap_offset,
                object_size,
            )
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Number of rows in the doubling table that contain direct blocks
    /// (i.e. rows where block_size ≤ max_direct_block_size).
    fn num_direct_rows(&self) -> u16 {
        let mut row = 0u16;
        while row < 256 {
            let size = self.block_size_for_row(row);
            if size > self.max_direct_block_size {
                break;
            }
            row += 1;
        }
        row
    }

    /// Block size for the given row index in the doubling table.
    ///
    /// Rows 0 and 1 both use `starting_block_size`; from row 2 onward
    /// the size doubles with each row.
    pub fn block_size_for_row(&self, row: u16) -> u64 {
        if row == 0 || row == 1 {
            self.starting_block_size
        } else {
            // Shift by (row - 1) to preserve the "two rows at initial size" rule.
            self.starting_block_size
                .checked_shl((row - 1) as u32)
                .unwrap_or(u64::MAX)
        }
    }

    /// Number of bytes used to encode the "Block Offset" field in FHDB and FHIB nodes.
    ///
    /// Per empirical analysis of h5py-generated HDF5 files, the block offset is
    /// stored in `soo` (size_of_offsets) bytes — not in `ceil(max_heap_size_bits/8)` bytes
    /// as the spec excerpt suggests.  Using `soo` gives the correct data-region start offset.
    fn block_offset_size(&self) -> usize {
        self.size_of_offsets as usize
    }

    /// Read `object_size` bytes starting at `heap_offset` from a direct block
    /// at `block_address`.
    fn read_from_direct_block(
        &self,
        block_address: u64,
        heap_offset: u64,
        object_size: usize,
    ) -> Result<Vec<u8>, OxiH5Error> {
        let base = block_address as usize;
        let d = self.file_data;

        // Signature "FHDB"
        let sig = d.get(base..base + 4).ok_or_else(|| {
            OxiH5Error::Format(format!("FractalHeap FHDB: truncated at {:#x}", base))
        })?;
        if sig != b"FHDB" {
            return Err(OxiH5Error::Format(format!(
                "FractalHeap FHDB: bad signature {:?} at {:#x}",
                sig, base
            )));
        }

        // The heap virtual address (heap_offset) is measured from the start of this
        // direct block itself (byte 0 of the "FHDB" signature), so the object is
        // simply at `block_address + heap_offset`.  The FHDB header (sig + version +
        // heap_header_addr + block_offset = typically 21 bytes for soo=8) is included
        // in the heap's address space, and heap IDs always point past it.
        let obj_start = base
            .checked_add(heap_offset as usize)
            .ok_or_else(|| OxiH5Error::Format("FractalHeap: heap_offset overflow".into()))?;
        let obj_end = obj_start
            .checked_add(object_size)
            .ok_or_else(|| OxiH5Error::Format("FractalHeap: object size overflow".into()))?;

        d.get(obj_start..obj_end)
            .ok_or_else(|| {
                OxiH5Error::Format(format!(
                    "FractalHeap: object [{obj_start}, {obj_end}) out of bounds (file len {})",
                    d.len()
                ))
            })
            .map(|s| s.to_vec())
    }

    /// Traverse an indirect block (FHIB) to find the object at `target_heap_offset`.
    ///
    /// The FHIB byte layout (soo=8):
    /// ```text
    /// base + 0..4:           "FHIB" signature
    /// base + 4:              version byte = 0
    /// base + 5..13:          heap header address (8 bytes, soo=8)
    /// base + 13..13+bos:     block_offset (bos = block_offset_size() bytes)
    /// base + 13+bos:         entries begin
    /// ```
    /// Each entry is `soo=8` bytes (little-endian u64 address).
    /// Rows `< num_direct_rows` are direct blocks; rows `>= num_direct_rows`
    /// are sub-indirect blocks, and recursion is used to traverse them.
    fn read_from_indirect_block(
        &self,
        block_address: u64,
        num_rows: u16,
        target_heap_offset: u64,
        object_size: usize,
    ) -> Result<Vec<u8>, OxiH5Error> {
        let soo: usize = self.size_of_offsets as usize;
        let base = block_address as usize;
        let d = self.file_data;

        // Validate signature.
        let sig = d
            .get(base..base + 4)
            .ok_or_else(|| OxiH5Error::Format(format!("FHIB: truncated at {:#x}", base)))?;
        if sig != b"FHIB" {
            return Err(OxiH5Error::Format(format!(
                "FHIB: bad signature {:?} at {:#x}",
                sig, base
            )));
        }
        // version byte at base+4 (skip)
        // heap_header_address at base+5 (soo bytes, skip)
        // block_offset at base+5+soo (bos bytes, skip)
        let bos = self.block_offset_size();
        let entries_start = base + 4 + 1 + soo + bos;

        let max_direct_rows = self.num_direct_rows();
        let tw = self.table_width as usize;

        // Compute cumulative heap offset up through each row.
        let mut cumulative: u64 = 0;
        for row in 0..(num_rows as usize) {
            let row_block_size = self.block_size_for_row(row as u16);
            for col in 0..tw {
                let entry_heap_start = cumulative.saturating_add(col as u64 * row_block_size);
                let entry_heap_end = entry_heap_start.saturating_add(row_block_size);

                if target_heap_offset >= entry_heap_start && target_heap_offset < entry_heap_end {
                    let entry_pos = entries_start + (row * tw + col) * soo;
                    let bytes: [u8; 8] = d
                        .get(entry_pos..entry_pos + soo)
                        .and_then(|b| b.try_into().ok())
                        .ok_or_else(|| {
                            OxiH5Error::Format(format!("FHIB: entry at {entry_pos} out of bounds"))
                        })?;
                    let addr = u64::from_le_bytes(bytes);

                    if addr == u64::MAX {
                        return Err(OxiH5Error::Format(
                            "FractalHeap: target heap offset points to unallocated block".into(),
                        ));
                    }

                    let within_block = target_heap_offset - entry_heap_start;

                    if (row as u16) < max_direct_rows {
                        // Direct block.
                        return self.read_from_direct_block(addr, within_block, object_size);
                    } else {
                        // Sub-indirect block: compute its nrows from its coverage size.
                        let sub_nrows = self.compute_indirect_nrows(row_block_size);
                        return self.read_from_indirect_block(
                            addr,
                            sub_nrows,
                            within_block,
                            object_size,
                        );
                    }
                }
            }
            cumulative = cumulative.saturating_add(self.table_width as u64 * row_block_size);
        }

        Err(OxiH5Error::Format(format!(
            "FractalHeap: heap offset {target_heap_offset:#x} not found in indirect block \
             at {block_address:#x} (num_rows={num_rows})",
        )))
    }

    /// Given a sub-indirect block that covers `block_size` total bytes, compute
    /// how many rows that block needs in its own doubling table.
    ///
    /// Finds smallest `n` such that the cumulative coverage
    /// `sum_{r=0}^{n-1}(table_width * block_size_for_row(r)) >= block_size`.
    fn compute_indirect_nrows(&self, block_size: u64) -> u16 {
        let tw = self.table_width as u64;
        let mut cumulative: u64 = 0;
        let mut n: u16 = 0;
        while n < 256 {
            cumulative = cumulative.saturating_add(tw * self.block_size_for_row(n));
            n += 1;
            if cumulative >= block_size {
                break;
            }
        }
        n
    }
}

// ---------------------------------------------------------------------------
// Little-endian byte helpers
// ---------------------------------------------------------------------------

#[inline]
fn read_u16(data: &[u8], off: usize) -> Result<u16, OxiH5Error> {
    data.get(off..off + 2)
        .ok_or_else(|| {
            OxiH5Error::Format(format!(
                "FractalHeap: read_u16 at offset {off} out of bounds"
            ))
        })?
        .try_into()
        .map(u16::from_le_bytes)
        .map_err(|_| OxiH5Error::Format("FractalHeap: read_u16 slice".into()))
}

#[inline]
fn read_u32(data: &[u8], off: usize) -> Result<u32, OxiH5Error> {
    data.get(off..off + 4)
        .ok_or_else(|| {
            OxiH5Error::Format(format!(
                "FractalHeap: read_u32 at offset {off} out of bounds"
            ))
        })?
        .try_into()
        .map(u32::from_le_bytes)
        .map_err(|_| OxiH5Error::Format("FractalHeap: read_u32 slice".into()))
}

#[inline]
fn read_u64(data: &[u8], off: usize) -> Result<u64, OxiH5Error> {
    data.get(off..off + 8)
        .ok_or_else(|| {
            OxiH5Error::Format(format!(
                "FractalHeap: read_u64 at offset {off} out of bounds"
            ))
        })?
        .try_into()
        .map(u64::from_le_bytes)
        .map_err(|_| OxiH5Error::Format("FractalHeap: read_u64 slice".into()))
}

/// Decode up to 8 bytes as a little-endian u64.
#[inline]
fn le_bytes_to_u64(bytes: &[u8]) -> u64 {
    let mut val = 0u64;
    for (i, &b) in bytes.iter().take(8).enumerate() {
        val |= (b as u64) << (i * 8);
    }
    val
}

/// Decode up to 8 bytes as a little-endian usize.
#[inline]
fn le_bytes_to_usize(bytes: &[u8]) -> usize {
    le_bytes_to_u64(bytes) as usize
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal FRHP header buffer for use in tests.
    ///
    /// Field offsets match the empirically-verified layout used by libhdf5 / h5py:
    /// - [108-109] = reserved (2 bytes, zero)
    /// - [110-111] = table_width
    /// - [112-119] = starting_block_size
    /// - [120-127] = max_direct_block_size
    /// - [128-129] = max_heap_size_bits
    /// - [130-131] = root_indirect_rows
    /// - [132-139] = root_block_address
    /// - [140-141] = current_rows
    #[allow(clippy::too_many_arguments)]
    fn make_minimal_frhp(
        heap_id_len: u8,
        max_managed_obj_size: u32,
        table_width: u16,
        starting_block_size: u64,
        max_direct_block_size: u64,
        max_heap_size_bits: u16,
        root_indirect_rows: u16,
        root_block_address: u64,
        current_rows: u16,
    ) -> Vec<u8> {
        let mut data = vec![0u8; 256];
        data[0..4].copy_from_slice(b"FRHP");
        data[4] = 0; // version
        data[5] = heap_id_len;
        data[6] = 0; // io_filter_encoded_len
        data[7] = 0; // flags
        data[8..12].copy_from_slice(&max_managed_obj_size.to_le_bytes());
        // Offsets 12..110 are other heap statistics (zeroed) + 2-byte reserved field at 108
        data[110..112].copy_from_slice(&table_width.to_le_bytes());
        data[112..120].copy_from_slice(&starting_block_size.to_le_bytes());
        data[120..128].copy_from_slice(&max_direct_block_size.to_le_bytes());
        data[128..130].copy_from_slice(&max_heap_size_bits.to_le_bytes());
        data[130..132].copy_from_slice(&root_indirect_rows.to_le_bytes());
        data[132..140].copy_from_slice(&root_block_address.to_le_bytes());
        data[140..142].copy_from_slice(&current_rows.to_le_bytes());
        data
    }

    #[test]
    fn test_bad_signature() {
        let data = vec![0u8; 256];
        assert!(FractalHeap::parse(&data, 0, 8).is_err());
    }

    #[test]
    fn test_unsupported_version() {
        let mut data = vec![0u8; 256];
        data[0..4].copy_from_slice(b"FRHP");
        data[4] = 1; // non-zero version
        let result = FractalHeap::parse(&data, 0, 8);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("version"));
    }

    #[test]
    fn test_io_filter_rejected() {
        let mut data = vec![0u8; 256];
        data[0..4].copy_from_slice(b"FRHP");
        data[4] = 0;
        data[6] = 1; // io_filter_len != 0
        let result = FractalHeap::parse(&data, 0, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_header_fields() {
        let raw = make_minimal_frhp(7, 256, 4, 512, 65536, 32, 0, u64::MAX, 0);
        let heap = FractalHeap::parse(&raw, 0, 8).expect("should parse");
        assert_eq!(heap.heap_id_len(), 7);
        assert_eq!(heap.table_width(), 4);
        assert_eq!(heap.starting_block_size, 512);
        assert_eq!(heap.max_direct_block_size, 65536);
        assert_eq!(heap.max_heap_size_bits, 32);
        assert_eq!(heap.root_block_address, u64::MAX);
    }

    #[test]
    fn test_empty_heap_returns_not_found() {
        // Root address == UNDEF → heap is empty.
        let raw = make_minimal_frhp(7, 256, 4, 512, 65536, 32, 0, u64::MAX, 0);
        let heap = FractalHeap::parse(&raw, 0, 8).unwrap();
        assert!(heap.read_object(0, 4).is_err());
    }

    #[test]
    fn test_parse_heap_id_managed() {
        // max_heap_size_bits = 16 → offset_size = 2 bytes
        // max_managed_obj_size = 255 → bits_needed = 8 → length_size = 1 byte
        // heap_id_len = 4 (1 type byte + 2 offset + 1 length)
        let raw = make_minimal_frhp(4, 255, 4, 512, 65536, 16, 0, u64::MAX, 0);
        let heap = FractalHeap::parse(&raw, 0, 8).unwrap();

        // Heap ID: type=0, offset=0x0102 (LE), length=42
        let id = [0x00u8, 0x02, 0x01, 0x2a];
        let (offset, length) = heap.parse_heap_id(&id).unwrap();
        assert_eq!(offset, 0x0102);
        assert_eq!(length, 42);
    }

    #[test]
    fn test_parse_heap_id_non_managed_rejected() {
        let raw = make_minimal_frhp(7, 256, 4, 512, 65536, 32, 0, u64::MAX, 0);
        let heap = FractalHeap::parse(&raw, 0, 8).unwrap();
        // Type bits = 1 → huge
        let id = [0x01u8, 0, 0, 0, 0, 0, 0];
        assert!(heap.parse_heap_id(&id).is_err());
    }

    #[test]
    fn test_read_object_from_direct_block() {
        // Build a file with an FRHP header and an FHDB direct block.
        // FRHP at offset 0; FHDB at offset 256.
        //
        // size_of_offsets = 8, so block_offset in FHDB is 8 bytes.
        // FHDB header = sig(4) + version(1) + heap_header_addr(8) + block_offset(8) = 21 bytes
        //
        // Heap virtual addresses are measured from the FHDB block base.
        // The 21-byte FHDB header occupies heap addresses 0..20.
        // Usable object storage starts at heap_offset=21.
        let fhdb_address: u64 = 256;
        let heap_offset_of_obj: u64 = 21; // first byte past the FHDB header

        // max_heap_size_bits = 8, current_rows = 0 → root is direct
        let mut file = make_minimal_frhp(7, 256, 4, 512, 65536, 8, 0, fhdb_address, 0);
        file.resize(1024, 0);

        // Build FHDB at offset 256:
        //   sig(4) + version(1) + heap_header_addr(8) + block_offset(soo=8) = 21 bytes header
        let fhdb_base = 256usize;
        file[fhdb_base..fhdb_base + 4].copy_from_slice(b"FHDB");
        file[fhdb_base + 4] = 0; // version
                                 // heap_header_addr (bytes 5..13) = 0 (points back to our FRHP)
        file[fhdb_base + 5..fhdb_base + 13].copy_from_slice(&0u64.to_le_bytes());
        // block_offset = 0 (8 bytes at fhdb_base+13..21)
        file[fhdb_base + 13..fhdb_base + 21].copy_from_slice(&0u64.to_le_bytes());
        // Write known pattern at the first usable position: fhdb_base + 21
        let obj_pos = fhdb_base + heap_offset_of_obj as usize;
        file[obj_pos..obj_pos + 8].copy_from_slice(b"DEADBEEF");

        let heap = FractalHeap::parse(&file, 0, 8).unwrap();
        // heap_offset_of_obj = 21 → object at fhdb_base + 21
        let obj = heap.read_object(heap_offset_of_obj, 8).unwrap();
        assert_eq!(&obj, b"DEADBEEF");
    }

    #[test]
    fn test_num_direct_rows() {
        // starting=512, max_direct=65536 → rows: 512,512,1024,2048,4096,8192,16384,32768,65536 = 9 rows
        let raw = make_minimal_frhp(7, 256, 4, 512, 65536, 32, 0, u64::MAX, 0);
        let heap = FractalHeap::parse(&raw, 0, 8).unwrap();
        let ndr = heap.num_direct_rows();
        // Row 0:512, row 1:512, row 2:1024, row 3:2048, row 4:4096,
        // row 5:8192, row 6:16384, row 7:32768, row 8:65536 (=max) → 9 rows
        assert_eq!(ndr, 9);
    }

    #[test]
    fn test_block_size_for_row() {
        let raw = make_minimal_frhp(7, 256, 4, 512, 65536, 32, 0, u64::MAX, 0);
        let heap = FractalHeap::parse(&raw, 0, 8).unwrap();
        assert_eq!(heap.block_size_for_row(0), 512);
        assert_eq!(heap.block_size_for_row(1), 512);
        assert_eq!(heap.block_size_for_row(2), 1024);
        assert_eq!(heap.block_size_for_row(3), 2048);
    }

    #[test]
    fn test_compute_indirect_nrows() {
        // starting=512, table_width=2
        // Row 0 covers: 2*512=1024
        // Row 1 covers: 2*512=1024 → cumulative 2048
        // Row 2 covers: 2*1024=2048 → cumulative 4096
        let raw = make_minimal_frhp(7, 256, 2, 512, 65536, 16, 0, u64::MAX, 0);
        let heap = FractalHeap::parse(&raw, 0, 8).unwrap();
        // block_size 1024 → needs 1 row (cumulative after row 0 = 1024 >= 1024)
        assert_eq!(heap.compute_indirect_nrows(1024), 1);
        // block_size 2048 → needs 2 rows (cumulative after row 1 = 2048 >= 2048)
        assert_eq!(heap.compute_indirect_nrows(2048), 2);
        // block_size 3000 → needs 3 rows (cumulative after row 2 = 4096 >= 3000)
        assert_eq!(heap.compute_indirect_nrows(3000), 3);
    }

    #[test]
    fn test_indirect_block_single_level() {
        // Layout (soo=8, bos=soo=8):
        //   FRHP at offset 0   (256+ bytes header)
        //   FHIB at offset 300
        //   FHDB_0_0 at offset 600   (row 0, col 0: heap range 0..512)
        //   FHDB_0_1 at offset 1200  (row 0, col 1: heap range 512..1024)
        //
        // table_width=2, starting_block_size=512, max_direct_block_size=65536
        // size_of_offsets=8, so block_offset in FHIB/FHDB = 8 bytes
        //
        // Target: heap_offset=600, size=4
        //   row=0 (cumulative=0), col=1 (600 >= 512, 600 < 1024)
        //   within_block = 600 - 512 = 88
        //   FHDB at 1200:
        //     heap virtual addresses are from block base → obj_start = 1200 + 88 = 1288
        //   object at 1288

        let tw: u16 = 2;
        let sbs: u64 = 512;
        let max_heap_size_bits: u16 = 16;
        let max_direct_block_size: u64 = 65536;
        let soo: usize = 8; // size_of_offsets

        let obj = [0xDE_u8, 0xAD, 0xBE, 0xEF];

        let fhib_addr: u64 = 300;
        let fhdb_0_0_addr: u64 = 600;
        let fhdb_0_1_addr: u64 = 1200;
        let heap_offset: u64 = 600; // row 0, col 1, within=88

        // Buffer large enough for all structures (1309 + 4 + margin)
        let mut buf = vec![0u8; 2000];

        // FRHP at offset 0
        let frhp = make_minimal_frhp(
            7,                     // heap_id_len
            1000,                  // max_managed_obj_size
            tw,                    // table_width
            sbs,                   // starting_block_size
            max_direct_block_size, // max_direct_block_size
            max_heap_size_bits,    // max_heap_size_bits
            2,                     // root_indirect_rows
            fhib_addr,             // root_block_address
            2,                     // current_rows
        );
        buf[..frhp.len()].copy_from_slice(&frhp);

        // FHIB at offset 300
        // Header: "FHIB"(4) + ver(1) + heap_hdr_addr(soo=8) + block_offset(soo=8) = 21 bytes
        let fhib_base = fhib_addr as usize;
        buf[fhib_base..fhib_base + 4].copy_from_slice(b"FHIB");
        buf[fhib_base + 4] = 0; // version
                                // heap_hdr_addr at fhib_base+5..+13 = 0 (points to FRHP)
                                // block_offset at fhib_base+13..21 = 0 (8 bytes)
        let entries_start = fhib_base + 4 + 1 + soo + soo; // = fhib_base + 21
                                                           // Row 0, col 0: fhdb_0_0_addr
        buf[entries_start..entries_start + soo].copy_from_slice(&fhdb_0_0_addr.to_le_bytes());
        // Row 0, col 1: fhdb_0_1_addr
        buf[entries_start + soo..entries_start + 2 * soo]
            .copy_from_slice(&fhdb_0_1_addr.to_le_bytes());
        // Row 1 entries: u64::MAX (unallocated)
        let undef = u64::MAX.to_le_bytes();
        buf[entries_start + 2 * soo..entries_start + 3 * soo].copy_from_slice(&undef);
        buf[entries_start + 3 * soo..entries_start + 4 * soo].copy_from_slice(&undef);

        // FHDB at fhdb_0_1_addr = 1200
        // Heap virtual addresses are measured from the FHDB block base.
        // within_block = heap_offset - entry_heap_start = 600 - 512 = 88
        // obj_start = fhdb_0_1_addr + within_block = 1200 + 88 = 1288
        let fhdb_base = fhdb_0_1_addr as usize;
        buf[fhdb_base..fhdb_base + 4].copy_from_slice(b"FHDB");
        buf[fhdb_base + 4] = 0; // version
        let obj_pos = fhdb_base + 88; // = 1288
        assert!(obj_pos + 4 <= buf.len(), "buffer too small");
        buf[obj_pos..obj_pos + 4].copy_from_slice(&obj);

        let heap = FractalHeap::parse(&buf, 0, 8).expect("parse FRHP");

        // Call read_from_indirect_block directly with 2 rows.
        let result = heap
            .read_from_indirect_block(fhib_addr, 2, heap_offset, 4)
            .expect("indirect traversal");

        assert_eq!(&result, &obj, "object bytes should match");
    }
}
