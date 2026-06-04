use crate::superblock::read_u64_le;
use oxih5_core::OxiH5Error;

/// Parsed local heap: borrows the NUL-terminated name strings directly from
/// the file bytes — no copy of the data segment is made.
///
/// The lifetime `'a` is tied to the `file_data` slice from which this heap was
/// parsed.  `name_at` returns `&'a str` slices that live as long as the
/// original file bytes, so callers that only need to compare or display names
/// pay no allocation at all.
pub struct LocalHeap<'a> {
    /// Slice of `file_data` covering the heap data segment only.
    pub data: &'a [u8],
}

impl<'a> LocalHeap<'a> {
    /// Return the NUL-terminated name string at `offset` bytes into the data segment.
    ///
    /// The returned `&str` borrows from the original `file_data` — no allocation.
    pub fn name_at(&self, offset: usize) -> Result<&'a str, OxiH5Error> {
        if offset >= self.data.len() {
            return Err(OxiH5Error::Format(format!(
                "heap name_at: offset {offset} >= data segment length {}",
                self.data.len()
            )));
        }
        let nul_pos = self.data[offset..]
            .iter()
            .position(|&b| b == 0)
            .ok_or_else(|| {
                OxiH5Error::Format(format!(
                    "heap name_at: no NUL terminator at offset {offset}"
                ))
            })?;
        std::str::from_utf8(&self.data[offset..offset + nul_pos]).map_err(|e| {
            OxiH5Error::Format(format!("heap name not valid UTF-8 at offset {offset}: {e}"))
        })
    }
}

/// Parse a local heap from `file_data` at the given absolute address.
///
/// Returns a [`LocalHeap`] that borrows the data segment directly from
/// `file_data` — the data segment bytes are **not** copied.
///
/// Local heap layout:
/// ```text
/// Offset  Size  Field
///  0       4     Signature "HEAP"
///  4       1     Version (must be 0)
///  5       3     Reserved
///  8       8     Data segment size (u64 LE)
/// 16       8     Free list head offset (u64 LE, offset into data segment)
/// 24       8     Data segment address (u64 LE, absolute file offset)
/// ```
pub fn parse(file_data: &[u8], heap_address: u64) -> Result<LocalHeap<'_>, OxiH5Error> {
    let off = usize::try_from(heap_address).map_err(|_| {
        OxiH5Error::Corrupted(format!(
            "heap address {heap_address} exceeds addressable range"
        ))
    })?;
    let off32 = off
        .checked_add(32)
        .ok_or_else(|| OxiH5Error::Corrupted(format!("heap address {heap_address} too large")))?;

    if off32 > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "heap at {heap_address}: header out of bounds (file len={})",
            file_data.len()
        )));
    }

    if &file_data[off..off + 4] != b"HEAP" {
        return Err(OxiH5Error::Format(format!(
            "no HEAP signature at {heap_address}: got {:?}",
            &file_data[off..off + 4]
        )));
    }

    let version = file_data[off + 4];
    if version != 0 {
        return Err(OxiH5Error::Format(format!(
            "unsupported local heap version: {version}"
        )));
    }

    let data_segment_size_raw = read_u64_le(file_data, off + 8)?;
    let data_segment_size = usize::try_from(data_segment_size_raw).map_err(|_| {
        OxiH5Error::Corrupted(format!(
            "heap data segment size {data_segment_size_raw} exceeds addressable range"
        ))
    })?;
    let data_segment_address_raw = read_u64_le(file_data, off + 24)?;
    let data_segment_address = usize::try_from(data_segment_address_raw).map_err(|_| {
        OxiH5Error::Corrupted(format!(
            "heap data segment address {data_segment_address_raw} exceeds addressable range"
        ))
    })?;
    let data_segment_end = data_segment_address
        .checked_add(data_segment_size)
        .ok_or_else(|| {
            OxiH5Error::Corrupted(format!(
                "heap data segment address+size overflows: \
                 {data_segment_address}+{data_segment_size}"
            ))
        })?;

    if data_segment_end > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "heap data segment at {data_segment_address}+{data_segment_size} out of bounds \
             (file len={})",
            file_data.len()
        )));
    }

    // Borrow the data segment directly — no copy (T7 optimization).
    let data = &file_data[data_segment_address..data_segment_end];
    Ok(LocalHeap { data })
}
