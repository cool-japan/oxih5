//! Extensible Array header ("EAHD") decoding and the derived block geometry.
//!
//! Everything in this file is arithmetic that libhdf5 recomputes at open time
//! from the five creation parameters stored in the header; none of it is on
//! disk.  Getting it wrong silently mis-attributes elements to chunks, so each
//! quantity is named after its `H5EApkg.h` counterpart.

use oxih5_core::OxiH5Error;

/// On-disk size of an extensible-array header with 8-byte offsets and lengths.
///
/// `signature(4) + version(1) + client_id(1) + element_size(1) +
///  max_nelmts_bits(1) + idx_blk_elmts(1) + data_blk_min_elmts(1) +
///  sup_blk_min_data_ptrs(1) + max_dblk_page_nelmts_bits(1) +
///  6 × length(8) + index_block_address(8) + checksum(4)`
pub(crate) const EA_HEADER_LEN: usize = 72;

/// Offset of the index-block address inside the header.
///
/// The six 8-byte statistics counters between the creation parameters and this
/// field are easy to overlook; omitting them puts the address at offset 28 and
/// makes every real extensible array decode against unrelated bytes.
const EA_INDEX_BLOCK_ADDR_OFF: usize = 60;

/// Size of an address/length field, and of the trailing Jenkins checksum.
pub(crate) const ADDR_SIZE: usize = 8;
/// Size of the trailing Jenkins-lookup3 checksum on every extensible-array block.
pub(crate) const CHECKSUM_SIZE: usize = 4;

/// HDF5's "undefined address" sentinel (all bits set).
pub(crate) const UNDEF_ADDR: u64 = u64::MAX;

/// Upper bound on the number of chunk records a single extensible array may
/// yield.  A crafted header can claim creation parameters that describe an
/// astronomically large array; this bounds the work and the allocation.
pub(crate) const EA_MAX_RECORDS: usize = 1 << 24;

/// Extensible-array element flavour, selected by the header's client ID.
///
/// These are `H5EA_cls_id_t` values.  Only the two dataset-chunk clients are
/// meaningful in an HDF5 file; `H5EA_CLS_TEST_ID` (2) exists solely for
/// libhdf5's own unit tests and never appears in a real file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EaClient {
    /// `H5EA_CLS_CHUNK_ID` — an unfiltered dataset chunk.  The element is a
    /// bare file address; the chunk's stored size is the full uncompressed
    /// chunk size and its filter mask is zero.
    Chunk,
    /// `H5EA_CLS_FILT_CHUNK_ID` — a filtered dataset chunk: file address, then
    /// the stored size in `size_len` little-endian bytes, then a 4-byte filter
    /// mask.  `size_len` is not stored anywhere: it is whatever is left of the
    /// element after the address and the mask.
    FilteredChunk { size_len: usize },
}

/// One entry of libhdf5's `hdr->sblk_info[]` table.
///
/// Super block *u* owns `ndblks` data blocks of `dblk_nelmts` elements each.
/// `start_idx` is the index of its first element *relative to the end of the
/// index block's inline elements*, and `start_dblk` the index of its first data
/// block in the array-wide data-block numbering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SuperBlockInfo {
    pub ndblks: u64,
    pub dblk_nelmts: u64,
    pub start_idx: u64,
    pub start_dblk: u64,
}

/// A decoded extensible-array header plus everything derived from it.
#[derive(Debug, Clone)]
pub(crate) struct EaHeader {
    /// File address the header was read from; every block back-points to it.
    pub header_address: u64,
    pub client: EaClient,
    /// Bytes per array element, as stored in the header.
    pub element_size: usize,
    /// Number of elements stored inline in the index block.
    pub idx_blk_elmts: u64,
    /// Address of the index block, or [`UNDEF_ADDR`] when none is allocated.
    pub index_block_address: u64,
    /// Width of the "block offset" field carried by data and super blocks:
    /// `ceil(max_nelmts_bits / 8)` (`H5EA_SIZEOF_OFFSET_BITS`).
    pub arr_off_size: usize,
    /// Number of data-block addresses stored directly in the index block.
    pub ndblk_addrs: usize,
    /// Number of super-block addresses stored in the index block.
    pub nsblk_addrs: usize,
    /// How many leading super blocks have their data blocks addressed straight
    /// from the index block (`H5EA_SBLK_FIRST_IDX`).
    pub iblk_nsblks: usize,
    /// Elements per data-block page (`1 << max_dblk_page_nelmts_bits`).  A data
    /// block is paged when its element count exceeds this.
    pub dblk_page_nelmts: u64,
    /// `hdr->sblk_info[]`, truncated at the first entry whose arithmetic would
    /// overflow `u64` (such a super block can never be reached in a real file).
    pub sblks: Vec<SuperBlockInfo>,
}

/// Exact base-2 logarithm of a power of two.
fn log2_of2(value: u64, what: &str) -> Result<u32, OxiH5Error> {
    if value == 0 || !value.is_power_of_two() {
        return Err(OxiH5Error::Format(format!(
            "EA: {what} must be a power of two, got {value}"
        )));
    }
    Ok(value.trailing_zeros())
}

impl EaHeader {
    /// Decode the header at `header_address`.
    pub(crate) fn parse(file_data: &[u8], header_address: u64) -> Result<Self, OxiH5Error> {
        let base = usize::try_from(header_address)
            .map_err(|_| OxiH5Error::Format("EA: header address exceeds address space".into()))?;
        let end = base
            .checked_add(EA_HEADER_LEN)
            .ok_or_else(|| OxiH5Error::Format("EA: header address overflow".into()))?;
        if end > file_data.len() {
            return Err(OxiH5Error::Format(format!(
                "EA: header at {base:#x} exceeds file length {}",
                file_data.len()
            )));
        }

        let sig = &file_data[base..base + 4];
        if sig != b"EAHD" {
            return Err(OxiH5Error::Format(format!(
                "EA: bad header signature {sig:?} at {base:#x}"
            )));
        }
        let version = file_data[base + 4];
        if version != 0 {
            return Err(OxiH5Error::Format(format!(
                "EA: unsupported header version {version}"
            )));
        }

        let client_id = file_data[base + 5];
        let element_size = file_data[base + 6] as usize;
        let max_nelmts_bits = file_data[base + 7];
        let idx_blk_elmts = file_data[base + 8] as u64;
        let data_blk_min_elmts = file_data[base + 9] as u64;
        let sup_blk_min_data_ptrs = file_data[base + 10] as u64;
        let max_dblk_page_nelmts_bits = file_data[base + 11];

        // The header's six statistics counters; only "Number of Elements"
        // (the sixth) is load-bearing here, as a bound on decoding work.
        let nelmts = read_u64(file_data, base + 52)?;
        let index_block_address = read_u64(file_data, base + EA_INDEX_BLOCK_ADDR_OFF)?;

        let client = classify_client(client_id, element_size)?;

        if max_nelmts_bits == 0 || max_nelmts_bits > 64 {
            return Err(OxiH5Error::Format(format!(
                "EA: max_nelmts_bits {max_nelmts_bits} out of range 1..=64"
            )));
        }
        let arr_off_size = (max_nelmts_bits as usize).div_ceil(8);

        let dbme_bits = log2_of2(data_blk_min_elmts, "data_blk_min_elmts")?;
        let sbmdp_bits = log2_of2(sup_blk_min_data_ptrs, "sup_blk_min_data_ptrs")?;
        if u32::from(max_nelmts_bits) < dbme_bits {
            return Err(OxiH5Error::Format(format!(
                "EA: max_nelmts_bits {max_nelmts_bits} below log2(data_blk_min_elmts) {dbme_bits}"
            )));
        }

        // hdr->nsblks = 1 + (max_nelmts_bits - log2(data_blk_min_elmts))
        let nsblks_total = (u32::from(max_nelmts_bits) - dbme_bits + 1) as usize;
        // iblock->nsblks = H5EA_SBLK_FIRST_IDX(sup_blk_min_data_ptrs)
        let iblk_nsblks = (2 * sbmdp_bits as usize).min(nsblks_total);
        // iblock->ndblk_addrs = 2 * (sup_blk_min_data_ptrs - 1)
        let ndblk_addrs = 2 * (sup_blk_min_data_ptrs as usize - 1);
        let nsblk_addrs = nsblks_total - iblk_nsblks;

        if max_dblk_page_nelmts_bits >= 64 {
            return Err(OxiH5Error::Format(format!(
                "EA: max_dblk_page_nelmts_bits {max_dblk_page_nelmts_bits} out of range"
            )));
        }
        let dblk_page_nelmts = 1u64 << max_dblk_page_nelmts_bits;

        // Every allocated slot occupies `element_size` bytes somewhere in the
        // file, so a header claiming more than the file could ever hold is
        // corrupt — reject it before any of it is walked.
        let capacity = (file_data.len() / element_size.max(1)) as u64;
        if nelmts > capacity {
            return Err(OxiH5Error::Format(format!(
                "EA: header claims {nelmts} elements of {element_size} bytes, more than the \
                 {} byte file can hold",
                file_data.len()
            )));
        }

        Ok(EaHeader {
            header_address,
            client,
            element_size,
            idx_blk_elmts,
            index_block_address,
            arr_off_size,
            ndblk_addrs,
            nsblk_addrs,
            iblk_nsblks,
            dblk_page_nelmts,
            sblks: super_block_table(nsblks_total, data_blk_min_elmts),
        })
    }

    /// Look up super block `u`, or report that the array does not describe one.
    pub(crate) fn super_block(&self, u: usize) -> Result<&SuperBlockInfo, OxiH5Error> {
        self.sblks.get(u).ok_or_else(|| {
            OxiH5Error::Format(format!(
                "EA: super block {u} is beyond the {} the header describes",
                self.sblks.len()
            ))
        })
    }

    /// Number of pages a data block of `dblk_nelmts` elements is split into,
    /// or `None` when the block is stored unpaged.
    pub(crate) fn dblk_npages(&self, dblk_nelmts: u64) -> Option<u64> {
        if dblk_nelmts > self.dblk_page_nelmts {
            Some(dblk_nelmts.div_ceil(self.dblk_page_nelmts))
        } else {
            None
        }
    }
}

/// Build `hdr->sblk_info[]`.
///
/// Super block *u* holds `2^(u/2)` data blocks of `2^((u+1)/2) *
/// data_blk_min_elmts` elements.  The table stops early if either quantity (or
/// the running element/data-block totals) would overflow `u64`: such a super
/// block would need more elements than the address space holds, so no real file
/// can reference it, and [`EaHeader::super_block`] turns a reference to one into
/// a typed error.
fn super_block_table(nsblks_total: usize, data_blk_min_elmts: u64) -> Vec<SuperBlockInfo> {
    let mut out: Vec<SuperBlockInfo> = Vec::with_capacity(nsblks_total);
    let mut start_idx: u64 = 0;
    let mut start_dblk: u64 = 0;
    for u in 0..nsblks_total {
        let (Some(ndblks), Some(pow)) = (
            1u64.checked_shl((u / 2) as u32),
            1u64.checked_shl(u.div_ceil(2) as u32),
        ) else {
            break;
        };
        let Some(dblk_nelmts) = pow.checked_mul(data_blk_min_elmts) else {
            break;
        };
        out.push(SuperBlockInfo {
            ndblks,
            dblk_nelmts,
            start_idx,
            start_dblk,
        });
        let Some(span) = ndblks.checked_mul(dblk_nelmts) else {
            break;
        };
        let (Some(next_idx), Some(next_dblk)) =
            (start_idx.checked_add(span), start_dblk.checked_add(ndblks))
        else {
            break;
        };
        start_idx = next_idx;
        start_dblk = next_dblk;
    }
    out
}

/// Decide how to read one array element from the header's client ID.
fn classify_client(client_id: u8, element_size: usize) -> Result<EaClient, OxiH5Error> {
    match client_id {
        0 => {
            if element_size != ADDR_SIZE {
                return Err(OxiH5Error::Format(format!(
                    "EA: unfiltered chunk client declares {element_size}-byte elements, \
                     expected a bare {ADDR_SIZE}-byte chunk address"
                )));
            }
            Ok(EaClient::Chunk)
        }
        1 => {
            let overhead = ADDR_SIZE + CHECKSUM_SIZE; // address + filter mask
            let size_len = element_size.saturating_sub(overhead);
            if size_len == 0 || size_len > 8 {
                return Err(OxiH5Error::Format(format!(
                    "EA: filtered chunk client declares {element_size}-byte elements, \
                     which leaves {size_len} bytes for the stored size (expected 1..=8)"
                )));
            }
            Ok(EaClient::FilteredChunk { size_len })
        }
        other => Err(OxiH5Error::NotImplemented(format!(
            "extensible array client ID {other} is not a dataset-chunk client \
             (0 = unfiltered chunk, 1 = filtered chunk)"
        ))),
    }
}

/// Read a little-endian `u64` at `offset`, or fail with a typed error.
pub(crate) fn read_u64(data: &[u8], offset: usize) -> Result<u64, OxiH5Error> {
    let end = offset
        .checked_add(ADDR_SIZE)
        .ok_or_else(|| OxiH5Error::Format("EA: address field offset overflow".into()))?;
    let bytes = data
        .get(offset..end)
        .ok_or_else(|| OxiH5Error::Format(format!("EA: address field at {offset} truncated")))?;
    let arr: [u8; 8] = bytes
        .try_into()
        .map_err(|_| OxiH5Error::Format("EA: address field slice".into()))?;
    Ok(u64::from_le_bytes(arr))
}

/// Read a little-endian unsigned integer of `width` (1..=8) bytes.
pub(crate) fn read_var_uint(data: &[u8], offset: usize, width: usize) -> Result<u64, OxiH5Error> {
    if width == 0 || width > 8 {
        return Err(OxiH5Error::Format(format!(
            "EA: variable-width integer of {width} bytes is out of range 1..=8"
        )));
    }
    let end = offset
        .checked_add(width)
        .ok_or_else(|| OxiH5Error::Format("EA: variable-width field offset overflow".into()))?;
    let bytes = data.get(offset..end).ok_or_else(|| {
        OxiH5Error::Format(format!(
            "EA: variable-width field at {offset} ({width} bytes) truncated"
        ))
    })?;
    let mut value = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        value |= u64::from(b) << (8 * i);
    }
    Ok(value)
}
