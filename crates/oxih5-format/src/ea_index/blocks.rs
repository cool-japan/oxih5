//! Traversal of the on-disk block structures: index block ("EAIB"), super
//! blocks ("EASB") and data blocks ("EADB"), including paged data blocks.
//!
//! The traversal never guesses how many elements a block holds — every count
//! comes from the super-block table derived in [`super::header`], so a block's
//! element region is bounded exactly rather than by scanning for the next
//! recognisable structure.

use oxih5_core::OxiH5Error;

use super::header::{
    read_u64, read_var_uint, EaHeader, ADDR_SIZE, CHECKSUM_SIZE, EA_MAX_RECORDS, UNDEF_ADDR,
};

/// Bytes before the first variable-length field of every extensible-array
/// block: `signature(4) + version(1) + client_id(1) + header_address(8)`.
const BLOCK_PREFIX: usize = 4 + 1 + 1 + ADDR_SIZE;

/// A contiguous run of array elements on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ElementRun {
    /// Byte offset of the first element in the file.
    pub offset: usize,
    /// Number of elements stored back to back from `offset`.
    pub count: u64,
    /// Array index of the element at `offset`.
    pub first_index: u64,
}

/// Validate a block's fixed prefix and return the offset just past it.
fn check_block_prefix(
    file_data: &[u8],
    address: u64,
    signature: &[u8; 4],
    header: &EaHeader,
    what: &str,
) -> Result<usize, OxiH5Error> {
    let base = usize::try_from(address).map_err(|_| {
        OxiH5Error::Format(format!(
            "EA: {what} address {address:#x} exceeds address space"
        ))
    })?;
    let end = base
        .checked_add(BLOCK_PREFIX)
        .ok_or_else(|| OxiH5Error::Format(format!("EA: {what} address overflow")))?;
    if end > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "EA: {what} at {base:#x} truncated (file len={})",
            file_data.len()
        )));
    }
    let sig = &file_data[base..base + 4];
    if sig != signature {
        return Err(OxiH5Error::Format(format!(
            "EA: bad {what} signature {sig:?} at {base:#x}"
        )));
    }
    let version = file_data[base + 4];
    if version != 0 {
        return Err(OxiH5Error::Format(format!(
            "EA: unsupported {what} version {version} at {base:#x}"
        )));
    }
    let back = read_u64(file_data, base + 6)?;
    if back != header.header_address {
        return Err(OxiH5Error::Format(format!(
            "EA: {what} at {base:#x} back-points to {back:#x}, not to its header at {:#x}",
            header.header_address
        )));
    }
    Ok(end)
}

/// The index block's contents: inline elements plus the two address arrays.
pub(crate) struct IndexBlock {
    /// Byte offset of the first inline element.
    pub inline_offset: usize,
    /// Data-block addresses stored directly in the index block.
    pub dblk_addrs: Vec<u64>,
    /// Super-block addresses stored in the index block.
    pub sblk_addrs: Vec<u64>,
}

/// Parse the index block ("EAIB") at the address recorded in the header.
pub(crate) fn parse_index_block(
    file_data: &[u8],
    header: &EaHeader,
) -> Result<IndexBlock, OxiH5Error> {
    let after_prefix = check_block_prefix(
        file_data,
        header.index_block_address,
        b"EAIB",
        header,
        "index block",
    )?;

    let inline_bytes = usize::try_from(header.idx_blk_elmts)
        .ok()
        .and_then(|n| n.checked_mul(header.element_size))
        .ok_or_else(|| {
            OxiH5Error::Format("EA: index block inline element area overflows".into())
        })?;
    let addr_bytes = header
        .ndblk_addrs
        .checked_add(header.nsblk_addrs)
        .and_then(|n| n.checked_mul(ADDR_SIZE))
        .ok_or_else(|| OxiH5Error::Format("EA: index block address array overflows".into()))?;
    let total = after_prefix
        .checked_add(inline_bytes)
        .and_then(|v| v.checked_add(addr_bytes))
        .and_then(|v| v.checked_add(CHECKSUM_SIZE))
        .ok_or_else(|| OxiH5Error::Format("EA: index block size overflows".into()))?;
    if total > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "EA: index block needs {total} bytes but the file holds {}",
            file_data.len()
        )));
    }

    let dblk_start = after_prefix + inline_bytes;
    let mut dblk_addrs = Vec::with_capacity(header.ndblk_addrs);
    for i in 0..header.ndblk_addrs {
        dblk_addrs.push(read_u64(file_data, dblk_start + i * ADDR_SIZE)?);
    }
    let sblk_start = dblk_start + header.ndblk_addrs * ADDR_SIZE;
    let mut sblk_addrs = Vec::with_capacity(header.nsblk_addrs);
    for i in 0..header.nsblk_addrs {
        sblk_addrs.push(read_u64(file_data, sblk_start + i * ADDR_SIZE)?);
    }

    Ok(IndexBlock {
        inline_offset: after_prefix,
        dblk_addrs,
        sblk_addrs,
    })
}

/// Collect every run of elements the array stores outside the index block.
///
/// Runs are produced in element order: first the data blocks whose addresses
/// live in the index block, then those reached through each super block.
pub(crate) fn collect_element_runs(
    file_data: &[u8],
    header: &EaHeader,
    index_block: &IndexBlock,
) -> Result<Vec<ElementRun>, OxiH5Error> {
    let mut runs = Vec::new();
    let mut budget = EA_MAX_RECORDS as u64;

    // Data blocks addressed straight from the index block: super blocks
    // 0..iblk_nsblks, each contributing `ndblks` consecutive addresses.
    let mut next = 0usize;
    for u in 0..header.iblk_nsblks {
        let sblk = *header.super_block(u)?;
        for k in 0..sblk.ndblks {
            let Some(&address) = index_block.dblk_addrs.get(next) else {
                break;
            };
            next += 1;
            if address == UNDEF_ADDR {
                continue;
            }
            let first_index =
                element_base(header.idx_blk_elmts, sblk.start_idx, k, sblk.dblk_nelmts)?;
            push_data_block(
                file_data,
                header,
                address,
                sblk.dblk_nelmts,
                first_index,
                None,
                &mut runs,
                &mut budget,
            )?;
        }
    }

    // Data blocks reached through a super block.
    for (j, &sblk_addr) in index_block.sblk_addrs.iter().enumerate() {
        if sblk_addr == UNDEF_ADDR {
            continue;
        }
        let u = header
            .iblk_nsblks
            .checked_add(j)
            .ok_or_else(|| OxiH5Error::Format("EA: super block index overflows".into()))?;
        parse_super_block(file_data, header, sblk_addr, u, &mut runs, &mut budget)?;
    }

    Ok(runs)
}

/// First array index of data block `k` of a super block.
fn element_base(
    idx_blk_elmts: u64,
    start_idx: u64,
    k: u64,
    dblk_nelmts: u64,
) -> Result<u64, OxiH5Error> {
    k.checked_mul(dblk_nelmts)
        .and_then(|v| v.checked_add(start_idx))
        .and_then(|v| v.checked_add(idx_blk_elmts))
        .ok_or_else(|| OxiH5Error::Format("EA: element index overflows".into()))
}

/// Parse one super block ("EASB") and append the runs of its data blocks.
fn parse_super_block(
    file_data: &[u8],
    header: &EaHeader,
    address: u64,
    u: usize,
    runs: &mut Vec<ElementRun>,
    budget: &mut u64,
) -> Result<(), OxiH5Error> {
    let sblk = *header.super_block(u)?;
    let after_prefix = check_block_prefix(file_data, address, b"EASB", header, "super block")?;

    // The super block records the array index its first element sits at.  It
    // is the one place the file states the element numbering explicitly, so a
    // mismatch means the block geometry derived from the creation parameters
    // does not describe this file and every offset would be wrong.
    let block_off = read_var_uint(file_data, after_prefix, header.arr_off_size)?;
    if block_off != sblk.start_idx {
        return Err(OxiH5Error::Format(format!(
            "EA: super block {u} at {address:#x} declares element offset {block_off}, \
             but its creation parameters place it at {}",
            sblk.start_idx
        )));
    }

    // A super block whose data blocks are paged carries their page-init
    // bitmasks, `ndblks × ceil(npages / 8)` bytes, before the addresses.
    let npages = header.dblk_npages(sblk.dblk_nelmts);
    let page_init_bytes = match npages {
        Some(n) => {
            let per_block = usize::try_from(n.div_ceil(8))
                .map_err(|_| OxiH5Error::Format("EA: page-init bitmask width overflows".into()))?;
            usize::try_from(sblk.ndblks)
                .ok()
                .and_then(|nd| nd.checked_mul(per_block))
                .ok_or_else(|| OxiH5Error::Format("EA: page-init bitmask area overflows".into()))?
        }
        None => 0,
    };

    let addr_area = usize::try_from(sblk.ndblks)
        .ok()
        .and_then(|nd| nd.checked_mul(ADDR_SIZE))
        .ok_or_else(|| {
            OxiH5Error::Format(format!(
                "EA: super block {u} claims {} data blocks, more than the file can address",
                sblk.ndblks
            ))
        })?;
    let bitmask_start = after_prefix + header.arr_off_size;
    let addrs_start = bitmask_start
        .checked_add(page_init_bytes)
        .ok_or_else(|| OxiH5Error::Format("EA: super block layout overflows".into()))?;
    let end = addrs_start
        .checked_add(addr_area)
        .and_then(|v| v.checked_add(CHECKSUM_SIZE))
        .ok_or_else(|| OxiH5Error::Format("EA: super block size overflows".into()))?;
    if end > file_data.len() {
        return Err(OxiH5Error::Format(format!(
            "EA: super block {u} at {address:#x} needs {end} bytes but the file holds {}",
            file_data.len()
        )));
    }
    let page_init = &file_data[bitmask_start..bitmask_start + page_init_bytes];

    for k in 0..sblk.ndblks {
        let k_usize = usize::try_from(k)
            .map_err(|_| OxiH5Error::Format("EA: data block index overflows".into()))?;
        let dblk_addr = read_u64(file_data, addrs_start + k_usize * ADDR_SIZE)?;
        if dblk_addr == UNDEF_ADDR {
            continue;
        }
        let first_index = element_base(header.idx_blk_elmts, sblk.start_idx, k, sblk.dblk_nelmts)?;
        // libhdf5 indexes the whole bitmask as one bit array:
        // `page_init_idx = dblk_idx × dblk_npages + page_idx`.
        let bit_base = npages.and_then(|n| k.checked_mul(n)).unwrap_or_default();
        push_data_block(
            file_data,
            header,
            dblk_addr,
            sblk.dblk_nelmts,
            first_index,
            npages.map(|n| PagedOwner {
                bitmap: page_init,
                bit_base,
                npages: n,
            }),
            runs,
            budget,
        )?;
    }
    Ok(())
}

/// Page bookkeeping a super block holds on behalf of one of its data blocks.
struct PagedOwner<'a> {
    bitmap: &'a [u8],
    bit_base: u64,
    npages: u64,
}

/// Validate a data block ("EADB") and append its element runs.
#[allow(clippy::too_many_arguments)]
fn push_data_block(
    file_data: &[u8],
    header: &EaHeader,
    address: u64,
    nelmts: u64,
    first_index: u64,
    paged: Option<PagedOwner<'_>>,
    runs: &mut Vec<ElementRun>,
    budget: &mut u64,
) -> Result<(), OxiH5Error> {
    let after_prefix = check_block_prefix(file_data, address, b"EADB", header, "data block")?;
    // The data block also stores a block offset here; unlike the super block's
    // it does not equal the element index for index-block-owned blocks, so it
    // is skipped rather than cross-checked.
    let after_off = after_prefix
        .checked_add(header.arr_off_size)
        .ok_or_else(|| OxiH5Error::Format("EA: data block layout overflows".into()))?;

    match paged {
        None => {
            if header.dblk_npages(nelmts).is_some() {
                // Only a super block carries the page-init bitmasks for its
                // data blocks; libhdf5's fixed extensible-array creation
                // parameters never page an index-block-owned data block
                // (those hold at most `2^(iblk_nsblks/2) × data_blk_min_elmts`
                // elements), so this combination has no verified encoding.
                return Err(OxiH5Error::NotImplemented(format!(
                    "extensible array: paged data block at {address:#x} owned by the index \
                     block ({nelmts} elements > {} per page)",
                    header.dblk_page_nelmts
                )));
            }
            let span = span_bytes(nelmts, header.element_size)?;
            let end = after_off
                .checked_add(span)
                .and_then(|v| v.checked_add(CHECKSUM_SIZE))
                .ok_or_else(|| OxiH5Error::Format("EA: data block size overflows".into()))?;
            if end > file_data.len() {
                return Err(OxiH5Error::Format(format!(
                    "EA: data block at {address:#x} needs {end} bytes but the file holds {}",
                    file_data.len()
                )));
            }
            spend(budget, nelmts)?;
            runs.push(ElementRun {
                offset: after_off,
                count: nelmts,
                first_index,
            });
        }
        Some(owner) => {
            // A paged data block stores its checksum straight after the prefix
            // and then one page per `dblk_page_nelmts` elements, each with its
            // own trailing checksum.  Pages whose bit is clear were never
            // written and hold whatever the file allocator left behind.
            let page_elems = header.dblk_page_nelmts;
            let page_span = span_bytes(page_elems, header.element_size)?
                .checked_add(CHECKSUM_SIZE)
                .ok_or_else(|| OxiH5Error::Format("EA: data block page size overflows".into()))?;
            let pages_start = after_off
                .checked_add(CHECKSUM_SIZE)
                .ok_or_else(|| OxiH5Error::Format("EA: data block layout overflows".into()))?;
            let mut remaining = nelmts;
            for page in 0..owner.npages {
                let page_usize = usize::try_from(page)
                    .map_err(|_| OxiH5Error::Format("EA: page index overflows".into()))?;
                let offset = page_usize
                    .checked_mul(page_span)
                    .and_then(|v| pages_start.checked_add(v))
                    .ok_or_else(|| OxiH5Error::Format("EA: page offset overflows".into()))?;
                let count = remaining.min(page_elems);
                remaining = remaining.saturating_sub(count);
                let end = span_bytes(count, header.element_size)?
                    .checked_add(offset)
                    .and_then(|v| v.checked_add(CHECKSUM_SIZE))
                    .ok_or_else(|| OxiH5Error::Format("EA: page extent overflows".into()))?;
                if end > file_data.len() {
                    return Err(OxiH5Error::Format(format!(
                        "EA: data block page {page} at {address:#x} needs {end} bytes but the \
                         file holds {}",
                        file_data.len()
                    )));
                }
                let bit = owner.bit_base.checked_add(page).ok_or_else(|| {
                    OxiH5Error::Format("EA: page-init bit index overflows".into())
                })?;
                if !bit_is_set(owner.bitmap, bit) {
                    continue;
                }
                spend(budget, count)?;
                let first = first_index
                    .checked_add(page.saturating_mul(page_elems))
                    .ok_or_else(|| {
                        OxiH5Error::Format("EA: paged element index overflows".into())
                    })?;
                runs.push(ElementRun {
                    offset,
                    count,
                    first_index: first,
                });
            }
        }
    }
    Ok(())
}

/// Byte span of `count` elements.
fn span_bytes(count: u64, element_size: usize) -> Result<usize, OxiH5Error> {
    usize::try_from(count)
        .ok()
        .and_then(|n| n.checked_mul(element_size))
        .ok_or_else(|| {
            OxiH5Error::Format(format!(
                "EA: {count} elements of {element_size} bytes exceed the address space"
            ))
        })
}

/// Charge `count` elements against the decoding budget.
fn spend(budget: &mut u64, count: u64) -> Result<(), OxiH5Error> {
    *budget = budget.checked_sub(count).ok_or_else(|| {
        OxiH5Error::Format(format!(
            "EA: the array describes more than {EA_MAX_RECORDS} elements"
        ))
    })?;
    Ok(())
}

/// Read bit `index` of a bitmask stored most-significant-bit first, as
/// `H5VM_bit_get` does.
fn bit_is_set(bitmap: &[u8], index: u64) -> bool {
    let byte = (index / 8) as usize;
    let shift = 7 - (index % 8) as u32;
    bitmap.get(byte).is_some_and(|b| (b >> shift) & 1 == 1)
}
