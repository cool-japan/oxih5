//! Extensible Array chunk index (HDF5 1.10+).
//!
//! libhdf5 selects an extensible array for a chunked dataset with exactly one
//! unlimited dimension — the shape `h5py`'s
//! `create_dataset(..., chunks=..., maxshape=(None, ...))` produces under
//! `libver='latest'`.
//!
//! # On-disk structure
//!
//! ```text
//! EAHD  header            creation parameters + statistics + index-block address
//!  └─ EAIB index block    inline elements, data-block addresses, super-block addresses
//!      ├─ EADB data block elements (or pages of elements, for a large block)
//!      └─ EASB super block page-init bitmasks + data-block addresses
//!          └─ EADB …
//! ```
//!
//! Elements are **not** self-describing.  An unfiltered array stores a bare
//! 8-byte chunk address per element; a filtered one stores the address, the
//! stored size in a variable number of bytes, and a 4-byte filter mask.  A
//! chunk's position comes from its element index combined with the dataset's
//! chunk geometry, which is why [`parse_extensible_array`] takes an
//! [`EaGeometry`] rather than just a rank.
//!
//! # Assumptions
//!
//! The header, index block, super blocks and data blocks are decoded for a file
//! whose superblock declares 8-byte offsets and lengths — the only combination
//! libhdf5 writes for the versions that use this index.  A file with narrower
//! addresses puts the index-block address elsewhere in the header and is
//! rejected at the "EAIB" signature check rather than mis-decoded.

mod blocks;
mod elements;
mod header;

#[cfg(test)]
mod tests;

use crate::btree_v2::ChunkRecord;
use oxih5_core::OxiH5Error;

pub use elements::EaGeometry;

use blocks::{collect_element_runs, parse_index_block};
use elements::{decode_element, ElementLocator};
use header::{EaHeader, UNDEF_ADDR};

/// Parse the extensible-array chunk index rooted at `header_address`.
///
/// Returns one [`ChunkRecord`] per allocated chunk, in element order.  An array
/// with no index block yet — a dataset created but never written — yields an
/// empty list rather than an error, matching libhdf5's "read the fill value"
/// behaviour.
///
/// Unallocated element slots, and chunks that fall outside the dataset's
/// current shape (which `H5Dset_extent` can leave behind after a shrink), are
/// skipped.
///
/// # Errors
///
/// Returns [`OxiH5Error::Format`] when a signature, version, back-pointer,
/// super-block element offset or block extent is inconsistent, and
/// [`OxiH5Error::NotImplemented`] for an array whose client ID is not one of
/// the two dataset-chunk clients.
pub fn parse_extensible_array(
    file_data: &[u8],
    header_address: u64,
    geom: &EaGeometry<'_>,
) -> Result<Vec<ChunkRecord>, OxiH5Error> {
    let hdr = EaHeader::parse(file_data, header_address)?;
    let locator = ElementLocator::new(geom)?;

    if hdr.index_block_address == UNDEF_ADDR {
        // No index block allocated: nothing has ever been written.
        return Ok(Vec::new());
    }

    let index_block = parse_index_block(file_data, &hdr)?;
    let mut records = Vec::new();

    // Elements 0..idx_blk_elmts live inline in the index block.
    for i in 0..hdr.idx_blk_elmts {
        let i_usize = usize::try_from(i)
            .map_err(|_| OxiH5Error::Format("EA: inline element index overflows".into()))?;
        let offset = index_block
            .inline_offset
            .checked_add(i_usize * hdr.element_size)
            .ok_or_else(|| OxiH5Error::Format("EA: inline element offset overflows".into()))?;
        push_record(
            file_data,
            &hdr,
            &locator,
            geom.chunk_bytes,
            offset,
            i,
            &mut records,
        )?;
    }

    for run in collect_element_runs(file_data, &hdr, &index_block)? {
        for i in 0..run.count {
            let i_usize = usize::try_from(i)
                .map_err(|_| OxiH5Error::Format("EA: element index overflows".into()))?;
            let offset = run
                .offset
                .checked_add(i_usize * hdr.element_size)
                .ok_or_else(|| OxiH5Error::Format("EA: element offset overflows".into()))?;
            let index = run
                .first_index
                .checked_add(i)
                .ok_or_else(|| OxiH5Error::Format("EA: element index overflows".into()))?;
            push_record(
                file_data,
                &hdr,
                &locator,
                geom.chunk_bytes,
                offset,
                index,
                &mut records,
            )?;
        }
    }

    Ok(records)
}

/// Decode one element and, if it is allocated and in range, record its chunk.
fn push_record(
    file_data: &[u8],
    hdr: &EaHeader,
    locator: &ElementLocator,
    chunk_bytes: u32,
    offset: usize,
    index: u64,
    records: &mut Vec<ChunkRecord>,
) -> Result<(), OxiH5Error> {
    let Some(elem) = decode_element(file_data, offset, hdr, chunk_bytes)? else {
        return Ok(());
    };
    let Some(offsets) = locator.offsets_for(index)? else {
        return Ok(());
    };
    records.push(ChunkRecord {
        address: elem.address,
        size: elem.size,
        filter_mask: elem.filter_mask,
        offsets,
    });
    Ok(())
}
