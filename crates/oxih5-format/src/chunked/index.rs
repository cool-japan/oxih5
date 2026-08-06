//! Chunk-index-type dispatch and resolution: turning a chunked layout's on-disk index (B-tree v1/v2, Fixed Array, Extensible Array, single-chunk, implicit) into ChunkRecords.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::btree_v2::{BTreeV2, ChunkRecord};
use crate::message::SingleChunkInfo;
use crate::{btree_v1_chunk, ea_index, fa_index};
use oxih5_core::OxiH5Error;
use std::sync::Arc;

use super::cache::ChunkIndexCache;

/// Index variety used by a chunked layout's chunk index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkIndex {
    /// Version-1 B-tree (layout v3 default, `libver='earliest'`).
    BTreeV1,
    /// Version-2 B-tree (layout v4, HDF5 1.10+).
    BTreeV2,
    /// Fixed array (layout v4, non-extensible chunked datasets).
    FixedArray,
    /// Extensible array (layout v4, single-extensible-dimension datasets).
    ExtensibleArray,
    /// Single chunk (layout v4): the dataset is exactly one chunk, so there is
    /// no index at all — the layout message's address *is* the chunk address.
    SingleChunk,
    /// Implicit (layout v4): no index either.  Every chunk is allocated up
    /// front, unfiltered, laid out back to back in row-major chunk order
    /// starting at the layout message's address.
    Implicit,
}
/// The dataspace geometry every chunked reader needs.
///
/// The two halves are grouped rather than passed separately because they are
/// only meaningful together — the maximum dimensions locate the unlimited
/// dimension *within* `dims` — and because splitting them pushes the readers
/// past the argument-count limit.
#[derive(Debug, Clone, Copy)]
pub struct DatasetShape<'a> {
    /// Current dataset shape in elements.
    pub dims: &'a [u64],
    /// Dataspace maximum dimensions, where `u64::MAX` marks an unlimited
    /// dimension, when the dataspace message carries them.  Only the
    /// extensible-array chunk index consults them, and only for rank ≥ 2.
    pub max_dims: Option<&'a [u64]>,
}

impl<'a> DatasetShape<'a> {
    /// A shape whose dataspace message records no maximum dimensions.
    pub fn fixed(dims: &'a [u64]) -> Self {
        DatasetShape {
            dims,
            max_dims: None,
        }
    }
}

/// Everything needed to turn a chunked layout's index into chunk records.
///
/// Grouped into one struct because six of these travel together through the
/// three readers (`read_chunked`, `read_chunked_slice`,
/// `read_chunked_hyperslab`), which previously each carried their own copy of
/// the resolution logic.
pub(crate) struct ChunkIndexQuery<'a> {
    pub index: ChunkIndex,
    /// Index root address, or chunk data address for single-chunk / implicit.
    pub index_address: u64,
    /// Per-dimension chunk shape in elements (element-size entry stripped).
    pub real_chunk_dims: &'a [u64],
    /// Full dataset shape in elements.
    pub dataset_dims: &'a [u64],
    /// Dataspace maximum dimensions (`u64::MAX` marks an unlimited dimension),
    /// when the dataspace message carries them.  Only the extensible-array
    /// index needs them, and only for rank ≥ 2, where the position of the
    /// unlimited dimension decides the element ordering.
    pub max_dims: Option<&'a [u64]>,
    pub elem_size: usize,
    /// Stored size + filter mask for a filtered single-chunk index.
    pub single_chunk: Option<SingleChunkInfo>,
}
/// Translate oxih5's internal `index_type` discriminant (as stored in
/// [`LayoutInfo::Chunked`]) into a [`ChunkIndex`].
///
/// These numbers are *this crate's* convention and deliberately differ from the
/// HDF5 on-disk indexing-type values; `message::parse_layout` performs that
/// translation when decoding a version-4 layout message.
pub(crate) fn index_of(index_type: u8, whose: &str) -> Result<ChunkIndex, OxiH5Error> {
    match index_type {
        0 => Ok(ChunkIndex::BTreeV1),
        1 => Ok(ChunkIndex::FixedArray),
        2 => Ok(ChunkIndex::ExtensibleArray),
        3 => Ok(ChunkIndex::BTreeV2),
        4 => Ok(ChunkIndex::SingleChunk),
        5 => Ok(ChunkIndex::Implicit),
        other => Err(OxiH5Error::Format(format!(
            "{whose}: unknown chunk index type {other}"
        ))),
    }
}

/// Resolve a chunk index into its chunk records.
///
/// `index_address` points at the index root; `ndims` is the *real* dataset
/// rank (not the layout's `dimensionality`, which is rank + 1).
///
/// Several index varieties need the dataset's chunk geometry to produce usable
/// records — the single-chunk and implicit indexes have no on-disk structure at
/// all, a version-2 B-tree stores *scaled* (chunk-grid) coordinates rather than
/// element offsets, and an extensible array stores no coordinates whatsoever.
/// Those are rejected here; go through `chunk_records` (as every reader in this
/// crate does) instead.
pub fn resolve_chunk_index(
    file_data: &[u8],
    index: ChunkIndex,
    index_address: u64,
    ndims: usize,
) -> Result<Vec<ChunkRecord>, OxiH5Error> {
    match index {
        ChunkIndex::BTreeV1 => btree_v1_chunk::parse(file_data, index_address, ndims),
        ChunkIndex::FixedArray => fa_index::parse_fixed_array(file_data, index_address, ndims),
        ChunkIndex::ExtensibleArray
        | ChunkIndex::BTreeV2
        | ChunkIndex::SingleChunk
        | ChunkIndex::Implicit => Err(OxiH5Error::Format(format!(
            "resolve_chunk_index: the {index:?} index needs the dataset's chunk geometry; \
             use chunk_records instead"
        ))),
    }
}

/// Number of chunks along each dimension: `ceil(dataset_dim / chunk_dim)`.
fn chunk_counts(dataset_dims: &[u64], chunk_dims: &[u64]) -> Result<Vec<u64>, OxiH5Error> {
    dataset_dims
        .iter()
        .zip(chunk_dims.iter())
        .map(|(&d, &c)| {
            if c == 0 {
                Err(OxiH5Error::Format(
                    "chunked layout: chunk dimension of zero".into(),
                ))
            } else {
                Ok(d.div_ceil(c))
            }
        })
        .collect()
}

/// Uncompressed byte size of one whole chunk.
fn chunk_byte_size(chunk_dims: &[u64], elem_size: usize) -> Result<usize, OxiH5Error> {
    let volume = chunk_dims
        .iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| OxiH5Error::Format("chunked layout: chunk volume overflows".into()))?;
    usize::try_from(volume)
        .ok()
        .and_then(|v| v.checked_mul(elem_size))
        .ok_or_else(|| OxiH5Error::Format("chunked layout: chunk size overflows".into()))
}

/// Build the single record described by a [`ChunkIndex::SingleChunk`] layout.
///
/// The chunk covers the whole dataset and therefore sits at offset zero in
/// every dimension.  Its stored size comes from the layout message when a
/// filter pipeline is present, and is the plain chunk size otherwise.
fn single_chunk_records(q: &ChunkIndexQuery<'_>) -> Result<Vec<ChunkRecord>, OxiH5Error> {
    let (size, filter_mask) = match q.single_chunk {
        Some(info) => (
            u32::try_from(info.stored_size).map_err(|_| {
                OxiH5Error::Format(format!(
                    "single-chunk layout: stored size {} exceeds u32",
                    info.stored_size
                ))
            })?,
            info.filter_mask,
        ),
        None => {
            let bytes = chunk_byte_size(q.real_chunk_dims, q.elem_size)?;
            let size = u32::try_from(bytes).map_err(|_| {
                OxiH5Error::Format(format!(
                    "single-chunk layout: chunk size {bytes} exceeds u32"
                ))
            })?;
            (size, 0)
        }
    };
    Ok(vec![ChunkRecord {
        address: q.index_address,
        size,
        filter_mask,
        offsets: vec![0; q.dataset_dims.len()],
    }])
}

/// Enumerate the records of a [`ChunkIndex::Implicit`] layout.
///
/// Implicit indexing is only chosen by libhdf5 when the dataset has fixed
/// dimensions, no filter pipeline and early allocation, so every chunk exists,
/// every chunk is stored uncompressed at its full size, and chunk *n* begins at
/// `address + n * chunk_bytes` with *n* running in row-major order over the
/// chunk grid.
fn implicit_records(q: &ChunkIndexQuery<'_>) -> Result<Vec<ChunkRecord>, OxiH5Error> {
    let ndims = q.dataset_dims.len();
    let counts = chunk_counts(q.dataset_dims, q.real_chunk_dims)?;
    let chunk_bytes = chunk_byte_size(q.real_chunk_dims, q.elem_size)?;
    let size = u32::try_from(chunk_bytes).map_err(|_| {
        OxiH5Error::Format(format!(
            "implicit chunk layout: chunk size {chunk_bytes} exceeds u32"
        ))
    })?;

    let total: u64 = counts
        .iter()
        .try_fold(1u64, |acc, &c| acc.checked_mul(c))
        .ok_or_else(|| OxiH5Error::Format("implicit chunk layout: chunk count overflows".into()))?;
    let total = usize::try_from(total).map_err(|_| {
        OxiH5Error::Format("implicit chunk layout: chunk count exceeds addressable range".into())
    })?;

    let mut records = Vec::with_capacity(total);
    let mut coords = vec![0u64; ndims];
    for n in 0..total {
        let byte_offset = (n as u64).checked_mul(chunk_bytes as u64).ok_or_else(|| {
            OxiH5Error::Format("implicit chunk layout: chunk offset overflows".into())
        })?;
        let address = q.index_address.checked_add(byte_offset).ok_or_else(|| {
            OxiH5Error::Format("implicit chunk layout: chunk address overflows".into())
        })?;
        // Chunk offsets are in *elements*, so scale the chunk-grid coordinate
        // by the chunk extent in each dimension.
        let offsets: Vec<u64> = coords
            .iter()
            .zip(q.real_chunk_dims.iter())
            .map(|(&c, &extent)| c.saturating_mul(extent))
            .collect();
        records.push(ChunkRecord {
            address,
            size,
            filter_mask: 0,
            offsets,
        });

        // Advance the row-major chunk-grid coordinate.
        for d in (0..ndims).rev() {
            coords[d] += 1;
            if coords[d] < counts[d] {
                break;
            }
            coords[d] = 0;
        }
    }
    Ok(records)
}

/// Resolve a chunked layout's index into chunk records, honouring the cache.
///
/// An index address of `u64::MAX` is HDF5's "undefined address": the dataset
/// has no chunks allocated yet (nothing was ever written to it).  That is not
/// an error — libhdf5 reads such a dataset as all fill value — so it yields an
/// empty record list and the caller's fill-initialised buffer stands.
pub(crate) fn chunk_records(
    file_data: &[u8],
    q: &ChunkIndexQuery<'_>,
    cache: Option<&ChunkIndexCache>,
) -> Result<Arc<Vec<ChunkRecord>>, OxiH5Error> {
    let ndims = q.dataset_dims.len();

    if q.index_address == u64::MAX {
        return Ok(Arc::new(Vec::new()));
    }

    let compute = || -> Result<Vec<ChunkRecord>, OxiH5Error> {
        match q.index {
            // The fixed-array index needs the chunk and dataset shapes to
            // reconstruct each chunk's offset from its linear array position.
            ChunkIndex::FixedArray => fa_index::parse_fixed_array_v4_with_dataset_dims(
                file_data,
                q.index_address,
                ndims,
                q.real_chunk_dims,
                q.dataset_dims,
                chunk_byte_size(q.real_chunk_dims, q.elem_size)?,
            ),
            // A v2 B-tree stores scaled coordinates, and an unfiltered record
            // omits the chunk size entirely; both come from the geometry.
            ChunkIndex::BTreeV2 => {
                let chunk_bytes = chunk_byte_size(q.real_chunk_dims, q.elem_size)?;
                let chunk_bytes = u32::try_from(chunk_bytes).map_err(|_| {
                    OxiH5Error::Format(format!(
                        "B-tree v2 chunk index: chunk size {chunk_bytes} exceeds u32"
                    ))
                })?;
                Ok(BTreeV2::parse(
                    file_data,
                    q.index_address,
                    ndims,
                    &crate::btree_v2::ChunkGeometry {
                        chunk_dims: q.real_chunk_dims,
                        chunk_bytes,
                    },
                )?
                .records()
                .to_vec())
            }
            // An extensible array stores no chunk coordinates at all: each
            // element's position comes from its linear index combined with the
            // chunk grid the *maximum* dimensions imply.
            ChunkIndex::ExtensibleArray => {
                let chunk_bytes = chunk_byte_size(q.real_chunk_dims, q.elem_size)?;
                let chunk_bytes = u32::try_from(chunk_bytes).map_err(|_| {
                    OxiH5Error::Format(format!(
                        "extensible array chunk index: chunk size {chunk_bytes} exceeds u32"
                    ))
                })?;
                ea_index::parse_extensible_array(
                    file_data,
                    q.index_address,
                    &ea_index::EaGeometry {
                        chunk_dims: q.real_chunk_dims,
                        dataset_dims: q.dataset_dims,
                        max_dims: q.max_dims,
                        chunk_bytes,
                    },
                )
            }
            ChunkIndex::SingleChunk => single_chunk_records(q),
            ChunkIndex::Implicit => implicit_records(q),
            other => resolve_chunk_index(file_data, other, q.index_address, ndims),
        }
    };

    match cache {
        Some(c) => c.get_or_insert((q.index_address, ndims), compute),
        None => Ok(Arc::new(compute()?)),
    }
}
