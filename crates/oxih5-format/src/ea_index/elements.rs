//! Element decoding and the element-index → chunk-offset mapping.
//!
//! An extensible-array element does **not** describe its own chunk position.
//! libhdf5 stores only what varies per chunk (the address, and for a filtered
//! dataset the stored size and filter mask) and recovers the position from the
//! element's linear index using the dataset's chunk geometry.  That mapping is
//! `H5VM_array_offset_pre` over *swizzled* coordinates: the unlimited dimension
//! is rotated to the front so that growing the dataset only ever appends to the
//! array.

use oxih5_core::OxiH5Error;

use super::header::{read_u64, read_var_uint, EaClient, EaHeader, ADDR_SIZE, UNDEF_ADDR};

/// The dataset geometry an extensible array needs in order to be decodable.
///
/// Every field comes from messages the caller has already parsed: the chunk
/// shape from the data-layout message, the current and maximum dimensions from
/// the dataspace message.
#[derive(Debug, Clone, Copy)]
pub struct EaGeometry<'a> {
    /// Per-dimension chunk shape in elements.  Must have the dataset's rank —
    /// strip the layout message's trailing element-size entry first.
    pub chunk_dims: &'a [u64],
    /// Current dataset shape in elements.
    pub dataset_dims: &'a [u64],
    /// Dataspace maximum dimensions, where `u64::MAX` marks an unlimited
    /// dimension.  Required for rank ≥ 2, because the position of the single
    /// unlimited dimension decides the element ordering; ignored for rank 1,
    /// where the element index *is* the chunk-grid coordinate.
    pub max_dims: Option<&'a [u64]>,
    /// Uncompressed byte size of one whole chunk.  Unfiltered arrays store only
    /// a chunk address, so this is where the record's size comes from.
    pub chunk_bytes: u32,
}

/// Turns a linear extensible-array element index into a chunk offset vector.
pub(crate) struct ElementLocator {
    chunk_dims: Vec<u64>,
    dataset_dims: Vec<u64>,
    /// Row-major strides over the *swizzled* chunk grid.  `strides[0]` is the
    /// span of one step along the unlimited dimension, so the unlimited
    /// dimension's own (unbounded) extent never enters the arithmetic.
    strides: Vec<u64>,
    /// Index of the unlimited dimension in unswizzled coordinates.
    unlim_dim: usize,
}

impl ElementLocator {
    /// Build the locator, validating the geometry it is handed.
    pub(crate) fn new(geom: &EaGeometry<'_>) -> Result<Self, OxiH5Error> {
        let rank = geom.dataset_dims.len();
        if rank == 0 {
            return Err(OxiH5Error::Format(
                "EA: a chunked dataset must have rank ≥ 1".into(),
            ));
        }
        if geom.chunk_dims.len() != rank {
            return Err(OxiH5Error::Format(format!(
                "EA: chunk rank {} does not match dataset rank {rank}",
                geom.chunk_dims.len()
            )));
        }
        if let Some(d) = geom.chunk_dims.iter().position(|&c| c == 0) {
            return Err(OxiH5Error::Format(format!(
                "EA: chunk dimension {d} is zero"
            )));
        }

        // Rank 1 is the case libhdf5 chooses an extensible array for most
        // often, and it needs no maximum dimensions at all: the element index
        // is the chunk-grid coordinate.
        if rank == 1 {
            return Ok(ElementLocator {
                chunk_dims: geom.chunk_dims.to_vec(),
                dataset_dims: geom.dataset_dims.to_vec(),
                strides: vec![1],
                unlim_dim: 0,
            });
        }

        let max_dims = geom.max_dims.ok_or_else(|| {
            OxiH5Error::Format(format!(
                "EA: a rank-{rank} extensible array needs the dataspace maximum dimensions \
                 to locate the unlimited dimension, but the dataspace message carries none"
            ))
        })?;
        if max_dims.len() != rank {
            return Err(OxiH5Error::Format(format!(
                "EA: maximum-dimension rank {} does not match dataset rank {rank}",
                max_dims.len()
            )));
        }
        let unlimited: Vec<usize> = (0..rank).filter(|&d| max_dims[d] == u64::MAX).collect();
        if unlimited.len() != 1 {
            return Err(OxiH5Error::Format(format!(
                "EA: an extensible array indexes a dataset with exactly one unlimited \
                 dimension, but {} of {rank} are unlimited",
                unlimited.len()
            )));
        }
        let unlim_dim = unlimited[0];

        // Chunk-grid extent per dimension, from the *maximum* dimensions: the
        // array must stay stable as the dataset grows, so the current shape
        // cannot be used.  The unlimited dimension's own extent is never
        // needed (see `strides`), so it is not computed.
        let mut grid = vec![0u64; rank];
        for (d, slot) in grid.iter_mut().enumerate() {
            if d == unlim_dim {
                continue;
            }
            let extent = max_dims[d].div_ceil(geom.chunk_dims[d]);
            if extent == 0 {
                return Err(OxiH5Error::Format(format!(
                    "EA: dimension {d} has a maximum size of zero, so it holds no chunks"
                )));
            }
            *slot = extent;
        }

        // Swizzle: rotate `unlim_dim` to position 0, sliding 0..unlim_dim down
        // one place (`H5VM_swizzle_coords`).  Note this is a rotation, not a
        // swap of positions 0 and `unlim_dim` — the two agree only for rank 2
        // or when the unlimited dimension is dimension 1.
        let mut swizzled = Vec::with_capacity(rank);
        swizzled.push(grid[unlim_dim]);
        swizzled.extend_from_slice(&grid[..unlim_dim]);
        swizzled.extend_from_slice(&grid[unlim_dim + 1..]);

        let mut strides = vec![1u64; rank];
        for d in (0..rank - 1).rev() {
            strides[d] = strides[d + 1].checked_mul(swizzled[d + 1]).ok_or_else(|| {
                OxiH5Error::Format(
                    "EA: the chunk grid implied by the maximum dimensions overflows".into(),
                )
            })?;
        }

        Ok(ElementLocator {
            chunk_dims: geom.chunk_dims.to_vec(),
            dataset_dims: geom.dataset_dims.to_vec(),
            strides,
            unlim_dim,
        })
    }

    /// Chunk offsets (in elements) for array element `index`.
    ///
    /// Returns `Ok(None)` when the chunk lies outside the dataset's *current*
    /// shape.  libhdf5 leaves such entries behind after `H5Dset_extent` shrinks
    /// a dataset and simply never looks them up; carrying them into the record
    /// list would make the readers scatter bytes past the output buffer.
    pub(crate) fn offsets_for(&self, index: u64) -> Result<Option<Vec<u64>>, OxiH5Error> {
        let rank = self.chunk_dims.len();
        let mut swizzled_scaled = vec![0u64; rank];
        let mut rem = index;
        for (d, (slot, &stride)) in swizzled_scaled
            .iter_mut()
            .zip(self.strides.iter())
            .enumerate()
        {
            if stride == 0 {
                return Err(OxiH5Error::Format(format!(
                    "EA: chunk-grid stride for dimension {d} is zero"
                )));
            }
            *slot = rem / stride;
            rem %= stride;
        }

        // Undo the rotation: dimensions before the unlimited one slid down by
        // one place, the unlimited one sat at the front.
        let mut scaled = Vec::with_capacity(rank);
        scaled.extend_from_slice(&swizzled_scaled[1..self.unlim_dim + 1]);
        scaled.push(swizzled_scaled[0]);
        scaled.extend_from_slice(&swizzled_scaled[self.unlim_dim + 1..]);

        let mut offsets = Vec::with_capacity(rank);
        for (d, (&coord, &extent)) in scaled.iter().zip(self.chunk_dims.iter()).enumerate() {
            let off = coord.checked_mul(extent).ok_or_else(|| {
                OxiH5Error::Format(format!(
                    "EA: chunk offset for dimension {d} overflows (element index {index})"
                ))
            })?;
            if off >= self.dataset_dims[d] {
                return Ok(None);
            }
            offsets.push(off);
        }
        Ok(Some(offsets))
    }
}

/// One decoded array element: what libhdf5 stores per chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DecodedElement {
    pub address: u64,
    pub size: u32,
    pub filter_mask: u32,
}

/// Decode the element at `offset`, or `None` when the slot is unallocated.
pub(crate) fn decode_element(
    file_data: &[u8],
    offset: usize,
    header: &EaHeader,
    chunk_bytes: u32,
) -> Result<Option<DecodedElement>, OxiH5Error> {
    let address = read_u64(file_data, offset)?;
    if address == UNDEF_ADDR {
        return Ok(None);
    }
    match header.client {
        EaClient::Chunk => Ok(Some(DecodedElement {
            address,
            size: chunk_bytes,
            filter_mask: 0,
        })),
        EaClient::FilteredChunk { size_len } => {
            let stored = read_var_uint(file_data, offset + ADDR_SIZE, size_len)?;
            let size = u32::try_from(stored).map_err(|_| {
                OxiH5Error::Format(format!(
                    "EA: filtered chunk at {address:#x} stores {stored} bytes, which exceeds u32"
                ))
            })?;
            let mask_at = offset + ADDR_SIZE + size_len;
            let bytes = file_data.get(mask_at..mask_at + 4).ok_or_else(|| {
                OxiH5Error::Format("EA: filtered element filter mask truncated".into())
            })?;
            let arr: [u8; 4] = bytes
                .try_into()
                .map_err(|_| OxiH5Error::Format("EA: filter mask slice".into()))?;
            Ok(Some(DecodedElement {
                address,
                size,
                filter_mask: u32::from_le_bytes(arr),
            }))
        }
    }
}
