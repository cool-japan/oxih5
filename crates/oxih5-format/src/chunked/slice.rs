//! Hyperslab-range chunk reading: only the chunk-grid cells overlapping a requested contiguous-range slice are resolved, read and filtered.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::btree_v2::ChunkRecord;
use crate::message::LayoutInfo;
// Only reachable through `assemble_chunks_slice`'s call from the
// non-parallel arm of `read_chunked_slice` below — gate the import to
// match, or it is (correctly) reported unused whenever the `parallel`
// feature is enabled.
#[cfg(not(feature = "parallel"))]
use crate::filters;
use oxih5_core::{FilterPipeline, OxiH5Error};
// Only the `#[cfg(feature = "parallel")]` chunk/decompression maps below need
// a `HashMap`; the non-parallel path (`assemble_chunks_slice`) does not.
#[cfg(feature = "parallel")]
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::cache::ChunkIndexCache;
use super::geometry::{flat_to_coords, row_major_strides};
// `apply_filters_to_chunk`/`read_chunk_bytes`/`scatter_chunk_slice` are only
// called from the `#[cfg(feature = "parallel")]` block below, so gate the
// imports to match — `scatter_chunk_slice` is defined under
// `any(feature = "parallel", test)` (wider than this file needs it), but
// this file's own use-site only ever needs it under `parallel`; matching
// that narrower condition (rather than the definition's own gate) keeps the
// import from going unused under a `test`-but-not-`parallel` build.
#[cfg(feature = "parallel")]
use super::geometry::{apply_filters_to_chunk, read_chunk_bytes, scatter_chunk_slice};
use super::index::{chunk_records, index_of, ChunkIndexQuery, DatasetShape};

/// Grouped parameters for [`read_chunked_slice`] to keep the argument count ≤ 7.
pub struct ChunkSliceParams<'a> {
    /// Element size in bytes.
    pub elem_size: usize,
    /// Optional per-element fill bytes for sparse chunks (length must equal `elem_size`).
    pub fill_value: Option<&'a [u8]>,
}
/// Parameters for [`assemble_chunks_slice`] that pack element config together
/// to keep the argument count under the clippy limit.
#[cfg(any(not(feature = "parallel"), test))]
pub(super) struct SliceElemConfig<'a> {
    pub(super) elem_size: usize,
    pub(super) fill_value: Option<&'a [u8]>,
}
/// Read only the chunks that overlap with `ranges` from a chunked dataset.
///
/// Returns a flat contiguous byte buffer of shape `[r0.len(), r1.len(), ..., rN-1.len()]`
/// in row-major (C) order, containing exactly the elements selected by `ranges`.
/// Each `ranges[i]` must satisfy `ranges[i].start <= ranges[i].end <= dataset_dims[i]`.
///
/// # Arguments
///
/// * `layout`       – the parsed chunked layout message
/// * `pipeline`     – the dataset's filter pipeline (empty ⇒ no filters)
/// * `shape`        – the dataset's current and maximum dimensions
/// * `params`       – element size and optional fill value (use `ChunkSliceParams`)
/// * `ranges`       – one `Range<u64>` per dimension specifying the requested sub-region
/// * `cache`        – optional pre-parsed chunk index cache; pass `None` to disable caching
pub fn read_chunked_slice(
    file_data: &[u8],
    layout: &LayoutInfo,
    pipeline: &FilterPipeline,
    shape: DatasetShape<'_>,
    params: ChunkSliceParams<'_>,
    ranges: &[std::ops::Range<u64>],
    cache: Option<&ChunkIndexCache>,
) -> Result<Vec<u8>, OxiH5Error> {
    let DatasetShape {
        dims: dataset_dims,
        max_dims,
    } = shape;
    let elem_size = params.elem_size;
    let fill_value = params.fill_value;
    let LayoutInfo::Chunked {
        data_address,
        dimensionality,
        chunk_dims,
        index_type,
        single_chunk,
    } = layout
    else {
        return Err(OxiH5Error::Format(
            "read_chunked_slice: layout is not chunked".into(),
        ));
    };

    let ndims = dataset_dims.len();

    if ranges.len() != ndims {
        return Err(OxiH5Error::Format(format!(
            "read_chunked_slice: {} ranges for {} dimensions",
            ranges.len(),
            ndims,
        )));
    }

    // Validate ranges and compute output shape.
    let mut out_dims = Vec::with_capacity(ndims);
    for (d, (r, &dim)) in ranges.iter().zip(dataset_dims.iter()).enumerate() {
        if r.start > r.end {
            return Err(OxiH5Error::Format(format!(
                "read_chunked_slice: range {}..{} is invalid (start > end) for dim {}",
                r.start, r.end, d
            )));
        }
        if r.end > dim {
            return Err(OxiH5Error::Format(format!(
                "read_chunked_slice: range {}..{} out of bounds for dim {} (size {})",
                r.start, r.end, d, dim
            )));
        }
        out_dims.push(r.end - r.start);
    }

    // Short-circuit: if any dimension has zero length, return an empty buffer.
    if out_dims.contains(&0) {
        return Ok(vec![]);
    }

    // Strip trailing element-size "dimension" from chunk_dims (layout v3/v4 convention).
    let real_chunk_dims: Vec<u64> = if chunk_dims.len() == ndims + 1 {
        chunk_dims[..ndims].to_vec()
    } else if chunk_dims.len() == ndims {
        chunk_dims.clone()
    } else {
        return Err(OxiH5Error::Format(format!(
            "read_chunked_slice: chunk_dims length {} incompatible with rank {} (dimensionality field = {})",
            chunk_dims.len(),
            ndims,
            dimensionality,
        )));
    };

    let index = index_of(*index_type, "read_chunked_slice")?;

    // Resolve all chunk records (with optional caching).
    let chunks_arc: Arc<Vec<ChunkRecord>> = chunk_records(
        file_data,
        &ChunkIndexQuery {
            index,
            index_address: *data_address,
            real_chunk_dims: &real_chunk_dims,
            dataset_dims,
            max_dims,
            elem_size,
            single_chunk: *single_chunk,
        },
        cache,
    )?;

    #[cfg(feature = "parallel")]
    {
        // Build a sparse-chunk fill buffer of the right size.
        let make_sparse_fill = |n_elems: usize| -> Vec<u8> {
            match fill_value {
                Some(fv) if fv.len() == elem_size && elem_size > 0 => fv
                    .iter()
                    .cycle()
                    .take(n_elems * elem_size)
                    .cloned()
                    .collect(),
                _ => vec![0u8; n_elems * elem_size],
            }
        };

        let out_dims: Vec<u64> = ranges.iter().map(|r| r.end - r.start).collect();
        let out_elems: u64 = out_dims.iter().product();
        let mut output = vec![0u8; out_elems as usize * elem_size];

        let chunk_volume: u64 = real_chunk_dims.iter().product();

        // Build chunk map: origin -> index in chunks_arc.
        let mut chunk_map: HashMap<Vec<u64>, usize> = HashMap::with_capacity(chunks_arc.len());
        for (i, cr) in chunks_arc.iter().enumerate() {
            if cr.offsets.len() >= ndims {
                chunk_map.insert(cr.offsets[..ndims].to_vec(), i);
            }
        }

        // Reject a zero chunk dimension before dividing by it below (see the
        // non-parallel `assemble_chunks_slice` path for the same guard).
        if let Some(d) = real_chunk_dims.iter().position(|&c| c == 0) {
            return Err(OxiH5Error::Format(format!(
                "read_chunked_slice: chunk dimension {d} is zero"
            )));
        }

        // Enumerate the chunk-grid cells that overlap the requested ranges.
        let first_ci: Vec<u64> = (0..ndims)
            .map(|d| ranges[d].start / real_chunk_dims[d])
            .collect();
        let last_ci: Vec<u64> = (0..ndims)
            .map(|d| (ranges[d].end - 1) / real_chunk_dims[d])
            .collect();
        let ci_counts: Vec<u64> = (0..ndims).map(|d| last_ci[d] - first_ci[d] + 1).collect();
        let total_cells: u64 = ci_counts.iter().product();
        let ci_strides = row_major_strides(&ci_counts);

        // Collect (record index, chunk offsets) for all intersecting cells that
        // have a real record present (sparse cells are skipped here and filled
        // from `fill_value` in phase 2 below). Carrying `rec_idx` through
        // directly — rather than re-deriving it from `origin` inside the
        // `map` closure below — avoids a second `chunk_map` lookup that could
        // only ever panic if this `filter_map` and that lookup disagreed, an
        // invariant that is easy to break under a future edit to either half.
        let present_cells: Vec<(usize, Vec<u64>)> = (0..total_cells as usize)
            .filter_map(|cell_flat| {
                let ci_rel = flat_to_coords(cell_flat, &ci_strides, ndims);
                let origin: Vec<u64> = (0..ndims)
                    .map(|d| (first_ci[d] + ci_rel[d]) * real_chunk_dims[d])
                    .collect();
                chunk_map
                    .get(&origin)
                    .map(|&rec_idx| (rec_idx, chunks_arc[rec_idx].offsets.clone()))
            })
            .collect();

        // Phase 1 (parallel): decompress only present (non-sparse) chunks.
        let decompressed: Vec<(Vec<u64>, Vec<u8>)> = present_cells
            .into_par_iter()
            .map(
                |(rec_idx, offsets)| -> Result<(Vec<u64>, Vec<u8>), OxiH5Error> {
                    let rec = &chunks_arc[rec_idx];
                    let raw = read_chunk_bytes(file_data, rec)?;
                    let data = apply_filters_to_chunk(
                        raw,
                        rec.filter_mask,
                        Some(pipeline),
                        elem_size,
                        Some(real_chunk_dims.iter().product::<u64>() as usize * elem_size),
                    )?;
                    Ok((offsets, data))
                },
            )
            .collect::<Result<Vec<_>, OxiH5Error>>()?;

        // Build a map from origin → decompressed data for scatter phase.
        let decomp_map: HashMap<Vec<u64>, Vec<u8>> = decompressed.into_iter().collect();

        // Phase 2 (sequential): iterate cells and scatter (sparse chunks → fill).
        for cell_flat in 0..total_cells as usize {
            let ci_rel = flat_to_coords(cell_flat, &ci_strides, ndims);
            let origin: Vec<u64> = (0..ndims)
                .map(|d| (first_ci[d] + ci_rel[d]) * real_chunk_dims[d])
                .collect();

            let chunk_data: std::borrow::Cow<[u8]> = if let Some(data) = decomp_map.get(&origin) {
                std::borrow::Cow::Borrowed(data.as_slice())
            } else {
                std::borrow::Cow::Owned(make_sparse_fill(chunk_volume as usize))
            };

            scatter_chunk_slice(
                &mut output,
                &origin,
                &chunk_data,
                &real_chunk_dims,
                ranges,
                elem_size,
            )?;
        }

        Ok(output)
    }

    #[cfg(not(feature = "parallel"))]
    assemble_chunks_slice(
        &chunks_arc,
        file_data,
        &real_chunk_dims,
        dataset_dims,
        ranges,
        SliceElemConfig {
            elem_size,
            fill_value,
        },
        |raw, mask| {
            if pipeline.filters.is_empty() {
                Ok(raw.to_vec())
            } else {
                filters::apply_pipeline_sized(
                    raw,
                    pipeline,
                    mask,
                    elem_size,
                    Some(real_chunk_dims.iter().product::<u64>() as usize * elem_size),
                )
            }
        },
    )
}

/// Assemble only the chunks that overlap with `ranges` into an output buffer
/// of shape `[r.len() for r in ranges]` (row-major).
///
/// For each chunk-grid cell that intersects the requested hyperslab, we read
/// and decompress the chunk, then scatter the overlapping elements into the
/// output buffer.  Absent (sparse) chunks are filled with `fill_value` or zero.
///
/// Used by the sequential (non-parallel) code path in [`read_chunked_slice`].
#[cfg(any(not(feature = "parallel"), test))]
pub(super) fn assemble_chunks_slice(
    chunks: &[ChunkRecord],
    file_data: &[u8],
    chunk_dims: &[u64],
    dataset_dims: &[u64],
    ranges: &[std::ops::Range<u64>],
    cfg: SliceElemConfig<'_>,
    apply_filters: impl Fn(&[u8], u32) -> Result<Vec<u8>, OxiH5Error>,
) -> Result<Vec<u8>, OxiH5Error> {
    let elem_size = cfg.elem_size;
    let fill_value = cfg.fill_value;
    let ndims = dataset_dims.len();

    // Derive output shape directly from ranges.
    let out_dims: Vec<u64> = ranges.iter().map(|r| r.end - r.start).collect();

    // Total output buffer size.
    let out_elems: u64 = out_dims.iter().product();
    let mut output = vec![0u8; out_elems as usize * elem_size];

    // Build a lookup map: chunk origin offsets → chunk record index.
    // Use a HashMap for O(1) lookup per chunk-grid cell.
    let mut chunk_map: std::collections::HashMap<Vec<u64>, usize> =
        std::collections::HashMap::with_capacity(chunks.len());
    for (i, cr) in chunks.iter().enumerate() {
        if cr.offsets.len() >= ndims {
            chunk_map.insert(cr.offsets[..ndims].to_vec(), i);
        }
    }

    // Row-major strides for the output buffer (in elements).
    let out_strides = row_major_strides(&out_dims);
    // Row-major strides for a single chunk (in elements).
    let chunk_strides = row_major_strides(chunk_dims);
    let chunk_volume: u64 = chunk_dims.iter().product();

    // Reject a zero chunk dimension before dividing by it below: a
    // crafted/corrupted layout message could claim a zero-sized chunk
    // dimension, which would otherwise panic with a divide-by-zero when
    // computing the overlapping chunk-grid cell range.
    if let Some(d) = chunk_dims.iter().position(|&c| c == 0) {
        return Err(OxiH5Error::Format(format!(
            "assemble_chunks_slice: chunk dimension {d} is zero"
        )));
    }

    // For dimension d, the range of chunk indices that overlap `ranges[d]` is:
    //   first_ci[d] = ranges[d].start / chunk_dims[d]
    //   last_ci[d]  = (ranges[d].end - 1) / chunk_dims[d]
    let first_ci: Vec<u64> = (0..ndims)
        .map(|d| ranges[d].start / chunk_dims[d])
        .collect();
    let last_ci: Vec<u64> = (0..ndims)
        .map(|d| (ranges[d].end - 1) / chunk_dims[d])
        .collect();

    // Count of chunk-grid cells per dimension.
    let ci_counts: Vec<u64> = (0..ndims).map(|d| last_ci[d] - first_ci[d] + 1).collect();
    let total_cells: u64 = ci_counts.iter().product();

    // Row-major strides for the chunk-grid cell index space.
    let ci_strides = row_major_strides(&ci_counts);

    // Iterate over every chunk-grid cell that overlaps the requested region.
    for cell_flat in 0..total_cells as usize {
        // Decode chunk-grid cell coordinates relative to `first_ci`.
        let ci_rel = flat_to_coords(cell_flat, &ci_strides, ndims);
        // Absolute chunk-grid cell coordinates.
        let ci: Vec<u64> = (0..ndims).map(|d| first_ci[d] + ci_rel[d]).collect();

        // Chunk origin in dataset element space.
        let origin: Vec<u64> = (0..ndims).map(|d| ci[d] * chunk_dims[d]).collect();

        // Overlap of this chunk with the requested ranges.
        let ovl_start: Vec<u64> = (0..ndims).map(|d| ranges[d].start.max(origin[d])).collect();
        let ovl_end: Vec<u64> = (0..ndims)
            .map(|d| ranges[d].end.min(origin[d] + chunk_dims[d]))
            .collect();

        // Skip degenerate overlaps (shouldn't happen given our ci range, but guard anyway).
        if (0..ndims).any(|d| ovl_start[d] >= ovl_end[d]) {
            continue;
        }

        // Find this chunk's record (if it exists; absent chunks stay zero).
        let maybe_record = chunk_map.get(&origin);

        // Decompress (or create fill-value buffer for sparse chunks).
        let chunk_data: Vec<u8> = if let Some(&rec_idx) = maybe_record {
            let cr = &chunks[rec_idx];
            let addr = cr.address as usize;
            let sz = cr.size as usize;
            let raw = file_data.get(addr..addr + sz).ok_or_else(|| {
                OxiH5Error::Format(format!(
                    "chunk at {:#x} size {} extends beyond file ({} bytes)",
                    addr,
                    sz,
                    file_data.len()
                ))
            })?;
            apply_filters(raw, cr.filter_mask)?
        } else {
            let n_elems = chunk_volume as usize;
            match fill_value {
                Some(fv) if fv.len() == elem_size && elem_size > 0 => fv
                    .iter()
                    .cycle()
                    .take(n_elems * elem_size)
                    .cloned()
                    .collect(),
                _ => vec![0u8; n_elems * elem_size],
            }
        };

        // Overlap shape (number of elements per dimension in the intersection).
        let ovl_shape: Vec<u64> = (0..ndims).map(|d| ovl_end[d] - ovl_start[d]).collect();
        let ovl_elems: u64 = ovl_shape.iter().product();

        // Strides for the overlap volume (row-major in overlap-local coords).
        let ovl_strides = row_major_strides(&ovl_shape);

        // Iterate over every element in the intersection rectangle.
        for ovl_flat in 0..ovl_elems as usize {
            // Decode overlap-local coords.
            let ovl_coords = flat_to_coords(ovl_flat, &ovl_strides, ndims);

            // Global element position.
            let global: Vec<u64> = (0..ndims).map(|d| ovl_start[d] + ovl_coords[d]).collect();

            // Position within the chunk (chunk-local coords).
            let in_chunk: Vec<u64> = (0..ndims).map(|d| global[d] - origin[d]).collect();

            // Flat index into the chunk data buffer.
            let chunk_flat: usize = in_chunk
                .iter()
                .zip(chunk_strides.iter())
                .map(|(&c, &s)| c as usize * s)
                .sum();

            // Position within the output buffer (output-local coords).
            let in_out: Vec<u64> = (0..ndims).map(|d| global[d] - ranges[d].start).collect();

            // Flat index into the output buffer.
            let out_flat: usize = in_out
                .iter()
                .zip(out_strides.iter())
                .map(|(&c, &s)| c as usize * s)
                .sum();

            let src_off = chunk_flat * elem_size;
            let dst_off = out_flat * elem_size;

            if src_off + elem_size <= chunk_data.len() && dst_off + elem_size <= output.len() {
                output[dst_off..dst_off + elem_size]
                    .copy_from_slice(&chunk_data[src_off..src_off + elem_size]);
            }
        }
    }

    Ok(output)
}
