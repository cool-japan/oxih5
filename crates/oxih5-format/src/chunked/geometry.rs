//! Shared low-level helpers used by every reader above: raw chunk I/O, filter-pipeline application, and row-major coordinate math.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::btree_v2::ChunkRecord;
use crate::filters;
use oxih5_core::{FilterPipeline, OxiH5Error};

/// Read the raw (compressed) bytes for a single chunk from the file buffer.
///
/// Returns a slice into `file_data` (zero-copy).  The caller is responsible
/// for applying the filter pipeline to obtain the decompressed element data.
pub(crate) fn read_chunk_bytes<'a>(
    file_data: &'a [u8],
    rec: &ChunkRecord,
) -> Result<&'a [u8], OxiH5Error> {
    let addr = rec.address as usize;
    let size = rec.size as usize;
    file_data.get(addr..addr + size).ok_or_else(|| {
        OxiH5Error::Format(format!(
            "chunk at {:#x} size {} extends beyond file ({} bytes)",
            addr,
            size,
            file_data.len(),
        ))
    })
}

/// Apply the filter pipeline (or identity) to raw chunk bytes.
///
/// When `pipeline` is `None` or has no filters, returns a copy of `raw`.
/// When filters are present, delegates to [`filters::apply_pipeline`].
///
/// Used by both the serial and parallel chunked-read paths.
pub(crate) fn apply_filters_to_chunk(
    raw: &[u8],
    filter_mask: u32,
    pipeline: Option<&FilterPipeline>,
    elem_size: usize,
    expected_out_len: Option<usize>,
) -> Result<Vec<u8>, OxiH5Error> {
    match pipeline {
        Some(p) if !p.filters.is_empty() => {
            filters::apply_pipeline_sized(raw, p, filter_mask, elem_size, expected_out_len)
        }
        _ => Ok(raw.to_vec()),
    }
}

/// Scatter elements from a decoded chunk into the full-dataset output buffer.
///
/// Uses the row-major strides of `dataset_dims` and `chunk_dims` to map each
/// chunk-local flat index to its dataset-absolute byte offset.  Out-of-bounds
/// (padding) elements are silently skipped.
///
/// Used by the `parallel` feature code path and unit tests.
#[cfg(any(feature = "parallel", test))]
pub(crate) fn scatter_chunk(
    output: &mut [u8],
    origin: &[u64],
    chunk_data: &[u8],
    chunk_dims: &[u64],
    dataset_dims: &[u64],
    elem_size: usize,
) -> Result<(), OxiH5Error> {
    let ndims = dataset_dims.len();
    let dataset_strides = row_major_strides(dataset_dims);
    let chunk_strides = row_major_strides(chunk_dims);
    let chunk_volume: u64 = chunk_dims.iter().product();
    let n_chunk_elems = (chunk_data.len() / elem_size).min(chunk_volume as usize);

    for flat_chunk_idx in 0..n_chunk_elems {
        let chunk_coords = flat_to_coords(flat_chunk_idx, &chunk_strides, ndims);

        let mut dataset_flat = 0usize;
        let mut in_bounds = true;

        for d in 0..ndims {
            let dataset_coord = origin[d] + chunk_coords[d];
            if dataset_coord >= dataset_dims[d] {
                in_bounds = false;
                break;
            }
            dataset_flat += dataset_coord as usize * dataset_strides[d];
        }

        if !in_bounds {
            continue;
        }

        let src_off = flat_chunk_idx * elem_size;
        let dst_off = dataset_flat * elem_size;

        if src_off + elem_size <= chunk_data.len() && dst_off + elem_size <= output.len() {
            output[dst_off..dst_off + elem_size]
                .copy_from_slice(&chunk_data[src_off..src_off + elem_size]);
        }
    }

    Ok(())
}

/// Scatter elements from a decoded chunk into a *slice* output buffer.
///
/// `ranges` defines the hyperslab — each element `ranges[d]` is `start..end`
/// in dataset coordinates.  The output buffer has shape `[r.len() for r in ranges]`
/// in row-major order.  Only elements within both the chunk and the hyperslab
/// are copied; all others are silently skipped.
///
/// Used by the `parallel` feature code path and unit tests.
#[cfg(any(feature = "parallel", test))]
pub(crate) fn scatter_chunk_slice(
    output: &mut [u8],
    origin: &[u64],
    chunk_data: &[u8],
    chunk_dims: &[u64],
    ranges: &[std::ops::Range<u64>],
    elem_size: usize,
) -> Result<(), OxiH5Error> {
    let ndims = ranges.len();
    let out_dims: Vec<u64> = ranges.iter().map(|r| r.end - r.start).collect();
    let out_strides = row_major_strides(&out_dims);
    let chunk_strides = row_major_strides(chunk_dims);

    // Overlap between chunk and requested hyperslab.
    let ovl_start: Vec<u64> = (0..ndims).map(|d| ranges[d].start.max(origin[d])).collect();
    let ovl_end: Vec<u64> = (0..ndims)
        .map(|d| ranges[d].end.min(origin[d] + chunk_dims[d]))
        .collect();

    if (0..ndims).any(|d| ovl_start[d] >= ovl_end[d]) {
        return Ok(());
    }

    let ovl_shape: Vec<u64> = (0..ndims).map(|d| ovl_end[d] - ovl_start[d]).collect();
    let ovl_elems: u64 = ovl_shape.iter().product();
    let ovl_strides = row_major_strides(&ovl_shape);

    for ovl_flat in 0..ovl_elems as usize {
        let ovl_coords = flat_to_coords(ovl_flat, &ovl_strides, ndims);

        let global: Vec<u64> = (0..ndims).map(|d| ovl_start[d] + ovl_coords[d]).collect();

        let in_chunk: Vec<u64> = (0..ndims).map(|d| global[d] - origin[d]).collect();
        let chunk_flat: usize = in_chunk
            .iter()
            .zip(chunk_strides.iter())
            .map(|(&c, &s)| c as usize * s)
            .sum();

        let in_out: Vec<u64> = (0..ndims).map(|d| global[d] - ranges[d].start).collect();
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

    Ok(())
}

/// Compute row-major strides for an N-dimensional shape.
///
/// stride\[d\] = product of dims\[d+1..N\], so that
/// `flat_index = sum_d(coord[d] * stride[d])`.
pub(crate) fn row_major_strides(dims: &[u64]) -> Vec<usize> {
    let n = dims.len();
    let mut strides = vec![1usize; n];
    if n == 0 {
        return strides;
    }
    for d in (0..n - 1).rev() {
        strides[d] = strides[d + 1] * dims[d + 1] as usize;
    }
    strides
}

/// Convert a flat (row-major) index back to per-dimension coordinates.
///
/// Uses the pre-computed `strides` vector (same convention as `row_major_strides`).
pub(crate) fn flat_to_coords(mut flat: usize, strides: &[usize], ndims: usize) -> Vec<u64> {
    let mut coords = vec![0u64; ndims];
    for d in 0..ndims {
        if let Some(q) = flat.checked_div(strides[d]) {
            coords[d] = q as u64;
            flat %= strides[d];
        }
    }
    coords
}
