//! Whole-dataset chunk reading: resolve the index, read and filter each chunk, and scatter the decoded chunks into a row-major output buffer sized for the full dataset.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::btree_v2::ChunkRecord;
use crate::message::LayoutInfo;
// Only reachable through `assemble_chunks_with_fill`'s call from the
// non-parallel arm of `read_chunked` below — gate the import to match, or it
// is (correctly) reported unused whenever the `parallel` feature is enabled.
#[cfg(not(feature = "parallel"))]
use crate::filters;
use oxih5_core::{FilterPipeline, OxiH5Error};
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::cache::ChunkIndexCache;
use super::geometry::{flat_to_coords, read_chunk_bytes, row_major_strides};
// `apply_filters_to_chunk` and `scatter_chunk` are only called from the
// `#[cfg(feature = "parallel")]` block below, so gate the imports to match —
// `scatter_chunk` is defined under `any(feature = "parallel", test)` (wider
// than this file needs it), but this file's own use-site only ever needs it
// under `parallel`; matching that narrower condition (rather than the
// definition's own gate) keeps the import from going unused under a
// `test`-but-not-`parallel` build.
#[cfg(feature = "parallel")]
use super::geometry::{apply_filters_to_chunk, scatter_chunk};
use super::index::{chunk_records, index_of, ChunkIndexQuery, DatasetShape};

/// Assemble chunks into a contiguous buffer of element data.
///
/// # Arguments
///
/// * `chunks`         – list of chunk records (address, size, filter_mask, offsets)
/// * `file_data`      – the full file buffer
/// * `chunk_dims`     – the size of each chunk in elements per dimension
/// * `dataset_dims`   – the full dataset dimensions in elements
/// * `elem_size`      – bytes per element
/// * `apply_filters`  – function to apply any enabled filters to raw chunk bytes,
///   receiving `(raw_bytes, filter_mask)` and returning
///   the decompressed/unfiltered element data
///
/// Returns a single contiguous byte buffer in row-major (C) order.
pub fn assemble_chunks(
    chunks: &[ChunkRecord],
    file_data: &[u8],
    chunk_dims: &[u64],
    dataset_dims: &[u64],
    elem_size: usize,
    apply_filters: impl Fn(&[u8], u32) -> Result<Vec<u8>, OxiH5Error>,
) -> Result<Vec<u8>, OxiH5Error> {
    let ndims = dataset_dims.len();
    if chunk_dims.len() != ndims {
        return Err(OxiH5Error::Format(format!(
            "assemble_chunks: chunk_dims ({}) and dataset_dims ({}) length mismatch",
            chunk_dims.len(),
            ndims,
        )));
    }
    if elem_size == 0 {
        return Err(OxiH5Error::Format(
            "assemble_chunks: elem_size must be > 0".into(),
        ));
    }

    let total_elems: u64 = dataset_dims.iter().product();
    let mut output = vec![0u8; total_elems as usize * elem_size];

    // Pre-compute row-major strides for the dataset (in elements).
    let dataset_strides = row_major_strides(dataset_dims);
    // Pre-compute row-major strides for the chunk (in elements).
    let chunk_strides = row_major_strides(chunk_dims);
    let chunk_volume: u64 = chunk_dims.iter().product();

    for chunk in chunks {
        // Validate that the chunk's offset vector has the right length.
        if chunk.offsets.len() < ndims {
            return Err(OxiH5Error::Format(format!(
                "chunk at {:#x}: offsets length {} < ndims {}",
                chunk.address,
                chunk.offsets.len(),
                ndims,
            )));
        }

        // Read and filter the raw chunk data.
        let raw = read_chunk_bytes(file_data, chunk)?;
        let chunk_data = apply_filters(raw, chunk.filter_mask)?;

        // Scatter each element in the chunk into the output buffer.
        let n_chunk_elems = (chunk_data.len() / elem_size).min(chunk_volume as usize);

        for flat_chunk_idx in 0..n_chunk_elems {
            // Decompose flat_chunk_idx into per-dimension chunk-local coords (row-major).
            let chunk_coords = flat_to_coords(flat_chunk_idx, &chunk_strides, ndims);

            // Compute the dataset-absolute coordinate for this element.
            let mut dataset_flat = 0usize;
            let mut in_bounds = true;

            for d in 0..ndims {
                let dataset_coord = chunk.offsets[d] + chunk_coords[d];
                if dataset_coord >= dataset_dims[d] {
                    // Padding element outside the dataset boundary.
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

            // Both bounds are guaranteed by construction above and by
            // n_chunk_elems ≤ chunk_data.len()/elem_size, but guard anyway.
            if src_off + elem_size <= chunk_data.len() && dst_off + elem_size <= output.len() {
                output[dst_off..dst_off + elem_size]
                    .copy_from_slice(&chunk_data[src_off..src_off + elem_size]);
            }
        }
    }

    Ok(output)
}

/// Read a complete chunked dataset into a single contiguous element buffer.
///
/// This is the high-level entry point used by the facade: it resolves the chunk
/// index, reads each chunk's raw bytes, applies the inverse filter pipeline, and
/// scatters the decoded chunks into a row-major output buffer sized for
/// `dataset_dims`.
///
/// * `layout`        – the parsed chunked layout message
/// * `pipeline`      – the dataset's filter pipeline (empty ⇒ no filters)
/// * `shape`         – the dataset's current and maximum dimensions
/// * `elem_size`     – element size in bytes
/// * `fill_value`    – optional per-element fill bytes (length must equal `elem_size`);
///   when `Some` the output buffer and any sparse chunks are initialised with
///   the tiled fill pattern rather than zeros.
/// * `cache`         – optional pre-parsed chunk index cache; when `Some` the
///   chunk records for this index address are computed at most once across
///   repeated calls.  Pass `None` to disable caching.
pub fn read_chunked(
    file_data: &[u8],
    layout: &LayoutInfo,
    pipeline: &FilterPipeline,
    shape: DatasetShape<'_>,
    elem_size: usize,
    fill_value: Option<&[u8]>,
    cache: Option<&ChunkIndexCache>,
) -> Result<Vec<u8>, OxiH5Error> {
    let DatasetShape {
        dims: dataset_dims,
        max_dims,
    } = shape;
    let LayoutInfo::Chunked {
        data_address,
        dimensionality,
        chunk_dims,
        index_type,
        single_chunk,
    } = layout
    else {
        return Err(OxiH5Error::Format(
            "read_chunked: layout is not chunked".into(),
        ));
    };

    let ndims = dataset_dims.len();

    // In layout v3/v4 the chunk-dims array carries an extra trailing element
    // ("element size") so its length is rank + 1.  Strip it to recover the
    // per-dimension chunk shape in *elements*.
    let real_chunk_dims: Vec<u64> = if chunk_dims.len() == ndims + 1 {
        chunk_dims[..ndims].to_vec()
    } else if chunk_dims.len() == ndims {
        chunk_dims.clone()
    } else {
        return Err(OxiH5Error::Format(format!(
            "read_chunked: chunk_dims length {} incompatible with rank {} (dimensionality field = {})",
            chunk_dims.len(),
            ndims,
            dimensionality,
        )));
    };

    let index = index_of(*index_type, "read_chunked")?;

    // Compute (or retrieve from cache) the chunk records.
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
        // Build the output buffer, tiled with the fill value if provided.
        let make_fill_buf = |n_elems: usize| -> Vec<u8> {
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

        let total_elems: u64 = dataset_dims.iter().product();
        let mut output = make_fill_buf(total_elems as usize);

        // Phase 1 (parallel): read + decompress each chunk concurrently.
        let decompressed: Vec<(Vec<u64>, Vec<u8>)> = chunks_arc
            .par_iter()
            .map(|rec| -> Result<(Vec<u64>, Vec<u8>), OxiH5Error> {
                let raw = read_chunk_bytes(file_data, rec)?;
                let data = apply_filters_to_chunk(
                    raw,
                    rec.filter_mask,
                    Some(pipeline),
                    elem_size,
                    Some(real_chunk_dims.iter().product::<u64>() as usize * elem_size),
                )?;
                Ok((rec.offsets.clone(), data))
            })
            .collect::<Result<Vec<_>, OxiH5Error>>()?;

        // Phase 2 (sequential): scatter each decompressed chunk into output.
        for (origin, chunk_data) in decompressed {
            scatter_chunk(
                &mut output,
                &origin,
                &chunk_data,
                &real_chunk_dims,
                dataset_dims,
                elem_size,
            )?;
        }

        Ok(output)
    }

    #[cfg(not(feature = "parallel"))]
    assemble_chunks_with_fill(
        &chunks_arc,
        file_data,
        &real_chunk_dims,
        dataset_dims,
        elem_size,
        fill_value,
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

/// Assemble a complete chunked dataset with optional fill value support.
///
/// Like [`assemble_chunks`] but initialises the output buffer and sparse chunks
/// with `fill_value` bytes instead of zeros.
///
/// Used by the sequential (non-parallel) code path in [`read_chunked`].
#[cfg(any(not(feature = "parallel"), test))]
pub(super) fn assemble_chunks_with_fill(
    chunks: &[ChunkRecord],
    file_data: &[u8],
    chunk_dims: &[u64],
    dataset_dims: &[u64],
    elem_size: usize,
    fill_value: Option<&[u8]>,
    apply_filters: impl Fn(&[u8], u32) -> Result<Vec<u8>, OxiH5Error>,
) -> Result<Vec<u8>, OxiH5Error> {
    let ndims = dataset_dims.len();
    if chunk_dims.len() != ndims {
        return Err(OxiH5Error::Format(format!(
            "assemble_chunks_with_fill: chunk_dims ({}) and dataset_dims ({}) length mismatch",
            chunk_dims.len(),
            ndims,
        )));
    }
    if elem_size == 0 {
        return Err(OxiH5Error::Format(
            "assemble_chunks_with_fill: elem_size must be > 0".into(),
        ));
    }

    let total_elems: u64 = dataset_dims.iter().product();
    let n_total = total_elems as usize;
    let mut output = match fill_value {
        Some(fv) if fv.len() == elem_size => fv
            .iter()
            .cycle()
            .take(n_total * elem_size)
            .cloned()
            .collect(),
        _ => vec![0u8; n_total * elem_size],
    };

    let dataset_strides = row_major_strides(dataset_dims);
    let chunk_strides = row_major_strides(chunk_dims);
    let chunk_volume: u64 = chunk_dims.iter().product();

    for chunk in chunks {
        if chunk.offsets.len() < ndims {
            return Err(OxiH5Error::Format(format!(
                "chunk at {:#x}: offsets length {} < ndims {}",
                chunk.address,
                chunk.offsets.len(),
                ndims,
            )));
        }
        let raw = read_chunk_bytes(file_data, chunk)?;
        let chunk_data = apply_filters(raw, chunk.filter_mask)?;
        let n_chunk_elems = (chunk_data.len() / elem_size).min(chunk_volume as usize);

        for flat_chunk_idx in 0..n_chunk_elems {
            let chunk_coords = flat_to_coords(flat_chunk_idx, &chunk_strides, ndims);
            let mut dataset_flat = 0usize;
            let mut in_bounds = true;

            for d in 0..ndims {
                let dataset_coord = chunk.offsets[d] + chunk_coords[d];
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
    }

    Ok(output)
}
