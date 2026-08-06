use crate::chunked::{
    apply_filters_to_chunk, chunk_records, flat_to_coords, index_of, read_chunk_bytes,
    row_major_strides, ChunkIndexCache, ChunkIndexQuery, ChunkSliceParams, DatasetShape,
};
use crate::hyperslab::{scatter_chunk_hyperslab, Hyperslab};
use crate::message::LayoutInfo;
use oxih5_core::{FilterPipeline, OxiH5Error};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Read only the chunks that overlap the hyperslab selection from a chunked dataset.
///
/// Returns a flat byte buffer shaped `selection.output_shape()` in row-major
/// (C) order.  For strided or blocked selections only the elements that pass
/// `selection.dims[d].contains(global[d])` for every dimension `d` are written
/// into the output buffer; all other bytes are left at their initial value of
/// zero (or the fill value if one is provided via `params`).
///
/// # Arguments
///
/// * `file_data`     – the complete HDF5 file buffer
/// * `layout`        – parsed chunked layout message (must be `LayoutInfo::Chunked`)
/// * `pipeline`      – the dataset's filter pipeline (empty ⇒ no filters)
/// * `shape`         – the dataset's current and maximum dimensions
/// * `params`        – element size + optional fill-value bytes
/// * `selection`     – N-dimensional hyperslab selection
/// * `cache`         – optional pre-parsed chunk index cache; pass `None` to disable
pub fn read_chunked_hyperslab(
    file_data: &[u8],
    layout: &LayoutInfo,
    pipeline: &FilterPipeline,
    shape: DatasetShape<'_>,
    params: ChunkSliceParams<'_>,
    selection: &Hyperslab,
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
            "read_chunked_hyperslab: layout is not chunked".into(),
        ));
    };

    let ndims = dataset_dims.len();

    if selection.dims.len() != ndims {
        return Err(OxiH5Error::Format(format!(
            "read_chunked_hyperslab: selection has {} dims but dataset has {}",
            selection.dims.len(),
            ndims,
        )));
    }
    // A caller-constructed `DimSelection { stride: 0, .. }` would otherwise
    // divide-by-zero panic inside `scatter_chunk_hyperslab` below; reject it
    // up front with a typed error instead.
    selection.validate()?;

    // Short-circuit: empty selection → nothing to read.
    if selection.is_empty() {
        return Ok(vec![]);
    }

    let out_shape = selection.output_shape();
    let out_elems: u64 = out_shape.iter().product();
    let bbox = selection.bounding_ranges();

    // Validate that the bounding box is within the dataset.
    for d in 0..ndims {
        if bbox[d].end > dataset_dims[d] {
            return Err(OxiH5Error::Format(format!(
                "read_chunked_hyperslab: bounding box dim {} end {} exceeds dataset dim {} ({})",
                d, bbox[d].end, d, dataset_dims[d],
            )));
        }
    }

    // Strip trailing element-size "dimension" from chunk_dims (layout v3/v4 convention).
    let real_chunk_dims: Vec<u64> = if chunk_dims.len() == ndims + 1 {
        chunk_dims[..ndims].to_vec()
    } else if chunk_dims.len() == ndims {
        chunk_dims.clone()
    } else {
        return Err(OxiH5Error::Format(format!(
            "read_chunked_hyperslab: chunk_dims length {} incompatible with rank {} (dimensionality field = {})",
            chunk_dims.len(),
            ndims,
            dimensionality,
        )));
    };

    // Reject a zero chunk dimension before dividing by it below (see
    // `read_chunked_slice` / `assemble_chunks_slice` in chunked.rs for the
    // same guard on the sibling slice readers): a crafted/corrupted layout
    // message could claim a zero-sized chunk dimension, which would
    // otherwise panic with a divide-by-zero when computing the overlapping
    // chunk-grid cell range below.
    if let Some(d) = real_chunk_dims.iter().position(|&c| c == 0) {
        return Err(OxiH5Error::Format(format!(
            "read_chunked_hyperslab: chunk dimension {d} is zero"
        )));
    }

    let index = index_of(*index_type, "read_chunked_hyperslab")?;

    // Resolve all chunk records (with optional caching).
    let chunks_arc: Arc<Vec<crate::btree_v2::ChunkRecord>> = chunk_records(
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

    // Build a lookup map: chunk origin → record index.
    let mut chunk_map: HashMap<Vec<u64>, usize> = HashMap::with_capacity(chunks_arc.len());
    for (i, cr) in chunks_arc.iter().enumerate() {
        if cr.offsets.len() >= ndims {
            chunk_map.insert(cr.offsets[..ndims].to_vec(), i);
        }
    }

    let chunk_volume: u64 = real_chunk_dims.iter().product();

    // Closure that builds a fill buffer for sparse chunks.
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

    // Enumerate the chunk-grid cells that overlap the bounding box.
    let first_ci: Vec<u64> = (0..ndims)
        .map(|d| bbox[d].start / real_chunk_dims[d])
        .collect();
    let last_ci: Vec<u64> = (0..ndims)
        .map(|d| bbox[d].end.saturating_sub(1) / real_chunk_dims[d])
        .collect();
    let ci_counts: Vec<u64> = (0..ndims).map(|d| last_ci[d] - first_ci[d] + 1).collect();
    let total_cells: u64 = ci_counts.iter().product();
    let ci_strides = row_major_strides(&ci_counts);

    #[cfg(feature = "parallel")]
    {
        // Allocate output buffer (zero-initialised).
        let mut output = vec![0u8; out_elems as usize * elem_size];

        // Phase 1 (parallel): collect and decompress all present chunks.
        let present_cells: Vec<(Vec<u64>, usize)> = (0..total_cells as usize)
            .filter_map(|cell_flat| {
                let ci_rel = flat_to_coords(cell_flat, &ci_strides, ndims);
                let origin: Vec<u64> = (0..ndims)
                    .map(|d| (first_ci[d] + ci_rel[d]) * real_chunk_dims[d])
                    .collect();
                chunk_map.get(&origin).map(|&rec_idx| (origin, rec_idx))
            })
            .collect();

        let decompressed: Vec<(Vec<u64>, Vec<u8>)> = present_cells
            .into_par_iter()
            .map(
                |(origin, rec_idx)| -> Result<(Vec<u64>, Vec<u8>), OxiH5Error> {
                    let rec = &chunks_arc[rec_idx];
                    let raw = read_chunk_bytes(file_data, rec)?;
                    let data = apply_filters_to_chunk(
                        raw,
                        rec.filter_mask,
                        Some(pipeline),
                        elem_size,
                        Some(real_chunk_dims.iter().product::<u64>() as usize * elem_size),
                    )?;
                    Ok((origin, data))
                },
            )
            .collect::<Result<Vec<_>, OxiH5Error>>()?;

        let decomp_map: HashMap<Vec<u64>, Vec<u8>> = decompressed.into_iter().collect();

        // Phase 2 (sequential): scatter each chunk into output.
        for cell_flat in 0..total_cells as usize {
            let ci_rel = flat_to_coords(cell_flat, &ci_strides, ndims);
            let chunk_origin: Vec<u64> = (0..ndims)
                .map(|d| (first_ci[d] + ci_rel[d]) * real_chunk_dims[d])
                .collect();

            let chunk_data: std::borrow::Cow<[u8]> =
                if let Some(data) = decomp_map.get(&chunk_origin) {
                    std::borrow::Cow::Borrowed(data.as_slice())
                } else {
                    std::borrow::Cow::Owned(make_sparse_fill(chunk_volume as usize))
                };

            scatter_chunk_hyperslab(
                &mut output,
                &chunk_data,
                &chunk_origin,
                &real_chunk_dims,
                selection,
                &out_shape,
                elem_size,
            )?;
        }

        Ok(output)
    }

    // Sequential path (non-parallel builds only).
    #[cfg(not(feature = "parallel"))]
    {
        let mut output = vec![0u8; out_elems as usize * elem_size];

        for cell_flat in 0..total_cells as usize {
            let ci_rel = flat_to_coords(cell_flat, &ci_strides, ndims);
            let chunk_origin: Vec<u64> = (0..ndims)
                .map(|d| (first_ci[d] + ci_rel[d]) * real_chunk_dims[d])
                .collect();

            let chunk_data: Vec<u8> = if let Some(&rec_idx) = chunk_map.get(&chunk_origin) {
                let rec = &chunks_arc[rec_idx];
                let raw = read_chunk_bytes(file_data, rec)?;
                if pipeline.filters.is_empty() {
                    raw.to_vec()
                } else {
                    apply_filters_to_chunk(
                        raw,
                        rec.filter_mask,
                        Some(pipeline),
                        elem_size,
                        Some(real_chunk_dims.iter().product::<u64>() as usize * elem_size),
                    )?
                }
            } else {
                make_sparse_fill(chunk_volume as usize)
            };

            scatter_chunk_hyperslab(
                &mut output,
                &chunk_data,
                &chunk_origin,
                &real_chunk_dims,
                selection,
                &out_shape,
                elem_size,
            )?;
        }

        Ok(output)
    }
}

/// Gather elements from a fully-loaded contiguous (or compact) dataset buffer
/// according to an N-dimensional hyperslab selection.
///
/// `full_data` must have exactly `dataset_dims.iter().product() * elem_size` bytes.
/// Returns a flat byte buffer shaped `selection.output_shape()` in row-major order.
pub fn gather_hyperslab_contiguous(
    full_data: &[u8],
    dataset_dims: &[u64],
    selection: &Hyperslab,
    elem_size: usize,
) -> Result<Vec<u8>, OxiH5Error> {
    let ndims = dataset_dims.len();

    if selection.dims.len() != ndims {
        return Err(OxiH5Error::Format(format!(
            "gather_hyperslab_contiguous: selection has {} dims but dataset has {}",
            selection.dims.len(),
            ndims,
        )));
    }
    // Keep this facade entry point consistent with `read_chunked_hyperslab`:
    // a zero stride has no valid meaning even though this function's own
    // arithmetic only divides by `block` (already guarded by `is_empty`
    // below), not `stride`.
    selection.validate()?;

    if selection.is_empty() {
        return Ok(vec![]);
    }

    let out_shape = selection.output_shape();
    let out_elems: u64 = out_shape.iter().product();

    let expected_bytes = dataset_dims.iter().product::<u64>() as usize * elem_size;
    if full_data.len() != expected_bytes {
        return Err(OxiH5Error::Format(format!(
            "gather_hyperslab_contiguous: full_data has {} bytes, expected {} (dims={:?}, elem_size={})",
            full_data.len(),
            expected_bytes,
            dataset_dims,
            elem_size,
        )));
    }

    let dataset_strides = row_major_strides(dataset_dims);
    let out_strides = row_major_strides(&out_shape);

    let mut output = vec![0u8; out_elems as usize * elem_size];

    for out_flat in 0..out_elems as usize {
        let out_coords = flat_to_coords(out_flat, &out_strides, ndims);

        // Map output coordinate to global dataset coordinate.
        let mut src_flat = 0usize;
        for d in 0..ndims {
            let sel = &selection.dims[d];
            let block_idx = out_coords[d] / sel.block;
            let in_block = out_coords[d] % sel.block;
            let global_d = sel.start + block_idx * sel.stride + in_block;
            src_flat += global_d as usize * dataset_strides[d];
        }

        let src_off = src_flat * elem_size;
        let dst_off = out_flat * elem_size;

        if src_off + elem_size <= full_data.len() && dst_off + elem_size <= output.len() {
            output[dst_off..dst_off + elem_size]
                .copy_from_slice(&full_data[src_off..src_off + elem_size]);
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperslab::DimSelection;

    // -----------------------------------------------------------------------
    // read_chunked_hyperslab: zero chunk dimension must error, not panic
    // -----------------------------------------------------------------------

    /// A crafted layout v3/v4 chunked message can claim a zero-sized chunk
    /// dimension (`message.rs::parse_layout` rejects this too, but
    /// `read_chunked_hyperslab` must not rely solely on that upstream check —
    /// see `test_chunked_slice_zero_chunk_dim_errors` in chunked.rs for the
    /// sibling slice readers). Before the fix, `first_ci`/`last_ci` divided
    /// `bbox[d].start` / `bbox[d].end` directly by `real_chunk_dims[d]`,
    /// panicking with a divide-by-zero the moment `File::dataset_hyperslab`
    /// was called on such a file.
    #[test]
    fn test_read_chunked_hyperslab_zero_chunk_dim_errors() {
        let file = vec![0u8; 64];
        let layout = LayoutInfo::Chunked {
            data_address: 0,
            dimensionality: 2,
            // chunk_dims[0] = 0 (zero geometric chunk dim), chunk_dims[1] = 4
            // (the implicit trailing element-size slot for a 1-D dataset).
            chunk_dims: vec![0, 4],
            index_type: 0,
            single_chunk: None,
        };
        let pipeline = FilterPipeline { filters: vec![] };
        let dataset_dims = [8u64];
        let r: std::ops::Range<u64> = 1..3;
        let selection = Hyperslab::contiguous(&[r]);

        let result = read_chunked_hyperslab(
            &file,
            &layout,
            &pipeline,
            DatasetShape::fixed(&dataset_dims),
            ChunkSliceParams {
                elem_size: 1,
                fill_value: None,
            },
            &selection,
            None,
        );

        assert!(
            result.is_err(),
            "zero chunk dimension must return an error, not panic"
        );
    }

    /// A caller-constructed `DimSelection { stride: 0, .. }` passed into the
    /// public `read_chunked_hyperslab` facade (the function backing
    /// `File::dataset_hyperslab`) must return a typed error instead of
    /// panicking with a divide-by-zero inside `scatter_chunk_hyperslab`.
    #[test]
    fn test_read_chunked_hyperslab_zero_stride_errors_not_panics() {
        let file = vec![0u8; 64];
        let layout = LayoutInfo::Chunked {
            data_address: 0,
            dimensionality: 2,
            chunk_dims: vec![4, 4], // valid, non-zero chunk geometry
            index_type: 0,
            single_chunk: None,
        };
        let pipeline = FilterPipeline { filters: vec![] };
        let dataset_dims = [8u64];
        let selection = Hyperslab {
            dims: vec![DimSelection {
                start: 0,
                stride: 0,
                count: 1,
                block: 1,
            }],
        };

        let result = read_chunked_hyperslab(
            &file,
            &layout,
            &pipeline,
            DatasetShape::fixed(&dataset_dims),
            ChunkSliceParams {
                elem_size: 1,
                fill_value: None,
            },
            &selection,
            None,
        );

        assert!(
            result.is_err(),
            "stride=0 must return an error, not panic with divide-by-zero"
        );
    }

    // -----------------------------------------------------------------------
    // gather_hyperslab_contiguous unit tests
    // -----------------------------------------------------------------------

    /// Contiguous selection (stride=1, block=1) should return exactly those elements.
    #[test]
    fn test_gather_contiguous_1d() {
        // Dataset: [0u8, 1, 2, 3, 4, 5, 6, 7], elem_size=1
        let data: Vec<u8> = (0u8..8).collect();
        let dataset_dims = [8u64];
        let r: std::ops::Range<u64> = 2..6;
        let selection = Hyperslab::contiguous(&[r]);

        let out = gather_hyperslab_contiguous(&data, &dataset_dims, &selection, 1)
            .expect("gather_contiguous_1d failed");

        assert_eq!(out, vec![2u8, 3, 4, 5]);
    }

    /// Strided 1-D selection: indices {1, 3, 5} from [10, 11, 12, 13, 14, 15].
    #[test]
    fn test_gather_strided_1d() {
        let data: Vec<u8> = (10u8..16).collect(); // [10,11,12,13,14,15]
        let dataset_dims = [6u64];
        let selection = Hyperslab {
            dims: vec![DimSelection {
                start: 1,
                stride: 2,
                count: 3,
                block: 1,
            }],
        };

        let out = gather_hyperslab_contiguous(&data, &dataset_dims, &selection, 1)
            .expect("gather_strided_1d failed");

        assert_eq!(out, vec![11u8, 13, 15]);
    }

    /// Block=2 selection: {0,1,4,5} from 6-element dataset.
    #[test]
    fn test_gather_block2_1d() {
        let data: Vec<u8> = (0u8..8).collect();
        let dataset_dims = [8u64];
        let selection = Hyperslab {
            dims: vec![DimSelection {
                start: 0,
                stride: 4,
                count: 2,
                block: 2,
            }],
        };

        let out = gather_hyperslab_contiguous(&data, &dataset_dims, &selection, 1)
            .expect("gather_block2_1d failed");

        assert_eq!(out, vec![0u8, 1, 4, 5]);
    }

    /// 2-D contiguous selection.
    #[test]
    fn test_gather_contiguous_2d() {
        // 4x4 dataset of u8 with values flat_index.
        let data: Vec<u8> = (0u8..16).collect();
        let dataset_dims = [4u64, 4];
        // Select rows 1..3, cols 1..3 → elements (1,1),(1,2),(2,1),(2,2) = 5,6,9,10
        let selection = Hyperslab::contiguous(&[1..3, 1..3]);

        let out = gather_hyperslab_contiguous(&data, &dataset_dims, &selection, 1)
            .expect("gather_2d failed");

        assert_eq!(out, vec![5u8, 6, 9, 10]);
    }

    /// Empty selection returns empty buffer.
    #[test]
    fn test_gather_empty_selection() {
        let data: Vec<u8> = (0u8..8).collect();
        let dataset_dims = [8u64];
        let selection = Hyperslab {
            dims: vec![DimSelection {
                start: 0,
                stride: 1,
                count: 0,
                block: 1,
            }],
        };

        let out = gather_hyperslab_contiguous(&data, &dataset_dims, &selection, 1)
            .expect("gather_empty failed");

        assert!(out.is_empty());
    }

    /// Wrong data length returns Err.
    #[test]
    fn test_gather_wrong_data_length() {
        let data = vec![0u8; 4]; // only 4 bytes but 8 expected
        let dataset_dims = [8u64];
        let r: std::ops::Range<u64> = 0..4;
        let selection = Hyperslab::contiguous(&[r]);

        let result = gather_hyperslab_contiguous(&data, &dataset_dims, &selection, 1);

        assert!(result.is_err(), "expected error for wrong data length");
    }
}
