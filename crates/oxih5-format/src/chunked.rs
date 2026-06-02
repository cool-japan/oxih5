use crate::btree_v2::{BTreeV2, ChunkRecord};
use crate::message::LayoutInfo;
use crate::{btree_v1_chunk, ea_index, fa_index, filters};
/// Chunk assembly: scatter-to-contiguous buffer reconstruction.
///
/// Given a list of chunk records (each with a file address, compressed size,
/// filter mask, and N-dimensional offset), this module reads and assembles
/// them into a single contiguous buffer that matches the full dataset shape.
use oxih5_core::{FilterPipeline, OxiH5Error};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// ChunkIndexCache
// ---------------------------------------------------------------------------

/// The inner storage type for [`ChunkIndexCache`].
type CacheMap = Arc<RwLock<HashMap<(u64, usize), Arc<Vec<ChunkRecord>>>>>;

/// Thread-safe cache from chunk-index address → resolved chunk records.
///
/// Keyed by `(index_address, num_dims)` to handle multi-dimensional datasets
/// unambiguously.  The same index address with different ranks would (in
/// principle) produce different record sets, so the rank is part of the key.
///
/// The cache is `Clone` (cheap: it clones the `Arc`, so all clones share the
/// same underlying storage) to allow `Group` to hold a reference to the same
/// cache as the parent `File`.
#[derive(Debug, Default, Clone)]
pub struct ChunkIndexCache {
    inner: CacheMap,
}

impl ChunkIndexCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the cached records for `key`, or compute + cache them on first access.
    ///
    /// `compute` is only called when the key is absent.  On success the result
    /// is stored and an `Arc` clone is returned; on failure the error propagates
    /// and nothing is stored.
    pub fn get_or_insert(
        &self,
        key: (u64, usize),
        compute: impl FnOnce() -> Result<Vec<ChunkRecord>, OxiH5Error>,
    ) -> Result<Arc<Vec<ChunkRecord>>, OxiH5Error> {
        // Fast path: read lock.
        {
            let guard = self
                .inner
                .read()
                .map_err(|_| OxiH5Error::Format("chunk cache read-lock poisoned".into()))?;
            if let Some(v) = guard.get(&key) {
                return Ok(Arc::clone(v));
            }
        }
        // Slow path: compute, then acquire write lock and store.
        let records = compute()?;
        let arc = Arc::new(records);
        let mut guard = self
            .inner
            .write()
            .map_err(|_| OxiH5Error::Format("chunk cache write-lock poisoned".into()))?;
        // Use `entry` to avoid overwriting a value inserted by another thread
        // between the read-unlock and the write-lock.
        let stored = guard.entry(key).or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(stored))
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

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
}

/// Resolve a chunk index into its chunk records.
///
/// `index_address` points at the index root; `ndims` is the *real* dataset
/// rank (not the layout's `dimensionality`, which is rank + 1).
pub fn resolve_chunk_index(
    file_data: &[u8],
    index: ChunkIndex,
    index_address: u64,
    ndims: usize,
) -> Result<Vec<ChunkRecord>, OxiH5Error> {
    match index {
        ChunkIndex::BTreeV1 => btree_v1_chunk::parse(file_data, index_address, ndims),
        ChunkIndex::BTreeV2 => Ok(BTreeV2::parse(file_data, index_address, ndims)?
            .records()
            .to_vec()),
        ChunkIndex::FixedArray => fa_index::parse_fixed_array(file_data, index_address, ndims),
        ChunkIndex::ExtensibleArray => {
            ea_index::parse_extensible_array(file_data, index_address, ndims)
        }
    }
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
/// * `dataset_dims`  – full dataset dimensions in elements
/// * `elem_size`     – element size in bytes
/// * `cache`         – optional pre-parsed chunk index cache; when `Some` the
///   chunk records for this index address are computed at most once across
///   repeated calls.  Pass `None` to disable caching.
pub fn read_chunked(
    file_data: &[u8],
    layout: &LayoutInfo,
    pipeline: &FilterPipeline,
    dataset_dims: &[u64],
    elem_size: usize,
    cache: Option<&ChunkIndexCache>,
) -> Result<Vec<u8>, OxiH5Error> {
    let LayoutInfo::Chunked {
        data_address,
        dimensionality,
        chunk_dims,
        index_type,
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

    // Layout v3 (the only chunked layout `parse_layout` currently emits) always
    // uses a version-1 B-tree, so `index_type` is hardcoded to 0 upstream and
    // only the `0` arm is reachable today.
    //
    // NOTE: these `index_type` values are *this crate's* internal convention,
    // NOT the HDF5 layout-v4 "indexing type" field (whose values differ:
    // 1=single-chunk, 2=implicit, 3=fixed-array, 4=extensible-array,
    // 5=B-tree-v2). When layout v4 parsing is added, translate the v4 field
    // into `ChunkIndex` directly rather than reusing these numbers.
    let index = match index_type {
        0 => ChunkIndex::BTreeV1,
        1 => ChunkIndex::FixedArray,
        2 => ChunkIndex::ExtensibleArray,
        3 => ChunkIndex::BTreeV2,
        other => {
            return Err(OxiH5Error::Format(format!(
                "read_chunked: unknown chunk index type {other}"
            )))
        }
    };

    // Compute (or retrieve from cache) the chunk records.
    //
    // Both the FixedArray special path and the generic `resolve_chunk_index`
    // path are wrapped inside a single closure so the cache key
    // `(data_address, ndims)` is sufficient regardless of index type.
    let chunks_arc: Arc<Vec<ChunkRecord>> = if let Some(c) = cache {
        let uncompressed_for_fa = real_chunk_dims.iter().product::<u64>() as usize * elem_size;
        let real_chunk_dims_clone = real_chunk_dims.clone();
        c.get_or_insert((*data_address, ndims), move || {
            if index == ChunkIndex::FixedArray {
                fa_index::parse_fixed_array_v4(
                    file_data,
                    *data_address,
                    ndims,
                    &real_chunk_dims_clone,
                    uncompressed_for_fa,
                )
            } else {
                resolve_chunk_index(file_data, index, *data_address, ndims)
            }
        })?
    } else {
        // No cache: compute directly.
        let records = if index == ChunkIndex::FixedArray {
            let uncompressed = real_chunk_dims.iter().product::<u64>() as usize * elem_size;
            fa_index::parse_fixed_array_v4(
                file_data,
                *data_address,
                ndims,
                &real_chunk_dims,
                uncompressed,
            )?
        } else {
            resolve_chunk_index(file_data, index, *data_address, ndims)?
        };
        Arc::new(records)
    };

    #[cfg(feature = "parallel")]
    {
        let total_elems: u64 = dataset_dims.iter().product();
        let mut output = vec![0u8; total_elems as usize * elem_size];

        // Phase 1 (parallel): read + decompress each chunk concurrently.
        let decompressed: Vec<(Vec<u64>, Vec<u8>)> = chunks_arc
            .par_iter()
            .map(|rec| -> Result<(Vec<u64>, Vec<u8>), OxiH5Error> {
                let raw = read_chunk_bytes(file_data, rec)?;
                let data = apply_filters_to_chunk(raw, rec.filter_mask, Some(pipeline), elem_size)?;
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
    assemble_chunks(
        &chunks_arc,
        file_data,
        &real_chunk_dims,
        dataset_dims,
        elem_size,
        |raw, mask| {
            if pipeline.filters.is_empty() {
                Ok(raw.to_vec())
            } else {
                filters::apply_pipeline(raw, pipeline, mask, elem_size)
            }
        },
    )
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
/// * `dataset_dims` – full dataset dimensions in elements
/// * `elem_size`    – element size in bytes
/// * `ranges`       – one `Range<u64>` per dimension specifying the requested sub-region
/// * `cache`        – optional pre-parsed chunk index cache; pass `None` to disable caching
pub fn read_chunked_slice(
    file_data: &[u8],
    layout: &LayoutInfo,
    pipeline: &FilterPipeline,
    dataset_dims: &[u64],
    elem_size: usize,
    ranges: &[std::ops::Range<u64>],
    cache: Option<&ChunkIndexCache>,
) -> Result<Vec<u8>, OxiH5Error> {
    let LayoutInfo::Chunked {
        data_address,
        dimensionality,
        chunk_dims,
        index_type,
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

    // Translate index_type (internal convention) to ChunkIndex.
    let index = match index_type {
        0 => ChunkIndex::BTreeV1,
        1 => ChunkIndex::FixedArray,
        2 => ChunkIndex::ExtensibleArray,
        3 => ChunkIndex::BTreeV2,
        other => {
            return Err(OxiH5Error::Format(format!(
                "read_chunked_slice: unknown chunk index type {other}"
            )))
        }
    };

    // Resolve all chunk records (with optional caching).
    let chunks_arc: Arc<Vec<ChunkRecord>> = if let Some(c) = cache {
        let uncompressed_for_fa = real_chunk_dims.iter().product::<u64>() as usize * elem_size;
        let real_chunk_dims_clone = real_chunk_dims.clone();
        c.get_or_insert((*data_address, ndims), move || {
            if index == ChunkIndex::FixedArray {
                crate::fa_index::parse_fixed_array_v4(
                    file_data,
                    *data_address,
                    ndims,
                    &real_chunk_dims_clone,
                    uncompressed_for_fa,
                )
            } else {
                resolve_chunk_index(file_data, index, *data_address, ndims)
            }
        })?
    } else {
        let records = if index == ChunkIndex::FixedArray {
            let uncompressed = real_chunk_dims.iter().product::<u64>() as usize * elem_size;
            crate::fa_index::parse_fixed_array_v4(
                file_data,
                *data_address,
                ndims,
                &real_chunk_dims,
                uncompressed,
            )?
        } else {
            resolve_chunk_index(file_data, index, *data_address, ndims)?
        };
        Arc::new(records)
    };

    #[cfg(feature = "parallel")]
    {
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

        // Collect origins for all intersecting cells (only those with a real record).
        let present_cells: Vec<(Vec<u64>, Vec<u64>)> = (0..total_cells as usize)
            .filter_map(|cell_flat| {
                let ci_rel = flat_to_coords(cell_flat, &ci_strides, ndims);
                let origin: Vec<u64> = (0..ndims)
                    .map(|d| (first_ci[d] + ci_rel[d]) * real_chunk_dims[d])
                    .collect();
                chunk_map
                    .get(&origin)
                    .map(|&rec_idx| (origin, chunks_arc[rec_idx].offsets.clone()))
            })
            .collect();

        // Phase 1 (parallel): decompress only present (non-sparse) chunks.
        let decompressed: Vec<(Vec<u64>, Vec<u8>)> = present_cells
            .into_par_iter()
            .map(
                |(origin, offsets)| -> Result<(Vec<u64>, Vec<u8>), OxiH5Error> {
                    let rec_idx = *chunk_map.get(&origin).expect("origin in map");
                    let rec = &chunks_arc[rec_idx];
                    let raw = read_chunk_bytes(file_data, rec)?;
                    let data =
                        apply_filters_to_chunk(raw, rec.filter_mask, Some(pipeline), elem_size)?;
                    Ok((offsets, data))
                },
            )
            .collect::<Result<Vec<_>, OxiH5Error>>()?;

        // Build a map from origin → decompressed data for scatter phase.
        let decomp_map: HashMap<Vec<u64>, Vec<u8>> = decompressed.into_iter().collect();

        // Phase 2 (sequential): iterate cells and scatter (sparse chunks → zero).
        for cell_flat in 0..total_cells as usize {
            let ci_rel = flat_to_coords(cell_flat, &ci_strides, ndims);
            let origin: Vec<u64> = (0..ndims)
                .map(|d| (first_ci[d] + ci_rel[d]) * real_chunk_dims[d])
                .collect();

            let chunk_data: std::borrow::Cow<[u8]> = if let Some(data) = decomp_map.get(&origin) {
                std::borrow::Cow::Borrowed(data.as_slice())
            } else {
                std::borrow::Cow::Owned(vec![0u8; chunk_volume as usize * elem_size])
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
        elem_size,
        ranges,
        |raw, mask| {
            if pipeline.filters.is_empty() {
                Ok(raw.to_vec())
            } else {
                filters::apply_pipeline(raw, pipeline, mask, elem_size)
            }
        },
    )
}

/// Assemble only the chunks that overlap with `ranges` into an output buffer
/// of shape `[r.len() for r in ranges]` (row-major).
///
/// For each chunk-grid cell that intersects the requested hyperslab, we read
/// and decompress the chunk, then scatter the overlapping elements into the
/// output buffer.  Absent (sparse) chunks are left as zero.
///
/// Used by the sequential (non-parallel) code path in [`read_chunked_slice`].
#[cfg(any(not(feature = "parallel"), test))]
fn assemble_chunks_slice(
    chunks: &[ChunkRecord],
    file_data: &[u8],
    chunk_dims: &[u64],
    dataset_dims: &[u64],
    elem_size: usize,
    ranges: &[std::ops::Range<u64>],
    apply_filters: impl Fn(&[u8], u32) -> Result<Vec<u8>, OxiH5Error>,
) -> Result<Vec<u8>, OxiH5Error> {
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

        // Decompress (or create zero-filled buffer for sparse chunks).
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
            // Sparse chunk — treat as all-zero (HDF5 default fill value).
            vec![0u8; chunk_volume as usize * elem_size]
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

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

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
/// Used by the `parallel` feature code path and unit tests.
#[cfg(any(feature = "parallel", test))]
pub(crate) fn apply_filters_to_chunk(
    raw: &[u8],
    filter_mask: u32,
    pipeline: Option<&FilterPipeline>,
    elem_size: usize,
) -> Result<Vec<u8>, OxiH5Error> {
    match pipeline {
        Some(p) if !p.filters.is_empty() => filters::apply_pipeline(raw, p, filter_mask, elem_size),
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
fn row_major_strides(dims: &[u64]) -> Vec<usize> {
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
fn flat_to_coords(mut flat: usize, strides: &[usize], ndims: usize) -> Vec<u64> {
    let mut coords = vec![0u64; ndims];
    for d in 0..ndims {
        if let Some(q) = flat.checked_div(strides[d]) {
            coords[d] = q as u64;
            flat %= strides[d];
        }
    }
    coords
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `ChunkRecord` with a synthetic address into a fake file buffer.
    fn make_chunk(addr: u64, offsets: Vec<u64>, data: &[u8], buf: &mut Vec<u8>) -> ChunkRecord {
        let addr_usize = addr as usize;
        // Ensure buffer is large enough.
        if addr_usize + data.len() > buf.len() {
            buf.resize(addr_usize + data.len(), 0);
        }
        buf[addr_usize..addr_usize + data.len()].copy_from_slice(data);
        ChunkRecord {
            address: addr,
            size: data.len() as u32,
            filter_mask: 0,
            offsets,
        }
    }

    fn no_filter(data: &[u8], _mask: u32) -> Result<Vec<u8>, OxiH5Error> {
        Ok(data.to_vec())
    }

    #[test]
    fn test_assemble_1d_two_chunks() {
        // 1D dataset of 8 u8 elements, chunked into 4-element chunks.
        // Chunk 0 at offset 0: elements [0,1,2,3]
        // Chunk 1 at offset 4: elements [4,5,6,7]
        let mut file = vec![0u8; 64];
        let c0 = make_chunk(0, vec![0], &[0_u8, 1, 2, 3], &mut file);
        let c1 = make_chunk(4, vec![4], &[4_u8, 5, 6, 7], &mut file);

        let result = assemble_chunks(
            &[c0, c1],
            &file,
            &[4], // chunk_dims
            &[8], // dataset_dims
            1,    // elem_size
            no_filter,
        )
        .expect("assemble failed");

        assert_eq!(result, vec![0_u8, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_assemble_2x2_chunks_into_4x4() {
        // 4×4 dataset of u8, chunked 2×2 (four chunks).
        // The expected output is identity [0..15] in row-major order.
        //
        // Chunk at offset (0,0) holds elements (0,0),(0,1),(1,0),(1,1)
        // which map to flat indices 0,1,4,5 → values 0,1,2,3.
        // Similarly for the other three chunks.
        let mut file = vec![0u8; 256];
        let chunks = vec![
            // chunk offset (0,0): data [0,1,2,3]
            make_chunk(0, vec![0, 0], &[0_u8, 1, 2, 3], &mut file),
            // chunk offset (0,2): data [4,5,6,7]
            make_chunk(4, vec![0, 2], &[4_u8, 5, 6, 7], &mut file),
            // chunk offset (2,0): data [8,9,10,11]
            make_chunk(8, vec![2, 0], &[8_u8, 9, 10, 11], &mut file),
            // chunk offset (2,2): data [12,13,14,15]
            make_chunk(12, vec![2, 2], &[12_u8, 13, 14, 15], &mut file),
        ];

        let result = assemble_chunks(
            &chunks,
            &file,
            &[2, 2], // chunk_dims
            &[4, 4], // dataset_dims
            1,       // elem_size
            no_filter,
        )
        .expect("assemble failed");

        assert_eq!(result.len(), 16);
        // Row 0: (0,0)=0, (0,1)=1, (0,2)=4, (0,3)=5
        assert_eq!(&result[0..4], &[0, 1, 4, 5]);
        // Row 1: (1,0)=2, (1,1)=3, (1,2)=6, (1,3)=7
        assert_eq!(&result[4..8], &[2, 3, 6, 7]);
        // Row 2: (2,0)=8, (2,1)=9, (2,2)=12, (2,3)=13
        assert_eq!(&result[8..12], &[8, 9, 12, 13]);
        // Row 3: (3,0)=10, (3,1)=11, (3,2)=14, (3,3)=15
        assert_eq!(&result[12..16], &[10, 11, 14, 15]);
    }

    #[test]
    fn test_assemble_with_filter_applied() {
        // 1D dataset of 4 i32 elements (16 bytes total), chunked 4 at once.
        // The "filter" doubles every byte.
        let elem_size = 4;
        let raw = vec![
            0x01_u8, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00,
            0x00, 0x00,
        ];
        let mut file = vec![0u8; 256];
        let chunk = make_chunk(0, vec![0], &raw, &mut file);

        let result = assemble_chunks(
            &[chunk],
            &file,
            &[4],
            &[4],
            elem_size,
            |data, _mask| Ok(data.to_vec()), // identity filter
        )
        .expect("assemble failed");

        // i32 values 1,2,3,4 in little-endian.
        assert_eq!(result[0..4], [0x01, 0x00, 0x00, 0x00]);
        assert_eq!(result[4..8], [0x02, 0x00, 0x00, 0x00]);
        assert_eq!(result[8..12], [0x03, 0x00, 0x00, 0x00]);
        assert_eq!(result[12..16], [0x04, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_assemble_padding_elements_ignored() {
        // 1D dataset of 3 elements (not a multiple of chunk size 4).
        // The chunk contains 4 raw elements but only 3 fit in the dataset.
        let mut file = vec![0u8; 64];
        let chunk = make_chunk(0, vec![0], &[10_u8, 20, 30, 99], &mut file);

        let result = assemble_chunks(
            &[chunk],
            &file,
            &[4], // chunk_dims
            &[3], // dataset_dims (3 < 4 → element 3 is padding)
            1,
            no_filter,
        )
        .expect("assemble failed");

        assert_eq!(result, vec![10_u8, 20, 30]);
    }

    #[test]
    fn test_assemble_dim_mismatch_errors() {
        let file = vec![0u8; 64];
        let chunk = ChunkRecord {
            address: 0,
            size: 4,
            filter_mask: 0,
            offsets: vec![0],
        };
        let result = assemble_chunks(
            &[chunk],
            &file,
            &[4],    // 1D chunk
            &[4, 4], // 2D dataset — mismatch!
            1,
            no_filter,
        );
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // read_chunked_slice unit tests
    // -----------------------------------------------------------------------

    /// Test: 4×4 u8 dataset chunked 2×2, request slice [1..3, 1..3].
    ///
    /// Expected result: elements (1,1)=5, (1,2)=6, (2,1)=9, (2,2)=10
    /// when the dataset contains consecutive u8 values 0..15 in row-major order.
    #[test]
    fn test_chunked_slice_2d() {
        // Build a 4×4 u8 dataset split into four 2×2 chunks.
        // Row-major element layout:
        //   row 0: 0  1  2  3
        //   row 1: 4  5  6  7
        //   row 2: 8  9 10 11
        //   row 3:12 13 14 15
        //
        // Chunk (0,0) offset (0,0): elements (0,0)=0,(0,1)=1,(1,0)=4,(1,1)=5  → [0,1,4,5]
        // Chunk (0,1) offset (0,2): elements (0,2)=2,(0,3)=3,(1,2)=6,(1,3)=7  → [2,3,6,7]
        // Chunk (1,0) offset (2,0): elements (2,0)=8,(2,1)=9,(3,0)=12,(3,1)=13 → [8,9,12,13]
        // Chunk (1,1) offset (2,2): elements (2,2)=10,(2,3)=11,(3,2)=14,(3,3)=15 → [10,11,14,15]

        let mut file = vec![0u8; 512];

        let c00 = make_chunk(0, vec![0, 0], &[0_u8, 1, 4, 5], &mut file);
        let c01 = make_chunk(4, vec![0, 2], &[2_u8, 3, 6, 7], &mut file);
        let c10 = make_chunk(8, vec![2, 0], &[8_u8, 9, 12, 13], &mut file);
        let c11 = make_chunk(12, vec![2, 2], &[10_u8, 11, 14, 15], &mut file);

        let chunk_dims = [2u64, 2];
        let dataset_dims = [4u64, 4];
        // Multi-element array avoids the single_range_in_vec_init lint.
        let ranges: [std::ops::Range<u64>; 2] = [1..3, 1..3];

        let result = assemble_chunks_slice(
            &[c00, c01, c10, c11],
            &file,
            &chunk_dims,
            &dataset_dims,
            1, // elem_size
            &ranges,
            no_filter,
        )
        .expect("assemble_chunks_slice failed");

        // Expected: row 0 of output = elements (1,1)=5, (1,2)=6
        //           row 1 of output = elements (2,1)=9, (2,2)=10
        assert_eq!(result.len(), 4, "output length mismatch");
        assert_eq!(result[0], 5, "element (1,1)");
        assert_eq!(result[1], 6, "element (1,2)");
        assert_eq!(result[2], 9, "element (2,1)");
        assert_eq!(result[3], 10, "element (2,2)");
    }

    /// Test: 1D dataset with boundary-spanning range.
    #[test]
    fn test_chunked_slice_1d_partial() {
        // 8-element u8 dataset, chunk size 4.
        // Chunk at offset 0: [0,1,2,3], chunk at offset 4: [4,5,6,7]
        let mut file = vec![0u8; 128];
        let c0 = make_chunk(0, vec![0], &[0_u8, 1, 2, 3], &mut file);
        let c1 = make_chunk(4, vec![4], &[4_u8, 5, 6, 7], &mut file);

        // Explicit binding avoids single_range_in_vec_init lint for a 1-element range.
        let r: std::ops::Range<u64> = 2..6;
        let ranges = [r]; // spans both chunks

        let result = assemble_chunks_slice(
            &[c0, c1],
            &file,
            &[4], // chunk_dims
            &[8], // dataset_dims
            1,    // elem_size
            &ranges,
            no_filter,
        )
        .expect("assemble_chunks_slice 1d failed");

        assert_eq!(result, vec![2_u8, 3, 4, 5]);
    }

    /// Test: empty range returns empty buffer.
    #[test]
    fn test_chunked_slice_empty_range() {
        let file = vec![0u8; 64];
        // Explicit binding avoids single_range_in_vec_init lint.
        let r: std::ops::Range<u64> = 3..3;
        let ranges = [r];
        let result = assemble_chunks_slice(&[], &file, &[4], &[8], 1, &ranges, no_filter)
            .expect("empty range failed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_row_major_strides_3d() {
        // Shape [2, 3, 4]: strides should be [12, 4, 1].
        let strides = row_major_strides(&[2, 3, 4]);
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_flat_to_coords_roundtrip() {
        let dims = [3u64, 4, 5];
        let strides = row_major_strides(&dims);
        for flat in 0..(3 * 4 * 5) {
            let coords = flat_to_coords(flat, &strides, 3);
            let reconstructed: usize = coords
                .iter()
                .zip(strides.iter())
                .map(|(&c, &s)| c as usize * s)
                .sum();
            assert_eq!(reconstructed, flat, "flat={flat}");
        }
    }

    // -----------------------------------------------------------------------
    // ChunkIndexCache unit tests
    // -----------------------------------------------------------------------

    /// Verify that `get_or_insert` calls `compute` exactly once for the same
    /// key, even when called a second time.
    #[test]
    fn test_chunk_cache_hit() {
        use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

        let counter = AtomicUsize::new(0);
        let cache = ChunkIndexCache::new();

        let key = (0x1000_u64, 2_usize);

        // First call — `compute` must run.
        let first = cache
            .get_or_insert(key, || {
                counter.fetch_add(1, Relaxed);
                Ok(vec![ChunkRecord {
                    address: 0,
                    size: 4,
                    filter_mask: 0,
                    offsets: vec![0, 0],
                }])
            })
            .expect("first get_or_insert failed");

        assert_eq!(
            counter.load(Relaxed),
            1,
            "compute should have been called once"
        );
        assert_eq!(first.len(), 1);

        // Second call with the same key — `compute` must NOT run again.
        let second = cache
            .get_or_insert(key, || {
                counter.fetch_add(1, Relaxed);
                Ok(vec![])
            })
            .expect("second get_or_insert failed");

        assert_eq!(
            counter.load(Relaxed),
            1,
            "compute should still have been called only once"
        );
        assert_eq!(second.len(), 1, "cached result should be returned");

        // Both Arcs must point to the same allocation.
        assert!(
            Arc::ptr_eq(&first, &second),
            "both Arc values should share the same backing allocation"
        );
    }

    /// Verify that different keys are cached independently.
    #[test]
    fn test_chunk_cache_different_keys() {
        use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

        let counter = AtomicUsize::new(0);
        let cache = ChunkIndexCache::new();

        let _ = cache
            .get_or_insert((0x1000_u64, 1_usize), || {
                counter.fetch_add(1, Relaxed);
                Ok(vec![ChunkRecord {
                    address: 0,
                    size: 1,
                    filter_mask: 0,
                    offsets: vec![0],
                }])
            })
            .expect("key-a failed");

        let _ = cache
            .get_or_insert((0x2000_u64, 1_usize), || {
                counter.fetch_add(1, Relaxed);
                Ok(vec![ChunkRecord {
                    address: 0,
                    size: 2,
                    filter_mask: 0,
                    offsets: vec![0],
                }])
            })
            .expect("key-b failed");

        assert_eq!(
            counter.load(Relaxed),
            2,
            "each distinct key should invoke compute once"
        );
    }

    // -----------------------------------------------------------------------
    // Parallel helper unit tests
    // -----------------------------------------------------------------------

    /// Verify `apply_filters_to_chunk` returns a copy of raw bytes when no pipeline is given.
    #[test]
    fn test_apply_filters_to_chunk_no_pipeline() {
        let raw = vec![1u8, 2, 3, 4];
        let result = apply_filters_to_chunk(&raw, 0, None, 1)
            .expect("apply_filters_to_chunk with no pipeline failed");
        assert_eq!(result, raw);
    }

    /// Verify `apply_filters_to_chunk` returns a copy of raw bytes for an empty pipeline.
    #[test]
    fn test_apply_filters_to_chunk_empty_pipeline() {
        use oxih5_core::FilterPipeline;
        let raw = vec![10u8, 20, 30];
        let pipeline = FilterPipeline { filters: vec![] };
        let result = apply_filters_to_chunk(&raw, 0, Some(&pipeline), 1)
            .expect("apply_filters_to_chunk with empty pipeline failed");
        assert_eq!(result, raw);
    }

    /// Verify `read_chunk_bytes` returns the correct slice.
    #[test]
    fn test_read_chunk_bytes() {
        let file = vec![0u8, 10, 20, 30, 40, 50];
        let rec = ChunkRecord {
            address: 2,
            size: 3,
            filter_mask: 0,
            offsets: vec![0],
        };
        let bytes = read_chunk_bytes(&file, &rec).expect("read_chunk_bytes failed");
        assert_eq!(bytes, &[20u8, 30, 40]);
    }

    /// Verify `scatter_chunk` places elements correctly in the output buffer.
    #[test]
    fn test_scatter_chunk_2d() {
        // 4x4 output, place a 2x2 chunk at origin (2, 2).
        let dataset_dims = [4u64, 4];
        let chunk_dims = [2u64, 2];
        let elem_size = 1;
        let mut output = vec![0u8; 16];
        let chunk_data = vec![11u8, 12, 13, 14];
        let origin = vec![2u64, 2];

        scatter_chunk(
            &mut output,
            &origin,
            &chunk_data,
            &chunk_dims,
            &dataset_dims,
            elem_size,
        )
        .expect("scatter_chunk failed");

        // Elements at (2,2)=11, (2,3)=12, (3,2)=13, (3,3)=14
        assert_eq!(output[2 * 4 + 2], 11);
        assert_eq!(output[2 * 4 + 3], 12);
        assert_eq!(output[3 * 4 + 2], 13);
        assert_eq!(output[3 * 4 + 3], 14);
        // All other elements stay zero.
        assert_eq!(output[0], 0);
    }

    /// Verify `scatter_chunk_slice` places elements correctly when only a sub-region is requested.
    #[test]
    fn test_scatter_chunk_slice_basic() {
        // 4x4 dataset, requesting slice [1..3, 1..3].
        // Chunk at origin (0, 0), 2x2, holds row-major values [0,1,4,5].
        let chunk_dims = [2u64, 2];
        let ranges: [std::ops::Range<u64>; 2] = [1..3, 1..3];
        let elem_size = 1;
        let mut output = vec![0u8; 4]; // 2x2 output
        let chunk_data = vec![0u8, 1, 4, 5]; // chunk at (0,0)
        let origin = vec![0u64, 0];

        scatter_chunk_slice(
            &mut output,
            &origin,
            &chunk_data,
            &chunk_dims,
            &ranges,
            elem_size,
        )
        .expect("scatter_chunk_slice failed");

        // Only element (1,1)=5 falls within the chunk (0..2, 0..2) ∩ slice (1..3, 1..3) = (1..2, 1..2).
        // In output coords: (1-1, 1-1) = (0, 0) → flat 0.
        assert_eq!(output[0], 5);
        // Other elements remain zero.
        assert_eq!(output[1], 0);
        assert_eq!(output[2], 0);
        assert_eq!(output[3], 0);
    }

    // -----------------------------------------------------------------------
    // Parallel vs sequential consistency test
    // -----------------------------------------------------------------------

    #[cfg(feature = "parallel")]
    #[test]
    fn test_chunked_parallel_matches_sequential() {
        // Build a synthetic 4x4 u8 dataset chunked 2x2 (four chunks).
        // Each chunk-local element layout (row-major within the chunk):
        //   chunk(0,0): [0,1,4,5]   (dataset positions (0,0),(0,1),(1,0),(1,1))
        //   chunk(0,2): [2,3,6,7]   (dataset positions (0,2),(0,3),(1,2),(1,3))
        //   chunk(2,0): [8,9,12,13] (dataset positions (2,0),(2,1),(3,0),(3,1))
        //   chunk(2,2): [10,11,14,15]
        let mut file = vec![0u8; 512];
        let chunks = vec![
            make_chunk(0, vec![0, 0], &[0_u8, 1, 4, 5], &mut file),
            make_chunk(4, vec![0, 2], &[2_u8, 3, 6, 7], &mut file),
            make_chunk(8, vec![2, 0], &[8_u8, 9, 12, 13], &mut file),
            make_chunk(12, vec![2, 2], &[10_u8, 11, 14, 15], &mut file),
        ];

        let chunk_dims = vec![2u64, 2];
        let dataset_dims = vec![4u64, 4];
        let elem_size = 1;

        // Sequential path via assemble_chunks.
        let seq = assemble_chunks(
            &chunks,
            &file,
            &chunk_dims,
            &dataset_dims,
            elem_size,
            no_filter,
        )
        .expect("sequential assemble_chunks failed");

        // Parallel path via the helper functions (mirrors what read_chunked does under
        // the `parallel` feature).
        let total_elems: u64 = dataset_dims.iter().product();
        let mut par_output = vec![0u8; total_elems as usize * elem_size];

        let decompressed: Vec<(Vec<u64>, Vec<u8>)> = chunks
            .par_iter()
            .map(|rec| -> Result<(Vec<u64>, Vec<u8>), OxiH5Error> {
                let raw = read_chunk_bytes(&file, rec)?;
                let data = apply_filters_to_chunk(raw, rec.filter_mask, None, elem_size)?;
                Ok((rec.offsets.clone(), data))
            })
            .collect::<Result<Vec<_>, OxiH5Error>>()
            .expect("parallel decompression failed");

        for (origin, chunk_data) in decompressed {
            scatter_chunk(
                &mut par_output,
                &origin,
                &chunk_data,
                &chunk_dims,
                &dataset_dims,
                elem_size,
            )
            .expect("scatter_chunk failed");
        }

        assert_eq!(
            seq, par_output,
            "parallel and sequential outputs must match exactly"
        );

        // Also verify the actual values are what we expect (row 0: 0,1,2,3 etc.)
        assert_eq!(&seq[0..4], &[0u8, 1, 2, 3]);
        assert_eq!(&seq[4..8], &[4u8, 5, 6, 7]);
        assert_eq!(&seq[8..12], &[8u8, 9, 10, 11]);
        assert_eq!(&seq[12..16], &[12u8, 13, 14, 15]);
    }
}
