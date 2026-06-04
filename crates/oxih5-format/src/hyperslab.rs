use oxih5_core::OxiH5Error;

/// A selection along a single dimension using the HDF5 hyperslab model.
///
/// Selects `count` blocks, each of `block` contiguous elements, with
/// `stride` elements between the *start* of consecutive blocks.  The
/// resulting selected element indices in that dimension are:
/// `{ start + k*stride + j : 0 <= k < count, 0 <= j < block }`.
///
/// The contiguous case (`stride == 1, block == 1`) is equivalent to a simple
/// `start .. start + count` range.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimSelection {
    pub start: u64,
    /// Gap between the start of consecutive blocks (>= 1).
    pub stride: u64,
    /// Number of blocks.
    pub count: u64,
    /// Elements per block (>= 1).
    pub block: u64,
}

impl DimSelection {
    /// Construct a contiguous selection equivalent to `r.start .. r.end`.
    pub fn contiguous(r: std::ops::Range<u64>) -> Self {
        Self {
            start: r.start,
            stride: 1,
            count: r.end - r.start,
            block: 1,
        }
    }

    /// Number of elements selected in this dimension.
    pub fn n_elements(&self) -> u64 {
        self.count * self.block
    }

    /// Inclusive last element index selected (if any).
    ///
    /// Returns `None` when `count == 0` or `block == 0`.
    fn last_element(&self) -> Option<u64> {
        if self.count == 0 || self.block == 0 {
            return None;
        }
        Some(self.start + (self.count - 1) * self.stride + (self.block - 1))
    }

    /// Check whether a global index `coord` is selected by this `DimSelection`.
    #[inline]
    pub fn contains(&self, coord: u64) -> bool {
        if self.count == 0 || self.block == 0 {
            return false;
        }
        if coord < self.start {
            return false;
        }
        let rel = coord - self.start;
        let block_idx = rel / self.stride;
        if block_idx >= self.count {
            return false;
        }
        (rel % self.stride) < self.block
    }
}

/// An N-dimensional HDF5 hyperslab selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hyperslab {
    pub dims: Vec<DimSelection>,
}

impl Hyperslab {
    /// Build a contiguous hyperslab from a slice of `Range<u64>`.
    pub fn contiguous(ranges: &[std::ops::Range<u64>]) -> Self {
        Self {
            dims: ranges
                .iter()
                .map(|r| DimSelection::contiguous(r.clone()))
                .collect(),
        }
    }

    /// Per-dimension element counts for the output buffer shape.
    pub fn output_shape(&self) -> Vec<u64> {
        self.dims.iter().map(|d| d.n_elements()).collect()
    }

    /// Conservative bounding box: smallest `start .. end` that encloses every
    /// selected element in each dimension.
    pub fn bounding_ranges(&self) -> Vec<std::ops::Range<u64>> {
        self.dims
            .iter()
            .map(|d| {
                if d.count == 0 || d.block == 0 {
                    return 0..0;
                }
                let end = d.last_element().map(|l| l + 1).unwrap_or(d.start);
                d.start..end
            })
            .collect()
    }

    /// True when the selection is equivalent to a simple contiguous slice
    /// (every `DimSelection` has `stride == 1` and `block == 1`).
    pub fn is_fully_contiguous(&self) -> bool {
        self.dims.iter().all(|d| d.stride == 1 && d.block == 1)
    }

    /// True when any dimension selects zero elements.
    pub fn is_empty(&self) -> bool {
        self.dims.iter().any(|d| d.count == 0 || d.block == 0)
    }
}

/// Scatter elements from a decoded chunk into a hyperslab-shaped output buffer.
///
/// For each element in the intersection of the chunk's spatial extent with the
/// bounding box of `selection`, the element's global coordinate is tested
/// against `selection`.  If selected, it is written to the appropriate position
/// in `output` (shaped `selection.output_shape()` in row-major order).
///
/// When `selection.is_fully_contiguous()` the result is identical to calling
/// `scatter_chunk_slice` with the corresponding range slice.
///
/// # Arguments
///
/// * `output`        – output byte buffer sized `product(out_shape) * elem_size`
/// * `chunk_data`    – decoded (decompressed) chunk bytes
/// * `chunk_origin`  – dataset-space origin of the chunk (one entry per dim)
/// * `chunk_dims`    – elements per chunk dimension
/// * `selection`     – hyperslab selection
/// * `out_shape`     – `selection.output_shape()` (caller pre-computes to avoid recomputation)
/// * `elem_size`     – bytes per element
pub fn scatter_chunk_hyperslab(
    output: &mut [u8],
    chunk_data: &[u8],
    chunk_origin: &[u64],
    chunk_dims: &[u64],
    selection: &Hyperslab,
    out_shape: &[u64],
    elem_size: usize,
) -> Result<(), OxiH5Error> {
    let ndims = selection.dims.len();
    if chunk_origin.len() != ndims || chunk_dims.len() != ndims || out_shape.len() != ndims {
        return Err(OxiH5Error::Format(
            "scatter_chunk_hyperslab: dimension mismatch".into(),
        ));
    }

    // Bounding box of the selection — used to clip the per-dim iteration range.
    let bbox = selection.bounding_ranges();

    // Overlap of the chunk with the bounding box.
    let ovl_start: Vec<u64> = (0..ndims)
        .map(|d| bbox[d].start.max(chunk_origin[d]))
        .collect();
    let ovl_end: Vec<u64> = (0..ndims)
        .map(|d| bbox[d].end.min(chunk_origin[d] + chunk_dims[d]))
        .collect();

    // Early exit when no overlap.
    if (0..ndims).any(|d| ovl_start[d] >= ovl_end[d]) {
        return Ok(());
    }

    let ovl_shape: Vec<u64> = (0..ndims).map(|d| ovl_end[d] - ovl_start[d]).collect();
    let ovl_elems: u64 = ovl_shape.iter().product();
    let ovl_strides = row_major_strides(&ovl_shape);

    let chunk_strides = row_major_strides(chunk_dims);
    let out_strides = row_major_strides(out_shape);

    // Cumulative output offset per selected element in each dimension.
    // out_offset[d][coord] = number of elements in dim d that are selected
    // before `coord`.  Built lazily into a running counter during iteration.
    // Because the output mapping is separable (row-major flat index =
    // sum_d(out_coord[d] * out_stride[d])), we can compute each dim's
    // output coordinate independently.

    for ovl_flat in 0..ovl_elems as usize {
        let ovl_coords = flat_to_coords(ovl_flat, &ovl_strides, ndims);

        // Global element coordinates.
        let global: Vec<u64> = (0..ndims).map(|d| ovl_start[d] + ovl_coords[d]).collect();

        // Check selection membership for every dimension.
        if (0..ndims).any(|d| !selection.dims[d].contains(global[d])) {
            continue;
        }

        // Chunk-local flat index (source).
        let in_chunk: Vec<u64> = (0..ndims).map(|d| global[d] - chunk_origin[d]).collect();
        let chunk_flat: usize = in_chunk
            .iter()
            .zip(chunk_strides.iter())
            .map(|(&c, &s)| c as usize * s)
            .sum();

        // Output-local coordinate per dimension.
        // For dimension d, the output coordinate is the number of selected
        // elements in [start_d .. global[d]).
        let out_coord: Vec<u64> = (0..ndims)
            .map(|d| {
                let sel = &selection.dims[d];
                let g = global[d];
                // Block index of this element.
                let rel = g - sel.start;
                let block_idx = rel / sel.stride;
                let in_block = rel % sel.stride;
                // Elements selected before this block.
                block_idx * sel.block + in_block
            })
            .collect();

        let out_flat: usize = out_coord
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

// ---------------------------------------------------------------------------
// Shared helpers (duplicated from chunked.rs to keep hyperslab self-contained)
// ---------------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // DimSelection / Hyperslab structural tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dim_selection_contiguous() {
        let ds = DimSelection::contiguous(3..8);
        assert_eq!(ds.start, 3);
        assert_eq!(ds.stride, 1);
        assert_eq!(ds.count, 5);
        assert_eq!(ds.block, 1);
        assert_eq!(ds.n_elements(), 5);
    }

    #[test]
    fn test_hyperslab_output_shape() {
        let hs = Hyperslab::contiguous(&[0..4, 2..6]);
        assert_eq!(hs.output_shape(), vec![4, 6 - 2]);
    }

    #[test]
    fn test_hyperslab_is_empty_cases() {
        let empty = Hyperslab {
            dims: vec![DimSelection {
                start: 0,
                stride: 1,
                count: 0,
                block: 1,
            }],
        };
        assert!(empty.is_empty());

        let zero_block = Hyperslab {
            dims: vec![DimSelection {
                start: 0,
                stride: 1,
                count: 5,
                block: 0,
            }],
        };
        assert!(zero_block.is_empty());

        let r: std::ops::Range<u64> = 0..3;
        let non_empty = Hyperslab::contiguous(&[r]);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_hyperslab_is_fully_contiguous() {
        let cont = Hyperslab::contiguous(&[1..5, 0..3]);
        assert!(cont.is_fully_contiguous());

        let strided = Hyperslab {
            dims: vec![DimSelection {
                start: 0,
                stride: 2,
                count: 3,
                block: 1,
            }],
        };
        assert!(!strided.is_fully_contiguous());
    }

    #[test]
    fn test_bounding_ranges_encloses_selected_coords() {
        // Stride=2, count=3, block=1 → selects {1, 3, 5}, last=5.
        let hs = Hyperslab {
            dims: vec![DimSelection {
                start: 1,
                stride: 2,
                count: 3,
                block: 1,
            }],
        };
        let bbox = hs.bounding_ranges();
        // Bbox must start <= 1 and end > 5.
        assert!(bbox[0].start <= 1);
        assert!(bbox[0].end > 5);
    }

    #[test]
    fn test_bounding_ranges_block2() {
        // start=0, stride=4, count=2, block=2 → selects {0,1,4,5}, last=5.
        let hs = Hyperslab {
            dims: vec![DimSelection {
                start: 0,
                stride: 4,
                count: 2,
                block: 2,
            }],
        };
        let bbox = hs.bounding_ranges();
        assert!(bbox[0].end > 5, "end={} should be > 5", bbox[0].end);
    }

    // -----------------------------------------------------------------------
    // scatter_chunk_hyperslab: 3-D contiguous slice crossing chunk boundaries
    // -----------------------------------------------------------------------

    #[test]
    fn test_scatter_3d_contiguous() {
        // Dataset 4×4×4, chunk 2×2×2, select [1..3, 1..3, 1..3] (2×2×2 output).
        // The dataset has values = flat_index (u8).
        let chunk_dims = [2u64, 2, 2];
        let elem_size = 1;

        // Dataset value at (r,c,d) = r*16 + c*4 + d.
        let row_major = |r: u64, c: u64, d: u64| -> u8 { (r * 16 + c * 4 + d) as u8 };

        let mut output = vec![0u8; 8]; // 2×2×2
        let selection = Hyperslab::contiguous(&[1..3, 1..3, 1..3]);
        let out_shape = selection.output_shape();

        // Chunk at origin (0,0,0): elements (0..2, 0..2, 0..2) in row-major chunk layout.
        let chunk_origin = [0u64, 0, 0];
        let mut chunk_data = vec![0u8; 8];
        for r in 0u64..2 {
            for c in 0u64..2 {
                for d in 0u64..2 {
                    let flat = (r * 4 + c * 2 + d) as usize;
                    chunk_data[flat] = row_major(r, c, d);
                }
            }
        }
        scatter_chunk_hyperslab(
            &mut output,
            &chunk_data,
            &chunk_origin,
            &chunk_dims,
            &selection,
            &out_shape,
            elem_size,
        )
        .expect("scatter 3d (0,0,0)");

        // Chunk at origin (0,0,2): elements (0..2, 0..2, 2..4).
        let chunk_origin_002 = [0u64, 0, 2];
        let mut chunk_002 = vec![0u8; 8];
        for r in 0u64..2 {
            for c in 0u64..2 {
                for d in 0u64..2 {
                    let flat = (r * 4 + c * 2 + d) as usize;
                    chunk_002[flat] = row_major(r, c, 2 + d);
                }
            }
        }
        scatter_chunk_hyperslab(
            &mut output,
            &chunk_002,
            &chunk_origin_002,
            &chunk_dims,
            &selection,
            &out_shape,
            elem_size,
        )
        .expect("scatter 3d (0,0,2)");

        // Chunk (0,2,0), (0,2,2), (2,0,0), (2,0,2), (2,2,0), (2,2,2).
        let mut other_origins: Vec<[u64; 3]> = Vec::new();
        for ri in [0u64, 2] {
            for ci in [0u64, 2] {
                for di in [0u64, 2] {
                    if ri == 0 && ci == 0 {
                        continue;
                    }
                    other_origins.push([ri, ci, di]);
                }
            }
        }
        for orig in &other_origins {
            let mut cd = vec![0u8; 8];
            for r in 0u64..2 {
                for c in 0u64..2 {
                    for d in 0u64..2 {
                        let flat = (r * 4 + c * 2 + d) as usize;
                        cd[flat] = row_major(orig[0] + r, orig[1] + c, orig[2] + d);
                    }
                }
            }
            scatter_chunk_hyperslab(
                &mut output,
                &cd,
                orig,
                &chunk_dims,
                &selection,
                &out_shape,
                elem_size,
            )
            .expect("scatter 3d other");
        }

        // Verify: output[(r-1, c-1, d-1)] = row_major(r, c, d) for r,c,d in 1..3.
        let out_strides = row_major_strides(&out_shape);
        for r in 1u64..3 {
            for c in 1u64..3 {
                for d in 1u64..3 {
                    let out_flat = ((r - 1) as usize * out_strides[0])
                        + ((c - 1) as usize * out_strides[1])
                        + ((d - 1) as usize * out_strides[2]);
                    assert_eq!(
                        output[out_flat],
                        row_major(r, c, d),
                        "mismatch at ({r},{c},{d})"
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Strided 1-D: picks {1, 3, 5} from 6-element dataset in 2-element chunks
    // -----------------------------------------------------------------------

    #[test]
    fn test_scatter_1d_strided() {
        // Dataset: [10, 11, 12, 13, 14, 15], chunks of 2 elements.
        // Selection: start=1, stride=2, count=3, block=1 → indices {1, 3, 5}.
        let selection = Hyperslab {
            dims: vec![DimSelection {
                start: 1,
                stride: 2,
                count: 3,
                block: 1,
            }],
        };
        let out_shape = selection.output_shape(); // [3]
        assert_eq!(out_shape, vec![3]);

        let mut output = vec![0u8; 3];
        let chunk_dims = [2u64];
        let elem_size = 1;

        // Chunk at origin 0: data [10, 11].
        scatter_chunk_hyperslab(
            &mut output,
            &[10u8, 11],
            &[0u64],
            &chunk_dims,
            &selection,
            &out_shape,
            elem_size,
        )
        .expect("scatter chunk 0");

        // Chunk at origin 2: data [12, 13].
        scatter_chunk_hyperslab(
            &mut output,
            &[12u8, 13],
            &[2u64],
            &chunk_dims,
            &selection,
            &out_shape,
            elem_size,
        )
        .expect("scatter chunk 2");

        // Chunk at origin 4: data [14, 15].
        scatter_chunk_hyperslab(
            &mut output,
            &[14u8, 15],
            &[4u64],
            &chunk_dims,
            &selection,
            &out_shape,
            elem_size,
        )
        .expect("scatter chunk 4");

        // Expect output = [11, 13, 15].
        assert_eq!(output, vec![11u8, 13, 15]);
    }

    // -----------------------------------------------------------------------
    // block=2 strided 2-D
    // -----------------------------------------------------------------------

    #[test]
    fn test_scatter_2d_block2() {
        // Dataset 8×1, chunk 2×1.
        // Selection dim0: start=0, stride=4, count=2, block=2 → indices {0,1,4,5}.
        //            dim1: contiguous 0..1.
        // Output shape = [4, 1], expected values [v[0], v[1], v[4], v[5]].
        let dataset_vals: Vec<u8> = (0u8..8).collect();
        let selection = Hyperslab {
            dims: vec![
                DimSelection {
                    start: 0,
                    stride: 4,
                    count: 2,
                    block: 2,
                },
                DimSelection::contiguous(0..1),
            ],
        };
        let out_shape = selection.output_shape(); // [4, 1]
        assert_eq!(out_shape, vec![4, 1]);

        let mut output = vec![0u8; 4];
        let chunk_dims = [2u64, 1];

        for chunk_start in (0u64..8).step_by(2) {
            let data = &dataset_vals[chunk_start as usize..chunk_start as usize + 2];
            scatter_chunk_hyperslab(
                &mut output,
                data,
                &[chunk_start, 0u64],
                &chunk_dims,
                &selection,
                &out_shape,
                1,
            )
            .expect("scatter block2");
        }

        // Indices selected: {0,1,4,5} → values 0,1,4,5.
        assert_eq!(output, vec![0u8, 1, 4, 5]);
    }

    // -----------------------------------------------------------------------
    // Trailing partial edge chunk (dataset not divisible by chunk size)
    // -----------------------------------------------------------------------

    #[test]
    fn test_scatter_partial_edge_chunk() {
        // Dataset of 5 elements, chunk size 3. Chunk at origin 3 has 2 real + 1 padding.
        // Selection: 0..5 (full).
        let r_full: std::ops::Range<u64> = 0..5;
        let selection = Hyperslab::contiguous(&[r_full]);
        let out_shape = selection.output_shape();
        let mut output = vec![0u8; 5];

        // Chunk 0 (origin 0): [10, 20, 30]
        scatter_chunk_hyperslab(
            &mut output,
            &[10u8, 20, 30],
            &[0u64],
            &[3u64],
            &selection,
            &out_shape,
            1,
        )
        .expect("edge scatter 0");

        // Chunk 1 (origin 3): [40, 50, 99] — 99 is padding (beyond dataset length).
        // The scatter should copy 40 and 50 only, because the selection only goes to 5.
        scatter_chunk_hyperslab(
            &mut output,
            &[40u8, 50, 99],
            &[3u64],
            &[3u64],
            &selection,
            &out_shape,
            1,
        )
        .expect("edge scatter 1");

        assert_eq!(&output, &[10u8, 20, 30, 40, 50]);
    }

    // -----------------------------------------------------------------------
    // Contiguous hyperslab result matches scatter_chunk_slice
    // -----------------------------------------------------------------------

    #[test]
    fn test_contiguous_matches_scatter_chunk_slice() {
        // 4×4 dataset, chunk 2×2, select [1..3, 1..3].
        let chunk_dims = [2u64, 2];
        let selection = Hyperslab::contiguous(&[1..3, 1..3]);
        let out_shape = selection.output_shape();
        let mut hyper_out = vec![0u8; 4];

        let chunk_data = [0u8, 1, 4, 5]; // chunk at (0,0)
        let origin = [0u64, 0];
        scatter_chunk_hyperslab(
            &mut hyper_out,
            &chunk_data,
            &origin,
            &chunk_dims,
            &selection,
            &out_shape,
            1,
        )
        .expect("hyperslab scatter");

        // Compare with the chunked::scatter_chunk_slice result.
        // Only element (1,1)=5 falls in the intersection of chunk(0..2,0..2) ∩ sel(1..3,1..3).
        assert_eq!(hyper_out[0], 5, "element (1,1) should be 5");
        assert_eq!(hyper_out[1], 0, "element (1,2) not in this chunk");
        assert_eq!(hyper_out[2], 0, "element (2,1) not in this chunk");
        assert_eq!(hyper_out[3], 0, "element (2,2) not in this chunk");
    }
}
