//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use crate::btree_v2::ChunkRecord;
use oxih5_core::OxiH5Error;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::sync::Arc;

use super::read::assemble_chunks_with_fill;
use super::slice::assemble_chunks_slice;
use super::*;

#[cfg(test)]
mod tests_2 {
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
            &ranges,
            SliceElemConfig {
                elem_size: 1,
                fill_value: None,
            },
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
            &ranges,
            SliceElemConfig {
                elem_size: 1,
                fill_value: None,
            },
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
        let result = assemble_chunks_slice(
            &[],
            &file,
            &[4],
            &[8],
            &ranges,
            SliceElemConfig {
                elem_size: 1,
                fill_value: None,
            },
            no_filter,
        )
        .expect("empty range failed");
        assert!(result.is_empty());
    }

    /// Regression test: a zero chunk dimension must be rejected with a typed
    /// error instead of panicking with a divide-by-zero when computing the
    /// overlapping chunk-grid cell range.
    #[test]
    fn test_chunked_slice_zero_chunk_dim_errors() {
        let file = vec![0u8; 64];
        let r: std::ops::Range<u64> = 1..3;
        let ranges = [r];
        let result = assemble_chunks_slice(
            &[],
            &file,
            &[0], // zero chunk dimension: must be rejected, not divided by
            &[8],
            &ranges,
            SliceElemConfig {
                elem_size: 1,
                fill_value: None,
            },
            no_filter,
        );
        assert!(
            result.is_err(),
            "zero chunk dimension must return an error, not panic"
        );
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
        let result = apply_filters_to_chunk(&raw, 0, None, 1, None)
            .expect("apply_filters_to_chunk with no pipeline failed");
        assert_eq!(result, raw);
    }

    /// Verify `apply_filters_to_chunk` returns a copy of raw bytes for an empty pipeline.
    #[test]
    fn test_apply_filters_to_chunk_empty_pipeline() {
        use oxih5_core::FilterPipeline;
        let raw = vec![10u8, 20, 30];
        let pipeline = FilterPipeline { filters: vec![] };
        let result = apply_filters_to_chunk(&raw, 0, Some(&pipeline), 1, None)
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
                let data = apply_filters_to_chunk(raw, rec.filter_mask, None, elem_size, None)?;
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

    // -----------------------------------------------------------------------
    // Fill value regression tests
    // -----------------------------------------------------------------------

    /// Sparse chunk is filled with the declared fill value instead of zero.
    #[test]
    fn test_assemble_slice_fill_value_sparse_chunk() {
        // 8-element u8 dataset, chunk size 4.
        // Only chunk at offset 4 is present ([40, 41, 42, 43]).
        // Chunk at offset 0 is absent (sparse).
        // Fill value = 0xFF.
        let mut file = vec![0u8; 64];
        let c1 = make_chunk(0, vec![4], &[40_u8, 41, 42, 43], &mut file);

        let fill: [u8; 1] = [0xFF];
        let r: std::ops::Range<u64> = 0..8;
        let ranges = [r];

        let result = assemble_chunks_slice(
            &[c1],
            &file,
            &[4u64], // chunk_dims
            &[8u64], // dataset_dims
            &ranges,
            SliceElemConfig {
                elem_size: 1,
                fill_value: Some(&fill),
            },
            no_filter,
        )
        .expect("fill value slice failed");

        // Elements 0..3 should be 0xFF (fill, sparse chunk).
        assert_eq!(&result[0..4], &[0xFF, 0xFF, 0xFF, 0xFF], "sparse fill");
        // Elements 4..7 from the present chunk.
        assert_eq!(&result[4..8], &[40, 41, 42, 43], "present chunk");
    }

    /// `assemble_chunks_with_fill` initialises output buffer with fill value.
    #[test]
    fn test_assemble_with_fill_init() {
        // 6-element dataset, only first 2 elements covered by a chunk.
        // Fill = 0xAB. Chunk at offset 0: [10, 20].
        let mut file = vec![0u8; 64];
        let c0 = make_chunk(0, vec![0], &[10_u8, 20], &mut file);

        let fill: [u8; 1] = [0xAB];
        let result = assemble_chunks_with_fill(
            &[c0],
            &file,
            &[2u64], // chunk_dims
            &[6u64], // dataset_dims
            1,
            Some(&fill),
            no_filter,
        )
        .expect("assemble_with_fill failed");

        // First 2 bytes from the chunk.
        assert_eq!(&result[0..2], &[10, 20]);
        // Remaining 4 bytes: no chunks present, so fill = 0xAB.
        assert_eq!(&result[2..], &[0xAB, 0xAB, 0xAB, 0xAB]);
    }
}
