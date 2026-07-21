//! What a dataset's data area holds, and how much room it needs.
//!
//! The writer computes every absolute address in a first pass and writes at
//! those addresses in a second, so the size of a dataset's data has to be known
//! before a byte is emitted.  For an uncompressed dataset that is
//! `raw.len()`; for a compressed one it is the length of the *compressed*
//! stream, which nothing can predict — so compression happens **here**, during
//! planning, and the plan carries the finished chunk images through to pass two.
//!
//! That is also what the chunk B-tree needs: a key records the bytes a chunk
//! occupies on disk, filters applied.  Compressing at emission time would mean
//! writing that key before the number in it existed.
//!
//! # Memory
//!
//! A compressed dataset is held twice while the file is being built: the
//! caller's `raw` bytes live in the [`DatasetDesc`] for the lifetime of the
//! `FileWriter`, and the compressed images live in the plan until `build`
//! returns.  Peak usage is therefore raw + compressed + the output buffer, and
//! the output buffer already holds a copy of everything.  This is deliberate:
//! the alternative is a third pass that re-compresses each chunk to emit it,
//! paying the CPU cost twice to save memory the output buffer has already
//! spent.

use std::borrow::Cow;

use oxih5_core::OxiH5Error;

use super::chunked;
use super::elem::VLEN_REF_SIZE;
use super::pad8;
use super::tree::{DatasetDesc, Filter};

/// One chunk's on-disk image.
pub(super) struct ChunkImage<'a> {
    /// Chunk origin in **elements**, one entry per dataspace dimension.
    pub(super) offsets: Vec<u64>,
    /// The bytes as they appear in the file: the raw tile, or the filter's
    /// output.  Borrowed when the whole dataset is one unfiltered chunk, which
    /// is the common case and copies nothing.
    pub(super) bytes: Cow<'a, [u8]>,
}

/// A dataset's data area.
pub(super) enum Payload<'a> {
    /// One unbroken run of the caller's little-endian bytes.
    Raw(&'a [u8]),
    /// One 16-byte global-heap reference per element.
    VlenRefs {
        /// Number of references, one per string.
        count: usize,
    },
    /// One image per chunk, in the order the B-tree files them.
    Chunked(Vec<ChunkImage<'a>>),
}

impl Payload<'_> {
    /// Bytes this payload occupies in the file, per-chunk padding included.
    ///
    /// The single derivation both passes use: pass one advances the layout
    /// cursor by exactly this, and [`reserve`] hands out chunk addresses from
    /// the same `pad8(len)` rule that produces it.
    pub(super) fn data_size(&self) -> usize {
        match self {
            Payload::Raw(bytes) => pad8(bytes.len()),
            Payload::VlenRefs { count } => pad8(count * VLEN_REF_SIZE),
            Payload::Chunked(images) => images.iter().map(|i| pad8(i.bytes.len())).sum(),
        }
    }
}

/// Build a dataset's payload, compressing it if it carries a filter.
///
/// `chunk_shape` is the completed chunk geometry from
/// [`chunked::chunk_shape_of`], not the caller's possibly-short request.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if the compression level is out of range and
/// `OxiH5Error::Corrupted` if the encoder itself fails — both straight from
/// `oxih5_format::filters::deflate_compress`.
pub(super) fn build<'a>(
    ds: &'a DatasetDesc,
    chunk_shape: &[usize],
) -> Result<Payload<'a>, OxiH5Error> {
    if let Some(strings) = &ds.vlen_strings {
        return Ok(Payload::VlenRefs {
            count: strings.len(),
        });
    }
    if ds.chunked().is_none() {
        return Ok(Payload::Raw(&ds.raw));
    }

    let elem_size = ds.elem_type.byte_size();
    let origins = chunked::chunk_origins(&ds.shape, chunk_shape);
    // One chunk that is exactly the dataset needs no rearranging, so it is
    // borrowed rather than rebuilt.  `cut_tile` would produce the same bytes.
    let verbatim = origins.len() == 1 && chunk_shape == ds.shape.as_slice();

    let mut images = Vec::with_capacity(origins.len());
    for offsets in origins {
        let tile: Cow<'a, [u8]> = if verbatim {
            Cow::Borrowed(ds.raw.as_slice())
        } else {
            Cow::Owned(cut_tile(
                &ds.raw,
                &ds.shape,
                chunk_shape,
                &offsets,
                elem_size,
            ))
        };
        let bytes = match ds.filter {
            None => tile,
            Some(Filter::Deflate { level }) => {
                Cow::Owned(oxih5_format::filters::deflate_compress(&tile, level)?)
            }
        };
        images.push(ChunkImage { offsets, bytes });
    }
    Ok(Payload::Chunked(images))
}

/// Reserve the payload's file space, advancing `current` past it; returns the
/// address of each chunk image, empty unless the payload is chunked.
pub(super) fn reserve(payload: &Payload<'_>, current: &mut usize) -> Vec<usize> {
    let start = *current;
    let mut chunk_addrs = Vec::new();
    if let Payload::Chunked(images) = payload {
        let mut at = start;
        for image in images {
            chunk_addrs.push(at);
            at += pad8(image.bytes.len());
        }
    }
    *current = start + payload.data_size();
    chunk_addrs
}

/// Row-major strides, in elements.
fn strides(dims: &[usize]) -> Vec<usize> {
    let mut out = vec![1usize; dims.len()];
    for d in (0..dims.len().saturating_sub(1)).rev() {
        out[d] = out[d + 1] * dims[d + 1];
    }
    out
}

/// Copy the chunk at `origin` out of `raw` as a **full-size** tile.
///
/// An edge chunk is stored at its full declared volume with the fill value —
/// zero, which is what this writer's fill-value message declares — in the part
/// that hangs over the dataset boundary.  Storing a short tile instead would
/// make the chunk's element count disagree with the chunk dimensions in the
/// layout message, and a reader that trusts the layout would read the next
/// chunk's data as this one's trailing elements.
fn cut_tile(
    raw: &[u8],
    shape: &[usize],
    chunk_shape: &[usize],
    origin: &[u64],
    elem_size: usize,
) -> Vec<u8> {
    let volume: usize = chunk_shape.iter().product();
    let mut tile = vec![0u8; volume * elem_size];
    let dataset_strides = strides(shape);
    let chunk_strides = strides(chunk_shape);

    for flat in 0..volume {
        let mut rest = flat;
        let mut source = 0usize;
        let mut inside = true;
        for d in 0..chunk_shape.len() {
            let local = rest / chunk_strides[d];
            rest %= chunk_strides[d];
            let absolute = origin[d] as usize + local;
            if absolute >= shape[d] {
                inside = false;
                break;
            }
            source += absolute * dataset_strides[d];
        }
        if !inside {
            continue;
        }
        let (from, to) = (source * elem_size, flat * elem_size);
        // Both ends are in range by construction; guarded so that a future
        // change to the geometry cannot turn a mistake into a panic.
        if from + elem_size <= raw.len() && to + elem_size <= tile.len() {
            tile[to..to + elem_size].copy_from_slice(&raw[from..from + elem_size]);
        }
    }
    tile
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::elem::ElemType;
    use crate::write::tree::Storage;

    fn dataset(
        shape: &[usize],
        raw: Vec<u8>,
        storage: Storage,
        filter: Option<Filter>,
    ) -> DatasetDesc {
        DatasetDesc {
            name: "ds".to_string(),
            raw,
            shape: shape.to_vec(),
            elem_type: ElemType::U8,
            attrs: Vec::new(),
            storage,
            filter,
            vlen_strings: None,
        }
    }

    fn chunked_desc(shape: &[usize], raw: Vec<u8>, filter: Option<Filter>) -> DatasetDesc {
        dataset(
            shape,
            raw,
            Storage::Chunked {
                chunk_shape: Vec::new(),
                unlimited_dim0: false,
            },
            filter,
        )
    }

    #[test]
    fn a_contiguous_payload_is_the_caller_bytes() {
        let ds = dataset(&[4], vec![1u8, 2, 3, 4], Storage::Contiguous, None);
        let payload = build(&ds, &[]).expect("build");
        let Payload::Raw(bytes) = &payload else {
            panic!("contiguous storage must not be chunked");
        };
        assert_eq!(*bytes, &[1u8, 2, 3, 4]);
        assert_eq!(payload.data_size(), 8, "4 bytes padded to 8");
    }

    /// One chunk covering the dataset must be the dataset's own bytes, not a
    /// rebuilt copy of them — that equality is what keeps an unfiltered chunked
    /// dataset byte-identical to what the writer emitted before tiling existed.
    #[test]
    fn one_unfiltered_chunk_is_the_dataset_verbatim() {
        let raw: Vec<u8> = (0..12u8).collect();
        let ds = chunked_desc(&[3, 4], raw.clone(), None);
        let payload = build(&ds, &[3, 4]).expect("build");
        let Payload::Chunked(images) = &payload else {
            panic!("chunked storage must produce chunks");
        };
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].offsets, vec![0u64, 0]);
        assert_eq!(images[0].bytes.as_ref(), raw.as_slice());
        assert!(
            matches!(images[0].bytes, Cow::Borrowed(_)),
            "a whole-dataset chunk must be borrowed, not rebuilt"
        );
    }

    /// The tiler is the part that has no second implementation to disagree
    /// with, so it is checked against hand-computed tiles on a shape that does
    /// not divide.
    #[test]
    fn edge_tiles_are_full_size_and_zero_padded() {
        // 1-D: 7 elements, chunks of 3 → [0,1,2] [3,4,5] [6,_,_]
        let raw: Vec<u8> = (1..=7u8).collect();
        assert_eq!(cut_tile(&raw, &[7], &[3], &[0], 1), vec![1, 2, 3]);
        assert_eq!(cut_tile(&raw, &[7], &[3], &[3], 1), vec![4, 5, 6]);
        assert_eq!(cut_tile(&raw, &[7], &[3], &[6], 1), vec![7, 0, 0]);

        // 2-D: shape [3, 3] of 1..9, chunks of [2, 2].
        let raw: Vec<u8> = (1..=9u8).collect();
        assert_eq!(
            cut_tile(&raw, &[3, 3], &[2, 2], &[0, 0], 1),
            vec![1, 2, 4, 5]
        );
        assert_eq!(
            cut_tile(&raw, &[3, 3], &[2, 2], &[0, 2], 1),
            vec![3, 0, 6, 0]
        );
        assert_eq!(
            cut_tile(&raw, &[3, 3], &[2, 2], &[2, 0], 1),
            vec![7, 8, 0, 0]
        );
        assert_eq!(
            cut_tile(&raw, &[3, 3], &[2, 2], &[2, 2], 1),
            vec![9, 0, 0, 0]
        );
    }

    /// Multi-byte elements must move whole, not byte by byte.
    #[test]
    fn tiles_move_whole_elements() {
        // Three u32 values 0x01020304, 0x05060708, 0x090a0b0c.
        let raw: Vec<u8> = vec![4, 3, 2, 1, 8, 7, 6, 5, 12, 11, 10, 9];
        assert_eq!(
            cut_tile(&raw, &[3], &[2], &[0], 4),
            vec![4, 3, 2, 1, 8, 7, 6, 5]
        );
        assert_eq!(
            cut_tile(&raw, &[3], &[2], &[2], 4),
            vec![12, 11, 10, 9, 0, 0, 0, 0],
            "the overhang is fill, not the neighbouring element"
        );
    }

    #[test]
    fn compression_happens_before_anything_is_sized() {
        // Highly compressible: 4 KiB of one repeated byte.
        let raw = vec![0x5Au8; 4096];
        let ds = chunked_desc(&[4096], raw.clone(), Some(Filter::Deflate { level: 6 }));
        let payload = build(&ds, &[4096]).expect("build");
        let Payload::Chunked(images) = &payload else {
            panic!("filtered storage must be chunked");
        };
        assert_eq!(images.len(), 1);
        assert!(
            images[0].bytes.len() < raw.len(),
            "4 KiB of one byte must compress: {} bytes",
            images[0].bytes.len()
        );
        assert_eq!(
            payload.data_size(),
            pad8(images[0].bytes.len()),
            "the reserved size is the *compressed* length"
        );

        // And it really is a zlib stream the reader can invert.
        let back = oxih5_format::filters::inflate_deflate(&images[0].bytes).expect("inflate");
        assert_eq!(back, raw);
    }

    #[test]
    fn an_out_of_range_level_is_reported_not_clamped() {
        let ds = chunked_desc(&[4], vec![0u8; 4], Some(Filter::Deflate { level: 10 }));
        assert!(build(&ds, &[4]).is_err());
    }

    /// A dataset with a zero-length dimension has no chunks and reserves
    /// nothing, so its B-tree has no entries to key.
    #[test]
    fn a_zero_length_dataset_reserves_nothing() {
        let ds = chunked_desc(&[0], Vec::new(), Some(Filter::Deflate { level: 6 }));
        let payload = build(&ds, &[1]).expect("build");
        let Payload::Chunked(images) = &payload else {
            panic!("chunked");
        };
        assert!(images.is_empty());
        assert_eq!(payload.data_size(), 0);

        let mut current = 200usize;
        assert!(reserve(&payload, &mut current).is_empty());
        assert_eq!(current, 200);
    }

    // -----------------------------------------------------------------------
    // W1e: compression, end to end
    //
    // These build real files rather than payloads.  They live beside the
    // compressor instead of in `write/mod.rs` because that file is 140 lines
    // from the project's 2000-line cap, and because what they actually test —
    // that the length reserved during planning is the length written — is this
    // module's contract.
    // -----------------------------------------------------------------------

    /// A compressed dataset must be smaller and read back bit-exact.
    ///
    /// The size assertion is made against the *same file built without the
    /// filter*, not against a remembered number, so it stays true as the rest
    /// of the writer changes — and it prices in the 2096-byte chunk index that
    /// compression drags along, which a chunk-level comparison would hide.
    #[test]
    fn w1e_deflate_roundtrip_f64() {
        let tmp = std::env::temp_dir().join("oxih5_test_w1e_deflate_f64.h5");
        // Repetitive enough to compress; varied enough that a stream truncated
        // to its first block would not still decode to the right values.
        let values: Vec<f64> = (0..4096).map(|i| f64::from(i % 17) * 0.25).collect();

        let mut w = crate::FileWriter::new();
        w.write_dataset_f64("readings", &values, &[4096])
            .expect("readings");
        let uncompressed_len = w.build_to_vec().expect("uncompressed build").len();

        w.set_deflate("readings", 6).expect("set_deflate");
        w.build(&tmp).expect("build");
        let compressed_len = std::fs::metadata(&tmp).expect("stat").len() as usize;

        let f = crate::File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        assert!(
            compressed_len < uncompressed_len,
            "compressed file is {compressed_len} bytes against {uncompressed_len} uncompressed — \
             32 KiB of highly repetitive f64 must beat the chunk index it pays for"
        );
        let ds = f.dataset("readings").expect("readings");
        assert_eq!(ds.shape, vec![4096usize]);
        assert_eq!(
            ds.as_f64().expect("as_f64"),
            values,
            "DEFLATE is lossless; every value must return bit-exact"
        );
    }

    /// Every level is a working codec, and a level out of range is refused
    /// before anything is written.
    #[test]
    fn w1e_deflate_every_level_roundtrips() {
        let values: Vec<i32> = (0..2048).map(|i| i % 5).collect();
        for level in 0..=super::super::MAX_DEFLATE_LEVEL {
            let tmp = std::env::temp_dir().join(format!("oxih5_test_w1e_deflate_level_{level}.h5"));
            let mut w = crate::FileWriter::new();
            w.write_dataset_i32("v", &values, &[2048]).expect("v");
            w.set_deflate("v", level).expect("set_deflate");
            w.build(&tmp).expect("build");

            let f = crate::File::open(&tmp).expect("open");
            let _ = std::fs::remove_file(&tmp);
            assert_eq!(
                f.dataset("v").expect("v").as_i32().expect("as_i32"),
                values,
                "level {level}"
            );
        }

        let mut w = crate::FileWriter::new();
        w.write_dataset_i32("v", &[1], &[1]).expect("v");
        let err = w
            .set_deflate("v", super::super::MAX_DEFLATE_LEVEL + 1)
            .expect_err("level 10 is not a zlib level");
        assert!(format!("{err}").contains("out of range"), "{err}");
    }

    /// A vlen-string dataset must be refused, not accepted and then unreadable.
    ///
    /// `reject_vlen_incompatible_filters` on the read side declines to decode a
    /// filtered vlen dataset, so a writer that accepted this would produce a
    /// file only it could not read back — the worst possible outcome, because
    /// nothing fails until the data is needed.
    #[test]
    fn w1e_vlen_string_deflate_rejected() {
        let mut w = crate::FileWriter::new();
        w.create_vlen_string_dataset("labels", &["alpha", "beta"])
            .expect("labels");
        let err = w
            .set_deflate("labels", 6)
            .expect_err("a vlen-string dataset cannot be filtered");
        assert!(
            format!("{err}").contains("global-heap references"),
            "the error must say why, not just that: {err}"
        );

        // The refusal leaves the dataset exactly as it was.
        let tmp = std::env::temp_dir().join("oxih5_test_w1e_vlen_unfiltered.h5");
        w.build(&tmp).expect("build");
        let f = crate::File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(
            f.dataset_strings("labels").expect("labels"),
            vec!["alpha", "beta"]
        );
    }

    /// `set_deflate` names a **dataset**; every other kind of path is an error
    /// with a reason attached.
    #[test]
    fn w1e_set_deflate_rejects_non_datasets() {
        let mut w = crate::FileWriter::new();
        w.write_dataset_f64("ds", &[1.0], &[1]).expect("ds");
        w.create_group("grp").expect("grp");

        assert!(
            matches!(w.set_deflate("nope", 6), Err(OxiH5Error::NotFound(_))),
            "a missing dataset is NotFound"
        );
        let Err(err) = w.set_deflate("grp", 6) else {
            panic!("a group is not a dataset");
        };
        assert!(format!("{err}").contains("is a group"), "{err}");
        assert!(w.set_deflate("/", 6).is_err(), "nor is the root group");
        assert!(w.set_deflate("a//b", 6).is_err(), "nor a malformed path");
    }

    /// A zero-length dataset compresses to nothing and indexes nothing.
    ///
    /// The interesting part is that it still has to be *valid*: a chunk extent
    /// of 0 is illegal in HDF5 and divides by zero in every chunk reader, so
    /// the geometry has to fall back to 1 while the chunk count stays 0.
    #[test]
    fn w1e_zero_length_chunked_dataset() {
        let tmp = std::env::temp_dir().join("oxih5_test_w1e_zero_length.h5");
        let mut w = crate::FileWriter::new();
        w.write_dataset_i32("empty", &[], &[0]).expect("empty");
        w.set_deflate("empty", 6).expect("set_deflate");
        // A populated dataset after it, so a mis-sized empty one would corrupt
        // something detectable rather than just ending the file early.
        w.write_dataset_f64("after", &[1.5, 2.5], &[2])
            .expect("after");
        w.build(&tmp).expect("build");

        let f = crate::File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let ds = f.dataset("empty").expect("empty");
        assert_eq!(ds.shape, vec![0usize]);
        assert!(ds.as_i32().expect("as_i32").is_empty());
        assert_eq!(
            f.dataset("after").expect("after").as_f64().expect("as_f64"),
            vec![1.5, 2.5]
        );
    }

    /// Compressing an unlimited dataset keeps it unlimited.
    ///
    /// `set_deflate` converts contiguous storage to chunked; an already-chunked
    /// dataset must keep the geometry and the `max_dim[0] = u64::MAX` it was
    /// created with, or a NetCDF record variable would quietly stop being one.
    #[test]
    fn w1e_deflate_preserves_an_unlimited_dimension() {
        let tmp = std::env::temp_dir().join("oxih5_test_w1e_deflate_unlimited.h5");
        let dtype = oxih5_core::Dtype::Float {
            size: 8,
            order: oxih5_core::ByteOrder::Little,
        };
        let values: Vec<f64> = (0..512).map(|i| f64::from(i % 3)).collect();
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut w = crate::FileWriter::new();
        w.create_dataset_unlimited("time", &[512], &[512], &dtype, &raw)
            .expect("time");
        w.set_deflate("time", 9).expect("set_deflate");
        w.build(&tmp).expect("build");

        let f = crate::File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let ds = f.dataset("time").expect("time");
        assert!(ds.is_unlimited(), "the unlimited dimension must survive");
        assert_eq!(ds.as_f64().expect("as_f64"), values);
    }

    // -----------------------------------------------------------------------
    // W1e2: real tiling
    // -----------------------------------------------------------------------

    /// Build a tiled, compressed dataset and read it back.
    ///
    /// `create_dataset_unlimited` is the only entry point that takes a chunk
    /// shape, so it is how a tiled dataset is requested; `set_deflate` then
    /// keeps that geometry rather than replacing it.
    fn tiled_roundtrip_i32(tag: &str, values: &[i32], shape: &[usize], chunk: &[usize]) {
        let tmp = std::env::temp_dir().join(format!("oxih5_test_w1e2_{tag}.h5"));
        let dtype = oxih5_core::Dtype::Int {
            size: 4,
            signed: true,
            order: oxih5_core::ByteOrder::Little,
        };
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut w = crate::FileWriter::new();
        w.create_dataset_unlimited("tiled", shape, chunk, &dtype, &raw)
            .expect("tiled");
        w.set_deflate("tiled", 6).expect("set_deflate");
        w.build(&tmp).expect("build");

        let f = crate::File::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let ds = f.dataset("tiled").expect("tiled");
        assert_eq!(ds.shape, shape.to_vec(), "{tag} shape");
        assert_eq!(
            ds.as_i32().expect("as_i32"),
            values,
            "{tag}: shape {shape:?} in chunks of {chunk:?}"
        );
    }

    /// A 1-D extent that the chunk size does not divide.
    ///
    /// Three chunks of three over seven elements: the last one is stored
    /// full-size with two fill elements hanging past the end.  A writer that
    /// stored a short final tile instead produces a chunk whose element count
    /// disagrees with the layout message, and the reader walks off into the
    /// next chunk's bytes.
    #[test]
    fn w1e2_multi_chunk_roundtrip_ragged_1d() {
        tiled_roundtrip_i32("ragged_1d", &(0..7i32).collect::<Vec<_>>(), &[7], &[3]);
        // Divisible, as a control: the same code path with no overhang at all.
        tiled_roundtrip_i32("even_1d", &(0..9i32).collect::<Vec<_>>(), &[9], &[3]);
    }

    /// A 2-D extent ragged in **both** dimensions: 3 × 2 chunks of `[2, 2]`
    /// over `[5, 3]`, so four of the six chunks hang over an edge.
    ///
    /// This is the shape that catches a row-major/column-major mix-up: a
    /// transposed chunk grid still yields fifteen values, just the wrong
    /// fifteen, and every one of them is a value that belongs in the dataset.
    #[test]
    fn w1e2_multi_chunk_roundtrip_ragged_2d() {
        tiled_roundtrip_i32(
            "ragged_2d",
            &(0..15i32).collect::<Vec<_>>(),
            &[5, 3],
            &[2, 2],
        );
        // A single ragged dimension each way, so a mistake that happens to be
        // symmetric in [5,3]/[2,2] still shows.
        tiled_roundtrip_i32(
            "ragged_rows",
            &(0..15i32).collect::<Vec<_>>(),
            &[5, 3],
            &[2, 3],
        );
        tiled_roundtrip_i32(
            "ragged_cols",
            &(0..15i32).collect::<Vec<_>>(),
            &[5, 3],
            &[5, 2],
        );
    }

    /// More chunks than one B-tree node holds, so the index grows a level.
    ///
    /// 150 chunks span three nodes: two leaves and a root.  A tree whose layout
    /// message pointed at the first leaf instead of the root would return the
    /// first 64 chunks and fill for the rest — which reads as a dataset that is
    /// two-thirds zeroes, not as an error.
    #[test]
    fn w1e2_multi_level_chunk_index_roundtrip() {
        let values: Vec<i32> = (0..300i32).map(|i| i * 3).collect();
        tiled_roundtrip_i32("multi_level", &values, &[300], &[2]);
    }

    /// Chunk addresses and the total reservation come from the same rule, so
    /// the last chunk always ends exactly where the cursor lands.
    #[test]
    fn reserved_chunk_addresses_tile_the_reserved_span() {
        let raw: Vec<u8> = (0..9u8).collect();
        let ds = chunked_desc(&[9], raw, None);
        // Force real tiles by asking for a geometry the payload builder will
        // honour even though `chunk_shape_of` would refuse it today.
        let payload = build(&ds, &[4]).expect("build");
        let Payload::Chunked(images) = &payload else {
            panic!("chunked");
        };
        assert_eq!(images.len(), 3, "ceil(9/4)");
        assert!(images.iter().all(|i| i.bytes.len() == 4), "full-size tiles");

        let mut current = 100usize;
        let addrs = reserve(&payload, &mut current);
        assert_eq!(addrs, vec![100, 108, 116]);
        assert_eq!(current, 100 + payload.data_size());
        assert_eq!(current, 124);
    }
}
