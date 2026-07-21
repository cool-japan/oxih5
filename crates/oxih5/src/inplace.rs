//! In-place dataset overwrite — replace a dataset's data bytes and nothing else.
//!
//! [`crate::FileWriter`] builds a file from scratch: whatever the caller does not
//! describe does not survive.  That is fine for files oxih5 authored, but it is
//! destructive for files it did not.  A MATLAB v7.3 `.mat` file (HDF5
//! underneath) carries `MATLAB_class` attributes, object references, a `#refs#`
//! group, cell arrays and compound structs; rebuilding such a file from an
//! application's in-memory model of "datasets, groups and simple attributes"
//! silently discards all of it.
//!
//! This module is the non-destructive alternative.  It resolves a dataset to its
//! object header, reads the Data Layout message (0x0008) to find where the raw
//! data lives, and writes the caller's bytes at exactly that offset.  Every
//! other byte of the file — superblock, group B-trees, local and global heaps,
//! object headers, attributes, and every other dataset — is left untouched.
//!
//! # The size invariant
//!
//! The supplied buffer must match the dataset's allocated size **exactly**.
//! That single rule is what makes the operation sound: because the byte count
//! never changes, nothing after the data area moves, so every address already
//! recorded elsewhere in the file stays valid.  A short or long buffer is
//! rejected rather than padded or truncated.
//!
//! # What cannot be overwritten this way
//!
//! Only a *contiguous* layout (class 1) is a single flat run of bytes.  Chunked,
//! compact and virtual datasets, filtered (compressed) data, and variable-length
//! data — whose data area holds 16-byte global-heap references rather than the
//! values themselves — are all rejected with a typed error.  See
//! [`write_dataset_in_place`] for the full list.

use oxih5_core::{ByteOrder, Dtype, OxiH5Error};
use oxih5_format::header;
use oxih5_format::message::{self, LayoutInfo};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

// ---------------------------------------------------------------------------
// DataExtent
// ---------------------------------------------------------------------------

/// The byte range a dataset's raw data occupies inside an HDF5 file.
///
/// Returned by [`dataset_data_extent`].  A successful lookup means the dataset
/// is overwritable in place, and that [`write_dataset_in_place`] will accept a
/// buffer of exactly [`DataExtent::size`] bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataExtent {
    /// Absolute byte offset of the dataset's first data byte within the file.
    pub address: u64,
    /// Number of bytes allocated to the dataset's data on disk.
    pub size: u64,
}

/// A resolved overwrite target: where the bytes live and how they are typed.
struct Target {
    extent: DataExtent,
    dtype: Dtype,
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

/// Build the error reported for a layout that is not a flat run of bytes.
fn unsupported_layout(dataset_path: &str, layout: &str, class: u8) -> OxiH5Error {
    OxiH5Error::NotImplemented(format!(
        "dataset '{dataset_path}' uses the {layout} layout (class {class}); in-place overwrite \
         requires a contiguous layout (class 1), whose data is a single flat run of bytes"
    ))
}

/// Resolve `dataset_path` to the file region that may be overwritten.
///
/// Rejects, in this order: a path naming a group; a filter pipeline; a
/// variable-length datatype; a non-contiguous layout; and an extent that does
/// not lie wholly inside the file.  Filters are checked before the layout
/// because a filter pipeline defeats a fixed-size raw overwrite whatever the
/// layout turns out to be, so naming the filter is the more useful diagnostic.
fn resolve_target(path: &Path, dataset_path: &str) -> Result<Target, OxiH5Error> {
    // A read-only mapping: only the pages holding the object header are faulted
    // in, so resolving a dataset in a multi-gigabyte .mat file costs a handful
    // of page faults rather than a full read.  The mapping is dropped when this
    // function returns — before `write_at` reopens the file for writing — which
    // is what keeps `open_mmap`'s "not modified while mapped" contract intact.
    let file = crate::open_mmap(path)?;
    let bytes: &[u8] = &file.data;
    let header_addr = file.resolve_dataset_header_addr(dataset_path)?;
    let messages = header::parse_messages(bytes, header_addr)?;

    // Groups carry a Symbol Table (0x0011) or Link Info (0x0002) message and
    // have no data area of their own.
    if messages
        .iter()
        .any(|m| m.msg_type == 0x0011 || m.msg_type == 0x0002)
    {
        return Err(OxiH5Error::TypeMismatch);
    }

    let mut datatype = None;
    let mut layout = None;
    let mut pipeline = None;
    for msg in &messages {
        match msg.msg_type {
            0x0003 => datatype = Some(message::parse_datatype(&msg.data)?),
            0x0008 => layout = Some(message::parse_layout(&msg.data)?),
            0x000B => pipeline = Some(message::parse_filter_pipeline(&msg.data)?),
            _ => {}
        }
    }

    if let Some(p) = &pipeline {
        if !p.filters.is_empty() {
            let ids: Vec<String> = p.filters.iter().map(|f| f.id.to_string()).collect();
            return Err(OxiH5Error::NotImplemented(format!(
                "dataset '{dataset_path}' has a filter pipeline (filter ids {}); filtered data \
                 cannot be overwritten with raw bytes at a fixed size",
                ids.join(", ")
            )));
        }
    }

    let dtype = datatype
        .ok_or_else(|| {
            OxiH5Error::Format(format!("no datatype message in dataset '{dataset_path}'"))
        })?
        .dtype;

    // Variable-length data stores 16-byte global-heap references in the data
    // area; overwriting those with values would orphan the heap objects.
    if crate::reader::is_vlen_dtype(&dtype) {
        return Err(OxiH5Error::NotImplemented(format!(
            "dataset '{dataset_path}' has variable-length datatype {dtype}; its data area holds \
             global-heap references rather than the values themselves"
        )));
    }

    let layout = layout.ok_or_else(|| {
        OxiH5Error::Format(format!("no layout message in dataset '{dataset_path}'"))
    })?;

    let (address, size) = match layout {
        LayoutInfo::Contiguous {
            data_address,
            data_size,
        } => (data_address, data_size),
        LayoutInfo::Compact { .. } => return Err(unsupported_layout(dataset_path, "compact", 0)),
        LayoutInfo::Chunked { .. } => return Err(unsupported_layout(dataset_path, "chunked", 2)),
        LayoutInfo::VirtualDataset { .. } => {
            return Err(unsupported_layout(dataset_path, "virtual", 3))
        }
    };

    // The write must land wholly inside the existing file: an extent running
    // past the end would grow the file, which is exactly what this API promises
    // never to do.
    let file_len = bytes.len() as u64;
    let end = address.checked_add(size).ok_or_else(|| {
        OxiH5Error::Format(format!(
            "dataset '{dataset_path}': data extent {address}+{size} overflows a 64-bit offset"
        ))
    })?;
    if end > file_len {
        return Err(OxiH5Error::Format(format!(
            "dataset '{dataset_path}': data extent {address}+{size} exceeds file size {file_len}"
        )));
    }

    Ok(Target {
        extent: DataExtent { address, size },
        dtype,
    })
}

/// Enforce the size invariant: supplied bytes must equal allocated bytes.
fn check_len(dataset_path: &str, supplied: usize, allocated: u64) -> Result<(), OxiH5Error> {
    if supplied as u64 != allocated {
        return Err(OxiH5Error::Format(format!(
            "dataset '{dataset_path}': in-place overwrite needs exactly {allocated} bytes but \
             {supplied} bytes were supplied; the two must match so that no address in the file \
             can shift"
        )));
    }
    Ok(())
}

/// Seek to `extent.address` and write `data`, touching no other byte.
///
/// Opened with `write(true)` alone — no `create`, no `truncate`, no `append` —
/// so the file's length and every byte outside `[address, address + len)` are
/// preserved.  Ordinary seek+write file I/O is used deliberately: a writable
/// memory map would require `unsafe`, and this crate is `#![deny(unsafe_code)]`
/// apart from the single read-only mapping in [`crate::open_mmap`].
fn write_at(path: &Path, extent: DataExtent, data: &[u8]) -> Result<(), OxiH5Error> {
    let mut file = std::fs::OpenOptions::new().write(true).open(path)?;
    file.seek(SeekFrom::Start(extent.address))?;
    file.write_all(data)?;
    file.flush()?;
    file.sync_all()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API — byte level
// ---------------------------------------------------------------------------

/// Locate the on-disk data range of a contiguous dataset.
///
/// Useful for sizing a buffer before calling [`write_dataset_in_place`], and for
/// asking whether a given dataset is overwritable at all: this function applies
/// every precondition the overwrite applies except the length check, so `Ok`
/// means an overwrite of exactly [`DataExtent::size`] bytes will be accepted.
///
/// # Errors
///
/// Returns [`OxiH5Error::NotFound`] if `dataset_path` does not exist;
/// [`OxiH5Error::TypeMismatch`] if it names a group rather than a dataset;
/// [`OxiH5Error::NotImplemented`] if the dataset is chunked, compact or virtual,
/// carries a filter pipeline, or has a variable-length datatype;
/// [`OxiH5Error::Format`] if the file is malformed or the recorded data extent
/// does not lie inside it; and [`OxiH5Error::Io`] if the file cannot be opened.
pub fn dataset_data_extent<P: AsRef<Path>>(
    path: P,
    dataset_path: &str,
) -> Result<DataExtent, OxiH5Error> {
    Ok(resolve_target(path.as_ref(), dataset_path)?.extent)
}

/// Overwrite a contiguous dataset's raw bytes in place, changing nothing else.
///
/// `data` must be exactly as long as the dataset's allocated data area.  Because
/// the length is unchanged, no address recorded anywhere in the file can shift:
/// attributes, object references, the `#refs#` group of a MATLAB v7.3 file, and
/// every other dataset survive untouched.  This is the operation the C `hdf5`
/// crate spells `Dataset::write_raw`.
///
/// The bytes are written verbatim, so the caller is responsible for encoding
/// them in the dataset's own datatype and byte order.  The
/// `write_dataset_in_place_*` wrappers (e.g. [`write_dataset_in_place_f64`]) do
/// that encoding, and additionally verify the element type.
///
/// ```no_run
/// # fn main() -> Result<(), oxih5::OxiH5Error> {
/// let bytes: Vec<u8> = [1.0f64, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect();
/// oxih5::write_dataset_in_place("measurements.mat", "/results/values", &bytes)?;
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// Returns [`OxiH5Error::NotFound`] if `dataset_path` does not exist;
/// [`OxiH5Error::TypeMismatch`] if it names a group rather than a dataset;
/// [`OxiH5Error::NotImplemented`] if the dataset's layout is chunked, compact or
/// virtual (none of which is a flat run of bytes), if it carries a filter
/// pipeline (message 0x000B — compressed data cannot be overwritten with raw
/// bytes at a fixed size), if its datatype is variable-length (the data area
/// holds global-heap references, not values), or if `dataset_path` reaches the
/// dataset through an **external link** (the target lives in another file and so
/// has no address in this one — reopen that file and overwrite it directly);
/// [`OxiH5Error::Format`] if `data.len()` differs from the dataset's allocated
/// size, or the file is malformed; and [`OxiH5Error::Io`] if the file cannot be
/// opened for writing.
///
/// On any error the file is left completely unmodified — every check runs before
/// the file is opened for writing.
pub fn write_dataset_in_place<P: AsRef<Path>>(
    path: P,
    dataset_path: &str,
    data: &[u8],
) -> Result<(), OxiH5Error> {
    let path = path.as_ref();
    let target = resolve_target(path, dataset_path)?;
    check_len(dataset_path, data.len(), target.extent.size)?;
    write_at(path, target.extent, data)
}

// ---------------------------------------------------------------------------
// Public API — typed wrappers
// ---------------------------------------------------------------------------

/// Element class a typed wrapper requires of the dataset's on-disk datatype.
#[derive(Clone, Copy)]
enum Expect {
    Float,
    Signed,
    Unsigned,
}

/// Check that `dtype` is the `width`-byte `want` class, returning its byte order.
fn expect_elem(dtype: &Dtype, want: Expect, width: usize) -> Result<ByteOrder, OxiH5Error> {
    match (want, dtype) {
        (Expect::Float, Dtype::Float { size, order }) if *size == width => Ok(*order),
        (
            Expect::Signed,
            Dtype::Int {
                size,
                signed: true,
                order,
            },
        ) if *size == width => Ok(*order),
        (
            Expect::Unsigned,
            Dtype::Int {
                size,
                signed: false,
                order,
            },
        ) if *size == width => Ok(*order),
        _ => Err(OxiH5Error::TypeMismatch),
    }
}

macro_rules! typed_in_place {
    ($name:ident, $ty:ty, $want:expr, $desc:literal) => {
        #[doc = concat!(
            "Overwrite a contiguous `", $desc, "` dataset in place with `values`.\n",
            "\n",
            "The dataset's on-disk element type must be `", $desc, "`, which catches the\n",
            "\"right size, wrong type\" mistake that [`write_dataset_in_place`] cannot see.\n",
            "The values are encoded in the dataset's *own* byte order, so a big-endian file\n",
            "is written correctly without the caller having to know.\n",
            "\n",
            "# Errors\n",
            "\n",
            "Returns [`OxiH5Error::TypeMismatch`] if `dataset_path` names a group, or if the\n",
            "dataset's element type is not `", $desc, "`.  Otherwise the same errors as\n",
            "[`write_dataset_in_place`]: [`OxiH5Error::NotFound`] if the path does not exist;\n",
            "[`OxiH5Error::NotImplemented`] for a chunked, compact or virtual layout, a filter\n",
            "pipeline, or a variable-length datatype; [`OxiH5Error::Format`] if `values` does\n",
            "not fill the dataset exactly; [`OxiH5Error::Io`] on I/O failure.\n",
            "\n",
            "On any error the file is left completely unmodified.",
        )]
        pub fn $name<P: AsRef<Path>>(
            path: P,
            dataset_path: &str,
            values: &[$ty],
        ) -> Result<(), OxiH5Error> {
            let path = path.as_ref();
            let target = resolve_target(path, dataset_path)?;
            let order = expect_elem(&target.dtype, $want, std::mem::size_of::<$ty>())?;
            let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<$ty>());
            for v in values {
                match order {
                    ByteOrder::Little => bytes.extend_from_slice(&v.to_le_bytes()),
                    ByteOrder::Big => bytes.extend_from_slice(&v.to_be_bytes()),
                }
            }
            check_len(dataset_path, bytes.len(), target.extent.size)?;
            write_at(path, target.extent, &bytes)
        }
    };
}

typed_in_place!(write_dataset_in_place_f32, f32, Expect::Float, "f32");
typed_in_place!(write_dataset_in_place_f64, f64, Expect::Float, "f64");
typed_in_place!(write_dataset_in_place_i8, i8, Expect::Signed, "i8");
typed_in_place!(write_dataset_in_place_i16, i16, Expect::Signed, "i16");
typed_in_place!(write_dataset_in_place_i32, i32, Expect::Signed, "i32");
typed_in_place!(write_dataset_in_place_i64, i64, Expect::Signed, "i64");
typed_in_place!(write_dataset_in_place_u8, u8, Expect::Unsigned, "u8");
typed_in_place!(write_dataset_in_place_u16, u16, Expect::Unsigned, "u16");
typed_in_place!(write_dataset_in_place_u32, u32, Expect::Unsigned, "u32");
typed_in_place!(write_dataset_in_place_u64, u64, Expect::Unsigned, "u64");

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FileWriter;

    /// A three-dataset file: `alpha` (the overwrite target) flanked by two
    /// siblings whose bytes must survive untouched.
    fn build_trio(tmp: &Path) {
        FileWriter::new()
            .write_dataset_f64("alpha", &[1.0, 2.0, 3.0, 4.0], &[4])
            .expect("alpha")
            .write_dataset_i32("beta", &[10, 20, 30], &[3])
            .expect("beta")
            .write_dataset_f32("gamma", &[0.5f32, 1.5], &[2])
            .expect("gamma")
            .build(tmp)
            .expect("build");
    }

    #[test]
    fn test_in_place_round_trip_leaves_siblings_intact() {
        let tmp = std::env::temp_dir().join("oxih5_inplace_round_trip.h5");
        build_trio(&tmp);

        let new_bytes: Vec<u8> = [9.0f64, 8.0, 7.0, 6.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        write_dataset_in_place(&tmp, "alpha", &new_bytes).expect("overwrite");

        let file = crate::open(&tmp).expect("reopen");
        let alpha = file.dataset("alpha").expect("alpha").as_f64().expect("f64");
        let beta = file.dataset("beta").expect("beta").as_i32().expect("i32");
        let gamma = file.dataset("gamma").expect("gamma").as_f32().expect("f32");
        drop(file);
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(alpha, vec![9.0, 8.0, 7.0, 6.0], "overwritten values");
        assert_eq!(beta, vec![10, 20, 30], "sibling 'beta' must be unchanged");
        assert_eq!(gamma, vec![0.5, 1.5], "sibling 'gamma' must be unchanged");
    }

    /// The strongest statement of the contract: the file differs before and
    /// after *only* inside the target dataset's own data range.
    #[test]
    fn test_in_place_disturbs_only_the_data_range() {
        let tmp = std::env::temp_dir().join("oxih5_inplace_byte_proof.h5");
        build_trio(&tmp);

        let extent = dataset_data_extent(&tmp, "alpha").expect("extent");
        let before = std::fs::read(&tmp).expect("read before");

        let new_bytes: Vec<u8> = [-1.0f64, -2.0, -3.0, -4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        write_dataset_in_place(&tmp, "alpha", &new_bytes).expect("overwrite");

        let after = std::fs::read(&tmp).expect("read after");
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(before.len(), after.len(), "file size must not change");
        let lo = extent.address as usize;
        let hi = lo + extent.size as usize;
        let differing: Vec<usize> = (0..before.len())
            .filter(|&i| before[i] != after[i])
            .collect();
        assert!(
            differing.iter().all(|&i| i >= lo && i < hi),
            "bytes changed outside [{lo}, {hi}): {:?}",
            differing
                .iter()
                .filter(|&&i| i < lo || i >= hi)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            &after[lo..hi],
            &new_bytes[..],
            "data range must hold values"
        );
        assert!(
            !differing.is_empty(),
            "the overwrite must have changed data"
        );
    }

    #[test]
    fn test_in_place_dataset_inside_sub_group() {
        let tmp = std::env::temp_dir().join("oxih5_inplace_subgroup.h5");
        let mut w = FileWriter::new();
        w.create_group("sensors").expect("group");
        w.write_group_dataset_f64("sensors", "temp", &[20.0, 21.0, 22.0], &[3])
            .expect("group ds");
        w.write_group_dataset_i32("sensors", "count", &[7, 8], &[2])
            .expect("sibling");
        w.build(&tmp).expect("build");

        write_dataset_in_place_f64(&tmp, "/sensors/temp", &[30.0, 31.0, 32.0]).expect("overwrite");

        let file = crate::open(&tmp).expect("reopen");
        let temp = file
            .dataset("/sensors/temp")
            .expect("temp")
            .as_f64()
            .expect("f64");
        let count = file
            .dataset("/sensors/count")
            .expect("count")
            .as_i32()
            .expect("i32");
        drop(file);
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(temp, vec![30.0, 31.0, 32.0]);
        assert_eq!(count, vec![7, 8], "sibling in the same group is unchanged");
    }

    #[test]
    fn test_in_place_size_mismatch_rejected() {
        let tmp = std::env::temp_dir().join("oxih5_inplace_size_mismatch.h5");
        build_trio(&tmp);

        // 'alpha' is 4 × f64 = 32 bytes; offer 24.
        let short = vec![0u8; 24];
        let err = write_dataset_in_place(&tmp, "alpha", &short).expect_err("must reject");
        let unchanged = crate::open(&tmp)
            .expect("reopen")
            .dataset("alpha")
            .expect("alpha")
            .as_f64()
            .expect("f64");
        let _ = std::fs::remove_file(&tmp);

        match err {
            OxiH5Error::Format(msg) => {
                assert!(
                    msg.contains("32"),
                    "message must name allocated size: {msg}"
                );
                assert!(msg.contains("24"), "message must name supplied size: {msg}");
            }
            other => panic!("expected Format, got {other:?}"),
        }
        assert_eq!(
            unchanged,
            vec![1.0, 2.0, 3.0, 4.0],
            "file must be untouched"
        );
    }

    #[test]
    fn test_in_place_group_path_rejected() {
        let tmp = std::env::temp_dir().join("oxih5_inplace_group_path.h5");
        let mut w = FileWriter::new();
        w.create_group("sensors").expect("group");
        w.write_group_dataset_f64("sensors", "temp", &[1.0], &[1])
            .expect("group ds");
        w.build(&tmp).expect("build");

        let err = write_dataset_in_place(&tmp, "sensors", &[0u8; 8]).expect_err("must reject");
        let _ = std::fs::remove_file(&tmp);

        assert!(
            matches!(err, OxiH5Error::TypeMismatch),
            "expected TypeMismatch for a group path, got {err:?}"
        );
    }

    #[test]
    fn test_in_place_vlen_string_rejected() {
        let tmp = std::env::temp_dir().join("oxih5_inplace_vlen.h5");
        let mut w = FileWriter::new();
        w.create_vlen_string_dataset("labels", &["one", "two"])
            .expect("vlen");
        w.build(&tmp).expect("build");

        let err = write_dataset_in_place(&tmp, "labels", &[0u8; 32]).expect_err("must reject");
        let _ = std::fs::remove_file(&tmp);

        match err {
            OxiH5Error::NotImplemented(msg) => assert!(
                msg.contains("variable-length"),
                "message must name the datatype class: {msg}"
            ),
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[test]
    fn test_in_place_chunked_layout_rejected() {
        let tmp = std::env::temp_dir().join("oxih5_inplace_chunked.h5");
        let raw: Vec<u8> = [1.0f64, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut w = FileWriter::new();
        w.create_dataset_unlimited(
            "stream",
            &[3],
            &[3],
            &Dtype::Float {
                size: 8,
                order: ByteOrder::Little,
            },
            &raw,
        )
        .expect("chunked");
        w.build(&tmp).expect("build");

        let err = write_dataset_in_place(&tmp, "stream", &raw).expect_err("must reject");
        let _ = std::fs::remove_file(&tmp);

        match err {
            OxiH5Error::NotImplemented(msg) => assert!(
                msg.contains("chunked"),
                "message must name the layout: {msg}"
            ),
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[test]
    fn test_in_place_typed_wrapper_rejects_wrong_dtype() {
        let tmp = std::env::temp_dir().join("oxih5_inplace_wrong_dtype.h5");
        build_trio(&tmp);

        // 'beta' is 3 × i32 = 12 bytes.  Three f32 values are also 12 bytes, so
        // only the dtype check can catch this.
        let err =
            write_dataset_in_place_f32(&tmp, "beta", &[1.0f32, 2.0, 3.0]).expect_err("must reject");
        let unchanged = crate::open(&tmp)
            .expect("reopen")
            .dataset("beta")
            .expect("beta")
            .as_i32()
            .expect("i32");
        let _ = std::fs::remove_file(&tmp);

        assert!(
            matches!(err, OxiH5Error::TypeMismatch),
            "expected TypeMismatch, got {err:?}"
        );
        assert_eq!(unchanged, vec![10, 20, 30], "file must be untouched");
    }

    #[test]
    fn test_in_place_missing_dataset_rejected() {
        let tmp = std::env::temp_dir().join("oxih5_inplace_missing.h5");
        build_trio(&tmp);
        let err = write_dataset_in_place(&tmp, "nope", &[0u8; 8]).expect_err("must reject");
        let _ = std::fs::remove_file(&tmp);
        assert!(
            matches!(err, OxiH5Error::NotFound(_)),
            "expected NotFound, got {err:?}"
        );
    }

    #[test]
    fn test_dataset_data_extent_reports_allocated_size() {
        let tmp = std::env::temp_dir().join("oxih5_inplace_extent.h5");
        build_trio(&tmp);
        let alpha = dataset_data_extent(&tmp, "alpha").expect("alpha extent");
        let beta = dataset_data_extent(&tmp, "beta").expect("beta extent");
        let len = std::fs::metadata(&tmp).expect("metadata").len();
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(alpha.size, 32, "4 × f64");
        assert_eq!(beta.size, 12, "3 × i32");
        assert!(alpha.address + alpha.size <= len, "extent inside the file");
        assert!(
            alpha.address + alpha.size <= beta.address || beta.address + beta.size <= alpha.address,
            "distinct datasets must not overlap: {alpha:?} vs {beta:?}"
        );
    }

    /// Exercise the float / signed / unsigned arms of `expect_elem` and three
    /// different element widths through the typed wrappers.
    #[test]
    fn test_in_place_typed_all_widths_round_trip() {
        let tmp = std::env::temp_dir().join("oxih5_inplace_widths.h5");
        FileWriter::new()
            .write_dataset_f32("f32s", &[1.0f32, 2.0], &[2])
            .expect("f32")
            .write_dataset_u8("u8s", &[3u8, 4], &[2])
            .expect("u8")
            .write_dataset_i64("i64s", &[5i64, 6], &[2])
            .expect("i64")
            .build(&tmp)
            .expect("build");

        write_dataset_in_place_f32(&tmp, "f32s", &[-7.5f32, -8.5]).expect("f32 overwrite");
        write_dataset_in_place_u8(&tmp, "u8s", &[200u8, 201]).expect("u8 overwrite");
        write_dataset_in_place_i64(&tmp, "i64s", &[-700i64, -800]).expect("i64 overwrite");

        let file = crate::open(&tmp).expect("reopen");
        let f32s = file
            .dataset("f32s")
            .expect("f32s")
            .as_f32()
            .expect("as_f32");
        let u8s = file.dataset("u8s").expect("u8s").data.clone();
        let i64s = file
            .dataset("i64s")
            .expect("i64s")
            .as_i64()
            .expect("as_i64");
        drop(file);
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(f32s, vec![-7.5, -8.5]);
        assert_eq!(u8s, vec![200, 201]);
        assert_eq!(i64s, vec![-700, -800]);
    }
}
