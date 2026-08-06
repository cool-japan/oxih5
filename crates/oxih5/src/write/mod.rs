//! HDF5 file writer — attribute, nested-group, link, and dataset support.
//!
//! Produces minimal, valid HDF5 files using superblock v0 and version-1 object
//! headers, with contiguous, chunked **and** compact data layouts.
//!
//! The file is a tree of [`tree::GroupNode`]s rooted at a nameless group, and
//! the root is planned and emitted by exactly the same code as every other
//! group — see [`plan`] for the layout pass and [`build`] for the emit pass.
//!
//! Every entry point takes a *path*, resolved by [`tree::split_path`]; writing
//! to `"/a/b/x"` creates the groups above `x`, matching h5py's
//! `create_intermediate_group=True`.
//!
//! # How a group stores its members
//!
//! Both of HDF5's unrelated encodings, chosen per group:
//!
//! * **Old-style** (the default): a version-1 B-tree over symbol table nodes
//!   plus a local heap — laid out the way libhdf5 lays it out, sorted by name
//!   and chunked into fixed-width nodes indexed by a B-tree that grows a level
//!   whenever one node's children run out (see [`btree_v1`]). Expresses hard
//!   links and soft links, and nothing else.
//! * **New-style** ([`link`]): a Link Info message and one Link message per
//!   member, or — past libhdf5's `max_compact` of 8 — a [`fractal_heap`] of
//!   link messages indexed by a type-5 [`btree_v2`] name index. This is the
//!   only encoding that can hold an **external link** or record **creation
//!   order**, so [`FileWriter::create_external_link`] and
//!   [`FileWriter::set_track_order`] select it, exactly as libhdf5 does.
//!
//! # Datasets
//!
//! A dataset can be DEFLATE-compressed with [`FileWriter::set_deflate`], which
//! converts it to chunked storage and attaches a filter pipeline message; see
//! [`chunked`] for the index that addresses the compressed chunks and
//! [`payload`] for why the compression happens during the layout pass.
//!
//! Element types come in two shapes. [`elem::ElemType`] is the closed, `Copy`
//! set whose datatype message has a fixed length — every numeric family in
//! either byte order, a fixed-length string, a vlen string, a boolean — and
//! [`dtype::EncodedDtype`] is everything whose message length depends on the
//! caller's description: compound records, variable-length sequences, arrays,
//! opaque blobs and bitfields, each encoded once when the dataset is declared.
//! [`tree::DatasetDesc::elem_size`] is the single definition of a dataset's
//! stride across both.
//!
//! Constraints:
//! - Filters: shuffle, DEFLATE and fletcher32, in libhdf5's canonical order;
//!   scaleoffset, nbit and szip have no write path
//! - Attribute types: fixed-length string, f64, i64, i32, their `f64`/`i64`/
//!   string vector forms, any numeric type in either byte order via
//!   [`FileWriter::write_numeric_attr`], and object-reference lists
//! - A dense group's links must fit one fractal-heap direct block, and its name
//!   index is a single B-tree leaf; both refuse rather than grow
//! - A structured datatype's members are inline scalars, not nested compound,
//!   array or variable-length types

mod api_attrs;
mod api_datasets;
mod api_dtypes;
mod api_filters;
mod api_groups;
mod api_links;
mod btree_v1;
mod btree_v2;
mod build;
mod checksum;
mod chunked;
mod dtype;
mod elem;
mod format;
mod fractal_heap;
mod link;
mod oh;
mod payload;
mod pipeline;
mod plan;
mod tree;

#[cfg(test)]
mod golden_tests;

use oxih5_core::OxiH5Error;
use std::path::Path;

use tree::GroupNode;

/// Highest DEFLATE level `oxiarc-deflate` accepts: 0 is stored, 9 is maximum.
const MAX_DEFLATE_LEVEL: u8 = 9;

// ---------------------------------------------------------------------------
// Writer-wide invariants
// ---------------------------------------------------------------------------

/// Round `n` up to the next multiple of 8, HDF5's universal alignment unit.
const fn pad8(n: usize) -> usize {
    (n + 7) & !7
}

/// Assert that an emitter filled exactly the space that was reserved for it.
///
/// The writer computes every absolute address in a first pass and writes at
/// those addresses in a second pass; there is no back-patching.  If a size
/// formula and its corresponding emitter ever disagree by a single byte, two
/// regions overlap and the result is a silently corrupt file.  This check costs
/// one integer comparison per structure and turns that into a typed error, so
/// it is deliberately **not** a `debug_assert!`.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if `wrote != reserved`.
fn check_size(what: &str, wrote: usize, reserved: usize) -> Result<(), OxiH5Error> {
    if wrote != reserved {
        return Err(OxiH5Error::Format(format!(
            "internal writer error: {what} wrote {wrote} bytes into {reserved} reserved"
        )));
    }
    Ok(())
}

/// Narrow `value` into a smaller on-disk integer field.
///
/// # Errors
///
/// Returns `OxiH5Error::Format`, naming `what`, if `value` does not fit.
fn narrow<T: TryFrom<usize>>(what: &str, value: usize) -> Result<T, OxiH5Error> {
    T::try_from(value).map_err(|_| {
        OxiH5Error::Format(format!(
            "internal writer error: {what} is {value}, too large for its on-disk field"
        ))
    })
}

/// Byte length of a dense dataset: `∏ shape × elem_size`, overflow-checked.
///
/// The shape product and the byte size are both formed with `checked_mul`, so a
/// shape whose element count or byte size exceeds `usize` becomes a typed error
/// rather than a debug-profile panic or a release-profile wraparound to a
/// smaller (and silently inconsistent) length. `what` names the caller for the
/// message. This is the single definition every dataset entry point uses to
/// size its raw buffer, so the overflow guard cannot drift between them.
///
/// # Errors
///
/// Returns `OxiH5Error::Format`, naming `what`, if either multiply overflows.
fn checked_byte_len(what: &str, shape: &[usize], elem_size: usize) -> Result<usize, OxiH5Error> {
    let n_elems = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| OxiH5Error::Format(format!("{what}: shape {shape:?} overflows usize")))?;
    n_elems
        .checked_mul(elem_size)
        .ok_or_else(|| OxiH5Error::Format(format!("{what}: byte size overflows usize")))
}

// ---------------------------------------------------------------------------
pub use api_datasets::NumericValues;

// FileWriter — public API
// ---------------------------------------------------------------------------

/// HDF5 file writer.
///
/// Supports:
/// - Contiguous and chunked (unlimited) datasets
/// - Sub-groups nested to any depth
/// - String, f64, i64, i32, `f64[]`, `i64[]`, string-vector and
///   object-reference-list attributes, on datasets **and** on groups
///
/// # Paths
///
/// Every method that names an object takes a path.  A leading `/` is optional
/// and changes nothing — paths are always resolved from the root group — and
/// `"/"` names the root group itself.  Writing to a path **creates the groups
/// above it**, matching h5py's `create_intermediate_group=True`:
///
/// ```no_run
/// use oxih5::FileWriter;
/// let path = std::env::temp_dir().join("nested.h5");
/// let mut w = FileWriter::new();
/// // Creates the groups `a` and `a/b` on the way.
/// w.write_dataset_f64("/a/b/x", &[1.0, 2.0], &[2]).unwrap();
/// w.write_string_attr("/a/b", "units", "km").unwrap();  // on the *group*
/// w.build(&path).unwrap();
/// ```
///
/// A name may be used once per group, by a dataset **or** by a sub-group: both
/// become links in the same symbol table, where one name can only mean one
/// thing.
///
/// # Example (builder pattern)
/// ```no_run
/// use oxih5::FileWriter;
/// let path = std::env::temp_dir().join("example.h5");
/// FileWriter::new()
///     .write_dataset_f32("data", &[1.0f32, 2.0, 3.0], &[3]).unwrap()
///     .build(&path)
///     .unwrap();
/// ```
pub struct FileWriter {
    /// The root group, and through it every group, dataset and attribute.
    root: GroupNode,
}

impl Default for FileWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl FileWriter {
    /// Create a new, empty file writer.
    pub fn new() -> Self {
        Self {
            root: GroupNode::new(""),
        }
    }

    /// The root group, for the layout pass.
    pub(super) fn root_node(&self) -> &GroupNode {
        &self.root
    }

    // -----------------------------------------------------------------------
    // Build
    // -----------------------------------------------------------------------

    /// Write the HDF5 file to disk.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if the file cannot be laid out — see
    /// [`Self::build_to_vec`] — or `OxiH5Error::Io` if it cannot be written.
    pub fn build(&mut self, path: impl AsRef<Path>) -> Result<(), OxiH5Error> {
        let bytes = self.build_bytes()?;
        std::fs::write(path, &bytes).map_err(OxiH5Error::Io)
    }

    /// Serialize the HDF5 file into a byte vector without writing to disk.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if a value overflows an on-disk field, if
    /// an object-reference attribute names a target that is not in the file, or
    /// if a group holds more links than the writer will lay out.
    pub fn build_to_vec(&mut self) -> Result<Vec<u8>, OxiH5Error> {
        self.build_bytes()
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Lay the file out and serialize it.
    ///
    /// See [`plan`] for the layout pass and [`build`] for the emit pass.
    fn build_bytes(&self) -> Result<Vec<u8>, OxiH5Error> {
        build::build_bytes(self)
    }
}
