//! `FileWriter` link-creation entry points — soft, external and hard-alias.
//!
//! Every other entry point on [`FileWriter`] creates an *object*: a dataset the
//! writer lays out, or a group it lays out.  The three here create a **name**
//! for something else, which is a different thing with three different
//! encodings and one shared constraint.
//!
//! # What the storage style has to do with it
//!
//! An old-style group indexes its members with a version-1 B-tree over symbol
//! table nodes, and a symbol table entry has no link-type field:
//!
//! * a **hard alias** is an ordinary entry whose object header address happens
//!   to be one another entry already names;
//! * a **soft link** is an entry whose *cache type* is 2, whose address is the
//!   undefined sentinel, and whose target path is interned in the group's local
//!   heap;
//! * an **external link** has no encoding at all.
//!
//! So [`FileWriter::create_external_link`] switches its group to link-message
//! storage. That is not a workaround: it is exactly what libhdf5 does — writing
//! an external link into a default (`libver='earliest'`) file converts the
//! whole group to Link messages, superblock v0 and object header v1 unchanged.
//!
//! [`FileWriter::set_track_order`] switches a group for the same kind of reason:
//! a symbol table is sorted by name and has nowhere to record the order its
//! members were created in.

use oxih5_core::OxiH5Error;

use super::tree::{link_insertion_point, split_path, GroupNode, LinkDesc, LinkKind};
use super::FileWriter;

/// Reject a link name that no HDF5 reader could address.
///
/// Path components are already validated by [`split_path`]; this covers the
/// remaining case, an empty final component, which `split_path` cannot see
/// because `""` and `"/"` yield no components at all.
fn check_link_target(what: &str, value: &str) -> Result<(), OxiH5Error> {
    if value.is_empty() {
        return Err(OxiH5Error::Format(format!("{what} must not be empty")));
    }
    if value.contains('\0') {
        return Err(OxiH5Error::Format(format!(
            "{what} contains a NUL byte, which HDF5 stores NUL-terminated and would truncate"
        )));
    }
    Ok(())
}

impl FileWriter {
    /// Create a **soft link** at `path` pointing at `target`.
    ///
    /// A soft link is a stored path, resolved when the file is read: the target
    /// need not exist, and one that does may be replaced later without touching
    /// the link.  `target` may be absolute (`/a/b`) or relative to the group
    /// that holds the link, which is how libhdf5 interprets it.
    ///
    /// The link is stored in whichever way its group is stored — a symbol table
    /// entry with cache type 2, or a Link message — so this never changes a
    /// group's storage style.
    ///
    /// ```no_run
    /// use oxih5::FileWriter;
    /// let path = std::env::temp_dir().join("softlink.h5");
    /// let mut w = FileWriter::new();
    /// w.write_dataset_f64("/data/values", &[1.0, 2.0], &[2]).unwrap();
    /// w.create_soft_link("/latest", "/data/values").unwrap();
    /// w.build(&path).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `path` is malformed or empty, if its
    /// final name is already used by a dataset, group or link in that group, or
    /// if `target` is empty or contains a NUL byte.
    pub fn create_soft_link(&mut self, path: &str, target: &str) -> Result<&mut Self, OxiH5Error> {
        check_link_target("a soft link target", target)?;
        let (parent, name) = link_insertion_point(&mut self.root, path)?;
        parent.push_link(LinkDesc {
            name: name.to_string(),
            kind: LinkKind::Soft {
                path: target.to_string(),
            },
            creation_order: 0,
        });
        Ok(self)
    }

    /// Create an **external link** at `path` into another HDF5 file.
    ///
    /// `file` is resolved relative to the directory of *this* file when the
    /// link is followed, and `target` is a path inside it.
    ///
    /// This moves the holding group to link-message storage, because a version-1
    /// symbol table entry cannot express an external link — the same conversion
    /// libhdf5 performs. Every other member of that group becomes a Link message
    /// too; nothing about them changes on read.
    ///
    /// ```no_run
    /// use oxih5::FileWriter;
    /// let path = std::env::temp_dir().join("external.h5");
    /// let mut w = FileWriter::new();
    /// w.create_external_link("/peer", "other.h5", "/data").unwrap();
    /// w.build(&path).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `path` is malformed or empty, if its
    /// final name is taken, or if `file` or `target` is empty or holds a NUL.
    pub fn create_external_link(
        &mut self,
        path: &str,
        file: &str,
        target: &str,
    ) -> Result<&mut Self, OxiH5Error> {
        check_link_target("an external link file name", file)?;
        check_link_target("an external link target", target)?;
        let (parent, name) = link_insertion_point(&mut self.root, path)?;
        parent.push_link(LinkDesc {
            name: name.to_string(),
            kind: LinkKind::External {
                file: file.to_string(),
                path: target.to_string(),
            },
            creation_order: 0,
        });
        // A symbol table has no encoding for this link, so the group moves.
        parent.require_link_messages();
        Ok(self)
    }

    /// Create a second **hard link** at `path` to the object at `target`.
    ///
    /// Both names then reach the same object header, and that header's
    /// reference count records it — the file says two names exist, which is
    /// what makes deleting one of them safe.
    ///
    /// `target` names an object this writer will lay out, by its path from the
    /// file root; it is resolved once the layout is settled, so it may be
    /// created before *or* after the link.
    ///
    /// ```no_run
    /// use oxih5::FileWriter;
    /// let path = std::env::temp_dir().join("alias.h5");
    /// let mut w = FileWriter::new();
    /// w.create_hard_link("/alias", "/data").unwrap();   // target not yet created
    /// w.write_dataset_i32("/data", &[1, 2, 3], &[3]).unwrap();
    /// w.build(&path).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `path` is malformed or empty, if its
    /// final name is taken, or if `target` is empty or holds a NUL byte.  A
    /// `target` that names no object in the finished file is reported by
    /// [`FileWriter::build`], since objects may be added after the link.
    pub fn create_hard_link(&mut self, path: &str, target: &str) -> Result<&mut Self, OxiH5Error> {
        check_link_target("a hard link target", target)?;
        // Validate the target's shape now, so a malformed path is reported at
        // the call that made it rather than at build time.
        split_path(target)?;
        let (parent, name) = link_insertion_point(&mut self.root, path)?;
        parent.push_link(LinkDesc {
            name: name.to_string(),
            kind: LinkKind::HardAlias {
                target: target.to_string(),
            },
            creation_order: 0,
        });
        Ok(self)
    }

    /// Store the group at `path` as **link messages**, tracking creation order.
    ///
    /// h5py's `track_order=True`.  An old-style group is indexed by name and
    /// has nowhere to record the order its members were created in; a new-style
    /// group gives every link an explicit creation-order field and declares
    /// that it tracks them, so a reader that walks the links in stored order
    /// sees them as they were made rather than alphabetically.
    ///
    /// `"/"` and `""` name the root group.
    ///
    /// Creation order is recorded but not separately *indexed*: this writer
    /// emits no type-6 version-2 B-tree, which is what libhdf5 does for
    /// `track_order` without `H5P_CRT_ORDER_INDEXED` as well.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `path` is malformed and
    /// `OxiH5Error::NotFound` if it names no group.
    pub fn set_track_order(&mut self, path: &str) -> Result<&mut Self, OxiH5Error> {
        let group = group_at(&mut self.root, path)?;
        group.track_order = true;
        group.require_link_messages();
        Ok(self)
    }

    /// Store the group at `path` as **link messages** without tracking order.
    ///
    /// The storage style h5py's `libver='latest'` selects.  Members keep their
    /// name ordering, so nothing about a read changes; what changes is that the
    /// group can then hold link kinds a symbol table cannot express.
    ///
    /// # Errors
    ///
    /// As [`Self::set_track_order`].
    pub fn set_link_storage(&mut self, path: &str) -> Result<&mut Self, OxiH5Error> {
        group_at(&mut self.root, path)?.require_link_messages();
        Ok(self)
    }
}

/// Resolve `path` to an existing group, without creating anything.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if `path` is malformed and
/// `OxiH5Error::NotFound` if it names no group.
fn group_at<'a>(root: &'a mut GroupNode, path: &str) -> Result<&'a mut GroupNode, OxiH5Error> {
    let segments = split_path(path)?;
    super::tree::group_mut(root, &segments, false)
}
