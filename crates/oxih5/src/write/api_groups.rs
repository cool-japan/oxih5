//! `FileWriter` group-creation and group-scoped convenience methods.
//!
//! `create_group` creates intermediate groups the way h5py does; the
//! `write_group_*` helpers are thin wrappers that join a group path and an
//! object name before delegating to the path-taking methods.  See [`super`]
//! for the overview.

use oxih5_core::OxiH5Error;

use super::tree::{insertion_point, GroupNode};
use super::FileWriter;

impl FileWriter {
    // -----------------------------------------------------------------------
    // Group creation
    // -----------------------------------------------------------------------

    /// Create a sub-group at `path`, creating any groups above it.
    ///
    /// `create_group("a/b/c")` creates `a`, `a/b` and `a/b/c`, matching h5py's
    /// `create_intermediate_group=True`.  Groups implied by a dataset path do
    /// not need to be created first.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `path` is empty or malformed, or if its
    /// final name is already used by a dataset or a group — including when the
    /// group already exists, which mirrors h5py.
    pub fn create_group(&mut self, path: &str) -> Result<(), OxiH5Error> {
        let (parent, name) = insertion_point(&mut self.root, path, "group")?;
        parent.push_group(GroupNode::new(name));
        Ok(())
    }

    /// Add a float64 dataset to the named group.
    ///
    /// A thin wrapper over [`Self::write_dataset_f64`] with the group and the
    /// dataset name given separately.
    ///
    /// # Errors
    ///
    /// As [`Self::write_dataset_f64`] for the joined path.
    pub fn write_group_dataset_f64(
        &mut self,
        group: &str,
        name: &str,
        data: &[f64],
        shape: &[usize],
    ) -> Result<(), OxiH5Error> {
        self.write_dataset_f64(&format!("{group}/{name}"), data, shape)?;
        Ok(())
    }

    /// Add an int32 dataset to the named group.
    ///
    /// A thin wrapper over [`Self::write_dataset_i32`] with the group and the
    /// dataset name given separately.
    ///
    /// # Errors
    ///
    /// As [`Self::write_dataset_i32`] for the joined path.
    pub fn write_group_dataset_i32(
        &mut self,
        group: &str,
        name: &str,
        data: &[i32],
        shape: &[usize],
    ) -> Result<(), OxiH5Error> {
        self.write_dataset_i32(&format!("{group}/{name}"), data, shape)?;
        Ok(())
    }

    /// Write a string attribute on an object inside a named group.
    ///
    /// A thin wrapper over [`Self::write_string_attr`] with the group and the
    /// object name given separately.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`] for the joined path.
    pub fn write_group_string_attr(
        &mut self,
        group_path: &str,
        obj_name: &str,
        attr_name: &str,
        value: &str,
    ) -> Result<(), OxiH5Error> {
        self.write_string_attr(&format!("{group_path}/{obj_name}"), attr_name, value)
    }
}
