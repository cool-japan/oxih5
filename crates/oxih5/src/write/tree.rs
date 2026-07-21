//! The writer's object model: a tree of groups, and the paths that navigate it.
//!
//! An HDF5 group is a recursive structure, and this module models it as one.
//! [`GroupNode`] holds datasets, sub-groups **and** attributes; the root group
//! is an ordinary node that happens to be nameless and pinned to a fixed
//! address, so it is planned and emitted by the same code as every other group
//! rather than by a parallel set of special cases.
//!
//! Every public entry point that takes a path resolves it here, through
//! [`split_path`] and [`group_mut`].  That matters twice over:
//!
//! * There is one definition of what a path *means*.  The five
//!   `name.contains('/')` rejections this replaced were five slightly different
//!   definitions of a name, none of which admitted a nested group.
//! * There is one definition of whether a name is *free* — [`name_taken`],
//!   which asks about datasets and sub-groups together.  Three of the old
//!   insertion points asked only about datasets, so a dataset could shadow a
//!   group: two links with the same name in one symbol table, which the format
//!   cannot represent and the reader resolves by whichever it meets first.

use oxih5_core::OxiH5Error;

use super::elem::{AttrDesc, ElemType};

/// Deepest path the writer will accept, counted in components.
///
/// Group nesting is planned, emitted and dropped by recursion, so an unbounded
/// path would turn a user-supplied string into a stack overflow.  Because every
/// group is reached by a path from the root, capping the path also caps the
/// tree: a group at depth *n* can only be created by a call carrying *n*
/// components.  64 is far past anything a real file uses.
pub(super) const MAX_PATH_DEPTH: usize = 64;

// ---------------------------------------------------------------------------
// Dataset descriptor
// ---------------------------------------------------------------------------

/// How a dataset's elements reach the file.
///
/// This replaced a `unlimited: bool` beside a `chunk_shape: Vec<usize>` that was
/// only meaningful when the flag was set — two fields that could disagree, and
/// one of which was silently ignored for a contiguous dataset.  Chunking and
/// unlimited extent are now what they actually are: one choice with its
/// parameters attached, and an unlimited dimension a *property* of chunked
/// storage rather than a synonym for it.
pub(crate) enum Storage {
    /// One unbroken run of raw bytes, addressed directly by the layout message.
    Contiguous,
    /// Tiled into chunks and indexed by a B-tree — the only layout HDF5 lets a
    /// filter or an unlimited dimension apply to.
    Chunked {
        /// Chunk extent per dimension, as the caller supplied it.  A short
        /// vector is completed from the dataset shape; see
        /// [`super::chunked::chunk_shape_of`], which is the single definition of
        /// what a chunked dataset's tiles actually measure.
        chunk_shape: Vec<usize>,
        /// Dimension 0 gets `max_dim[0] = u64::MAX` in the dataspace message.
        unlimited_dim0: bool,
    },
}

/// A filter applied to every chunk on its way to disk.
///
/// One variant today; the enum exists so that the *pipeline* — a filter list
/// with an order that has to be inverted on read — has somewhere to grow into
/// without another round of `Option<bool>` fields.
#[derive(Clone, Copy)]
pub(crate) enum Filter {
    /// zlib DEFLATE, HDF5 filter id 1, at the given level (`0..=9`).
    Deflate {
        /// zlib compression level, recorded in the pipeline message's client
        /// data exactly as libhdf5 records it.
        level: u8,
    },
}

/// One dataset, as the caller described it.
pub(crate) struct DatasetDesc {
    /// Link name within its group; never empty and never contains `'/'`.
    pub(crate) name: String,
    /// Raw little-endian payload, empty for a vlen-string dataset.
    pub(crate) raw: Vec<u8>,
    /// Current extent of each dimension.
    pub(crate) shape: Vec<usize>,
    /// On-disk element type.
    pub(crate) elem_type: ElemType,
    /// Attributes attached to the dataset's object header.
    pub(crate) attrs: Vec<AttrDesc>,
    /// Contiguous or chunked, and with what geometry.
    pub(crate) storage: Storage,
    /// Filter applied to every chunk; `None` for an unfiltered dataset.
    ///
    /// Only ever `Some` alongside [`Storage::Chunked`]: HDF5 has nowhere to
    /// record a per-chunk filtered length for a contiguous dataset, so
    /// [`super::FileWriter::set_deflate`] converts the storage as it sets this.
    pub(crate) filter: Option<Filter>,
    /// W0d: strings for a VLen-string dataset.  `None` for all other types.
    /// When `Some`, `elem_type` must be `ElemType::VlenStr`.
    pub(crate) vlen_strings: Option<Vec<String>>,
}

impl DatasetDesc {
    /// Byte size of the dataset's *unfiltered* element data.
    ///
    /// For VlenStr datasets this is `n_strings × 16` (one 16-byte global-heap
    /// reference per element); for all other datasets it is `raw.len()`.  A
    /// chunked dataset's on-disk footprint is **not** this — see
    /// [`super::payload::Payload`], which is what the layout pass reserves.
    pub(crate) fn data_len(&self) -> usize {
        match &self.vlen_strings {
            Some(strings) => strings.len() * 16,
            None => self.raw.len(),
        }
    }

    /// The chunk shape and unlimited flag, for a chunked dataset.
    pub(crate) fn chunked(&self) -> Option<(&[usize], bool)> {
        match &self.storage {
            Storage::Contiguous => None,
            Storage::Chunked {
                chunk_shape,
                unlimited_dim0,
            } => Some((chunk_shape, *unlimited_dim0)),
        }
    }

    /// Does dimension 0 have an unlimited maximum extent?
    pub(crate) fn unlimited_dim0(&self) -> bool {
        matches!(
            self.storage,
            Storage::Chunked {
                unlimited_dim0: true,
                ..
            }
        )
    }
}

// ---------------------------------------------------------------------------
// Group node
// ---------------------------------------------------------------------------

/// One group of the file: its datasets, its sub-groups, and its attributes.
///
/// The root group is a `GroupNode` whose name is empty.  Nothing else
/// distinguishes it: it carries attributes like any group, its links are sorted
/// and chunked like any group's, and only its object-header *address* is fixed
/// by the format.
pub(crate) struct GroupNode {
    /// Link name within the parent group; empty for the root group.
    pub(crate) name: String,
    /// Datasets directly in this group, in declaration order.
    pub(crate) datasets: Vec<DatasetDesc>,
    /// Sub-groups, in creation order.
    pub(crate) groups: Vec<GroupNode>,
    /// Attributes attached to this group's object header.
    pub(crate) attrs: Vec<AttrDesc>,
}

impl GroupNode {
    /// Create an empty group named `name`; pass `""` for the root group.
    pub(super) fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            datasets: Vec::new(),
            groups: Vec::new(),
            attrs: Vec::new(),
        }
    }
}

/// Is `name` already used by a dataset or a sub-group of `node`?
///
/// Both kinds become links in the same symbol table, where one name can only
/// mean one thing, so both kinds have to be consulted at every insertion point.
pub(super) fn name_taken(node: &GroupNode, name: &str) -> bool {
    node.datasets.iter().any(|ds| ds.name == name) || node.groups.iter().any(|grp| grp.name == name)
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// Split an HDF5 path into its components.
///
/// A single leading `/` is optional and changes nothing: paths are always
/// resolved from the root group.  `"/"` and `""` both name the root group
/// itself and yield no components.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if a component is empty (`"a//b"`, `"a/"`,
/// `"//a"`), if a component is `"."` or `".."` — the writer has no notion of a
/// current or parent directory and silently treating them as names would create
/// links no reader can address — or if the path is deeper than
/// [`MAX_PATH_DEPTH`].
pub(super) fn split_path(path: &str) -> Result<Vec<&str>, OxiH5Error> {
    // Exactly one leading separator is stripped, so `"//a"` still carries an
    // empty component and is rejected below rather than silently accepted.
    let rest = path.strip_prefix('/').unwrap_or(path);
    if rest.is_empty() {
        return Ok(Vec::new());
    }

    let segments: Vec<&str> = rest.split('/').collect();
    if segments.len() > MAX_PATH_DEPTH {
        return Err(OxiH5Error::Format(format!(
            "path '{path}' is {} components deep, over the writer's limit of {MAX_PATH_DEPTH}",
            segments.len()
        )));
    }
    for segment in &segments {
        if segment.is_empty() {
            return Err(OxiH5Error::Format(format!(
                "path '{path}' has an empty component"
            )));
        }
        if *segment == "." || *segment == ".." {
            return Err(OxiH5Error::Format(format!(
                "path '{path}' has a '{segment}' component, which the writer does not resolve"
            )));
        }
    }
    Ok(segments)
}

/// Resolve `segments` to a group, relative to `node`.
///
/// With `create = true` this is h5py's `create_intermediate_group=True`: every
/// missing component becomes a new, empty group.  With `create = false` a
/// missing component is an error, which is what attribute lookup wants — an
/// attribute on a path that does not exist is a caller mistake, not a request
/// to build the path.
///
/// # Errors
///
/// Returns `OxiH5Error::NotFound` if a component does not exist and `create` is
/// false, and `OxiH5Error::Format` if a component names an existing dataset,
/// which can never also be a group.
pub(super) fn group_mut<'a>(
    node: &'a mut GroupNode,
    segments: &[&str],
    create: bool,
) -> Result<&'a mut GroupNode, OxiH5Error> {
    let Some((head, tail)) = segments.split_first() else {
        return Ok(node);
    };

    let index = match node.groups.iter().position(|grp| grp.name == *head) {
        Some(index) => index,
        None => {
            if !create {
                return Err(OxiH5Error::NotFound(format!("group '{head}' not found")));
            }
            if node.datasets.iter().any(|ds| ds.name == *head) {
                return Err(OxiH5Error::Format(format!(
                    "'{head}' is a dataset and cannot also be a group"
                )));
            }
            node.groups.push(GroupNode::new(head));
            node.groups.len() - 1
        }
    };

    match node.groups.get_mut(index) {
        Some(child) => group_mut(child, tail, create),
        // Unreachable: `index` was either found in, or just pushed onto, this
        // very vector.  Reported rather than indexed so that a future edit
        // cannot turn it into a panic.
        None => Err(OxiH5Error::Format(format!(
            "internal writer error: group '{head}' vanished during path resolution"
        ))),
    }
}

/// Resolve `path` to the group that will hold a new object and the name it will
/// take there, creating intermediate groups on the way.
///
/// This is the single insertion point behind every `write_dataset_*` /
/// `create_*` entry point, which is what makes the duplicate check uniform.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if `path` is malformed (see [`split_path`]),
/// names nothing at all (`""`, `"/"`), or if the final name is already used by
/// a dataset or a sub-group; `OxiH5Error::NotFound` never escapes, since
/// intermediate groups are created.
pub(super) fn insertion_point<'a, 'p>(
    root: &'a mut GroupNode,
    path: &'p str,
    what: &str,
) -> Result<(&'a mut GroupNode, &'p str), OxiH5Error> {
    let segments = split_path(path)?;
    let Some((&name, parents)) = segments.split_last() else {
        return Err(OxiH5Error::Format(format!("{what} name must not be empty")));
    };

    let parent = group_mut(root, parents, true)?;
    if name_taken(parent, name) {
        return Err(OxiH5Error::Format(format!(
            "'{path}' already exists: a dataset or group named '{name}' is already in that group"
        )));
    }
    Ok((parent, name))
}

/// Resolve `path` to the attribute list of a dataset or a group, at any depth.
///
/// `"/"` and `""` name the root group.  Otherwise the final component is looked
/// up as a **dataset first** and only then as a sub-group: a bare name has
/// always meant a root dataset, and callers that pass one must keep getting
/// one even once a group of the same name could exist — which, thanks to
/// [`name_taken`], it cannot.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if `path` is malformed, and
/// `OxiH5Error::NotFound` if it names no dataset or group.
pub(super) fn attrs_mut<'a>(
    root: &'a mut GroupNode,
    path: &str,
) -> Result<&'a mut Vec<AttrDesc>, OxiH5Error> {
    let segments = split_path(path)?;
    let Some((&name, parents)) = segments.split_last() else {
        return Ok(&mut root.attrs);
    };

    let parent = group_mut(root, parents, false)?;

    // Positions first, borrows second: taking the mutable borrow inside the
    // lookup would hold an immutable borrow of `parent` across it.
    if let Some(index) = parent.datasets.iter().position(|ds| ds.name == name) {
        if let Some(ds) = parent.datasets.get_mut(index) {
            return Ok(&mut ds.attrs);
        }
    }
    if let Some(index) = parent.groups.iter().position(|grp| grp.name == name) {
        if let Some(grp) = parent.groups.get_mut(index) {
            return Ok(&mut grp.attrs);
        }
    }
    Err(OxiH5Error::NotFound(format!(
        "'{path}' names no dataset or group"
    )))
}

/// Resolve `path` to a dataset, at any depth.
///
/// Unlike [`attrs_mut`] this never falls back to a group: a caller that wants to
/// change how a *dataset* is stored has named a dataset or made a mistake, and
/// silently succeeding on a group would leave the request with no effect at all.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if `path` is malformed or names the root group,
/// and `OxiH5Error::NotFound` if it names no dataset — including when it names a
/// group, which is reported as such rather than as a missing object.
pub(super) fn dataset_mut<'a>(
    root: &'a mut GroupNode,
    path: &str,
) -> Result<&'a mut DatasetDesc, OxiH5Error> {
    let segments = split_path(path)?;
    let Some((&name, parents)) = segments.split_last() else {
        return Err(OxiH5Error::Format(
            "the root group is not a dataset".to_string(),
        ));
    };

    let parent = group_mut(root, parents, false)?;
    // Position first, borrow second: taking the mutable borrow inside the lookup
    // would hold an immutable borrow of `parent` across it.
    if let Some(index) = parent.datasets.iter().position(|ds| ds.name == name) {
        if let Some(ds) = parent.datasets.get_mut(index) {
            return Ok(ds);
        }
    }
    if parent.groups.iter().any(|grp| grp.name == name) {
        return Err(OxiH5Error::Format(format!(
            "'{path}' is a group, not a dataset"
        )));
    }
    Err(OxiH5Error::NotFound(format!("'{path}' names no dataset")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> GroupNode {
        GroupNode::new("")
    }

    fn dataset(name: &str) -> DatasetDesc {
        DatasetDesc {
            name: name.to_string(),
            raw: Vec::new(),
            shape: vec![0],
            elem_type: ElemType::U8,
            attrs: Vec::new(),
            storage: Storage::Contiguous,
            filter: None,
            vlen_strings: None,
        }
    }

    #[test]
    fn a_leading_slash_is_optional_and_the_root_has_no_components() {
        assert_eq!(split_path("a/b/c").expect("relative"), vec!["a", "b", "c"]);
        assert_eq!(split_path("/a/b/c").expect("absolute"), vec!["a", "b", "c"]);
        assert_eq!(split_path("x").expect("bare"), vec!["x"]);
        assert!(split_path("/").expect("root").is_empty());
        assert!(split_path("").expect("empty").is_empty());
    }

    #[test]
    fn malformed_paths_are_rejected() {
        for path in ["a//b", "a/", "//a", "a/./b", "a/../b", "/.", ".."] {
            assert!(
                split_path(path).is_err(),
                "'{path}' should have been rejected"
            );
        }
    }

    #[test]
    fn paths_deeper_than_the_cap_are_rejected() {
        let ok = vec!["g"; MAX_PATH_DEPTH].join("/");
        assert_eq!(split_path(&ok).expect("at the cap").len(), MAX_PATH_DEPTH);

        let too_deep = vec!["g"; MAX_PATH_DEPTH + 1].join("/");
        let err = split_path(&too_deep).expect_err("over the cap");
        assert!(
            format!("{err}").contains("over the writer's limit"),
            "{err}"
        );
    }

    #[test]
    fn intermediate_groups_are_created_on_demand() {
        let mut root = tree();
        let deep = group_mut(&mut root, &["a", "b", "c"], true).expect("create");
        deep.datasets.push(dataset("x"));

        // Same walk again must land on the same node, not build a second one.
        let again = group_mut(&mut root, &["a", "b", "c"], true).expect("revisit");
        assert_eq!(again.datasets.len(), 1);
        assert_eq!(root.groups.len(), 1);
        assert_eq!(root.groups[0].name, "a");
    }

    #[test]
    fn lookup_without_create_reports_a_missing_group() {
        let mut root = tree();
        let Err(err) = group_mut(&mut root, &["nope"], false) else {
            panic!("a missing group must not be created");
        };
        assert!(matches!(err, OxiH5Error::NotFound(_)), "{err}");
        assert!(root.groups.is_empty(), "a failed lookup must not mutate");
    }

    #[test]
    fn a_dataset_name_cannot_be_walked_through_as_a_group() {
        let mut root = tree();
        root.datasets.push(dataset("a"));
        let Err(err) = group_mut(&mut root, &["a", "b"], true) else {
            panic!("a dataset must not be walked through");
        };
        assert!(format!("{err}").contains("cannot also be a group"), "{err}");
    }

    #[test]
    fn name_taken_sees_both_kinds() {
        let mut root = tree();
        root.datasets.push(dataset("ds"));
        root.groups.push(GroupNode::new("grp"));
        assert!(name_taken(&root, "ds"));
        assert!(name_taken(&root, "grp"));
        assert!(!name_taken(&root, "other"));
    }

    #[test]
    fn insertion_refuses_a_name_of_either_kind() {
        let mut root = tree();
        root.datasets.push(dataset("ds"));
        root.groups.push(GroupNode::new("grp"));
        assert!(insertion_point(&mut root, "ds", "dataset").is_err());
        assert!(insertion_point(&mut root, "grp", "dataset").is_err());
        assert!(insertion_point(&mut root, "", "dataset").is_err());
        assert!(insertion_point(&mut root, "/", "dataset").is_err());

        let (parent, name) = insertion_point(&mut root, "/a/b/fresh", "dataset").expect("fresh");
        assert_eq!(name, "fresh");
        assert!(parent.datasets.is_empty());
    }

    #[test]
    fn attrs_resolve_to_root_datasets_and_groups_at_any_depth() {
        let mut root = tree();
        {
            let (parent, name) = insertion_point(&mut root, "a/b/x", "dataset").expect("x");
            parent.datasets.push(dataset(name));
        }
        root.datasets.push(dataset("top"));

        // Root group, both spellings.
        assert!(attrs_mut(&mut root, "/").is_ok());
        assert!(attrs_mut(&mut root, "").is_ok());
        // A bare name still finds the root dataset.
        assert!(attrs_mut(&mut root, "top").is_ok());
        assert!(attrs_mut(&mut root, "/top").is_ok());
        // A nested group, and a nested dataset.
        assert!(attrs_mut(&mut root, "a/b").is_ok());
        assert!(attrs_mut(&mut root, "/a/b/x").is_ok());
        // Nothing of that name anywhere.
        assert!(attrs_mut(&mut root, "a/b/nope").is_err());
        assert!(attrs_mut(&mut root, "nope/x").is_err());
    }

    /// Dataset lookup is stricter than attribute lookup: it never settles for a
    /// group, and it says so rather than reporting the object as missing.
    #[test]
    fn dataset_lookup_refuses_groups_and_the_root() {
        let mut root = tree();
        {
            let (parent, name) = insertion_point(&mut root, "a/b/x", "dataset").expect("x");
            parent.datasets.push(dataset(name));
        }
        root.datasets.push(dataset("top"));
        root.groups.push(GroupNode::new("grp"));

        assert_eq!(dataset_mut(&mut root, "top").expect("top").name, "top");
        assert_eq!(dataset_mut(&mut root, "/a/b/x").expect("nested").name, "x");

        let Err(err) = dataset_mut(&mut root, "grp") else {
            panic!("a group is not a dataset");
        };
        assert!(format!("{err}").contains("is a group"), "{err}");
        assert!(matches!(
            dataset_mut(&mut root, "/"),
            Err(OxiH5Error::Format(_))
        ));
        assert!(matches!(
            dataset_mut(&mut root, "nope"),
            Err(OxiH5Error::NotFound(_))
        ));
        assert!(matches!(
            dataset_mut(&mut root, "a/b/nope"),
            Err(OxiH5Error::NotFound(_))
        ));
        // A missing *parent* must not be created on the way to a lookup.
        assert!(dataset_mut(&mut root, "fresh/x").is_err());
        assert_eq!(
            root.groups
                .iter()
                .map(|g| g.name.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "grp"],
            "a failed lookup must not create the groups on its path"
        );
    }

    /// A dataset and a group of the same name in one group resolve differently,
    /// so the tie-break has to be pinned even though `name_taken` makes the
    /// situation unreachable through the public API.
    #[test]
    fn attribute_lookup_prefers_a_dataset_over_a_group() {
        let mut root = tree();
        root.groups.push(GroupNode::new("dup"));
        root.datasets.push(dataset("dup"));

        let attrs = attrs_mut(&mut root, "dup").expect("dup");
        attrs.push(AttrDesc {
            name: "marker".to_string(),
            kind: super::super::elem::AttrKind::I32(1),
        });
        assert_eq!(root.datasets[0].attrs.len(), 1, "dataset must win");
        assert!(root.groups[0].attrs.is_empty());
    }
}
