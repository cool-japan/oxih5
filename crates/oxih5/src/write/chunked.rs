//! Chunk geometry, and the B-tree v1 type-1 index that addresses it.
//!
//! Two things live here because they are the same question asked twice: *where
//! does chunk `i` start* decides both the layout message's chunk-dimension
//! vector and the key the B-tree files that chunk under, and the two have to
//! agree or libhdf5 looks for a chunk that is not there.
//!
//! # Node width is not negotiable
//!
//! A B-tree v1 node is read as a fixed-size image whose length libhdf5 computes
//! *before* it has seen the node:
//!
//! ```text
//! sizeof_rnode = 24 + two_k * 8 + (two_k + 1) * sizeof_rkey
//! ```
//!
//! `two_k` is `2 × btree_k[H5B_CHUNK]`, and for a superblock v0 or v1 file that
//! `K` is not read from the file at all — it is `HDF5_BTREE_CHUNK_IK_DEF`, a
//! compile-time **32**.  Those superblocks carry `leaf_node_K` (bytes 16..18)
//! and the *group* B-tree's `internal_node_K` (bytes 18..20), and nothing that
//! describes this tree.  So every chunk B-tree node in a file we write must be
//! [`chunk_node_size`] bytes wide however few chunks it holds, with the slots
//! past `entries_used` zero-filled.
//!
//! Sizing a node for its contents instead is what produced
//! `addr overflow, addr = 3000, size = 2096, eoa = 3112`: an 80-byte node for a
//! one-dimensional dataset, asked for as 2096 bytes, running off the end of the
//! file.  The 2096 in that message is exactly `chunk_node_size(1)`, which is how
//! the constant below was confirmed rather than guessed.

use oxih5_core::OxiH5Error;

use super::format::{fill_zero, write_u16_le, write_u32_le, write_u64_le};
use super::narrow;
use super::tree::DatasetDesc;

/// Children one chunk B-tree node is sized for: `2 × btree_k[H5B_CHUNK]`.
///
/// See the module documentation for why this is a constant and not a field.
pub(super) const CHUNK_NODE_MAX_CHILDREN: usize = 64;

/// Size of the B-tree v1 node header: signature, type, level, entry count, and
/// the two sibling addresses.
const NODE_PREFIX: usize = 24;

/// Size of a child pointer — one 8-byte file address.
const CHILD_SIZE: usize = 8;

/// The undefined-address sentinel, used for both sibling links.
const UNDEFINED_ADDR: u64 = u64::MAX;

/// Size of one chunk key: byte count, filter mask, and one offset per
/// dimension **plus one** — HDF5 stores a trailing element-offset dimension
/// that is always zero.
const fn chunk_key_size(ndims: usize) -> usize {
    8 + (ndims + 1) * 8
}

/// On-disk size of a chunk B-tree node for a dataset of `ndims` real
/// dimensions.
///
/// ```text
/// header (24) | key[0] child[0] … key[63] child[63] | key[64]
/// ```
pub(super) const fn chunk_node_size(ndims: usize) -> usize {
    NODE_PREFIX
        + (CHUNK_NODE_MAX_CHILDREN + 1) * chunk_key_size(ndims)
        + CHUNK_NODE_MAX_CHILDREN * CHILD_SIZE
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// The chunk shape a chunked dataset is actually stored with, in elements.
///
/// A **short** `chunk_shape` is completed from the dataset shape, not from 1s.
/// That is not a convenience: `oxinetcdf` passes `&[shape[0].max(1)]` for a
/// variable of any rank, so a 2-D unlimited variable of shape `[5, 3]` used to
/// *declare* chunk dimensions `[5, 1]` — five elements — while the one chunk on
/// disk held all fifteen.  Every reader clamps a chunk to its declared volume,
/// so twelve of the fifteen values read back as zero, from a file that opened
/// without complaint.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if the dataset has no dimensions — HDF5 has no
/// chunked scalar dataspace — or if a chunk extent is 0, which HDF5 forbids and
/// which divides by zero in every chunk-index reader.
pub(super) fn chunk_shape_of(ds: &DatasetDesc) -> Result<Vec<usize>, OxiH5Error> {
    let Some((requested, _)) = ds.chunked() else {
        return Ok(Vec::new());
    };
    if ds.shape.is_empty() {
        return Err(OxiH5Error::Format(format!(
            "dataset '{}': chunked storage needs at least one dimension",
            ds.name
        )));
    }

    let mut shape = Vec::with_capacity(ds.shape.len());
    for (d, &extent) in ds.shape.iter().enumerate() {
        // A zero-length dimension holds no elements but still needs a chunk
        // extent: HDF5 rejects a chunk dimension of 0, and dividing by one is
        // how a reader finds which chunk a coordinate belongs to.
        let whole = extent.max(1);
        let chunk = requested.get(d).copied().unwrap_or(whole);
        if chunk == 0 {
            return Err(OxiH5Error::Format(format!(
                "dataset '{}': chunk dimension {d} is 0",
                ds.name
            )));
        }
        shape.push(chunk);
    }
    Ok(shape)
}

/// The chunk-dimension vector a chunked layout message stores: one entry per
/// dataspace dimension, followed by the element size, exactly as HDF5 encodes
/// it.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if a chunk extent or the element size does not
/// fit the 32-bit on-disk field.
pub(super) fn layout_chunk_dims(
    chunk_shape: &[usize],
    elem_size: usize,
) -> Result<Vec<u32>, OxiH5Error> {
    let mut dims = Vec::with_capacity(chunk_shape.len() + 1);
    for &extent in chunk_shape {
        dims.push(narrow("chunk dimension", extent)?);
    }
    dims.push(narrow("chunk element size", elem_size)?);
    Ok(dims)
}

/// Number of chunks along each dimension — `ceil(extent / chunk)`, and so 0
/// wherever the extent is 0.
fn chunk_counts(shape: &[usize], chunk_shape: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .zip(chunk_shape)
        .map(|(&extent, &chunk)| extent.div_ceil(chunk))
        .collect()
}

/// The dataset extent rounded **up** to a chunk boundary: the terminal B-tree
/// key.
///
/// `H5B_find` binary-searches with `cmp3(key[i], wanted, key[i+1])`, which
/// reports "found" only while `wanted < key[i+1]`.  The key past the last child
/// is therefore an exclusive upper bound over the whole node, and it has to sit
/// on a chunk boundary because `H5D__btree_decode_key` divides every offset it
/// reads by the chunk extent to recover a scaled coordinate.
pub(super) fn end_offsets(shape: &[usize], chunk_shape: &[usize]) -> Vec<u64> {
    chunk_counts(shape, chunk_shape)
        .iter()
        .zip(chunk_shape)
        .map(|(&count, &chunk)| (count * chunk) as u64)
        .collect()
}

/// Every chunk origin, in elements, in the order the B-tree stores them.
///
/// Row-major over the chunk grid: the last dimension varies fastest.  A dataset
/// with a zero-length dimension has **no** chunks at all — not one empty one —
/// because HDF5 only allocates a chunk once something is written into it.
pub(super) fn chunk_origins(shape: &[usize], chunk_shape: &[usize]) -> Vec<Vec<u64>> {
    let counts = chunk_counts(shape, chunk_shape);
    if counts.contains(&0) {
        return Vec::new();
    }
    let total: usize = counts.iter().product();

    let mut origins = Vec::with_capacity(total);
    for flat in 0..total {
        let mut rest = flat;
        let mut origin = vec![0u64; counts.len()];
        for d in (0..counts.len()).rev() {
            origin[d] = ((rest % counts[d]) * chunk_shape[d]) as u64;
            rest /= counts[d];
        }
        origins.push(origin);
    }
    origins
}

// ---------------------------------------------------------------------------
// The node
// ---------------------------------------------------------------------------

/// One chunk, as the B-tree files it.
pub(super) struct ChunkEntry {
    /// Chunk origin in **elements**, one entry per dataspace dimension.
    ///
    /// Always a multiple of the chunk extent in that dimension:
    /// `H5D__btree_decode_key` divides by the extent and asserts the remainder
    /// is zero.
    pub(super) offsets: Vec<u64>,
    /// File address of the chunk's bytes.
    pub(super) addr: u64,
    /// Bytes the chunk occupies on disk, after any filter.
    pub(super) nbytes: u32,
    /// Per-chunk filter mask; bit *i* set means filter *i* was skipped for this
    /// chunk and must not be inverted on read.
    pub(super) filter_mask: u32,
}

/// One key as it goes on disk.
///
/// At level 0 this describes a chunk; above it, only the `offsets` are ever
/// read — `H5D__btree_cmp3` compares nothing else, and `H5B_iterate` hands keys
/// to its callback only at level 0 — so an internal key carries `nbytes = 0` to
/// say plainly that there is no chunk behind it.
#[derive(Clone, Copy)]
struct NodeKey<'a> {
    /// Bytes the chunk occupies on disk; 0 for a boundary key.
    nbytes: u32,
    /// Per-chunk filter mask; 0 for a boundary key.
    filter_mask: u32,
    /// Chunk origin in elements, one entry per dataspace dimension.
    offsets: &'a [u64],
}

/// Write one chunk key at `at`.
///
/// The trailing element-offset dimension is left at 0, which is what libhdf5
/// writes for a data chunk and what it decodes to a scaled coordinate of 0 for
/// a boundary key as well.
fn write_key(buf: &mut [u8], at: usize, key: NodeKey<'_>) {
    write_u32_le(buf, at, key.nbytes);
    write_u32_le(buf, at + 4, key.filter_mask);
    for (d, &offset) in key.offsets.iter().enumerate() {
        write_u64_le(buf, at + 8 + d * 8, offset);
    }
}

/// Write one chunk B-tree node at `addr`; returns bytes written, always
/// [`chunk_node_size`].
///
/// `keys[i]` is the **inclusive lower bound** of `children[i]` and `terminal`
/// the **exclusive upper bound** of the whole node — the direction
/// `H5D__btree_cmp3` uses, and the opposite of the group B-tree's convention in
/// [`super::btree_v1`], where a key is an exclusive lower bound.  Getting that
/// backwards does not fail loudly; it makes half the chunks unfindable.
///
/// An **empty** node gets no keys at all, not even the terminal one:
/// `H5B__cache_deserialize` decodes the final key only `if (bt->nchildren > 0)`.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if the node holds more children than
/// [`CHUNK_NODE_MAX_CHILDREN`], if the counts do not agree, if the level or
/// child count does not fit its on-disk field, or if a key does not have one
/// offset per dimension.
#[allow(clippy::too_many_arguments)]
fn write_node(
    buf: &mut [u8],
    addr: usize,
    ndims: usize,
    level: usize,
    left_sibling: u64,
    right_sibling: u64,
    children: &[u64],
    keys: &[NodeKey<'_>],
    terminal: NodeKey<'_>,
) -> Result<usize, OxiH5Error> {
    if children.len() > CHUNK_NODE_MAX_CHILDREN || children.len() != keys.len() {
        return Err(OxiH5Error::Format(format!(
            "internal writer error: chunk B-tree node with {} children and {} keys, \
             capacity {CHUNK_NODE_MAX_CHILDREN}",
            children.len(),
            keys.len()
        )));
    }
    for key in keys.iter().chain(std::iter::once(&terminal)) {
        if key.offsets.len() != ndims {
            return Err(OxiH5Error::Format(format!(
                "internal writer error: chunk B-tree key has {} offsets, expected {ndims}",
                key.offsets.len()
            )));
        }
    }

    let total = chunk_node_size(ndims);
    fill_zero(buf, addr, total);

    buf[addr..addr + 4].copy_from_slice(b"TREE");
    buf[addr + 4] = 1; // node type = 1 (raw data chunks)
    buf[addr + 5] = narrow::<u8>("chunk B-tree node level", level)?;
    write_u16_le(
        buf,
        addr + 6,
        narrow::<u16>("chunk B-tree entry count", children.len())?,
    );
    write_u64_le(buf, addr + 8, left_sibling);
    write_u64_le(buf, addr + 16, right_sibling);

    let key_size = chunk_key_size(ndims);
    let stride = key_size + CHILD_SIZE;
    for (i, (&child, &key)) in children.iter().zip(keys).enumerate() {
        let slot = addr + NODE_PREFIX + i * stride;
        write_key(buf, slot, key);
        write_u64_le(buf, slot + key_size, child);
    }
    if !children.is_empty() {
        write_key(buf, addr + NODE_PREFIX + children.len() * stride, terminal);
    }

    Ok(total)
}

// ---------------------------------------------------------------------------
// ChunkTree — one dataset's whole chunk index
// ---------------------------------------------------------------------------

/// Deepest chunk B-tree the writer will build.
///
/// Structural insurance only: [`MAX_CHUNKS`] bites first.  Level *L* addresses
/// `64^(L+1)` chunks — 64, then 4 096, then 262 144, then 16.7 million — so
/// four levels already cover everything [`MAX_CHUNKS`] permits.
const MAX_CHUNK_LEVELS: usize = 8;

/// Largest number of chunks the writer will index for one dataset.
///
/// A resource guard rather than a format limit: 16.7 million chunks is a
/// multi-gigabyte image and a quarter of a million index nodes, so refusing up
/// front turns an out-of-memory abort into a typed error.
const MAX_CHUNKS: usize = 1 << 24;

/// The planned chunk index of one dataset: levels of fixed-width B-tree nodes,
/// leaves first.
///
/// Built by [`ChunkTree::plan`] from the chunk count alone — so the layout pass
/// can reserve space in the same step that computes it — then given its
/// addresses by [`ChunkTree::assign`], then emitted by [`ChunkTree::write`].
pub(super) struct ChunkTree {
    /// Real dataset rank; the key width follows from it.
    ndims: usize,
    /// Number of chunks the tree indexes.
    n_chunks: usize,
    /// Node count of each level, leaves first; the last is always 1.
    levels: Vec<usize>,
    /// Address of the first node of each level, parallel to `levels`.
    level_addr: Vec<usize>,
    /// Address of the root node — what the layout message points at.
    root_addr: usize,
}

impl ChunkTree {
    /// Plan the index of a dataset holding `n_chunks` chunks.
    ///
    /// A dataset with **no** chunks still gets one empty leaf, because the
    /// layout message has to point at something: HDF5's own writer leaves the
    /// address undefined instead, but that is a second shape for every reader
    /// to handle for no gain here.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `n_chunks` exceeds [`MAX_CHUNKS`] or the
    /// tree would be deeper than [`MAX_CHUNK_LEVELS`].
    pub(super) fn plan(ndims: usize, n_chunks: usize) -> Result<Self, OxiH5Error> {
        if n_chunks > MAX_CHUNKS {
            return Err(OxiH5Error::Format(format!(
                "dataset needs {n_chunks} chunks, over the writer's limit of {MAX_CHUNKS}"
            )));
        }
        let mut nodes = n_chunks.div_ceil(CHUNK_NODE_MAX_CHILDREN).max(1);
        let mut levels = vec![nodes];
        while nodes > 1 {
            if levels.len() >= MAX_CHUNK_LEVELS {
                return Err(OxiH5Error::Format(format!(
                    "chunk B-tree for {n_chunks} chunks would exceed {MAX_CHUNK_LEVELS} levels"
                )));
            }
            nodes = nodes.div_ceil(CHUNK_NODE_MAX_CHILDREN);
            levels.push(nodes);
        }
        Ok(Self {
            ndims,
            n_chunks,
            level_addr: vec![0; levels.len()],
            levels,
            root_addr: 0,
        })
    }

    /// Total bytes the tree's nodes occupy.
    pub(super) fn bytes(&self) -> usize {
        self.levels.iter().sum::<usize>() * chunk_node_size(self.ndims)
    }

    /// Place the planned nodes from `base`, leaves first.
    pub(super) fn assign(&mut self, base: usize) {
        let node_size = chunk_node_size(self.ndims);
        let mut addr = base;
        for (slot, &count) in self.level_addr.iter_mut().zip(self.levels.iter()) {
            *slot = addr;
            addr += count * node_size;
        }
        // `plan` guarantees at least one level, whose last entry is the single
        // root node.
        self.root_addr = self.level_addr.last().copied().unwrap_or(base);
    }

    /// Address of the tree's root node — what the chunked layout message points
    /// at.
    pub(super) fn root_addr(&self) -> usize {
        self.root_addr
    }

    /// Emit every node of the tree; returns bytes written, always
    /// [`ChunkTree::bytes`].
    ///
    /// `entries` must be sorted the way [`chunk_origins`] produces them —
    /// row-major over the chunk grid, dimension 0 most significant. Nothing
    /// downstream detects that they are not: an out-of-order node still parses,
    /// and `H5B_find`'s binary search simply fails to locate some of the chunks
    /// in it.
    ///
    /// `end_offsets` closes the rightmost node of every level.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `entries` does not match the planned
    /// chunk count, if an offset vector has the wrong rank, or if a node field
    /// overflows its on-disk width.
    pub(super) fn write(
        &self,
        buf: &mut [u8],
        entries: &[ChunkEntry],
        end_offsets: &[u64],
    ) -> Result<usize, OxiH5Error> {
        if entries.len() != self.n_chunks {
            return Err(OxiH5Error::Format(format!(
                "internal writer error: chunk index planned for {} chunks but got {}",
                self.n_chunks,
                entries.len()
            )));
        }
        if end_offsets.len() != self.ndims {
            return Err(OxiH5Error::Format(format!(
                "internal writer error: chunk index terminal key has {} offsets, expected {}",
                end_offsets.len(),
                self.ndims
            )));
        }

        let node_size = chunk_node_size(self.ndims);
        let bound = NodeKey {
            nbytes: 0,
            filter_mask: 0,
            offsets: end_offsets,
        };

        // Level 0's children are the chunks themselves; each level above folds
        // the level below into its child list.
        let mut children: Vec<u64> = entries.iter().map(|entry| entry.addr).collect();
        let mut keys: Vec<NodeKey<'_>> = entries
            .iter()
            .map(|entry| NodeKey {
                nbytes: entry.nbytes,
                filter_mask: entry.filter_mask,
                offsets: &entry.offsets,
            })
            .collect();

        let mut wrote = 0usize;
        for (level, &base) in self.level_addr.iter().enumerate() {
            let n_nodes = children.len().div_ceil(CHUNK_NODE_MAX_CHILDREN).max(1);
            let mut next_children = Vec::with_capacity(n_nodes);
            let mut next_keys = Vec::with_capacity(n_nodes);

            for i in 0..n_nodes {
                let lo = i * CHUNK_NODE_MAX_CHILDREN;
                let hi = ((i + 1) * CHUNK_NODE_MAX_CHILDREN).min(children.len());
                let addr = base + i * node_size;
                // Siblings are linked within a level only; the ends are
                // undefined.  Neither oxih5's reader nor `H5B_find` follows
                // them — a lookup descends from the root — but libhdf5 walks
                // them for insertion and removal.
                let left = if i == 0 {
                    UNDEFINED_ADDR
                } else {
                    (addr - node_size) as u64
                };
                let right = if i + 1 == n_nodes {
                    UNDEFINED_ADDR
                } else {
                    (addr + node_size) as u64
                };
                // This node's exclusive upper bound is the smallest chunk it
                // does *not* hold: the next node's first key, or — for the
                // rightmost node — the dataset extent.
                let terminal = keys.get(hi).copied().unwrap_or(bound);
                wrote += write_node(
                    buf,
                    addr,
                    self.ndims,
                    level,
                    left,
                    right,
                    &children[lo..hi],
                    &keys[lo..hi],
                    terminal,
                )?;

                next_children.push(addr as u64);
                // A node's own lower bound is its first child's: every chunk
                // beneath it is at or after that origin.
                next_keys.push(NodeKey {
                    nbytes: 0,
                    filter_mask: 0,
                    offsets: keys.get(lo).map_or(end_offsets, |key| key.offsets),
                });
            }

            children = next_children;
            keys = next_keys;
        }
        Ok(wrote)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write::elem::ElemType;
    use crate::write::tree::Storage;

    fn chunked(shape: &[usize], chunk_shape: &[usize]) -> DatasetDesc {
        DatasetDesc {
            name: "ds".to_string(),
            raw: Vec::new(),
            shape: shape.to_vec(),
            elem_type: ElemType::F64,
            attrs: Vec::new(),
            storage: Storage::Chunked {
                chunk_shape: chunk_shape.to_vec(),
                unlimited_dim0: false,
            },
            filter: None,
            vlen_strings: None,
        }
    }

    /// The width libhdf5 asks the file for, arrived at from its own formula.
    ///
    /// 2096 is the number that appeared in `addr overflow, addr = 3000,
    /// size = 2096, eoa = 3112`, so this is the measured constant, not a
    /// restatement of the code above.
    #[test]
    fn node_width_matches_the_size_libhdf5_reads() {
        assert_eq!(chunk_node_size(1), 2096);
        for ndims in 1..5usize {
            let key = 8 + (ndims + 1) * 8;
            assert_eq!(chunk_node_size(ndims), 24 + 65 * key + 64 * 8);
        }
    }

    #[test]
    fn a_short_chunk_shape_is_completed_from_the_dataset_shape() {
        // What `oxinetcdf` passes: one chunk extent for a 2-D variable.
        let ds = chunked(&[5, 3], &[5]);
        assert_eq!(chunk_shape_of(&ds).expect("completed"), vec![5, 3]);
        // …and the empty vector, which is what `set_deflate` stores.
        let ds = chunked(&[5, 3], &[]);
        assert_eq!(chunk_shape_of(&ds).expect("defaulted"), vec![5, 3]);
        // A zero-length dimension still gets a chunk extent of 1.
        let ds = chunked(&[0], &[]);
        assert_eq!(chunk_shape_of(&ds).expect("empty"), vec![1]);
    }

    #[test]
    fn degenerate_chunk_shapes_are_rejected() {
        let zero = chunked(&[4], &[0]);
        let err = chunk_shape_of(&zero).expect_err("a 0 chunk extent divides by zero");
        assert!(format!("{err}").contains("is 0"), "{err}");

        let scalar = chunked(&[], &[]);
        let err = chunk_shape_of(&scalar).expect_err("no chunked scalar dataspace");
        assert!(format!("{err}").contains("at least one dimension"), "{err}");
    }

    /// A requested tiling is honoured verbatim — including a chunk larger than
    /// the extent, which HDF5 stores as one chunk of mostly fill.
    #[test]
    fn a_tiling_request_is_honoured_verbatim() {
        assert_eq!(chunk_shape_of(&chunked(&[7], &[3])).expect("1-D"), vec![3]);
        assert_eq!(
            chunk_shape_of(&chunked(&[5, 3], &[2, 2])).expect("2-D"),
            vec![2, 2]
        );
        assert_eq!(
            chunk_shape_of(&chunked(&[4], &[8])).expect("oversized"),
            vec![8],
            "a chunk wider than the extent is one chunk with fill in the overhang"
        );
        // Completion still applies per dimension, so a short vector may mix a
        // tiled dimension with a whole one.
        assert_eq!(
            chunk_shape_of(&chunked(&[6, 4], &[2])).expect("short"),
            vec![2, 4]
        );
    }

    #[test]
    fn one_chunk_covers_a_whole_dataset() {
        assert_eq!(chunk_origins(&[5, 3], &[5, 3]), vec![vec![0u64, 0]]);
        assert_eq!(end_offsets(&[5, 3], &[5, 3]), vec![5u64, 3]);
    }

    /// The grid arithmetic is exercised on non-divisible shapes even while the
    /// writer only emits one chunk, so that the emitter is the only thing left
    /// to change when tiling lands.
    #[test]
    fn the_chunk_grid_rounds_up_and_orders_row_major() {
        assert_eq!(
            chunk_origins(&[7], &[3]),
            vec![vec![0u64], vec![3], vec![6]]
        );
        assert_eq!(end_offsets(&[7], &[3]), vec![9u64]);

        // 3 × 2 chunks over [5, 3]; last dimension fastest.
        assert_eq!(
            chunk_origins(&[5, 3], &[2, 2]),
            vec![
                vec![0u64, 0],
                vec![0, 2],
                vec![2, 0],
                vec![2, 2],
                vec![4, 0],
                vec![4, 2],
            ]
        );
        assert_eq!(end_offsets(&[5, 3], &[2, 2]), vec![6u64, 4]);
    }

    /// Nothing is allocated for a dataset with a zero-length dimension, and the
    /// terminal key degenerates with it.
    #[test]
    fn a_zero_length_dimension_yields_no_chunks() {
        assert!(chunk_origins(&[0], &[1]).is_empty());
        assert!(chunk_origins(&[5, 0], &[5, 1]).is_empty());
        assert_eq!(end_offsets(&[0], &[1]), vec![0u64]);
    }

    /// A chunk at `origin` of `nbytes` bytes, addressed at `addr`.
    fn entry(origin: &[u64], addr: u64, nbytes: u32) -> ChunkEntry {
        ChunkEntry {
            offsets: origin.to_vec(),
            addr,
            nbytes,
            filter_mask: 0,
        }
    }

    /// Plan, place and emit a whole index into a fresh buffer padded with
    /// `0xAA`, so that a write past the tree is visible.
    fn build_tree(ndims: usize, entries: &[ChunkEntry], end: &[u64]) -> (Vec<u8>, ChunkTree) {
        let mut tree = ChunkTree::plan(ndims, entries.len()).expect("plan");
        tree.assign(0);
        let mut buf = vec![0xAAu8; tree.bytes() + 16];
        assert_eq!(
            tree.write(&mut buf, entries, end).expect("write"),
            tree.bytes(),
            "the emitter must fill exactly what the plan reserved"
        );
        (buf, tree)
    }

    #[test]
    fn a_node_is_full_width_and_zero_filled_past_its_entries() {
        let ndims = 1usize;
        let total = chunk_node_size(ndims);
        let (buf, tree) = build_tree(ndims, &[entry(&[0], 0xABCD, 256)], &[4]);
        assert_eq!(tree.bytes(), total, "one chunk needs one node");
        assert_eq!(tree.root_addr(), 0);

        assert_eq!(&buf[0..4], b"TREE");
        assert_eq!(buf[4], 1, "node type 1 = raw data chunks");
        assert_eq!(buf[5], 0, "leaf");
        assert_eq!(u16::from_le_bytes([buf[6], buf[7]]), 1);
        let u64_at = |off: usize| u64::from_le_bytes(buf[off..off + 8].try_into().expect("8"));
        assert_eq!(u64_at(8), u64::MAX, "left sibling undefined");
        assert_eq!(u64_at(16), u64::MAX, "right sibling undefined");

        // key[0] then child[0] then key[1].
        let key_size = chunk_key_size(ndims);
        assert_eq!(u32::from_le_bytes(buf[24..28].try_into().expect("4")), 256);
        assert_eq!(u64_at(32), 0, "chunk origin");
        assert_eq!(u64_at(40), 0, "trailing element offset");
        assert_eq!(u64_at(24 + key_size), 0xABCD, "child = chunk address");
        let terminal = 24 + key_size + CHILD_SIZE;
        assert_eq!(
            u32::from_le_bytes(buf[terminal..terminal + 4].try_into().expect("4")),
            0,
            "the terminal key describes no chunk, so its byte count is 0"
        );
        assert_eq!(u64_at(terminal + 8), 4, "terminal key = rounded-up extent");

        // Everything past the terminal key is zero, and nothing past the node
        // was touched.
        assert!(
            buf[terminal + key_size..total].iter().all(|&b| b == 0),
            "unused children must be zero-filled"
        );
        assert!(
            buf[total..].iter().all(|&b| b == 0xAA),
            "wrote past the node"
        );
    }

    #[test]
    fn an_empty_node_carries_no_terminal_key() {
        let total = chunk_node_size(2);
        let (buf, tree) = build_tree(2, &[], &[0, 0]);
        assert_eq!(tree.bytes(), total, "an empty index is still one node");
        assert_eq!(u16::from_le_bytes([buf[6], buf[7]]), 0);
        assert!(
            buf[24..total].iter().all(|&b| b == 0),
            "an empty node has no keys at all"
        );
    }

    #[test]
    fn mismatched_offset_vectors_are_refused() {
        let mut tree = ChunkTree::plan(2, 1).expect("plan");
        tree.assign(0);
        let mut buf = vec![0u8; tree.bytes()];

        let wrong = [entry(&[0], 8, 1)];
        assert!(tree.write(&mut buf, &wrong, &[1, 1]).is_err(), "rank 1 key");

        let right = [entry(&[0, 0], 8, 1)];
        assert!(
            tree.write(&mut buf, &right, &[1]).is_err(),
            "rank 1 terminal"
        );
        assert!(tree.write(&mut buf, &right, &[1, 1]).is_ok());

        // And a chunk count the plan was not built for.
        assert!(tree.write(&mut buf, &[], &[1, 1]).is_err());
    }

    // -----------------------------------------------------------------------
    // W1e2: trees deeper than one node
    // -----------------------------------------------------------------------

    /// Level counts follow the 64-way fan-out, and every level is one node
    /// wide at the top.
    #[test]
    fn tree_depth_follows_the_chunk_count() {
        let levels = |n: usize| ChunkTree::plan(1, n).expect("plan").levels;
        assert_eq!(levels(0), vec![1], "an empty index still owns a leaf");
        assert_eq!(levels(1), vec![1]);
        assert_eq!(levels(64), vec![1], "exactly one full leaf");
        assert_eq!(levels(65), vec![2, 1], "the 65th chunk grows a level");
        assert_eq!(levels(4096), vec![64, 1]);
        assert_eq!(levels(4097), vec![65, 2, 1]);

        // Sizes and the root address follow from the level list.
        let mut tree = ChunkTree::plan(1, 65).expect("plan");
        tree.assign(1000);
        assert_eq!(tree.bytes(), 3 * chunk_node_size(1));
        assert_eq!(
            tree.root_addr(),
            1000 + 2 * chunk_node_size(1),
            "the root is the *last* level; pointing the layout message at the \
             first leaf would hide every chunk but the first 64"
        );
        assert!(ChunkTree::plan(1, MAX_CHUNKS + 1).is_err());
    }

    /// A two-level tree's keys must bracket its children correctly.
    ///
    /// `key[i]` is the inclusive lower bound of child `i` and `key[i+1]` the
    /// exclusive upper bound — the opposite of the group B-tree's convention.
    /// Reversing them leaves a tree that parses perfectly and loses chunks.
    #[test]
    fn a_two_level_tree_brackets_every_child() {
        // 100 chunks of a 1-D dataset with chunk extent 1 → leaves of 64 + 36.
        let entries: Vec<ChunkEntry> = (0..100u64)
            .map(|i| entry(&[i], 10_000 + i * 8, 8))
            .collect();
        let (buf, tree) = build_tree(1, &entries, &[100]);
        assert_eq!(tree.bytes(), 3 * chunk_node_size(1));

        let node_size = chunk_node_size(1);
        let key_size = chunk_key_size(1);
        let u16_at = |off: usize| u16::from_le_bytes([buf[off], buf[off + 1]]);
        let u64_at = |off: usize| u64::from_le_bytes(buf[off..off + 8].try_into().expect("8"));
        let key_offset = |node: usize, i: usize| node + NODE_PREFIX + i * (key_size + CHILD_SIZE);
        let child_at = |node: usize, i: usize| u64_at(key_offset(node, i) + key_size);

        // Two leaves at level 0, then the root at level 1.
        let (leaf0, leaf1, root) = (0, node_size, 2 * node_size);
        assert_eq!(buf[leaf0 + 5], 0, "leaf level");
        assert_eq!(buf[root + 5], 1, "root level");
        assert_eq!(u16_at(leaf0 + 6), 64);
        assert_eq!(u16_at(leaf1 + 6), 36);
        assert_eq!(u16_at(root + 6), 2);
        assert_eq!(tree.root_addr(), root);

        // Leaf 0 holds chunks 0..64 and closes at 64 — the first chunk it does
        // *not* hold, not the last one it does.
        assert_eq!(u64_at(key_offset(leaf0, 0) + 8), 0);
        assert_eq!(u64_at(key_offset(leaf0, 63) + 8), 63);
        assert_eq!(
            u64_at(key_offset(leaf0, 64) + 8),
            64,
            "a leaf's terminal key is the next leaf's first chunk"
        );
        // Leaf 1 holds 64..100 and closes at the dataset extent.
        assert_eq!(u64_at(key_offset(leaf1, 0) + 8), 64);
        assert_eq!(u64_at(key_offset(leaf1, 35) + 8), 99);
        assert_eq!(u64_at(key_offset(leaf1, 36) + 8), 100);

        // The root brackets both leaves and points at them, in order.
        assert_eq!(u64_at(key_offset(root, 0) + 8), 0);
        assert_eq!(u64_at(key_offset(root, 1) + 8), 64);
        assert_eq!(u64_at(key_offset(root, 2) + 8), 100);
        assert_eq!(child_at(root, 0), leaf0 as u64);
        assert_eq!(child_at(root, 1), leaf1 as u64);
        // An internal key describes no chunk.
        assert_eq!(
            u32::from_le_bytes(
                buf[key_offset(root, 0)..key_offset(root, 0) + 4]
                    .try_into()
                    .expect("4")
            ),
            0
        );

        // Siblings are linked within the level, and the ends are undefined.
        assert_eq!(u64_at(leaf0 + 8), u64::MAX);
        assert_eq!(u64_at(leaf0 + 16), leaf1 as u64);
        assert_eq!(u64_at(leaf1 + 8), leaf0 as u64);
        assert_eq!(u64_at(leaf1 + 16), u64::MAX);
        assert_eq!((u64_at(root + 8), u64_at(root + 16)), (u64::MAX, u64::MAX));

        // Every chunk address is still reachable, exactly once.
        let reached: Vec<u64> = (0..2)
            .flat_map(|leaf| {
                let node = if leaf == 0 { leaf0 } else { leaf1 };
                (0..u16_at(node + 6) as usize).map(move |i| child_at(node, i))
            })
            .collect();
        assert_eq!(
            reached,
            entries.iter().map(|e| e.addr).collect::<Vec<_>>(),
            "the leaves must cover every chunk, in order"
        );
        assert!(buf[tree.bytes()..].iter().all(|&b| b == 0xAA));
    }

    // -----------------------------------------------------------------------
    // W1e: the node as it lands in a real file
    // -----------------------------------------------------------------------

    /// Address of the first chunk B-tree node in a built file.
    ///
    /// Found by scanning rather than by walking the layout message, so the test
    /// does not depend on the message encoder it is meant to be independent of.
    fn find_chunk_node(bytes: &[u8]) -> Option<usize> {
        (0..bytes.len().saturating_sub(8))
            .find(|&i| &bytes[i..i + 4] == b"TREE" && bytes[i + 4] == 1)
    }

    /// The key past the last chunk carries the dataset extent, and no chunk.
    ///
    /// This is the field that used to be left entirely zero.  `H5B_find`
    /// binary-searches with `cmp3(key[i], wanted, key[i+1])` and reports a hit
    /// only while `wanted < key[i+1]`, so an all-zero terminal key describes an
    /// empty range: libhdf5 walked into the node, compared `[0,0]` against
    /// `[0,0]`, concluded the chunk it wanted was past the end, and reported
    /// the dataset as unallocated.
    #[test]
    fn w1e_chunk_btree_terminal_key_is_extent() {
        let dtype = oxih5_core::Dtype::Int {
            size: 4,
            signed: true,
            order: oxih5_core::ByteOrder::Little,
        };
        let raw: Vec<u8> = (0..15i32).flat_map(i32::to_le_bytes).collect();
        let mut w = crate::FileWriter::new();
        // A deliberately *short* chunk shape, the way `oxinetcdf` passes one.
        w.create_dataset_unlimited("grid", &[5, 3], &[5], &dtype, &raw)
            .expect("grid");
        let bytes = w.build_to_vec().expect("build");

        let node = find_chunk_node(&bytes).expect("a chunk B-tree node in the file");
        let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4"));
        let u64_at = |off: usize| u64::from_le_bytes(bytes[off..off + 8].try_into().expect("8"));

        assert_eq!(u16::from_le_bytes([bytes[node + 6], bytes[node + 7]]), 1);
        let key_size = chunk_key_size(2);

        // key[0]: the one chunk, at the origin, holding all 15 elements.
        let key0 = node + NODE_PREFIX;
        assert_eq!(u32_at(key0), 60, "15 × 4 bytes");
        assert_eq!(u32_at(key0 + 4), 0, "no filter was skipped");
        assert_eq!(
            (u64_at(key0 + 8), u64_at(key0 + 16)),
            (0, 0),
            "chunk origin"
        );

        // key[1]: the extent — and emphatically not zero.
        let key1 = key0 + key_size + CHILD_SIZE;
        assert_eq!(u32_at(key1), 0, "a terminal key describes no chunk");
        assert_eq!(
            (u64_at(key1 + 8), u64_at(key1 + 16)),
            (5, 3),
            "the terminal key must be the dataset extent, not [0, 0] — an empty \
             search range is what made libhdf5 report the chunk as missing"
        );
        assert_eq!(u64_at(key1 + 24), 0, "the trailing element offset stays 0");

        // The chunk really is where its child pointer says, and the node
        // occupies the full width libhdf5 will read.
        let child = u64_at(key0 + key_size) as usize;
        assert_eq!(
            &bytes[child..child + 4],
            &raw[..4],
            "child[0] → chunk bytes"
        );
        assert!(
            node + chunk_node_size(2) <= bytes.len(),
            "the node must fit inside the file libhdf5 is told the size of: node at \
             {node}, width {}, file {} bytes",
            chunk_node_size(2),
            bytes.len()
        );
    }

    /// A dataset with no elements still gets a valid, empty node.
    #[test]
    fn w1e_zero_length_dataset_indexes_nothing() {
        let mut w = crate::FileWriter::new();
        w.write_dataset_i32("empty", &[], &[0]).expect("empty");
        w.set_deflate("empty", 6).expect("set_deflate");
        let bytes = w.build_to_vec().expect("build");

        let node = find_chunk_node(&bytes).expect("a chunk B-tree node");
        assert_eq!(
            u16::from_le_bytes([bytes[node + 6], bytes[node + 7]]),
            0,
            "no elements, no chunks"
        );
        assert!(
            bytes[node + NODE_PREFIX..node + chunk_node_size(1)]
                .iter()
                .all(|&b| b == 0),
            "an empty node's keys are all absent, not merely zero-valued"
        );
    }

    /// One node holds at most 64 children; the 65th chunk grows a level rather
    /// than overflowing the node.
    #[test]
    fn a_node_refuses_more_chunks_than_it_can_hold() {
        let mut buf = vec![0u8; chunk_node_size(1)];
        let over: Vec<ChunkEntry> = (0..=CHUNK_NODE_MAX_CHILDREN as u64)
            .map(|i| entry(&[i], 0, 0))
            .collect();
        let keys: Vec<NodeKey<'_>> = over
            .iter()
            .map(|e| NodeKey {
                nbytes: e.nbytes,
                filter_mask: e.filter_mask,
                offsets: &e.offsets,
            })
            .collect();
        let children = vec![0u64; over.len()];
        assert!(
            write_node(
                &mut buf,
                0,
                1,
                0,
                UNDEFINED_ADDR,
                UNDEFINED_ADDR,
                &children,
                &keys,
                keys[0],
            )
            .is_err(),
            "a single node must refuse a 65th child"
        );

        // Through the tree, the same 65 chunks simply gain a level.
        let (_, tree) = build_tree(1, &over, &[65]);
        assert_eq!(tree.bytes(), 3 * chunk_node_size(1));
    }
}
