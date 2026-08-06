//! Version-2 B-tree writer — the *link name index* of a dense group.
//!
//! A dense group keeps its link messages in a fractal heap (see
//! [`super::fractal_heap`]) and finds them again through a **type-5 version-2
//! B-tree**: one record per link, keyed by the Jenkins lookup3 hash of the link
//! name, carrying the heap ID that retrieves the message.
//!
//! # Why the records must be sorted, and by what
//!
//! `H5B2__locate_record` **binary-searches** a node by that hash.  A tree whose
//! records are in any other order does not fail loudly — libhdf5 simply fails
//! to find some links, while a linear enumeration (ours, and `h5ls`) still
//! reports every one of them.  So [`NameIndexWriter::plan`] sorts, and the sort
//! key is the hash first and the name second, which makes the order total even
//! for two names that collide.
//!
//! # The subset this writer emits
//!
//! One leaf node, no internal nodes.  A version-2 B-tree's node size is a
//! per-tree creation parameter that the header declares and the reader obeys,
//! so a single leaf sized to hold every record is a *legal* tree rather than a
//! truncated one — and it is the only depth `oxih5_format::btree_v2` needs to
//! walk, since the heap it indexes is itself capped at one direct block.
//!
//! # On-disk form
//!
//! ```text
//! header "BTHD"                       leaf "BTLF"
//!   0.. 4  signature                    0.. 4  signature
//!   4      version = 0                  4      version = 0
//!   5      type = 5 (link name)         5      type = 5
//!   6..10  node size                    6..    records
//!  10..12  record size = 4 + heap ID    ..     metadata checksum
//!  12..14  depth = 0
//!  14      split percent = 100        record
//!  15      merge percent = 40           0.. 4  name hash (lookup3)
//!  16..24  root node address            4..    heap ID
//!  24..26  records in the root
//!  26..34  records in the tree
//!  34..38  metadata checksum
//! ```
//!
//! The checksum of each structure covers the bytes *before* it — for the leaf
//! that is the prefix plus the records, not the whole padded node — which is
//! what `H5B2__cache_leaf_serialize` computes.

use oxih5_core::OxiH5Error;

use super::checksum::{link_name_hash, metadata_checksum};
use super::format::{fill_zero, write_u16_le, write_u32_le, write_u64_le};
use super::{check_size, narrow};

/// Version-2 B-tree type code for a group's link name index.
const BTREE_TYPE_LINK_NAME: u8 = 5;

/// Size of the "BTHD" header, checksum included.
pub(super) const HEADER_SIZE: usize = 38;

/// Byte offset of the header checksum within the header.
const HEADER_CHECKSUM_OFFSET: usize = 34;

/// Bytes before the first record of a leaf node: signature, version, type.
const LEAF_PREFIX: usize = 6;

/// Bytes of metadata checksum at the end of a node's live region.
const CHECKSUM_SIZE: usize = 4;

/// Node size granularity; libhdf5 uses 512 for a link name index.
const NODE_SIZE_UNIT: usize = 512;

/// Percentage fill at which libhdf5 splits a node, as the header records it.
const SPLIT_PERCENT: u8 = 100;
/// Percentage fill at which libhdf5 merges nodes, as the header records it.
const MERGE_PERCENT: u8 = 40;

/// One record of a link name index: the key, and what it retrieves.
#[derive(Debug)]
struct NameRecord {
    /// Jenkins lookup3 of the link name — the binary-search key.
    hash: u32,
    /// The link name itself, only ever used to break a hash tie.
    name: String,
    /// Fractal-heap ID of the link message this record indexes.
    heap_id: Vec<u8>,
}

/// A link name index holding a fixed set of records, laid out but not placed.
///
/// Mirrors [`super::fractal_heap::FractalHeapWriter`]: [`NameIndexWriter::plan`]
/// settles every size, [`NameIndexWriter::assign`] fixes the addresses, and
/// [`NameIndexWriter::write`] emits — so the enclosing two-pass writer can
/// reserve the tree's bytes before knowing where it goes.
#[derive(Debug)]
pub(super) struct NameIndexWriter {
    /// Records, sorted ascending by `(hash, name)`.
    records: Vec<NameRecord>,
    /// Bytes of one record: the 4-byte hash plus a heap ID.
    record_size: usize,
    /// Declared node size; a multiple of [`NODE_SIZE_UNIT`] large enough for
    /// every record plus the leaf prefix and checksum.
    node_size: usize,
    /// Address of the "BTHD" header; 0 until assigned.
    header_addr: usize,
    /// Address of the single "BTLF" leaf; 0 until assigned.
    leaf_addr: usize,
}

impl NameIndexWriter {
    /// Lay out an index over `entries`, each a `(link name, heap ID)` pair.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if the heap IDs are not all the same width
    /// — the header declares one record size for the whole tree, and a reader
    /// that trusts it would slice every later record at the wrong offset — or
    /// if the tree holds more records than its 16-bit count field can express.
    pub(super) fn plan(entries: Vec<(String, Vec<u8>)>) -> Result<Self, OxiH5Error> {
        let heap_id_len = entries.first().map_or(0, |(_, id)| id.len());
        if let Some((name, id)) = entries.iter().find(|(_, id)| id.len() != heap_id_len) {
            return Err(OxiH5Error::Format(format!(
                "internal writer error: link '{name}' has a {}-byte heap ID where the \
                 name index declares {heap_id_len}",
                id.len()
            )));
        }
        let _: u16 = narrow("link name index record count", entries.len())?;

        let mut records: Vec<NameRecord> = entries
            .into_iter()
            .map(|(name, heap_id)| NameRecord {
                hash: link_name_hash(&name),
                name,
                heap_id,
            })
            .collect();
        // Hash first, name second: the tie-break keeps the order total, so the
        // emitted bytes do not depend on the order the caller happened to use.
        records.sort_by(|a, b| a.hash.cmp(&b.hash).then_with(|| a.name.cmp(&b.name)));

        let record_size = 4 + heap_id_len;
        let live = LEAF_PREFIX + records.len() * record_size + CHECKSUM_SIZE;
        let node_size = live.div_ceil(NODE_SIZE_UNIT).max(1) * NODE_SIZE_UNIT;

        Ok(Self {
            records,
            record_size,
            node_size,
            header_addr: 0,
            leaf_addr: 0,
        })
    }

    /// Total bytes the index occupies: header, then the leaf node.
    pub(super) fn bytes(&self) -> usize {
        HEADER_SIZE + self.node_size
    }

    /// Place the index at `base`, header first.
    pub(super) fn assign(&mut self, base: usize) {
        self.header_addr = base;
        self.leaf_addr = base + HEADER_SIZE;
    }

    /// Address of the "BTHD" header — what a Link Info message points at.
    pub(super) fn header_addr(&self) -> u64 {
        self.header_addr as u64
    }

    /// Emit the header and the leaf node; returns bytes written.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if a count overflows its on-disk field or
    /// if the emitted bytes do not match what [`Self::bytes`] reserved.
    pub(super) fn write(&self, buf: &mut [u8]) -> Result<usize, OxiH5Error> {
        let nrecords: u16 = narrow("link name index record count", self.records.len())?;
        let record_size: u16 = narrow("link name index record size", self.record_size)?;
        let node_size: u32 = narrow("link name index node size", self.node_size)?;

        // -- Leaf node. -------------------------------------------------------
        let leaf = self.leaf_addr;
        fill_zero(buf, leaf, self.node_size);
        buf[leaf..leaf + 4].copy_from_slice(b"BTLF");
        buf[leaf + 4] = 0x00; // version 0
        buf[leaf + 5] = BTREE_TYPE_LINK_NAME;
        for (i, record) in self.records.iter().enumerate() {
            let at = leaf + LEAF_PREFIX + i * self.record_size;
            write_u32_le(buf, at, record.hash);
            buf[at + 4..at + 4 + record.heap_id.len()].copy_from_slice(&record.heap_id);
        }
        // The checksum covers only the live prefix — signature through the last
        // record — and is written immediately after it, never at the node's end.
        let live = LEAF_PREFIX + self.records.len() * self.record_size;
        let leaf_checksum = metadata_checksum(&buf[leaf..leaf + live]);
        write_u32_le(buf, leaf + live, leaf_checksum);

        // -- Header. ----------------------------------------------------------
        let base = self.header_addr;
        fill_zero(buf, base, HEADER_SIZE);
        buf[base..base + 4].copy_from_slice(b"BTHD");
        buf[base + 4] = 0x00; // version 0
        buf[base + 5] = BTREE_TYPE_LINK_NAME;
        write_u32_le(buf, base + 6, node_size);
        write_u16_le(buf, base + 10, record_size);
        write_u16_le(buf, base + 12, 0); // depth: a single leaf is the root
        buf[base + 14] = SPLIT_PERCENT;
        buf[base + 15] = MERGE_PERCENT;
        write_u64_le(buf, base + 16, leaf as u64);
        write_u16_le(buf, base + 24, nrecords);
        write_u64_le(buf, base + 26, u64::from(nrecords));
        let header_checksum = metadata_checksum(&buf[base..base + HEADER_CHECKSUM_OFFSET]);
        write_u32_le(buf, base + HEADER_CHECKSUM_OFFSET, header_checksum);

        let wrote = self.bytes();
        check_size("link name index", leaf + self.node_size - base, wrote)?;
        Ok(wrote)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxih5_format::btree_v2::parse_name_index;

    fn entries(names: &[&str]) -> Vec<(String, Vec<u8>)> {
        names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                (
                    (*name).to_string(),
                    vec![0x00, i as u8, 0, 0, 0, 0x0f, 0x00],
                )
            })
            .collect()
    }

    /// Every heap ID must come back out of the reader that walks a name index.
    #[test]
    fn heap_ids_round_trip_through_the_reader() {
        let source = entries(&["ext", "ds00", "ds01", "ds02", "alpha", "zeta"]);
        let mut index = NameIndexWriter::plan(source.clone()).expect("plan");
        index.assign(0x800);
        let mut buf = vec![0u8; 0x800 + index.bytes()];
        assert_eq!(index.write(&mut buf).expect("write"), index.bytes());

        let mut got = parse_name_index(&buf, 0x800, 7).expect("parse");
        let mut want: Vec<Vec<u8>> = source.into_iter().map(|(_, id)| id).collect();
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want);
    }

    /// Records must ascend by name hash, because libhdf5 binary-searches them.
    #[test]
    fn records_ascend_by_name_hash() {
        let index = NameIndexWriter::plan(entries(&[
            "zeta", "alpha", "ds07", "ds00", "middle", "ds11",
        ]))
        .expect("plan");
        let hashes: Vec<u32> = index.records.iter().map(|r| r.hash).collect();
        let mut sorted = hashes.clone();
        sorted.sort_unstable();
        assert_eq!(hashes, sorted, "records must be in ascending hash order");
        // And the hashes are the ones libhdf5 computed for these very names.
        let ds07 = index
            .records
            .iter()
            .find(|r| r.name == "ds07")
            .expect("ds07");
        assert_eq!(ds07.hash, 0x0613_6c62);
    }

    /// The header's declared geometry must match the leaf the writer emitted,
    /// and the checksum must be the one libhdf5 would compute over the *live*
    /// region only — the reader verifies neither, so this test has to.
    #[test]
    fn header_geometry_and_checksums_match_libhdf5() {
        let mut index = NameIndexWriter::plan(entries(&["a", "b", "c"])).expect("plan");
        index.assign(0x40);
        let mut buf = vec![0u8; 0x40 + index.bytes()];
        index.write(&mut buf).expect("write");

        assert_eq!(&buf[0x40..0x44], b"BTHD");
        assert_eq!(buf[0x45], 5, "type 5 = link name index");
        assert_eq!(
            u32::from_le_bytes(buf[0x46..0x4a].try_into().expect("4 bytes")) as usize,
            NODE_SIZE_UNIT
        );
        assert_eq!(u16::from_le_bytes([buf[0x4a], buf[0x4b]]), 11, "4 + 7");
        assert_eq!(u16::from_le_bytes([buf[0x4c], buf[0x4d]]), 0, "depth");
        assert_eq!(buf[0x4e], SPLIT_PERCENT);
        assert_eq!(buf[0x4f], MERGE_PERCENT);
        assert_eq!(u16::from_le_bytes([buf[0x58], buf[0x59]]), 3, "root nrec");

        let header = &buf[0x40..0x40 + HEADER_SIZE];
        assert_eq!(
            u32::from_le_bytes(
                header[HEADER_CHECKSUM_OFFSET..HEADER_CHECKSUM_OFFSET + 4]
                    .try_into()
                    .expect("4 bytes")
            ),
            metadata_checksum(&header[..HEADER_CHECKSUM_OFFSET])
        );

        let leaf = 0x40 + HEADER_SIZE;
        let live = LEAF_PREFIX + 3 * 11;
        assert_eq!(
            u32::from_le_bytes(
                buf[leaf + live..leaf + live + 4]
                    .try_into()
                    .expect("4 bytes")
            ),
            metadata_checksum(&buf[leaf..leaf + live])
        );
    }

    /// A leaf grows past one node-size unit rather than overrunning it.
    #[test]
    fn the_node_grows_to_hold_every_record() {
        let names: Vec<String> = (0..100).map(|i| format!("link{i:04}")).collect();
        let index = NameIndexWriter::plan(
            names
                .iter()
                .map(|n| (n.clone(), vec![0u8; 7]))
                .collect::<Vec<_>>(),
        )
        .expect("plan");
        assert!(index.node_size >= LEAF_PREFIX + 100 * 11 + CHECKSUM_SIZE);
        assert_eq!(index.node_size % NODE_SIZE_UNIT, 0);
    }

    /// Ragged heap IDs would desynchronise every record after the first.
    #[test]
    fn mismatched_heap_id_widths_are_rejected() {
        let err = NameIndexWriter::plan(vec![
            ("a".to_string(), vec![0u8; 7]),
            ("b".to_string(), vec![0u8; 8]),
        ])
        .expect_err("must reject");
        assert!(format!("{err}").contains("heap ID"), "{err}");
    }

    /// An empty index is a valid, walkable tree with no records.
    #[test]
    fn an_empty_index_is_still_walkable() {
        let mut index = NameIndexWriter::plan(Vec::new()).expect("plan");
        index.assign(0x20);
        let mut buf = vec![0u8; 0x20 + index.bytes()];
        index.write(&mut buf).expect("write");
        assert_eq!(
            parse_name_index(&buf, 0x20, 0).expect("parse"),
            Vec::<Vec<u8>>::new()
        );
    }
}
