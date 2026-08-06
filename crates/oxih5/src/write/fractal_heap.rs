//! Fractal heap writer — the object store behind a *dense* new-style group.
//!
//! Once a group holds more links than fit comfortably in its object header,
//! libhdf5 moves them out: each link message becomes an opaque object in a
//! **fractal heap**, and a version-2 B-tree (see [`super::btree_v2`]) indexes
//! those objects by the hash of the link name.  The Link Info message then
//! names the two structures instead of carrying any links itself.
//!
//! # The subset this writer emits, and why it is the right one
//!
//! A fractal heap is a doubling table of direct and indirect blocks, and a
//! general writer would have to grow one.  This writer emits the *root direct
//! block* case only — `Current # of Rows = 0`, the whole heap in one block —
//! and sizes that block so every object fits.  That is not a simplification of
//! the format: it is exactly the shape libhdf5 produces for a group whose links
//! fit one block, which is every group up to roughly four thousand members, and
//! it is the shape `oxih5_format::FractalHeap::read_object` resolves directly.
//! A request that would need an indirect root is refused with a typed error
//! rather than written wrong.
//!
//! # On-disk form
//!
//! The header ("FRHP", 146 bytes at `size_of_offsets = size_of_lengths = 8`):
//!
//! ```text
//!   0.. 4  signature "FRHP"          72.. 78  size of huge objects
//!   4      version = 0               86.. 94  number of huge objects
//!   5.. 7  heap ID length = 7        94..102  size of tiny objects
//!   7.. 9  I/O filter length = 0    102..110  number of tiny objects
//!   9      flags = 0x02             110..112  table width = 4
//!  10.. 14 max managed object size  112..120  starting block size
//!  14.. 22 next huge object ID      120..128  max direct block size = 65536
//!  22.. 30 huge object B-tree       128..130  max heap size, in bits = 32
//!  30.. 38 free space in blocks     130..132  starting rows in root = 1
//!  38.. 46 free space manager       132..140  address of the root block
//!  46.. 54 managed space in heap    140..142  current rows in root = 0
//!  54.. 62 allocated managed space  142..146  metadata checksum
//!  62.. 70 direct block iterator
//!  70.. 78 number of managed objects
//! ```
//!
//! and the root direct block ("FHDB"), whose 21-byte header is counted *inside*
//! the heap's own address space — a heap offset is measured from byte 0 of the
//! signature, so the first object sits at offset 21:
//!
//! ```text
//!   0.. 4  signature "FHDB"
//!   4      version = 0
//!   5.. 13 address of the heap header
//!  13.. 17 block offset (`max heap size bits / 8` bytes) = 0
//!  17.. 21 metadata checksum (present because flags bit 1 is set)
//!  21..    objects, packed with no alignment or padding between them
//! ```
//!
//! Every constant and offset here is pinned against a heap libhdf5 2.0.0 wrote.

use oxih5_core::OxiH5Error;

use super::checksum::metadata_checksum;
use super::format::{fill_zero, write_u16_le, write_u32_le, write_u64_le};
use super::{check_size, pad8};

/// Bytes in every heap ID this writer produces: 1 type + 4 offset + 2 length.
///
/// libhdf5 derives the same 7 from `max heap size = 32 bits` and a two-byte
/// object length, and `FractalHeap::parse_heap_id` re-derives the split the
/// same way, so the three agree by construction rather than by coincidence.
pub(super) const HEAP_ID_LEN: u8 = 7;

/// Bits of heap virtual address space, and hence 4 bytes of heap ID offset.
const MAX_HEAP_SIZE_BITS: u16 = 32;

/// Bytes of a heap ID given over to the object's heap offset.
const ID_OFFSET_BYTES: usize = (MAX_HEAP_SIZE_BITS as usize).div_ceil(8);

/// Bytes of a heap ID given over to the object's length.
const ID_LENGTH_BYTES: usize = HEAP_ID_LEN as usize - 1 - ID_OFFSET_BYTES;

/// Columns of the doubling table; libhdf5's default.
const TABLE_WIDTH: u16 = 4;

/// Smallest direct block libhdf5 allocates, and this writer's floor.
const MIN_BLOCK_SIZE: u64 = 512;

/// Largest direct block the doubling table may address.
const MAX_DIRECT_BLOCK_SIZE: u64 = 65536;

/// Heap header flags: bit 1 = direct blocks carry a metadata checksum.
const FLAG_CHECKSUM_DIRECT_BLOCKS: u8 = 0x02;

/// Total size of the "FRHP" header, checksum included.
pub(super) const HEADER_SIZE: usize = 146;

/// Byte offset of the header checksum within the header.
const HEADER_CHECKSUM_OFFSET: usize = 142;

/// Size of the "FHDB" direct block header, checksum included.
///
/// `4 (signature) + 1 (version) + 8 (heap header address) + 4 (block offset) +
/// 4 (checksum)`.
const DIRECT_BLOCK_HEADER: usize = 21;

/// Largest single object a heap ID's two length bytes can describe.
const MAX_OBJECT_SIZE: usize = (1 << (8 * ID_LENGTH_BYTES)) - 1;

/// A fractal heap holding a fixed set of objects, laid out but not yet placed.
///
/// Built by [`FractalHeapWriter::plan`], which settles every size; the caller
/// then assigns addresses with [`FractalHeapWriter::assign`] and emits with
/// [`FractalHeapWriter::write`].  That split is what lets the enclosing
/// two-pass writer reserve the heap's bytes before it knows where anything is.
#[derive(Debug)]
pub(super) struct FractalHeapWriter {
    /// Byte length of each object, in insertion order; heap IDs index into this.
    ///
    /// Lengths rather than bytes, deliberately.  A group's link messages are
    /// sized during the layout pass but their hard-link *addresses* are not
    /// known until every object header has been placed, so the heap is planned
    /// from what is final and handed the bytes at [`Self::write`] time — where
    /// the lengths are checked against these, not trusted.
    sizes: Vec<usize>,
    /// Heap offset of each object, parallel to `sizes`.
    offsets: Vec<u32>,
    /// Size of the root direct block, a power of two ≥ [`MIN_BLOCK_SIZE`].
    block_size: u64,
    /// Address of the "FRHP" header; 0 until assigned.
    header_addr: usize,
    /// Address of the root "FHDB" block; 0 until assigned.
    block_addr: usize,
}

impl FractalHeapWriter {
    /// Lay out a heap holding objects of `sizes`, in the order given.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if a single object is longer than a heap
    /// ID's length field can describe, or if the objects together do not fit
    /// one maximum-size direct block — the point at which the heap would need
    /// an indirect root, which this writer does not emit.
    pub(super) fn plan(sizes: &[usize]) -> Result<Self, OxiH5Error> {
        let sizes = sizes.to_vec();
        let mut offsets = Vec::with_capacity(sizes.len());
        let mut cursor = DIRECT_BLOCK_HEADER;
        for &size in &sizes {
            if size > MAX_OBJECT_SIZE {
                return Err(OxiH5Error::Format(format!(
                    "fractal heap object of {size} bytes exceeds the {MAX_OBJECT_SIZE}-byte \
                     maximum a {HEAP_ID_LEN}-byte heap ID can describe"
                )));
            }
            offsets.push(u32::try_from(cursor).map_err(|_| {
                OxiH5Error::Format("fractal heap: object offset overflows 32 bits".to_string())
            })?);
            cursor += size;
        }

        // Rows 0 and 1 of the doubling table share the starting block size, so
        // a root that *is* a direct block is exactly one starting block.
        let mut block_size = MIN_BLOCK_SIZE;
        while (block_size as usize) < cursor {
            block_size *= 2;
            if block_size > MAX_DIRECT_BLOCK_SIZE {
                return Err(OxiH5Error::Format(format!(
                    "dense group storage needs {cursor} bytes of fractal heap, over the \
                     {MAX_DIRECT_BLOCK_SIZE}-byte root direct block this writer emits"
                )));
            }
        }

        Ok(Self {
            sizes,
            offsets,
            block_size,
            header_addr: 0,
            block_addr: 0,
        })
    }

    /// Total bytes the heap occupies: header, then the root direct block.
    ///
    /// The block is padded out to an 8-byte boundary so that whatever the
    /// enclosing writer places next stays aligned; the block's *declared* size
    /// is unaffected, because a power of two ≥ 512 is already aligned.
    pub(super) fn bytes(&self) -> usize {
        HEADER_SIZE + pad8(self.block_size as usize)
    }

    /// Place the heap at `base`, header first.
    pub(super) fn assign(&mut self, base: usize) {
        self.header_addr = base;
        self.block_addr = base + HEADER_SIZE;
    }

    /// Address of the "FRHP" header — what a Link Info message points at.
    pub(super) fn header_addr(&self) -> u64 {
        self.header_addr as u64
    }

    /// The heap ID of the object at `index`, as a version-2 B-tree record
    /// stores it.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `index` names no object, which would
    /// mean the index and the heap were built from different link lists.
    pub(super) fn heap_id(&self, index: usize) -> Result<Vec<u8>, OxiH5Error> {
        let offset = self.offsets.get(index).copied().ok_or_else(|| {
            OxiH5Error::Format(format!(
                "internal writer error: fractal heap has no object {index}"
            ))
        })?;
        let len = self.sizes.get(index).copied().unwrap_or(0);

        let mut id = vec![0u8; HEAP_ID_LEN as usize];
        // Version 0 in the upper six bits, type 0 (managed) in the lower two.
        id[0] = 0x00;
        id[1..1 + ID_OFFSET_BYTES].copy_from_slice(&offset.to_le_bytes()[..ID_OFFSET_BYTES]);
        id[1 + ID_OFFSET_BYTES..].copy_from_slice(&(len as u64).to_le_bytes()[..ID_LENGTH_BYTES]);
        Ok(id)
    }

    /// Emit the heap header and its root direct block; returns bytes written.
    ///
    /// `objects` must be the same objects, in the same order and of the same
    /// lengths, that [`Self::plan`] was sized from — which is checked here
    /// rather than assumed, because a heap ID names an *offset* and a single
    /// object that grew between the two passes would move every object after
    /// it out from under its own ID.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `objects` disagrees with the plan, if a
    /// field overflows its on-disk width, or if the block does not fill exactly
    /// the space that was reserved for it.
    pub(super) fn write(&self, buf: &mut [u8], objects: &[Vec<u8>]) -> Result<usize, OxiH5Error> {
        if objects.len() != self.sizes.len() {
            return Err(OxiH5Error::Format(format!(
                "internal writer error: fractal heap was planned for {} objects and given {}",
                self.sizes.len(),
                objects.len()
            )));
        }
        for (index, (object, &size)) in objects.iter().zip(&self.sizes).enumerate() {
            if object.len() != size {
                return Err(OxiH5Error::Format(format!(
                    "internal writer error: fractal heap object {index} was planned at {size} \
                     bytes and is {} bytes",
                    object.len()
                )));
            }
        }

        let block_size = usize::try_from(self.block_size)
            .map_err(|_| OxiH5Error::Format("fractal heap: block size out of range".to_string()))?;
        let used: usize = DIRECT_BLOCK_HEADER + self.sizes.iter().sum::<usize>();
        let free = block_size - used;

        // -- Root direct block, written first so its checksum is final. -------
        let block_base = self.block_addr;
        fill_zero(buf, block_base, pad8(block_size));
        buf[block_base..block_base + 4].copy_from_slice(b"FHDB");
        buf[block_base + 4] = 0x00; // version 0
        write_u64_le(buf, block_base + 5, self.header_addr as u64);
        // Block offset within the heap's address space: this *is* the root, so 0.
        // Its width is `max heap size bits / 8`, not `size_of_offsets`.
        fill_zero(buf, block_base + 13, ID_OFFSET_BYTES);
        // The checksum field stays zero while the checksum is computed over the
        // whole block, which is what `H5HF__cache_dblock_serialize` does.
        for (object, &offset) in objects.iter().zip(&self.offsets) {
            let at = block_base + offset as usize;
            buf[at..at + object.len()].copy_from_slice(object);
        }
        let block_checksum = metadata_checksum(&buf[block_base..block_base + block_size]);
        write_u32_le(buf, block_base + 17, block_checksum);

        // -- Header. ----------------------------------------------------------
        let base = self.header_addr;
        fill_zero(buf, base, HEADER_SIZE);
        buf[base..base + 4].copy_from_slice(b"FRHP");
        buf[base + 4] = 0x00; // version 0
        write_u16_le(buf, base + 5, u16::from(HEAP_ID_LEN));
        write_u16_le(buf, base + 7, 0); // no I/O filters
        buf[base + 9] = FLAG_CHECKSUM_DIRECT_BLOCKS;
        // Objects larger than this go to the "huge" object B-tree; sizing it at
        // the block size means every object this writer stores is managed.
        write_u32_le(
            buf,
            base + 10,
            u32::try_from(MAX_DIRECT_BLOCK_SIZE).unwrap_or(u32::MAX),
        );
        write_u64_le(buf, base + 14, 0); // next huge object ID
        write_u64_le(buf, base + 22, u64::MAX); // huge object B-tree: none
        write_u64_le(buf, base + 30, free as u64); // free space in managed blocks
        write_u64_le(buf, base + 38, u64::MAX); // free space manager: none
        write_u64_le(buf, base + 46, self.block_size); // managed space in heap
        write_u64_le(buf, base + 54, self.block_size); // allocated managed space
        write_u64_le(buf, base + 62, 0); // direct block allocation iterator
        write_u64_le(buf, base + 70, self.sizes.len() as u64);
        write_u64_le(buf, base + 78, 0); // size of huge objects
        write_u64_le(buf, base + 86, 0); // number of huge objects
        write_u64_le(buf, base + 94, 0); // size of tiny objects
        write_u64_le(buf, base + 102, 0); // number of tiny objects
        write_u16_le(buf, base + 110, TABLE_WIDTH);
        write_u64_le(buf, base + 112, self.block_size); // starting block size
        write_u64_le(buf, base + 120, MAX_DIRECT_BLOCK_SIZE);
        write_u16_le(buf, base + 128, MAX_HEAP_SIZE_BITS);
        write_u16_le(buf, base + 130, 1); // starting rows in the root indirect block
        write_u64_le(buf, base + 132, self.block_addr as u64);
        write_u16_le(buf, base + 140, 0); // current rows: the root is direct
        let header_checksum = metadata_checksum(&buf[base..base + HEADER_CHECKSUM_OFFSET]);
        write_u32_le(buf, base + HEADER_CHECKSUM_OFFSET, header_checksum);

        let wrote = self.bytes();
        check_size("fractal heap", block_base + pad8(block_size) - base, wrote)?;
        Ok(wrote)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxih5_format::fractal_heap::FractalHeap;

    fn objects(sizes: &[usize]) -> Vec<Vec<u8>> {
        sizes
            .iter()
            .enumerate()
            .map(|(i, &n)| vec![i as u8 + 1; n])
            .collect()
    }

    /// The header the writer emits must decode field-for-field through the
    /// reader, and every object must come back byte-identical through its own
    /// heap ID — the same two calls `group::list_new_style_links` makes.
    #[test]
    fn objects_round_trip_through_the_reader() {
        let contents = objects(&[23, 15, 15, 40, 7]);
        let sizes: Vec<usize> = contents.iter().map(Vec::len).collect();
        let mut heap = FractalHeapWriter::plan(&sizes).expect("plan");
        heap.assign(0x400);
        let mut buf = vec![0u8; 0x400 + heap.bytes()];
        assert_eq!(
            heap.write(&mut buf, &contents).expect("write"),
            heap.bytes()
        );

        let parsed = FractalHeap::parse(&buf, 0x400, 8).expect("parse");
        assert_eq!(parsed.heap_id_len(), HEAP_ID_LEN);
        assert_eq!(parsed.table_width(), TABLE_WIDTH);
        assert_eq!(parsed.header_address(), 0x400);

        for (i, want) in contents.iter().enumerate() {
            let id = heap.heap_id(i).expect("heap id");
            let (offset, size) = parsed.parse_heap_id(&id).expect("parse id");
            assert_eq!(size, want.len(), "object {i} length");
            let got = parsed.read_object(offset, size).expect("read");
            assert_eq!(&got, want, "object {i} bytes");
        }
    }

    /// The first object sits at offset 21 — the direct block header is inside
    /// the heap's own address space — and objects pack with no padding.
    #[test]
    fn objects_pack_from_offset_twenty_one_with_no_gaps() {
        let heap = FractalHeapWriter::plan(&[23, 15, 15]).expect("plan");
        assert_eq!(heap.offsets, vec![21, 44, 59]);
        // Heap ID layout: type byte, 4-byte offset, 2-byte length.
        assert_eq!(heap.heap_id(1).expect("id"), vec![0x00, 44, 0, 0, 0, 15, 0]);
    }

    /// The block size is the smallest power of two ≥ 512 that holds everything,
    /// which for the libhdf5 file this was pinned against is exactly 512.
    #[test]
    fn the_block_size_is_the_smallest_power_of_two_that_fits() {
        // 21 + 23 + 12*15 = 224 bytes of content: libhdf5 chose 512 too.
        let mut sizes = vec![23usize];
        sizes.extend(std::iter::repeat_n(15usize, 12));
        let heap = FractalHeapWriter::plan(&sizes).expect("plan");
        assert_eq!(heap.block_size, 512);

        // One byte past 512 doubles it.
        let heap = FractalHeapWriter::plan(&[512 - DIRECT_BLOCK_HEADER + 1]).expect("plan");
        assert_eq!(heap.block_size, 1024);

        // And a heap that cannot fit one maximum-size block is refused rather
        // than written with an indirect root this writer cannot emit.
        let too_big = vec![2000usize; 40];
        let err = FractalHeapWriter::plan(&too_big).expect_err("must refuse");
        assert!(format!("{err}").contains("root direct block"), "{err}");
    }

    /// A heap with no objects still has a valid header and an empty block.
    #[test]
    fn an_empty_heap_is_still_well_formed() {
        let mut heap = FractalHeapWriter::plan(&[]).expect("plan");
        heap.assign(0x100);
        let mut buf = vec![0u8; 0x100 + heap.bytes()];
        heap.write(&mut buf, &[]).expect("write");
        let parsed = FractalHeap::parse(&buf, 0x100, 8).expect("parse");
        assert_eq!(parsed.root_indirect_rows(), 1);
        assert!(heap.heap_id(0).is_err(), "no object 0 to name");
    }

    /// Both checksums are computed the way libhdf5 computes them: the header's
    /// over its first 142 bytes, the block's over the *whole* block with the
    /// checksum field zeroed.  Recomputing them here is the only way to catch a
    /// writer that fills the field with something plausible but wrong — the
    /// oxih5 reader does not verify either one.
    #[test]
    fn both_checksums_are_the_ones_libhdf5_would_compute() {
        let contents = objects(&[23, 15]);
        let mut heap = FractalHeapWriter::plan(&[23, 15]).expect("plan");
        heap.assign(0x200);
        let mut buf = vec![0u8; 0x200 + heap.bytes()];
        heap.write(&mut buf, &contents).expect("write");

        let header = &buf[0x200..0x200 + HEADER_SIZE];
        assert_eq!(
            u32::from_le_bytes(
                header[HEADER_CHECKSUM_OFFSET..HEADER_CHECKSUM_OFFSET + 4]
                    .try_into()
                    .expect("4 bytes")
            ),
            metadata_checksum(&header[..HEADER_CHECKSUM_OFFSET])
        );

        let block_base = 0x200 + HEADER_SIZE;
        let block_size = heap.block_size as usize;
        let stored = u32::from_le_bytes(
            buf[block_base + 17..block_base + 21]
                .try_into()
                .expect("4 bytes"),
        );
        let mut zeroed = buf[block_base..block_base + block_size].to_vec();
        zeroed[17..21].fill(0);
        assert_eq!(stored, metadata_checksum(&zeroed));
    }

    /// An object longer than the two-byte length field is refused up front.
    #[test]
    fn an_object_over_the_heap_id_length_field_is_rejected() {
        let err = FractalHeapWriter::plan(&[MAX_OBJECT_SIZE + 1]).expect_err("must refuse");
        assert!(format!("{err}").contains("heap ID can describe"), "{err}");

        // And a set of objects that does not match the plan is refused at write
        // time, rather than shifting every object out from under its heap ID.
        let mut heap = FractalHeapWriter::plan(&[8, 8]).expect("plan");
        heap.assign(0);
        let mut buf = vec![0u8; heap.bytes()];
        let err = heap
            .write(&mut buf, &[vec![0u8; 8], vec![0u8; 9]])
            .expect_err("length drift");
        assert!(format!("{err}").contains("planned at 8 bytes"), "{err}");
        let err = heap
            .write(&mut buf, &[vec![0u8; 8]])
            .expect_err("count drift");
        assert!(format!("{err}").contains("given 1"), "{err}");
    }
}
