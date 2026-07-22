//! HDF5 Global Heap Collection (GCOL) writer — W0d.
//!
//! [`GlobalHeapWriter`] accumulates heap objects (arbitrary byte slices or
//! UTF-8 strings) and serializes them into one or more canonical GCOL
//! collections, exactly as described in the HDF5 file-format specification
//! §III.E and as produced by libhdf5/h5py.
//!
//! ## GCOL on-disk format
//!
//! ```text
//! Bytes  0– 3: signature "GCOL"
//! Byte      4: version = 1
//! Bytes  5– 7: reserved (zero)
//! Bytes  8–15: total collection size (LE u64, includes this header)
//!
//! Followed by N object entries (1-indexed), in insertion order:
//!   Bytes +0– +1: heap object index (u16 LE, 1-based)
//!   Bytes +2– +3: reference count  (u16 LE)
//!   Bytes +4– +7: reserved         (zero)
//!   Bytes +8–+15: object size      (u64 LE, byte count of object data)
//!   Bytes +16…:   object data
//!   [padding to 8-byte boundary]
//!
//! Followed — when at least 16 bytes remain — by ONE free-space object
//! (heap index 0):
//!   Bytes +0– +1: index      = 0
//!   Bytes +2– +3: ref_count  = 0
//!   Bytes +4– +7: reserved   = 0
//!   Bytes +8–+15: object size = collection_size − offset_of_this_object
//!                 (LE u64, *includes* this 16-byte header)
//! ```
//!
//! ## libhdf5 conformance (why this is not the naive layout)
//!
//! Three rules make a collection readable by libhdf5/h5py — all verified by
//! hex-dumping heaps written by h5py 3.16 / HDF5 2.0.0:
//!
//! * **Minimum size ([`H5HG_MINSIZE`] = 4096).**  `H5HG__cache_heap_deserialize`
//!   rejects any collection whose declared size is below 4096 *before* it
//!   deserialises a single object, so a tight 56/104/240-byte heap makes every
//!   small vlen-string dataset unreadable.  The collection is padded up to the
//!   next power of two, clamped to `[4096, 65536]`.
//! * **A real free-space object, never a size-0 terminator.**  libhdf5 advances
//!   its parse cursor by the free-space object's size; a size-0 object at index
//!   0 never advances the cursor and hangs the deserialiser forever.  When at
//!   least 16 bytes remain past the last real object, this writer emits one
//!   index-0 free-space object whose size is `collection_size − its_offset`
//!   (the size of the remaining space, header included).  When fewer than 16
//!   bytes remain — or the collection is exactly full — no free-space object is
//!   emitted (there is no room for its header); the trailing bytes are left as
//!   zero padding, exactly as h5py does.
//! * **One collection per 65 536 bytes.**  A collection is capped at
//!   [`H5HG_MAXSIZE`] = 65536 bytes (and never more than
//!   [`MAX_OBJECTS_PER_COLLECTION`] = 65535 objects, since heap object indices
//!   are 16-bit with 0 reserved for free space).  Objects that overflow the cap
//!   spill into a fresh collection; each vlen reference carries the (collection
//!   address, object index) of the collection that actually holds it.  A single
//!   object larger than the cap is placed alone in its own oversized
//!   collection, matching libhdf5.
//!
//! ## Vlen string convention
//!
//! Each string is stored as its raw UTF-8 bytes with **no** NUL terminator and
//! the heap object's size is the string length (`strlen`), matching
//! `H5T__vlen_disk_write`.  The on-disk vlen reference pointing to a heap object
//! has the layout:
//! ```text
//! [0–3]:   seq_len (u32 LE) = string_len   (strlen, no NUL)
//! [4–11]:  heap_addr (u64 LE) = absolute address of the GCOL in the file
//! [12–15]: obj_idx (u32 LE) = 1-based GCOL index within that collection
//! ```

/// HDF5 global-heap minimum collection size (`H5HG_MINSIZE`).  A collection
/// smaller than this is rejected by libhdf5 before any object is read.
pub const H5HG_MINSIZE: usize = 4096;

/// HDF5 global-heap maximum single-collection size (`H5HG_MAXSIZE`).  Content
/// beyond this spills into the next collection.
pub const H5HG_MAXSIZE: usize = 65536;

/// Maximum number of real objects one collection may index.  Heap object
/// indices are 16-bit and index 0 is reserved for the free-space object, so the
/// largest usable real index is 65535.
pub const MAX_OBJECTS_PER_COLLECTION: usize = 65535;

/// Where one heap object landed once the writer split its objects across
/// collections.
///
/// The collection *address* is not known until the file is laid out, so this
/// carries the collection's 0-based ordinal instead; the caller resolves it to
/// an absolute address at emit time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeapObjectLocation {
    /// 0-based ordinal of the collection that holds the object.
    pub collection: u32,
    /// 1-based object index within that collection (never 0).
    pub index: u32,
}

/// HDF5 Global Heap Collection writer.
///
/// Heap objects are stored in insertion order and assigned 1-based *global*
/// ordinals (across every collection) as they are added.  Call
/// [`GlobalHeapWriter::build_collections`] to consume the writer and produce
/// one serialised collection per 65 536 bytes of content, together with the
/// per-object [`HeapObjectLocation`] map the caller needs to fill in each vlen
/// reference.
///
/// # Example
/// ```rust
/// use oxih5_format::GlobalHeapWriter;
///
/// let mut w = GlobalHeapWriter::new();
/// let idx = w.write_string("hello");
/// assert_eq!(idx, 1);
/// let bytes = w.build();
/// assert_eq!(&bytes[0..4], b"GCOL");
/// ```
#[derive(Debug, Default)]
pub struct GlobalHeapWriter {
    objects: Vec<Vec<u8>>,
}

impl GlobalHeapWriter {
    /// Create a new, empty writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `data` as a new heap object and return its 1-based global ordinal.
    pub fn write_bytes(&mut self, data: &[u8]) -> u32 {
        self.objects.push(data.to_vec());
        self.objects.len() as u32
    }

    /// Store a UTF-8 string as a heap object.
    ///
    /// The string is stored as its raw bytes with **no** NUL terminator; the
    /// heap object's size is `strlen`, matching libhdf5's `H5T__vlen_disk_write`.
    /// Returns the 1-based global ordinal.
    pub fn write_string(&mut self, s: &str) -> u32 {
        self.write_bytes(s.as_bytes())
    }

    /// Return `true` if no objects have been added yet.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Return the number of objects stored.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Serialise all stored objects into one or more GCOL collections.
    ///
    /// Objects are packed greedily, in insertion order, into collections of at
    /// most [`H5HG_MAXSIZE`] bytes (and [`MAX_OBJECTS_PER_COLLECTION`] objects);
    /// a single object larger than the cap gets its own oversized collection.
    ///
    /// Returns the serialised collections (in address order) and a parallel
    /// map, indexed by 0-based global ordinal, describing which collection and
    /// which 1-based local index each object landed at.  The caller places the
    /// collections at consecutive file addresses and uses the map to fill in
    /// every vlen reference's `(heap_addr, obj_idx)`.
    pub fn build_collections(self) -> (Vec<Vec<u8>>, Vec<HeapObjectLocation>) {
        let mut collections: Vec<Vec<u8>> = Vec::new();
        let mut locations: Vec<HeapObjectLocation> = Vec::with_capacity(self.objects.len());

        // Objects assigned to the collection currently being filled.
        let mut current: Vec<&[u8]> = Vec::new();
        // Content bytes used so far in `current` (header included).
        let mut content_used = COLLECTION_HEADER;

        for obj in &self.objects {
            let need = align8_usize(OBJECT_HEADER.saturating_add(obj.len()));
            let would_use = content_used.saturating_add(need);
            let over_size = would_use > H5HG_MAXSIZE;
            let over_count = current.len() >= MAX_OBJECTS_PER_COLLECTION;

            if !current.is_empty() && (over_size || over_count) {
                collections.push(serialize_collection(&current));
                current.clear();
                content_used = COLLECTION_HEADER;
            }

            locations.push(HeapObjectLocation {
                collection: collections.len() as u32,
                index: (current.len() + 1) as u32,
            });
            current.push(obj);
            content_used = content_used.saturating_add(need);
        }

        if !current.is_empty() {
            collections.push(serialize_collection(&current));
        }

        (collections, locations)
    }

    /// Serialise every stored object into a single GCOL collection.
    ///
    /// A convenience for callers (and tests) that know their objects fit in one
    /// collection; the general path is [`Self::build_collections`].  The
    /// returned collection is padded and terminated exactly like every other
    /// collection this writer produces.
    pub fn build(self) -> Vec<u8> {
        let refs: Vec<&[u8]> = self.objects.iter().map(Vec::as_slice).collect();
        serialize_collection(&refs)
    }
}

/// A reference to an object in a Global Heap Collection.
///
/// `collection_addr` must be filled in by the caller after the GCOL bytes
/// have been placed at a known absolute address in the target file.
#[derive(Debug, Clone, Copy)]
pub struct GlobalHeapRef {
    /// Absolute file offset of the GCOL (filled in after layout).
    pub collection_addr: u64,
    /// 1-based index of the referenced object within this collection.
    pub object_idx: u32,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Bytes of the collection header (signature, version, reserved, size).
const COLLECTION_HEADER: usize = 16;

/// Bytes of a per-object header (index, ref count, reserved, size).
const OBJECT_HEADER: usize = 16;

/// Serialise one collection from `objects` (in order), applying the
/// libhdf5-conformant sizing, free-space, and padding rules.
fn serialize_collection(objects: &[&[u8]]) -> Vec<u8> {
    // Content end = header + Σ align8(obj_header + obj_len); the offset at which
    // the free-space object (if any) begins.
    let content_end: usize = objects.iter().fold(COLLECTION_HEADER, |acc, obj| {
        acc.saturating_add(align8_usize(OBJECT_HEADER.saturating_add(obj.len())))
    });
    let declared = collection_size(content_end);

    let mut out = Vec::with_capacity(declared);

    // ---- GCOL header (16 bytes) ----
    out.extend_from_slice(b"GCOL"); // signature
    out.push(1u8); // version
    out.extend_from_slice(&[0u8; 3]); // reserved
    out.extend_from_slice(&(declared as u64).to_le_bytes()); // collection size

    // ---- Object entries ----
    for (i, obj) in objects.iter().enumerate() {
        let idx = (i as u16).saturating_add(1); // 1-based; never overflows a capped collection
        out.extend_from_slice(&idx.to_le_bytes()); // object index
        out.extend_from_slice(&1u16.to_le_bytes()); // reference count = 1
        out.extend_from_slice(&[0u8; 4]); // reserved
        out.extend_from_slice(&(obj.len() as u64).to_le_bytes()); // object size = strlen
        out.extend_from_slice(obj); // data (no NUL terminator)

        // Pad to 8-byte boundary.
        let remainder = out.len() % 8;
        if remainder != 0 {
            out.resize(out.len() + (8 - remainder), 0u8);
        }
    }

    // At this point `out.len()` equals `content_end`.
    debug_assert_eq!(out.len(), content_end);

    // ---- Free-space object (only when >= 16 bytes remain) ----
    let leftover = declared.saturating_sub(content_end);
    if leftover >= OBJECT_HEADER {
        out.extend_from_slice(&0u16.to_le_bytes()); // index 0 = free space
        out.extend_from_slice(&0u16.to_le_bytes()); // ref count 0
        out.extend_from_slice(&[0u8; 4]); // reserved
        out.extend_from_slice(&(leftover as u64).to_le_bytes()); // size incl. this header
    }

    // Zero-pad up to the declared collection size (free-space body, or the
    // < 16-byte tail that has no room for a free-space header).
    out.resize(declared, 0u8);
    out
}

/// The declared size of a collection whose content occupies `content_end`
/// bytes, matching libhdf5/h5py: the next power of two, clamped to
/// `[H5HG_MINSIZE, H5HG_MAXSIZE]`; a single object larger than the cap yields a
/// tightly-sized oversized collection.
fn collection_size(content_end: usize) -> usize {
    if content_end <= H5HG_MAXSIZE {
        let mut size = H5HG_MINSIZE;
        while size < content_end {
            size <<= 1;
        }
        size
    } else {
        // Oversized single object: content_end is already 8-aligned, so it is
        // exactly full with no free-space object — as h5py writes it.
        align8_usize(content_end)
    }
}

/// Round `n` up to the next multiple of 8 (saturating on overflow).
#[inline]
fn align8_usize(n: usize) -> usize {
    n.saturating_add(7) & !7
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::global_heap::GlobalHeap;

    /// Every collection this writer produces must be at least `H5HG_MINSIZE`,
    /// with the declared size matching the byte length.
    fn assert_conformant(bytes: &[u8]) {
        assert_eq!(&bytes[0..4], b"GCOL", "signature");
        assert_eq!(bytes[4], 1, "version");
        let declared = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        assert_eq!(declared, bytes.len(), "declared size == byte length");
        assert!(declared >= H5HG_MINSIZE, "declared {declared} >= 4096");
        assert_eq!(declared % 8, 0, "8-aligned");
    }

    /// Writer output must parse correctly with the existing `GlobalHeap` reader
    /// and store strings with NO NUL terminator (strlen sizing).
    #[test]
    fn round_trip_strings() {
        let mut w = GlobalHeapWriter::new();
        let i1 = w.write_string("hello");
        let i2 = w.write_string("world");
        let i3 = w.write_string("foo");

        assert_eq!(i1, 1);
        assert_eq!(i2, 2);
        assert_eq!(i3, 3);

        let bytes = w.build();
        assert_conformant(&bytes);
        let heap = GlobalHeap::parse(&bytes, 0).expect("parse GCOL");

        // Each string is stored WITHOUT a NUL terminator (B009).
        assert_eq!(heap.object(1).expect("obj1"), b"hello");
        assert_eq!(heap.object(2).expect("obj2"), b"world");
        assert_eq!(heap.object(3).expect("obj3"), b"foo");
        assert!(heap.object(4).is_err(), "index 4 should not exist");
    }

    #[test]
    fn small_collection_is_padded_to_minsize() {
        let mut w = GlobalHeapWriter::new();
        w.write_string("alpha");
        w.write_string("beta");
        w.write_string("gamma");
        let bytes = w.build();
        assert_eq!(bytes.len(), H5HG_MINSIZE, "3 short strings pad to 4096");
        assert_conformant(&bytes);
    }

    /// The trailing 16 bytes must NOT be a size-0 free object (the B002 hang);
    /// a padded collection must carry a real free-space object whose size is
    /// `declared - offset`.
    #[test]
    fn free_space_object_has_real_size() {
        let mut w = GlobalHeapWriter::new();
        w.write_string("hello");
        let bytes = w.build();
        // Walk to the free-space object.
        let mut pos = 16usize;
        loop {
            let idx = u16::from_le_bytes(bytes[pos..pos + 2].try_into().unwrap());
            let size = u64::from_le_bytes(bytes[pos + 8..pos + 16].try_into().unwrap()) as usize;
            if idx == 0 {
                assert_eq!(size, bytes.len() - pos, "free size = declared - offset");
                assert!(size >= 16, "free object never size 0");
                break;
            }
            pos = (pos + 16 + size + 7) & !7;
            assert!(pos < bytes.len(), "must reach the free-space object");
        }
    }

    #[test]
    fn round_trip_raw_bytes() {
        let mut w = GlobalHeapWriter::new();
        let idx = w.write_bytes(b"rawdata");
        assert_eq!(idx, 1);

        let bytes = w.build();
        let heap = GlobalHeap::parse(&bytes, 0).expect("parse");
        assert_eq!(heap.object(1).expect("obj1"), b"rawdata");
    }

    #[test]
    fn empty_gcol_is_valid_and_padded() {
        let w = GlobalHeapWriter::new();
        assert!(w.is_empty());
        let bytes = w.build();
        // No objects still yields a valid, minimum-size collection.
        assert_eq!(bytes.len(), H5HG_MINSIZE);
        assert_conformant(&bytes);
        let heap = GlobalHeap::parse(&bytes, 0).expect("parse empty");
        assert!(heap.object(1).is_err());
    }

    #[test]
    fn collection_size_in_header_equals_byte_length() {
        let mut w = GlobalHeapWriter::new();
        w.write_string("test_string");
        let bytes = w.build();
        let size_in_header = u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes"));
        assert_eq!(size_in_header as usize, bytes.len());
    }

    #[test]
    fn multiple_strings_round_trip() {
        let inputs = ["", "a", "longer string with spaces", "unicode: \u{00e9}"];
        let mut w = GlobalHeapWriter::new();
        let mut indices = Vec::new();
        for s in &inputs {
            indices.push(w.write_string(s));
        }

        let bytes = w.build();
        let heap = GlobalHeap::parse(&bytes, 0).expect("parse");

        for (i, (&idx, expected)) in indices.iter().zip(inputs.iter()).enumerate() {
            let stored = heap.object(idx as u16).expect("object");
            let decoded = std::str::from_utf8(stored).expect("utf8");
            assert_eq!(
                decoded, *expected,
                "mismatch at index {i}: got {decoded:?} expected {expected:?}"
            );
        }
    }

    #[test]
    fn gcol_placed_at_offset_parseable() {
        let mut w = GlobalHeapWriter::new();
        w.write_string("offset_test");
        let gcol = w.build();

        // Place a 32-byte preamble before the GCOL.
        let offset = 32usize;
        let mut buf = vec![0u8; offset + gcol.len()];
        buf[offset..].copy_from_slice(&gcol);

        let heap = GlobalHeap::parse(&buf, offset as u64).expect("parse at offset");
        assert_eq!(heap.object(1).expect("obj1"), b"offset_test");
    }

    #[test]
    fn is_empty_and_len() {
        let mut w = GlobalHeapWriter::new();
        assert!(w.is_empty());
        assert_eq!(w.len(), 0);

        w.write_bytes(b"x");
        assert!(!w.is_empty());
        assert_eq!(w.len(), 1);

        w.write_string("y");
        assert_eq!(w.len(), 2);
    }

    /// The declared-size formula must reproduce libhdf5/h5py: next power of two,
    /// clamped to [4096, 65536], and oversized objects tightly sized.
    #[test]
    fn collection_size_matches_h5py() {
        assert_eq!(collection_size(16), 4096);
        assert_eq!(collection_size(88), 4096);
        assert_eq!(collection_size(4072), 4096);
        assert_eq!(collection_size(4096), 4096);
        assert_eq!(collection_size(4120), 8192);
        assert_eq!(collection_size(12816), 16384);
        assert_eq!(collection_size(18016), 32768);
        assert_eq!(collection_size(33616), 65536);
        assert_eq!(collection_size(64016), 65536);
        assert_eq!(collection_size(65536), 65536);
        // Oversized single object (content_end already 8-aligned).
        assert_eq!(collection_size(100032), 100032);
    }

    /// A full collection (content_end == declared) carries no free-space object,
    /// which the reader tolerates by reaching heap_end.
    #[test]
    fn exactly_full_collection_has_no_free_object() {
        // 170 two-byte strings: 16 + 170*align8(18)=16+170*24 = 4096, exactly full.
        let mut w = GlobalHeapWriter::new();
        for _ in 0..170 {
            w.write_string("ab");
        }
        let bytes = w.build();
        assert_eq!(bytes.len(), 4096);
        let heap = GlobalHeap::parse(&bytes, 0).expect("parse full");
        assert_eq!(heap.object(1).expect("obj1"), b"ab");
        assert_eq!(heap.object(170).expect("obj170"), b"ab");
    }

    /// More than one collection's worth of objects must split, and every object
    /// gets a location within index range and a readable collection.
    #[test]
    fn splits_across_collections() {
        let mut w = GlobalHeapWriter::new();
        let n = 6000usize; // 6000 * align8(16+9)=32 = 192016 > 65536 -> >=3 collections
        for i in 0..n {
            w.write_string(&format!("item-{i:04}"));
        }
        let (collections, locations) = w.build_collections();
        assert!(collections.len() >= 3, "must split: {}", collections.len());
        assert_eq!(locations.len(), n);

        // Every location points at a real collection, a valid index, and the
        // right string.
        for (i, loc) in locations.iter().enumerate() {
            assert!((loc.collection as usize) < collections.len());
            assert!(loc.index >= 1 && (loc.index as usize) <= MAX_OBJECTS_PER_COLLECTION);
            let heap = GlobalHeap::parse(&collections[loc.collection as usize], 0)
                .expect("parse split collection");
            let got = heap
                .object(loc.index as u16)
                .expect("obj in split collection");
            assert_eq!(got, format!("item-{i:04}").as_bytes());
        }
        for c in &collections {
            assert_conformant(c);
            assert!(c.len() <= H5HG_MAXSIZE, "capped at 65536");
        }
    }

    /// An object larger than the cap gets its own oversized collection.
    #[test]
    fn oversized_object_gets_own_collection() {
        let mut w = GlobalHeapWriter::new();
        w.write_string("small");
        let big = "Z".repeat(100_000);
        w.write_string(&big);
        w.write_string("after");
        let (collections, locations) = w.build_collections();
        assert_eq!(locations.len(), 3);
        // The big object is alone in its collection, which exceeds the cap.
        let big_loc = locations[1];
        assert_eq!(big_loc.index, 1);
        let heap = GlobalHeap::parse(&collections[big_loc.collection as usize], 0).expect("parse");
        assert_eq!(heap.object(1).expect("big").len(), 100_000);
        assert!(collections[big_loc.collection as usize].len() > H5HG_MAXSIZE);
    }
}
