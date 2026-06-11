//! HDF5 Global Heap Collection (GCOL) writer — W0d.
//!
//! [`GlobalHeapWriter`] accumulates heap objects (arbitrary byte slices or
//! UTF-8 strings) and serializes them into the canonical GCOL binary format
//! described in the HDF5 file-format specification §III.E.
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
//!   Bytes +2– +3: reference count  (u16 LE, = 1)
//!   Bytes +4– +7: reserved         (zero)
//!   Bytes +8–+15: object size      (u64 LE, byte count of object data)
//!   Bytes +16…:   object data
//!   [padding to 8-byte boundary]
//!
//! Followed by NIL terminator (16 zero bytes):
//!   index=0, ref_count=0, reserved=0, size=0
//! ```
//!
//! Total collection size = 16 (header) + Σ align8(16 + obj_len) + 16 (NIL).
//!
//! ## Vlen string convention
//!
//! Each string is stored with a NUL terminator (added by [`GlobalHeapWriter::write_string`]).
//! The on-disk vlen reference pointing to a heap object has the layout:
//! ```text
//! [0–3]:  seq_len (u32 LE) = string_len + 1  (includes NUL)
//! [4–5]:  obj_idx (u16 LE) = 1-based GCOL index
//! [6–7]:  reserved         = 0
//! [8–15]: heap_addr (u64 LE) = absolute address of the GCOL in the file
//! ```

/// HDF5 Global Heap Collection writer.
///
/// Heap objects are stored in insertion order and assigned 1-based indices
/// starting from 1.  Call [`GlobalHeapWriter::build`] to consume the writer
/// and produce the fully serialised GCOL byte vector.
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

    /// Store `data` as a new heap object and return its 1-based index.
    pub fn write_bytes(&mut self, data: &[u8]) -> u32 {
        self.objects.push(data.to_vec());
        self.objects.len() as u32
    }

    /// Store a NUL-terminated UTF-8 string as a heap object.
    ///
    /// The NUL terminator (`\0`) is appended automatically and is included in
    /// the heap object's stored byte count.  Returns the 1-based object index.
    pub fn write_string(&mut self, s: &str) -> u32 {
        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0u8); // NUL terminator
        self.write_bytes(&bytes)
    }

    /// Return `true` if no objects have been added yet.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Return the number of objects stored.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Serialise all stored objects into a GCOL byte vector.
    ///
    /// The returned vector is complete and self-contained — no post-build
    /// patching is required.  The "collection size" field in the header is
    /// set to `bytes.len()`.
    pub fn build(self) -> Vec<u8> {
        // Pre-compute total size:
        //   16 (header) + Σ align8(16 + obj_len) + 16 (NIL terminator)
        let body_size: usize = self
            .objects
            .iter()
            .map(|o| align8_usize(16usize.saturating_add(o.len())))
            .sum();
        let total: usize = 16usize.saturating_add(body_size).saturating_add(16);

        let mut out = Vec::with_capacity(total);

        // ---- GCOL header (16 bytes) ----
        out.extend_from_slice(b"GCOL"); // signature
        out.push(1u8); // version
        out.extend_from_slice(&[0u8; 3]); // reserved
        out.extend_from_slice(&(total as u64).to_le_bytes()); // collection size

        // ---- Object entries ----
        for (i, obj) in self.objects.iter().enumerate() {
            let idx = (i as u16).saturating_add(1); // 1-based
            out.extend_from_slice(&idx.to_le_bytes()); // object index
            out.extend_from_slice(&1u16.to_le_bytes()); // reference count = 1
            out.extend_from_slice(&[0u8; 4]); // reserved
            out.extend_from_slice(&(obj.len() as u64).to_le_bytes()); // object size
            out.extend_from_slice(obj); // data

            // Pad to 8-byte boundary
            let remainder = out.len() % 8;
            if remainder != 0 {
                let padding = 8 - remainder;
                out.extend(std::iter::repeat(0u8).take(padding));
            }
        }

        // ---- NIL terminator (16 zero bytes) ----
        out.extend_from_slice(&[0u8; 16]);

        out
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

/// Round `n` up to the next multiple of 8 (saturating on overflow).
#[inline]
fn align8_usize(n: usize) -> usize {
    n.saturating_add(7) & !7
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::global_heap::GlobalHeap;

    /// Writer output must parse correctly with the existing `GlobalHeap` reader.
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
        let heap = GlobalHeap::parse(&bytes, 0).expect("parse GCOL");

        // Each string is stored with NUL terminator.
        assert_eq!(heap.object(1).expect("obj1"), b"hello\0");
        assert_eq!(heap.object(2).expect("obj2"), b"world\0");
        assert_eq!(heap.object(3).expect("obj3"), b"foo\0");
        assert!(heap.object(4).is_err(), "index 4 should not exist");
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
    fn empty_gcol_is_valid() {
        let w = GlobalHeapWriter::new();
        assert!(w.is_empty());
        let bytes = w.build();
        // Minimum size: 16 (header) + 16 (NIL) = 32
        assert_eq!(bytes.len(), 32);
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
            // Strip the NUL terminator that write_string appended.
            let trimmed = stored.split(|&b| b == 0).next().unwrap_or(stored);
            let decoded = std::str::from_utf8(trimmed).expect("utf8");
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
        assert_eq!(heap.object(1).expect("obj1"), b"offset_test\0");
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
}
