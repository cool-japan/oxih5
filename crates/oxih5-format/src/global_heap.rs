use oxih5_core::OxiH5Error;

/// Software decode of an IEEE 754 half-precision float (binary16) to f32.
/// Exported for use by `values.rs` to avoid a circular dependency on oxih5-core.
pub fn f16_to_f32_pub(bits: u16) -> f32 {
    oxih5_core::f16_to_f32(bits)
}

/// Build a minimal in-memory GCOL collection for unit tests.
///
/// Only compiled in test configurations; exposed as `pub` so sibling modules'
/// `#[cfg(test)]` blocks can call it without duplicating the builder.
#[cfg(test)]
pub fn build_gcol_for_test(objects: &[(u16, &[u8])]) -> Vec<u8> {
    build_gcol_bytes(objects)
}

/// Internal GCOL builder used by tests.
#[cfg(test)]
fn build_gcol_bytes(objects: &[(u16, &[u8])]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"GCOL");
    data.push(1); // version
    data.extend_from_slice(&[0u8; 3]); // reserved
    let size_pos = data.len();
    data.extend_from_slice(&[0u8; 8]);

    for (idx, obj_data) in objects {
        data.extend_from_slice(&idx.to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes()); // ref_count
        data.extend_from_slice(&[0u8; 4]); // reserved
        data.extend_from_slice(&(obj_data.len() as u64).to_le_bytes());
        data.extend_from_slice(obj_data);
        let pad = (8 - (data.len() % 8)) % 8;
        data.extend(std::iter::repeat(0u8).take(pad));
    }
    data.extend_from_slice(&[0u8; 16]); // NIL terminator
    let total = data.len() as u64;
    data[size_pos..size_pos + 8].copy_from_slice(&total.to_le_bytes());
    data
}

/// Global heap collection — stores variable-length data referenced by VLen datatypes.
pub struct GlobalHeap {
    /// Map from heap object index to object data.
    objects: std::collections::HashMap<u16, Vec<u8>>,
}

impl GlobalHeap {
    /// Parse a global heap collection from `file_data` at the given absolute address.
    pub fn parse(file_data: &[u8], collection_address: u64) -> Result<Self, OxiH5Error> {
        let base = usize::try_from(collection_address).map_err(|_| {
            OxiH5Error::Corrupted(format!(
                "global heap address {collection_address} exceeds addressable range"
            ))
        })?;
        let base4 = base.checked_add(4).ok_or_else(|| {
            OxiH5Error::Corrupted(format!(
                "global heap address {collection_address} too large"
            ))
        })?;

        // "GCOL" signature (4 bytes)
        let sig = file_data
            .get(base..base4)
            .ok_or_else(|| OxiH5Error::Format("GlobalHeap: truncated at signature".into()))?;
        if sig != b"GCOL" {
            return Err(OxiH5Error::Format(format!(
                "GlobalHeap: bad signature {:?}",
                sig
            )));
        }

        // version (1 byte) — must be 1
        let version = *file_data
            .get(base4)
            .ok_or_else(|| OxiH5Error::Format("GlobalHeap: missing version".into()))?;
        if version != 1 {
            return Err(OxiH5Error::Format(format!(
                "GlobalHeap: unsupported version {}",
                version
            )));
        }

        // reserved (3 bytes), collection size (8 bytes)
        let base8 = base
            .checked_add(8)
            .ok_or_else(|| OxiH5Error::Corrupted("GlobalHeap: base+8 overflows".into()))?;
        let base16 = base
            .checked_add(16)
            .ok_or_else(|| OxiH5Error::Corrupted("GlobalHeap: base+16 overflows".into()))?;
        let collection_size_raw = u64::from_le_bytes(
            file_data
                .get(base8..base16)
                .ok_or_else(|| OxiH5Error::Format("GlobalHeap: truncated at size".into()))?
                .try_into()
                .map_err(|_| OxiH5Error::Format("GlobalHeap: size bytes".into()))?,
        );
        let collection_size = usize::try_from(collection_size_raw).map_err(|_| {
            OxiH5Error::Corrupted(format!(
                "GlobalHeap: collection size {collection_size_raw} exceeds addressable range"
            ))
        })?;

        let heap_end = base.checked_add(collection_size).ok_or_else(|| {
            OxiH5Error::Corrupted(format!(
                "GlobalHeap: base+collection_size overflows: {base}+{collection_size}"
            ))
        })?;
        let mut pos = base16; // first object starts at base + 16
        let mut objects = std::collections::HashMap::new();

        // Parse heap objects until we reach the end or hit a NIL terminator
        while let Some(pos_end) = pos.checked_add(16) {
            if pos_end > heap_end || pos >= file_data.len() {
                break;
            }
            // Heap object header: index (2), ref count (2), reserved (4), object size (8)
            let idx = u16::from_le_bytes(
                file_data
                    .get(pos..pos + 2)
                    .ok_or_else(|| OxiH5Error::Format("GlobalHeap: object index".into()))?
                    .try_into()
                    .map_err(|_| OxiH5Error::Format("GlobalHeap: index bytes".into()))?,
            );

            if idx == 0 {
                // NIL terminator — end of collection
                break;
            }

            // ref_count at pos+2 (2 bytes) — skip
            // reserved at pos+4 (4 bytes) — skip
            let obj_size_raw = u64::from_le_bytes(
                file_data
                    .get(pos + 8..pos + 16)
                    .ok_or_else(|| OxiH5Error::Format("GlobalHeap: object size".into()))?
                    .try_into()
                    .map_err(|_| OxiH5Error::Format("GlobalHeap: size bytes".into()))?,
            );
            let obj_size = usize::try_from(obj_size_raw).map_err(|_| {
                OxiH5Error::Corrupted(format!(
                    "GlobalHeap: object {idx} size {obj_size_raw} exceeds addressable range"
                ))
            })?;

            pos = pos_end; // advance past object header (pos + 16)

            let data_end = pos.checked_add(obj_size).ok_or_else(|| {
                OxiH5Error::Corrupted(format!(
                    "GlobalHeap: object {idx} data overflows: pos={pos} size={obj_size}"
                ))
            })?;
            let data = file_data
                .get(pos..data_end)
                .ok_or_else(|| {
                    OxiH5Error::Format(format!("GlobalHeap: object {} data truncated", idx))
                })?
                .to_vec();
            objects.insert(idx, data);

            // Advance to the next object.  HDF5 aligns each heap object on an
            // 8-byte boundary *relative to the collection's start* — the
            // collection itself is not guaranteed to begin at an 8-byte file
            // offset (libhdf5 / netCDF-C routinely place a GCOL at an odd file
            // address, e.g. 4763).  Aligning the absolute file position instead
            // of the collection-relative offset silently truncated every object
            // after the first whenever `base % 8 != 0`, so a multi-object
            // collection reported "object N not found" for N >= 2.
            let rel = data_end.checked_sub(base).ok_or_else(|| {
                OxiH5Error::Corrupted("GlobalHeap: object end precedes collection base".into())
            })?;
            let rel_aligned = rel
                .checked_add(7)
                .ok_or_else(|| OxiH5Error::Corrupted("GlobalHeap: alignment overflows".into()))?
                & !7usize;
            pos = base.checked_add(rel_aligned).ok_or_else(|| {
                OxiH5Error::Corrupted("GlobalHeap: aligned position overflows".into())
            })?;
        }

        Ok(Self { objects })
    }

    /// Retrieve the data for a heap object by index.
    pub fn object(&self, index: u16) -> Result<&[u8], OxiH5Error> {
        self.objects
            .get(&index)
            .map(Vec::as_slice)
            .ok_or_else(|| OxiH5Error::Format(format!("GlobalHeap: object {} not found", index)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_gcol(objects: &[(u16, &[u8])]) -> Vec<u8> {
        build_gcol_bytes(objects)
    }

    #[test]
    fn test_global_heap_basic() {
        let gcol = build_gcol(&[(1, b"hello"), (2, b"world!")]);
        let heap = GlobalHeap::parse(&gcol, 0).unwrap();
        assert_eq!(heap.object(1).unwrap(), b"hello");
        assert_eq!(heap.object(2).unwrap(), b"world!");
        assert!(heap.object(3).is_err());
    }

    #[test]
    fn test_global_heap_bad_signature() {
        let mut gcol = build_gcol(&[(1, b"test")]);
        gcol[0] = b'X'; // corrupt signature
        assert!(GlobalHeap::parse(&gcol, 0).is_err());
    }

    #[test]
    fn test_global_heap_nil_terminator() {
        let gcol = build_gcol(&[]);
        let heap = GlobalHeap::parse(&gcol, 0).unwrap();
        assert!(heap.object(1).is_err());
    }

    #[test]
    fn test_global_heap_bad_version() {
        let mut gcol = build_gcol(&[(1, b"data")]);
        gcol[4] = 2; // unsupported version
        assert!(GlobalHeap::parse(&gcol, 0).is_err());
    }

    #[test]
    fn test_global_heap_offset() {
        // Collection placed at offset 16 in a larger buffer
        let gcol = build_gcol(&[(1, b"offset_test")]);
        let mut buf = vec![0u8; 16];
        buf.extend_from_slice(&gcol);
        let heap = GlobalHeap::parse(&buf, 16).unwrap();
        assert_eq!(heap.object(1).unwrap(), b"offset_test");
    }

    /// Build a libhdf5-conformant collection (B001/B002 layout): objects, then a
    /// real free-space object at index 0 whose size is `declared - offset`, then
    /// zero padding up to `declared` (>= 4096).
    fn build_gcol_new_format(objects: &[(u16, &[u8])], declared: usize) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(b"GCOL");
        data.push(1);
        data.extend_from_slice(&[0u8; 3]);
        data.extend_from_slice(&(declared as u64).to_le_bytes());
        for (idx, obj) in objects {
            data.extend_from_slice(&idx.to_le_bytes());
            data.extend_from_slice(&1u16.to_le_bytes()); // ref_count
            data.extend_from_slice(&[0u8; 4]);
            data.extend_from_slice(&(obj.len() as u64).to_le_bytes());
            data.extend_from_slice(obj);
            let pad = (8 - (data.len() % 8)) % 8;
            data.extend(std::iter::repeat(0u8).take(pad));
        }
        let free_off = data.len();
        let leftover = declared - free_off;
        if leftover >= 16 {
            data.extend_from_slice(&0u16.to_le_bytes()); // index 0 = free space
            data.extend_from_slice(&0u16.to_le_bytes()); // ref_count 0
            data.extend_from_slice(&[0u8; 4]);
            data.extend_from_slice(&(leftover as u64).to_le_bytes());
        }
        data.resize(declared, 0u8);
        data
    }

    /// B002 reader compat: a collection whose leftover is a real (non-zero)
    /// free-space object must read its real objects and stop at index 0 — not
    /// hang or mistake the free space for data.
    #[test]
    fn reads_new_format_with_real_free_space_object() {
        let gcol = build_gcol_new_format(&[(1, b"alpha"), (2, b"beta"), (3, b"gamma")], 4096);
        assert_eq!(gcol.len(), 4096);
        let heap = GlobalHeap::parse(&gcol, 0).unwrap();
        assert_eq!(heap.object(1).unwrap(), b"alpha");
        assert_eq!(heap.object(2).unwrap(), b"beta");
        assert_eq!(heap.object(3).unwrap(), b"gamma");
        assert!(heap.object(4).is_err());
    }

    /// A completely-full collection carries no free-space object at all; the
    /// reader must stop when it reaches the declared end.
    #[test]
    fn reads_full_collection_without_free_object() {
        // 16 header + align8(16+2)=24 -> content_end 40; declare exactly 40.
        let gcol = build_gcol_new_format(&[(1, b"ab")], 40);
        assert_eq!(gcol.len(), 40);
        // No index-0 object present.
        let heap = GlobalHeap::parse(&gcol, 0).unwrap();
        assert_eq!(heap.object(1).unwrap(), b"ab");
        assert!(heap.object(2).is_err());
    }

    /// B009 reader compat: 0.2.1-era heaps store strings with a trailing NUL and
    /// a size-0 terminator; the reader must still return the raw stored bytes so
    /// the string layer can trim the terminator.
    #[test]
    fn reads_old_format_nul_terminated_objects() {
        let gcol = build_gcol(&[(1, b"hello\0"), (2, b"world\0")]);
        let heap = GlobalHeap::parse(&gcol, 0).unwrap();
        assert_eq!(heap.object(1).unwrap(), b"hello\0");
        assert_eq!(heap.object(2).unwrap(), b"world\0");
    }

    /// W3 regression: HDF5 aligns heap objects on an 8-byte boundary *relative
    /// to the collection start*, and the collection is not guaranteed to begin
    /// at an 8-byte file offset.  netCDF-C, for example, places a multi-object
    /// GCOL at file offset 4763 (`% 8 == 3`); aligning the absolute file
    /// position dropped every object after the first ("object 2 not found").
    /// Placing the same collection at a `% 8 != 0` file offset must still read
    /// objects 2 and 3.
    #[test]
    fn reads_multi_object_collection_at_unaligned_base() {
        let gcol = build_gcol(&[(1, b"time-axis"), (2, "café温度".as_bytes()), (3, b"z")]);
        // Prepend 3 pad bytes so the collection base is not 8-byte aligned.
        for pad in [1usize, 3, 5, 7] {
            let mut buf = vec![0u8; pad];
            buf.extend_from_slice(&gcol);
            let heap = GlobalHeap::parse(&buf, pad as u64)
                .unwrap_or_else(|e| panic!("parse at base {pad}: {e}"));
            assert_eq!(heap.object(1).unwrap(), b"time-axis", "base {pad}");
            assert_eq!(heap.object(2).unwrap(), "café温度".as_bytes(), "base {pad}");
            assert_eq!(heap.object(3).unwrap(), b"z", "base {pad}");
        }
    }
}
