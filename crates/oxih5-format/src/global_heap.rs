use oxih5_core::OxiH5Error;

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

            // Advance to next object (8-byte aligned)
            pos = data_end;
            // Align pos to 8-byte boundary
            pos = pos.saturating_add(7) & !7;
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
        // Build a minimal GCOL collection in memory for testing
        let mut data = Vec::new();
        data.extend_from_slice(b"GCOL");
        data.push(1); // version
        data.extend_from_slice(&[0u8; 3]); // reserved
                                           // placeholder for collection size (will fill in at end)
        let size_pos = data.len();
        data.extend_from_slice(&[0u8; 8]);

        for (idx, obj_data) in objects {
            data.extend_from_slice(&idx.to_le_bytes());
            data.extend_from_slice(&1u16.to_le_bytes()); // ref_count
            data.extend_from_slice(&[0u8; 4]); // reserved
            data.extend_from_slice(&(obj_data.len() as u64).to_le_bytes());
            data.extend_from_slice(obj_data);
            // pad to 8-byte alignment
            let pad = (8 - (data.len() % 8)) % 8;
            data.extend(std::iter::repeat(0u8).take(pad));
        }

        // NIL terminator
        data.extend_from_slice(&[0u8; 16]);

        // Fill in collection size
        let total = data.len() as u64;
        data[size_pos..size_pos + 8].copy_from_slice(&total.to_le_bytes());
        data
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
}
