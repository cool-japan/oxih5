use oxih5_core::{FilterInfo, FilterPipeline, OxiH5Error};

/// HDF5 standard filter identifiers (see the HDF5 "Registered Filters" list).
pub mod filter_id {
    /// Deflate / gzip (zlib-wrapped DEFLATE, RFC 1950/1951).
    pub const DEFLATE: u16 = 1;
    /// Shuffle (byte transpose).
    pub const SHUFFLE: u16 = 2;
    /// Fletcher-32 checksum.
    pub const FLETCHER32: u16 = 3;
    /// SZIP (not supported in Pure Rust).
    pub const SZIP: u16 = 4;
    /// N-bit packing.
    pub const NBIT: u16 = 5;
    /// Scale + offset.
    pub const SCALEOFFSET: u16 = 6;
}

/// Decompress an HDF5 *deflate* (filter id 1) chunk.
///
/// HDF5's deflate filter stores a complete zlib stream (RFC 1950: 2-byte header,
/// raw DEFLATE body, 4-byte Adler-32 trailer).  Decompression is delegated to
/// the COOLJAPAN Pure-Rust [`oxiarc_deflate`] crate — never flate2 / miniz.
pub fn inflate_deflate(data: &[u8]) -> Result<Vec<u8>, OxiH5Error> {
    oxiarc_deflate::zlib_decompress(data)
        .map_err(|e| OxiH5Error::Corrupted(format!("deflate filter: zlib inflate failed: {e}")))
}

/// Apply the inverse of an HDF5 filter pipeline to a single raw chunk.
///
/// On write, HDF5 applies filters in the order they appear in the pipeline
/// message; on read we must apply the inverse of each filter in **reverse**
/// order.  Filters whose bit is set in `filter_mask` were skipped for this
/// particular chunk and are therefore not reversed.
///
/// * `raw`         – the raw on-disk chunk bytes (after reading from the file)
/// * `pipeline`    – the dataset's filter pipeline (forward / write order)
/// * `filter_mask` – per-chunk bitmask; bit *i* set ⇒ filter *i* was disabled
/// * `elem_size`   – element size in bytes (needed by the shuffle filter)
///
/// Returns the fully decoded element bytes for the chunk.
pub fn apply_pipeline(
    raw: &[u8],
    pipeline: &FilterPipeline,
    filter_mask: u32,
    elem_size: usize,
) -> Result<Vec<u8>, OxiH5Error> {
    let mut data = raw.to_vec();
    // Reverse order: the last filter applied on write is undone first on read.
    for (idx, filter) in pipeline.filters.iter().enumerate().rev() {
        // A set mask bit means this filter was skipped for this chunk.
        if idx < 32 && (filter_mask >> idx) & 1 == 1 {
            continue;
        }
        data = apply_one_inverse(&data, filter, elem_size)?;
    }
    Ok(data)
}

/// Undo a single filter (the inverse transform used on read).
fn apply_one_inverse(
    data: &[u8],
    filter: &FilterInfo,
    elem_size: usize,
) -> Result<Vec<u8>, OxiH5Error> {
    match filter.id {
        filter_id::DEFLATE => inflate_deflate(data),
        filter_id::SHUFFLE => {
            // The shuffle filter's element size is the first client-data word
            // when present; fall back to the dataset element size otherwise.
            let es = filter
                .client_data
                .first()
                .map(|&v| v as usize)
                .filter(|&v| v > 0)
                .unwrap_or(elem_size);
            if es == 0 {
                return Ok(data.to_vec());
            }
            unshuffle(data, es)
        }
        filter_id::FLETCHER32 => verify_fletcher32(data),
        filter_id::NBIT => {
            // For simple integer nbit: client_data = [count, class=0, sizeof, sign,
            // byte_order, precision, bit_offset, ...]
            if filter.client_data.len() >= 7 && filter.client_data[1] == 0 {
                let sizeof_elem = filter.client_data[2] as usize;
                let precision = filter.client_data[5];
                let bit_offset = filter.client_data[6];
                if sizeof_elem > 0 && precision > 0 {
                    return unpack_nbit(data, sizeof_elem, precision, bit_offset);
                }
            }
            // Fallback for complex types (compound, array, float nbit) or missing descriptor.
            Err(OxiH5Error::UnsupportedFilter(format!(
                "filter id {} (nbit): unsupported type descriptor (class={:?})",
                filter.id,
                filter.client_data.get(1),
            )))
        }
        filter_id::SCALEOFFSET => {
            if data.is_empty() {
                return Ok(Vec::new());
            }
            let min_bits = data[0] as u32;
            if elem_size == 0 {
                return Err(OxiH5Error::Format("scaleoffset: element size is 0".into()));
            }
            if min_bits == 0 {
                // Constant chunk: all elements equal min_val, but we don't know count.
                // Strip the 1-byte header and return the minimum value repeated.
                // The caller (chunked.rs) knows the expected output size; we just return
                // the minimum value padded to a multiple of elem_size.
                // In practice this means: return min_val bytes (the next elem_size bytes).
                if data.len() < 1 + elem_size {
                    return Err(OxiH5Error::Format(
                        "scaleoffset: chunk too short for constant min_val".into(),
                    ));
                }
                // Return just the min_val — chunked.rs will need to replicate it.
                // But we don't know n_elems here. Return raw bytes after header.
                return Ok(data[1..].to_vec());
            }
            if data.len() < 1 + elem_size {
                return Err(OxiH5Error::Format(
                    "scaleoffset: chunk too short for per-chunk header".into(),
                ));
            }
            let min_val_bytes = &data[1..1 + elem_size];
            let packed_data = &data[1 + elem_size..];
            decode_scaleoffset_int(packed_data, elem_size, min_bits, min_val_bytes)
        }
        filter_id::SZIP => {
            #[cfg(feature = "szip")]
            {
                decode_szip(data.to_vec(), filter)
            }
            #[cfg(not(feature = "szip"))]
            {
                Err(OxiH5Error::UnsupportedFilter(format!(
                    "szip filter (id {}) requires the `szip` Cargo feature",
                    filter.id
                )))
            }
        }
        other => Err(OxiH5Error::UnsupportedFilter(format!(
            "unknown filter id {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// SZIP filter helper
// ---------------------------------------------------------------------------

/// Decode a single HDF5 szip-compressed chunk.
///
/// HDF5 wraps the raw AEC bitstream with a 4-byte little-endian
/// `uncompressed_byte_count` header.  The filter parameters come from
/// `filter.client_data` in the order specified by H5Zszip.c:
///   `[options_mask, bits_per_pixel, pixels_per_block, pixels_per_scanline]`.
#[cfg(feature = "szip")]
fn decode_szip(data: Vec<u8>, filter: &FilterInfo) -> Result<Vec<u8>, OxiH5Error> {
    use oxiarc_szip::{decode as szip_decode, SzipParams};

    // HDF5 szip client_data layout (per HDF5 spec H5Zszip.c):
    //   [0] = options_mask        (u32)
    //   [1] = bits_per_pixel      (u32)
    //   [2] = pixels_per_block    (u32)
    //   [3] = pixels_per_scanline (u32) = RSI sample count
    //
    // HDF5 szip options_mask bits:
    //   0x01 = SZ_ALLOW_K13   (k13 mode)
    //   0x02 = SZ_CHIP        (hardware CHIPS encoder)
    //   0x04 = SZ_EC          (error correction)
    //   0x08 = SZ_LSB         (LSB data byte order)
    //   0x10 = SZ_MSB         (MSB data byte order — compressed stream bit order)
    //   0x20 = SZ_NN          (nearest-neighbor preprocessing)
    //   0x80 = SZ_RAW         (raw mode, no HDF5 framing header)
    if filter.client_data.len() < 4 {
        return Err(OxiH5Error::Corrupted(format!(
            "szip filter client_data too short: {} entries, need 4",
            filter.client_data.len()
        )));
    }

    let options_mask = filter.client_data[0];
    let bits_per_pixel = filter.client_data[1] as u8;
    let pixels_per_block = filter.client_data[2];
    let pixels_per_scanline = filter.client_data[3];

    const SZ_NN_MASK: u32 = 0x20;
    const SZ_MSB_MASK: u32 = 0x10;
    const SZ_RAW_MASK: u32 = 0x80;

    let nn_preprocess = (options_mask & SZ_NN_MASK) != 0;
    let msb = (options_mask & SZ_MSB_MASK) != 0;
    let raw_mode = (options_mask & SZ_RAW_MASK) != 0;

    // Parse the HDF5 szip framing: a leading little-endian u32 = uncompressed byte count.
    // Then the raw AEC bitstream follows.
    if raw_mode {
        // RAW mode: no framing header; the entire `data` is the AEC bitstream.
        // We cannot know the expected output length without additional context.
        return Err(OxiH5Error::UnsupportedFilter(
            "szip RAW mode (no HDF5 framing) not yet supported — dataset uses an unusual szip variant".into(),
        ));
    }

    if data.len() < 4 {
        return Err(OxiH5Error::Corrupted(
            "szip compressed data too short: missing 4-byte uncompressed length header".into(),
        ));
    }

    let uncompressed_bytes = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let aec_stream = &data[4..];

    if bits_per_pixel == 0 {
        return Err(OxiH5Error::Corrupted(
            "szip filter client_data[1] (bits_per_pixel) is 0 — cannot decode".into(),
        ));
    }

    let bytes_per_sample = (bits_per_pixel as usize).div_ceil(8);
    let samples = uncompressed_bytes
        .checked_div(bytes_per_sample)
        .unwrap_or(0);

    let params = SzipParams {
        bits_per_pixel,
        pixels_per_block,
        samples,
        reference_sample_interval: pixels_per_scanline,
        msb,
        nn_preprocess,
        rsi_byte_align: false,
    };

    let decoded = szip_decode(aec_stream, &params)
        .map_err(|e| OxiH5Error::Corrupted(format!("szip decode error: {e}")))?;

    if decoded.len() != uncompressed_bytes {
        return Err(OxiH5Error::Corrupted(format!(
            "szip decoded {} bytes but header said {}",
            decoded.len(),
            uncompressed_bytes
        )));
    }

    Ok(decoded)
}

/// Un-shuffle bytes: reverses the HDF5 shuffle filter.
///
/// The shuffle filter reorders bytes so that the i-th byte of each element
/// are grouped together, improving compression ratios.
///
/// Given `n_elems` elements of `elem_size` bytes, the shuffled layout groups
/// all byte-positions together:
/// ```text
/// shuffled: [byte[0] of elem[0], byte[0] of elem[1], ..., byte[1] of elem[0], ...]
/// original: [elem[0] bytes, elem[1] bytes, ...]
/// ```
///
/// `elem_size`: number of bytes per element (must be > 0, must divide `data.len()`).
pub fn unshuffle(data: &[u8], elem_size: usize) -> Result<Vec<u8>, OxiH5Error> {
    if elem_size == 0 {
        return Err(OxiH5Error::Format(
            "shuffle: element size must be > 0".into(),
        ));
    }
    if data.len() % elem_size != 0 {
        return Err(OxiH5Error::Format(format!(
            "shuffle: data length {} not divisible by element size {}",
            data.len(),
            elem_size
        )));
    }
    let n_elems = data.len() / elem_size;
    let mut out = vec![0u8; data.len()];
    for byte_pos in 0..elem_size {
        for elem_idx in 0..n_elems {
            out[elem_idx * elem_size + byte_pos] = data[byte_pos * n_elems + elem_idx];
        }
    }
    Ok(out)
}

/// Verify and strip the Fletcher-32 checksum from chunk data.
///
/// The last 4 bytes of `data` are the stored checksum (little-endian).
/// Returns the data with those 4 bytes removed, or an error if the
/// checksum does not match.
pub fn verify_fletcher32(data: &[u8]) -> Result<Vec<u8>, OxiH5Error> {
    if data.len() < 4 {
        return Err(OxiH5Error::Corrupted(
            "fletcher32: data too short to contain checksum".into(),
        ));
    }
    let (payload, checksum_bytes) = data.split_at(data.len() - 4);
    let stored = u32::from_le_bytes([
        checksum_bytes[0],
        checksum_bytes[1],
        checksum_bytes[2],
        checksum_bytes[3],
    ]);
    let computed = fletcher32_compute(payload);
    if stored != computed {
        return Err(OxiH5Error::Corrupted(format!(
            "fletcher32 mismatch: stored={stored:#010x}, computed={computed:#010x}"
        )));
    }
    Ok(payload.to_vec())
}

/// Compute the Fletcher-32 checksum over `data`.
///
/// Fletcher-32 operates on 16-bit words (big-endian interpretation of each pair).
/// If the data length is odd, the last byte is treated as a 16-bit word padded
/// on the right with a zero byte.
fn fletcher32_compute(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;
    let mut i = 0;
    while i + 1 < data.len() {
        let word = u16::from_be_bytes([data[i], data[i + 1]]) as u32;
        sum1 = (sum1 + word) % 65535;
        sum2 = (sum2 + sum1) % 65535;
        i += 2;
    }
    if i < data.len() {
        // Odd byte: treat as high byte of a 16-bit word, low byte = 0.
        let word = (data[i] as u32) << 8;
        sum1 = (sum1 + word) % 65535;
        sum2 = (sum2 + sum1) % 65535;
    }
    (sum2 << 16) | sum1
}

// ---------------------------------------------------------------------------
// Nbit filter helpers
// ---------------------------------------------------------------------------

/// Un-pack the HDF5 nbit filter output for simple integer elements.
///
/// The nbit filter packs `bits_per_elem` significant bits per element into a
/// dense bit stream (most-significant-bit first within each byte).  This
/// function extracts each element and places it at `bit_offset` within an
/// `elem_size_bytes`-wide little-endian output word.
///
/// # Parameters
/// * `data`           – packed bit stream from the filter.
/// * `elem_size_bytes` – output element width in bytes (1, 2, 4, or 8).
/// * `bits_per_elem`  – number of packed bits per element (1 ≤ n ≤ elem_size_bytes * 8).
/// * `bit_offset`     – where the bits land within the output element (LE, starting from LSB).
///
/// The total number of output elements is `(data.len() * 8) / bits_per_elem`.
///
/// # Errors
/// Returns [`OxiH5Error::Format`] if parameters are out of range.
///
/// # Note
/// Full nbit support (compound types, arrays, floating-point nbit) requires
/// parsing the nbit `client_data` descriptor from the filter pipeline message.
/// This helper covers the common case of a single atomic integer type.
pub fn unpack_nbit(
    data: &[u8],
    elem_size_bytes: usize,
    bits_per_elem: u32,
    bit_offset: u32,
) -> Result<Vec<u8>, OxiH5Error> {
    if elem_size_bytes == 0 {
        return Err(OxiH5Error::Format(
            "nbit: elem_size_bytes must be > 0".into(),
        ));
    }
    if bits_per_elem == 0 {
        return Err(OxiH5Error::Format("nbit: bits_per_elem must be > 0".into()));
    }
    let max_bits = elem_size_bytes.saturating_mul(8) as u32;
    if bits_per_elem > max_bits {
        return Err(OxiH5Error::Format(format!(
            "nbit: bits_per_elem {bits_per_elem} exceeds element size {elem_size_bytes} * 8 = {max_bits}"
        )));
    }
    if bit_offset >= max_bits {
        return Err(OxiH5Error::Format(format!(
            "nbit: bit_offset {bit_offset} >= element size in bits {max_bits}"
        )));
    }

    let total_bits = data.len().saturating_mul(8);
    let n_elems = total_bits / bits_per_elem as usize;
    let mut out = vec![0u8; n_elems * elem_size_bytes];

    for i in 0..n_elems {
        let bit_start = i * bits_per_elem as usize;
        let mut value: u64 = 0;
        for b in 0..bits_per_elem as usize {
            let byte_idx = (bit_start + b) / 8;
            // HDF5 nbit: MSB first within each byte.
            let bit_in_byte = 7 - ((bit_start + b) % 8);
            if byte_idx < data.len() {
                let bit = ((data[byte_idx] >> bit_in_byte) & 1) as u64;
                // Place in value: bit `b` maps to bit (bits_per_elem - 1 - b)
                // so bit `0` is the most significant packed bit.
                value |= bit << (bits_per_elem as usize - 1 - b);
            }
        }

        // Shift into position and write as little-endian.
        let value_placed = value << bit_offset;
        let value_bytes = value_placed.to_le_bytes();
        let copy_len = elem_size_bytes.min(8);
        out[i * elem_size_bytes..i * elem_size_bytes + copy_len]
            .copy_from_slice(&value_bytes[..copy_len]);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Scaleoffset filter helpers
// ---------------------------------------------------------------------------

/// Decode the integer variant of the HDF5 scaleoffset filter.
///
/// The scaleoffset filter reduces precision by subtracting a minimum value
/// and storing the difference using only `min_bits` bits per element.  On
/// decode, the inverse is applied:
///
/// ```text
/// output[i] = packed_value[i] + minimum_value
/// ```
///
/// # Parameters
/// * `data`          – the packed data bytes (nbit-packed with `min_bits` per element).
/// * `elem_size`     – output element width in bytes (1, 2, 4, or 8).
/// * `min_bits`      – bits used per packed element (from the per-chunk header).
/// * `min_val_bytes` – the stored minimum value, little-endian, `elem_size` bytes long.
///
/// If `elem_size` is 0 or `min_bits` is 0, the input is returned unchanged.
///
/// # Note
/// The per-chunk minimum value and `min_bits` are embedded in the chunk data
/// stream (not in the filter `client_data`), so callers must parse the
/// per-chunk scaleoffset header before calling this function.
pub fn decode_scaleoffset_int(
    data: &[u8],
    elem_size: usize,
    min_bits: u32,
    min_val_bytes: &[u8],
) -> Result<Vec<u8>, OxiH5Error> {
    if elem_size == 0 || min_bits == 0 {
        // Pass-through: no scaling applied.
        return Ok(data.to_vec());
    }

    // Unpack min_bits-per-element packed integers into elem_size-wide output.
    let unpacked = unpack_nbit(data, elem_size, min_bits, 0)?;
    let n_elems = unpacked.len() / elem_size;
    let mut out = vec![0u8; unpacked.len()];

    // Read the minimum value (little-endian signed integer).
    let min_val: i64 = match elem_size {
        1 => min_val_bytes.first().copied().unwrap_or(0) as i8 as i64,
        2 => {
            let lo = min_val_bytes.first().copied().unwrap_or(0);
            let hi = min_val_bytes.get(1).copied().unwrap_or(0);
            i16::from_le_bytes([lo, hi]) as i64
        }
        4 => {
            let b: [u8; 4] = [
                min_val_bytes.first().copied().unwrap_or(0),
                min_val_bytes.get(1).copied().unwrap_or(0),
                min_val_bytes.get(2).copied().unwrap_or(0),
                min_val_bytes.get(3).copied().unwrap_or(0),
            ];
            i32::from_le_bytes(b) as i64
        }
        8 => {
            let b: [u8; 8] = [
                min_val_bytes.first().copied().unwrap_or(0),
                min_val_bytes.get(1).copied().unwrap_or(0),
                min_val_bytes.get(2).copied().unwrap_or(0),
                min_val_bytes.get(3).copied().unwrap_or(0),
                min_val_bytes.get(4).copied().unwrap_or(0),
                min_val_bytes.get(5).copied().unwrap_or(0),
                min_val_bytes.get(6).copied().unwrap_or(0),
                min_val_bytes.get(7).copied().unwrap_or(0),
            ];
            i64::from_le_bytes(b)
        }
        _ => 0,
    };

    for i in 0..n_elems {
        let src = &unpacked[i * elem_size..(i + 1) * elem_size];
        let packed_val: i64 = match elem_size {
            1 => src[0] as i64,
            2 => i16::from_le_bytes([src[0], src[1]]) as i64,
            4 => i32::from_le_bytes([src[0], src[1], src[2], src[3]]) as i64,
            8 => i64::from_le_bytes([
                src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7],
            ]),
            _ => 0,
        };
        let result = packed_val.wrapping_add(min_val);
        let result_bytes = result.to_le_bytes();
        out[i * elem_size..(i + 1) * elem_size].copy_from_slice(&result_bytes[..elem_size]);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inflate_deflate_roundtrip() {
        // Compress with oxiarc-deflate's zlib encoder, then inflate via our
        // HDF5 deflate-filter wrapper.
        let original: Vec<u8> = (0u8..=255).cycle().take(1000).collect();
        let compressed = oxiarc_deflate::zlib_compress(&original, 6).expect("zlib compress");
        let result = inflate_deflate(&compressed).expect("inflate_deflate");
        assert_eq!(result, original);
    }

    #[test]
    fn test_inflate_deflate_bad_data_errors() {
        // Not a valid zlib stream.
        assert!(inflate_deflate(&[0xFF, 0xFF, 0xFF, 0xFF]).is_err());
    }

    #[test]
    fn test_apply_pipeline_deflate_only() {
        let original = b"the quick brown fox jumps over the lazy dog".to_vec();
        let compressed = oxiarc_deflate::zlib_compress(&original, 6).expect("compress");
        let pipeline = FilterPipeline {
            filters: vec![FilterInfo {
                id: filter_id::DEFLATE,
                name: Some("deflate".into()),
                flags: 0,
                client_data: vec![6],
            }],
        };
        let out = apply_pipeline(&compressed, &pipeline, 0, 1).expect("apply");
        assert_eq!(out, original);
    }

    #[test]
    fn test_apply_pipeline_shuffle_then_deflate() {
        // Write order: shuffle (filter[0]) then deflate (filter[1]).
        // Read order must invert: inflate first, then unshuffle.
        let elem_size = 4usize;
        // 3 i32 elements: 1, 2, 3 (little-endian).
        let original = vec![
            1u8, 0, 0, 0, //
            2, 0, 0, 0, //
            3, 0, 0, 0,
        ];
        // Apply shuffle, then compress the shuffled bytes.
        let shuffled = shuffle_for_test(&original, elem_size);
        let compressed = oxiarc_deflate::zlib_compress(&shuffled, 6).expect("compress");

        let pipeline = FilterPipeline {
            filters: vec![
                FilterInfo {
                    id: filter_id::SHUFFLE,
                    name: Some("shuffle".into()),
                    flags: 0,
                    client_data: vec![elem_size as u32],
                },
                FilterInfo {
                    id: filter_id::DEFLATE,
                    name: Some("deflate".into()),
                    flags: 0,
                    client_data: vec![6],
                },
            ],
        };
        let out = apply_pipeline(&compressed, &pipeline, 0, elem_size).expect("apply");
        assert_eq!(out, original);
    }

    #[test]
    fn test_apply_pipeline_filter_mask_skips() {
        // A pipeline with a single deflate filter, but the per-chunk mask
        // disables it (bit 0 set) → data passes through unchanged.
        let raw = b"uncompressed".to_vec();
        let pipeline = FilterPipeline {
            filters: vec![FilterInfo {
                id: filter_id::DEFLATE,
                name: None,
                flags: 0,
                client_data: vec![],
            }],
        };
        let out = apply_pipeline(&raw, &pipeline, 0b1, 1).expect("apply");
        assert_eq!(out, raw);
    }

    #[test]
    fn test_apply_pipeline_unknown_filter_errors() {
        let pipeline = FilterPipeline {
            filters: vec![FilterInfo {
                id: 999,
                name: Some("bogus".into()),
                flags: 0,
                client_data: vec![],
            }],
        };
        assert!(matches!(
            apply_pipeline(b"x", &pipeline, 0, 1),
            Err(OxiH5Error::UnsupportedFilter(_))
        ));
    }

    /// Forward shuffle (inverse of `unshuffle`) — test helper only.
    fn shuffle_for_test(data: &[u8], elem_size: usize) -> Vec<u8> {
        let n_elems = data.len() / elem_size;
        let mut out = vec![0u8; data.len()];
        for byte_pos in 0..elem_size {
            for elem_idx in 0..n_elems {
                out[byte_pos * n_elems + elem_idx] = data[elem_idx * elem_size + byte_pos];
            }
        }
        out
    }

    #[test]
    fn test_unshuffle_roundtrip() {
        // 2 elements of 4 bytes each: [01 02 03 04] [05 06 07 08]
        // Shuffled layout groups all byte-0s, then byte-1s, etc.:
        //   [01 05 | 02 06 | 03 07 | 04 08]
        // Unshuffling should recover:
        //   [01 02 03 04 | 05 06 07 08]
        let shuffled = vec![0x01_u8, 0x05, 0x02, 0x06, 0x03, 0x07, 0x04, 0x08];
        let original = vec![0x01_u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let result = unshuffle(&shuffled, 4).expect("unshuffle failed");
        assert_eq!(result, original);
    }

    #[test]
    fn test_unshuffle_elem_size_1_is_identity() {
        let data = vec![1u8, 2, 3, 4, 5];
        let result = unshuffle(&data, 1).expect("unshuffle failed");
        assert_eq!(result, data);
    }

    #[test]
    fn test_unshuffle_zero_elem_size_errors() {
        let data = vec![1u8, 2, 3, 4];
        assert!(unshuffle(&data, 0).is_err());
    }

    #[test]
    fn test_unshuffle_misaligned_errors() {
        // 5 bytes is not divisible by 4.
        let data = vec![1u8, 2, 3, 4, 5];
        assert!(unshuffle(&data, 4).is_err());
    }

    #[test]
    fn test_fletcher32_valid() {
        let payload = b"abcdefgh";
        let checksum = fletcher32_compute(payload);
        let mut with_checksum = payload.to_vec();
        with_checksum.extend_from_slice(&checksum.to_le_bytes());
        let result = verify_fletcher32(&with_checksum).expect("verify failed");
        assert_eq!(result, payload);
    }

    #[test]
    fn test_fletcher32_corrupted() {
        let payload = b"hello world";
        let mut with_bad = payload.to_vec();
        with_bad.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
        assert!(verify_fletcher32(&with_bad).is_err());
    }

    #[test]
    fn test_fletcher32_too_short() {
        assert!(verify_fletcher32(&[0u8; 3]).is_err());
    }

    #[test]
    fn test_unshuffle_2d_elements() {
        // 3 elements of 2 bytes each: [AA BB] [CC DD] [EE FF]
        // Shuffled: [AA CC EE | BB DD FF]
        let shuffled = vec![0xAA_u8, 0xCC, 0xEE, 0xBB, 0xDD, 0xFF];
        let expected = vec![0xAA_u8, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        let result = unshuffle(&shuffled, 2).expect("unshuffle failed");
        assert_eq!(result, expected);
    }

    // -----------------------------------------------------------------------
    // nbit helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_unpack_nbit_8bit_identity() {
        // When bits_per_elem == 8 and elem_size_bytes == 1, output must equal input.
        let data = vec![10u8, 20, 30, 255];
        let out = unpack_nbit(&data, 1, 8, 0).expect("unpack_nbit failed");
        assert_eq!(out, data, "8-bit nbit should be identity");
    }

    #[test]
    fn test_unpack_nbit_4bit_to_u8() {
        // Pack four 4-bit values into 2 bytes:
        //   values: [3, 7, 15, 0]
        //   binary: 0011 0111 1111 0000
        //   bytes:  0x37, 0xF0
        let packed = vec![0x37_u8, 0xF0];
        let out = unpack_nbit(&packed, 1, 4, 0).expect("unpack_nbit failed");
        assert_eq!(out.len(), 4, "expected 4 output bytes");
        assert_eq!(out[0], 3, "first nibble should be 3");
        assert_eq!(out[1], 7, "second nibble should be 7");
        assert_eq!(out[2], 15, "third nibble should be 15");
        assert_eq!(out[3], 0, "fourth nibble should be 0");
    }

    #[test]
    fn test_unpack_nbit_2bit_to_u8() {
        // 4 two-bit values packed into 1 byte: 0b11_10_01_00 = 0xE4
        //   values: [3, 2, 1, 0]
        let packed = vec![0b11_10_01_00_u8];
        let out = unpack_nbit(&packed, 1, 2, 0).expect("unpack_nbit failed");
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], 3);
        assert_eq!(out[1], 2);
        assert_eq!(out[2], 1);
        assert_eq!(out[3], 0);
    }

    #[test]
    fn test_unpack_nbit_bit_offset() {
        // Extract one 4-bit value from a single byte, place at bit_offset=4.
        // Input: 0xA0 = 1010_0000 → first nibble is 10 (0xA).
        // Placed at bit_offset=4 in a u8: 0xA << 4 = 0xA0.
        let packed = vec![0xA0_u8];
        let out = unpack_nbit(&packed, 1, 4, 4).expect("unpack_nbit failed");
        // 2 elements fit in 8 bits / 4 bits_per_elem
        assert_eq!(out.len(), 2);
        // First element: value = 0b1010 = 10, shifted by 4 → 0b1010_0000 = 0xA0
        assert_eq!(out[0], 0xA0u8);
    }

    #[test]
    fn test_unpack_nbit_invalid_params() {
        // elem_size_bytes == 0
        assert!(unpack_nbit(&[0u8], 0, 4, 0).is_err());
        // bits_per_elem == 0
        assert!(unpack_nbit(&[0u8], 1, 0, 0).is_err());
        // bits_per_elem > elem_size_bytes * 8
        assert!(unpack_nbit(&[0u8], 1, 9, 0).is_err());
    }

    #[test]
    fn test_unpack_nbit_empty_input() {
        let out = unpack_nbit(&[], 1, 4, 0).expect("should succeed on empty");
        assert!(out.is_empty());
    }

    // -----------------------------------------------------------------------
    // scaleoffset helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_decode_scaleoffset_int_zero_params_passthrough() {
        // min_bits == 0 → data passes through unchanged.
        let data = vec![1u8, 2, 3];
        let out = decode_scaleoffset_int(&data, 1, 0, &[0]).expect("decode failed");
        assert_eq!(out, data);
    }

    #[test]
    fn test_decode_scaleoffset_int_u8_values() {
        // 3 u8 values [0, 1, 2] packed as 2-bit values.
        // Packed: 0b00_01_10_00 = 0x18 (last two bits pad to fill byte)
        // Wait: values [0, 1, 2] in 2-bit MSB-first packing:
        //   value 0 → 00, value 1 → 01, value 2 → 10
        //   packed: 0b00_01_10_xx where xx = 00 padding → byte = 0b00_01_10_00 = 0x18
        // min_val = 5 → output should be [5, 6, 7]
        // But 0x18 = 0b0001_1000 : first nibble = 0, but we're doing 2-bit fields:
        //   bits [7,6] = 00 → 0
        //   bits [5,4] = 01 → 1
        //   bits [3,2] = 10 → 2
        //   bits [1,0] = 00 → pad (4th elem, ignored)
        let packed = vec![0x18_u8]; // 0b0001_1000
        let min_val = [5u8]; // min = 5
        let out = decode_scaleoffset_int(&packed, 1, 2, &min_val).expect("decode failed");
        // 8 bits / 2 = 4 elements; values [0,1,2,0] + min_val(5) = [5,6,7,5]
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], 5, "elem 0 should be 5 (0+5)");
        assert_eq!(out[1], 6, "elem 1 should be 6 (1+5)");
        assert_eq!(out[2], 7, "elem 2 should be 7 (2+5)");
        assert_eq!(out[3], 5, "elem 3 (pad) should be 5 (0+5)");
    }

    #[test]
    fn test_decode_scaleoffset_int_i16_values() {
        // 2 i16 values with min_bits=4, min_val=-3 (i16).
        // packed values [2, 5] stored as 4-bit:
        //   byte = 0b0010_0101 = 0x25
        // output = [2+(-3), 5+(-3)] = [-1, 2]
        let packed = vec![0b0010_0101_u8];
        let min_val_i16: i16 = -3;
        let min_bytes = min_val_i16.to_le_bytes();
        let out = decode_scaleoffset_int(&packed, 2, 4, &min_bytes).expect("decode failed");
        // 2 i16 elements (4 bytes total)
        assert_eq!(out.len(), 4);
        let v0 = i16::from_le_bytes([out[0], out[1]]);
        let v1 = i16::from_le_bytes([out[2], out[3]]);
        assert_eq!(v0, -1, "first i16 should be -1");
        assert_eq!(v1, 2, "second i16 should be 2");
    }

    #[test]
    fn test_nbit_filter_fallback_unsupported_on_empty_client_data() {
        // Without a valid client_data descriptor (< 7 elements), nbit should
        // return UnsupportedFilter.
        use oxih5_core::{FilterInfo, FilterPipeline};
        let pipeline = FilterPipeline {
            filters: vec![FilterInfo {
                id: filter_id::NBIT,
                name: Some("nbit".into()),
                flags: 0,
                client_data: vec![],
            }],
        };
        let result = apply_pipeline(b"x", &pipeline, 0, 1);
        assert!(
            matches!(result, Err(OxiH5Error::UnsupportedFilter(_))),
            "nbit with empty client_data should return UnsupportedFilter"
        );
    }

    #[test]
    fn test_nbit_filter_fallback_unsupported_on_non_integer_class() {
        // class != 0 (e.g. class=1 for float) should still return UnsupportedFilter.
        use oxih5_core::{FilterInfo, FilterPipeline};
        let pipeline = FilterPipeline {
            filters: vec![FilterInfo {
                id: filter_id::NBIT,
                name: Some("nbit".into()),
                flags: 0,
                // class=1 (float), not integer
                client_data: vec![7, 1, 4, 0, 0, 23, 0],
            }],
        };
        let result = apply_pipeline(b"x", &pipeline, 0, 4);
        assert!(
            matches!(result, Err(OxiH5Error::UnsupportedFilter(_))),
            "nbit with float class should return UnsupportedFilter"
        );
    }

    #[test]
    fn test_apply_pipeline_nbit_integer() {
        // Simulate nbit-compressed i16 data: precision=12, bit_offset=0, sizeof=2
        // Pack [1u16, 2, 4095, 0] into 12 bits each = 6 bytes
        // Then verify apply_pipeline with NBIT filter unwraps them
        let values: Vec<u16> = vec![1, 2, 4095, 0];
        let elem_size = 2usize;
        let precision = 12usize;

        // Pack the values manually (MSB-first bit packing, 12 bits each)
        let packed = {
            let total_bits = values.len() * precision;
            let mut out = vec![0u8; total_bits.div_ceil(8)];
            for (i, &v) in values.iter().enumerate() {
                let bit_start = i * precision;
                for b in 0..precision {
                    let bit = ((v >> (precision - 1 - b)) & 1) as u8;
                    let byte_idx = (bit_start + b) / 8;
                    let bit_in_byte = 7 - ((bit_start + b) % 8);
                    out[byte_idx] |= bit << bit_in_byte;
                }
            }
            out
        };

        let pipeline = FilterPipeline {
            filters: vec![FilterInfo {
                id: filter_id::NBIT,
                name: Some("nbit".into()),
                flags: 0,
                // client_data: [count=7, class=0(int), sizeof=2, sign=1, byte_order=0,
                //               precision=12, bit_offset=0]
                client_data: vec![7, 0, 2, 1, 0, 12, 0],
            }],
        };

        let out = apply_pipeline(&packed, &pipeline, 0, elem_size).expect("nbit apply");
        let out_u16: Vec<u16> = out
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(out_u16, values);
    }

    #[test]
    fn test_apply_pipeline_scaleoffset_integer() {
        // Simulate scaleoffset on i32 data: values [100, 101, 102, 103]
        // min_val = 100, packed values = [0, 1, 2, 3] using 2 bits each
        let elem_size = 4usize;
        let min_bits: u32 = 2; // 2 bits for values 0..3
        let min_val: i32 = 100;
        let relative_values: Vec<u32> = vec![0, 1, 2, 3];

        // Pack relative values into 2-bit nbit stream (MSB first)
        let total_bits = relative_values.len() * min_bits as usize;
        let mut packed = vec![0u8; total_bits.div_ceil(8)];
        for (i, &v) in relative_values.iter().enumerate() {
            let bit_start = i * min_bits as usize;
            for b in 0..min_bits as usize {
                let bit = ((v >> (min_bits as usize - 1 - b)) & 1) as u8;
                let byte_idx = (bit_start + b) / 8;
                let bit_in_byte = 7 - ((bit_start + b) % 8);
                packed[byte_idx] |= bit << bit_in_byte;
            }
        }

        // Build per-chunk scaleoffset header: [min_bits(1)] + [min_val LE(4)] + packed_data
        let mut chunk_data = vec![min_bits as u8];
        chunk_data.extend_from_slice(&min_val.to_le_bytes());
        chunk_data.extend_from_slice(&packed);

        let pipeline = FilterPipeline {
            filters: vec![FilterInfo {
                id: filter_id::SCALEOFFSET,
                name: Some("scaleoffset".into()),
                flags: 0,
                client_data: vec![0, 0], // scale_type=0, scale_factor=0
            }],
        };

        let out = apply_pipeline(&chunk_data, &pipeline, 0, elem_size).expect("scaleoffset apply");
        let out_i32: Vec<i32> = out
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(out_i32, vec![100, 101, 102, 103]);
    }

    // -----------------------------------------------------------------------
    // szip filter tests (require the `szip` feature)
    // -----------------------------------------------------------------------

    #[cfg(feature = "szip")]
    mod szip_tests {
        use super::*;
        use oxih5_core::FilterInfo;

        fn make_szip_filter(options_mask: u32, bpp: u32, ppb: u32, pps: u32) -> FilterInfo {
            FilterInfo {
                id: 4, // filter_id::SZIP
                name: Some("szip".to_string()),
                flags: 0,
                client_data: vec![options_mask, bpp, ppb, pps],
            }
        }

        #[test]
        fn szip_decode_all_zeros_8bpp() {
            use oxiarc_szip::{encode as szip_encode, SzipParams};

            let samples: Vec<u64> = vec![0u64; 64];
            let params = SzipParams {
                bits_per_pixel: 8,
                pixels_per_block: 8,
                samples: 64,
                reference_sample_interval: 8,
                msb: false,
                nn_preprocess: false,
                rsi_byte_align: false,
            };

            let compressed = szip_encode(&samples, &params).expect("encode");

            // Build HDF5 framing: 4-byte LE uncompressed byte count + AEC stream.
            let uncompressed_bytes = 64u32; // 64 samples × 1 byte/sample for 8bpp
            let mut hdf5_stream = uncompressed_bytes.to_le_bytes().to_vec();
            hdf5_stream.extend_from_slice(&compressed);

            // options_mask: SZ_MSB_MASK=0x10 not set (msb=false), SZ_NN_MASK=0x20 not set.
            let filter = make_szip_filter(0, 8, 8, 8);
            let decoded = decode_szip(hdf5_stream, &filter).expect("decode");

            assert_eq!(decoded.len(), 64);
            assert!(
                decoded.iter().all(|&b| b == 0),
                "all-zeros round-trip failed"
            );
        }

        #[test]
        fn szip_decode_malformed_framing_returns_err_not_panic() {
            let filter = make_szip_filter(0, 8, 8, 8);

            // Too short for 4-byte header.
            let result = decode_szip(vec![0x01, 0x02], &filter);
            assert!(result.is_err(), "short input must return Err");

            // 4-byte header claims 1000 bytes but decoded length will differ.
            let mut data = 1000u32.to_le_bytes().to_vec();
            data.push(0);
            let result = decode_szip(data, &filter);
            assert!(result.is_err(), "mismatch must return Err");
        }

        #[test]
        fn szip_missing_client_data_returns_err() {
            let filter = FilterInfo {
                id: 4,
                name: None,
                flags: 0,
                client_data: vec![0, 8], // only 2 entries, need 4
            };
            let result = decode_szip(vec![0u8; 64], &filter);
            assert!(result.is_err(), "missing client_data must return Err");
        }
    }
}
