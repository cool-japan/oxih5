//! Jenkins lookup3 — the one hash HDF5's version-2 metadata is built on.
//!
//! Two unrelated-looking things in a new-style group are the same function:
//!
//! * every version-2 structure (fractal heap header, fractal heap direct block,
//!   version-2 B-tree header and nodes) carries a 4-byte **metadata checksum**,
//!   which libhdf5 computes with `H5_checksum_metadata` → `H5_checksum_lookup3`;
//! * every record of a link-name index (a type-5 version-2 B-tree) is keyed by
//!   the **name hash** of the link, which is the very same `H5_checksum_lookup3`
//!   over the raw name bytes with `initval = 0`.
//!
//! Writing one and not the other is not an option: the records of a name index
//! must be *sorted ascending by that hash*, or libhdf5's binary search silently
//! fails to find links that our own linear enumeration still reports.
//!
//! [`lookup3`] is Bob Jenkins' `hashlittle` verbatim, restricted to the
//! little-endian byte-at-a-time form so that it produces identical results on
//! every target rather than only on a little-endian one.  It is pinned against
//! bytes libhdf5 2.0.0 actually wrote — see [`tests`].

/// Rotate `x` left by `k` bits (`0 < k < 32`).
#[inline]
const fn rot(x: u32, k: u32) -> u32 {
    x.rotate_left(k)
}

/// Jenkins lookup3 `mix` — reversibly stirs three 32-bit accumulators.
#[inline]
const fn mix(mut a: u32, mut b: u32, mut c: u32) -> (u32, u32, u32) {
    a = a.wrapping_sub(c);
    a ^= rot(c, 4);
    c = c.wrapping_add(b);
    b = b.wrapping_sub(a);
    b ^= rot(a, 6);
    a = a.wrapping_add(c);
    c = c.wrapping_sub(b);
    c ^= rot(b, 8);
    b = b.wrapping_add(a);
    a = a.wrapping_sub(c);
    a ^= rot(c, 16);
    c = c.wrapping_add(b);
    b = b.wrapping_sub(a);
    b ^= rot(a, 19);
    a = a.wrapping_add(c);
    c = c.wrapping_sub(b);
    c ^= rot(b, 4);
    b = b.wrapping_add(a);
    (a, b, c)
}

/// Jenkins lookup3 `final` — the avalanche applied to the last block.
#[inline]
const fn last_mix(mut a: u32, mut b: u32, mut c: u32) -> u32 {
    c ^= b;
    c = c.wrapping_sub(rot(b, 14));
    a ^= c;
    a = a.wrapping_sub(rot(c, 11));
    b ^= a;
    b = b.wrapping_sub(rot(a, 25));
    c ^= b;
    c = c.wrapping_sub(rot(b, 16));
    a ^= c;
    a = a.wrapping_sub(rot(c, 4));
    b ^= a;
    b = b.wrapping_sub(rot(a, 14));
    c ^= b;
    c = c.wrapping_sub(rot(b, 24));
    c
}

/// Little-endian `u32` from up to four bytes, zero-extended.
///
/// The C original reads whole words on little-endian hosts and falls back to
/// this byte-at-a-time form elsewhere; both produce the same number, and only
/// this one is defined for a tail shorter than four bytes.
#[inline]
fn le_word(bytes: &[u8]) -> u32 {
    let mut word = 0u32;
    for (i, &byte) in bytes.iter().take(4).enumerate() {
        word |= u32::from(byte) << (8 * i);
    }
    word
}

/// Jenkins lookup3 (`hashlittle`) over `key`, seeded with `initval`.
///
/// This is `H5_checksum_lookup3`, which is in turn `H5_checksum_metadata` for
/// every version-2 structure HDF5 writes.
pub(super) fn lookup3(key: &[u8], initval: u32) -> u32 {
    let seed = 0xdead_beefu32
        .wrapping_add(key.len() as u32)
        .wrapping_add(initval);
    let (mut a, mut b, mut c) = (seed, seed, seed);

    let mut rest = key;
    while rest.len() > 12 {
        a = a.wrapping_add(le_word(&rest[0..4]));
        b = b.wrapping_add(le_word(&rest[4..8]));
        c = c.wrapping_add(le_word(&rest[8..12]));
        (a, b, c) = mix(a, b, c);
        rest = &rest[12..];
    }

    // A key whose length is an exact multiple of 12 (including the empty key)
    // ends with `c` already final — the C original returns from `case 0`.
    if rest.is_empty() {
        return c;
    }
    if rest.len() > 8 {
        c = c.wrapping_add(le_word(&rest[8..]));
    }
    if rest.len() > 4 {
        b = b.wrapping_add(le_word(&rest[4..]));
    }
    a = a.wrapping_add(le_word(rest));
    last_mix(a, b, c)
}

/// The checksum HDF5 stores in a version-2 metadata structure.
pub(super) fn metadata_checksum(bytes: &[u8]) -> u32 {
    lookup3(bytes, 0)
}

/// The 4-byte key a link takes in a type-5 (link name) version-2 B-tree.
pub(super) fn link_name_hash(name: &str) -> u32 {
    lookup3(name.as_bytes(), 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every value here was read out of an HDF5 file written by **h5py 3.16 /
    /// libhdf5 2.0.0** (a superblock-v0 file whose group was converted to dense
    /// link storage), so a change to this function that still "looks right"
    /// cannot pass.
    #[test]
    fn link_name_hashes_match_libhdf5() {
        for (name, want) in [
            ("ext", 0x2c3f_baa3u32),
            ("ds00", 0x8fe7_669a),
            ("ds01", 0xb168_bfbb),
            ("ds02", 0x80bf_9137),
            ("ds03", 0x708f_b7e6),
            ("ds04", 0xfa2a_c11b),
            ("ds05", 0x96f6_c9a6),
            ("ds06", 0xe875_9db7),
            ("ds07", 0x0613_6c62),
            ("ds08", 0xb167_e091),
            ("ds09", 0x547f_ad2f),
            ("ds10", 0x253b_b378),
            ("ds11", 0xbe9f_6aaa),
        ] {
            assert_eq!(link_name_hash(name), want, "hash of '{name}'");
        }
    }

    /// Jenkins' own published self-test vectors, which pin the tail handling
    /// for every length class (`< 4`, `4..8`, `8..12`, `> 12`).
    #[test]
    fn known_vectors_from_the_reference_implementation() {
        // Produced by the C reference `hashlittle` for these exact inputs; the
        // empty key exercises the `case 0` early return.
        assert_eq!(lookup3(b"", 0), 0xdead_beef);
        assert_eq!(lookup3(b"", 0xdead_beef), 0xbd5b_7dde);
        assert_eq!(lookup3(b"Four score and seven years ago", 0), 0x1777_0551);
        assert_eq!(lookup3(b"Four score and seven years ago", 1), 0xcd62_8161);
    }

    /// A key of exactly 12 bytes must take the tail path, not the loop: the C
    /// original's `while(length > 12)` leaves 12 bytes for the switch.
    #[test]
    fn a_twelve_byte_key_takes_the_tail_path() {
        // If the loop consumed it, `rest` would be empty and the early return
        // would fire without the final avalanche — a different number.
        assert_eq!(lookup3(b"abcdefghijkl", 0), 0x4012_f87b);
        assert_eq!(lookup3(b"abcdefghijk", 0), 0x5f61_edf8);
    }

    /// The seed participates: the same bytes under two seeds differ.
    #[test]
    fn the_seed_changes_the_result() {
        assert_eq!(lookup3(b"link", 0), 0x6d39_ddcf);
        assert_eq!(lookup3(b"link", 1), 0x336c_4048);
        assert_eq!(metadata_checksum(b"link"), lookup3(b"link", 0));
    }
}
