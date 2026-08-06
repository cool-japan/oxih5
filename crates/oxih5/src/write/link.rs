//! Link messages (0x0006) and the two messages that frame them.
//!
//! An HDF5 group stores its members one of two unrelated ways, and this module
//! is the *new-style* one: a Link Info message (0x0002) declaring where the
//! links live, a Group Info message (0x000A), and — for a small group — one
//! **Link message** per member right there in the object header.
//!
//! The writer emits new-style storage only where the old style cannot express
//! what was asked for.  A version-1 symbol table entry has no link-type field:
//! it distinguishes a hard link from a soft link by its *cache type* and has no
//! encoding at all for an external link, so a group holding one must be
//! new-style.  libhdf5 does exactly this — creating an external link in a
//! default (`libver='earliest'`) file converts the whole group to link
//! messages, superblock v0 and object header v1 unchanged, which is the shape
//! this module reproduces.
//!
//! # On-disk form
//!
//! ```text
//! version  1 byte   = 1
//! flags    1 byte   bits 0-1: name length width, 0→1, 1→2, 2→4, 3→8 bytes
//!                   bit 2:    creation order field present (8 bytes)
//!                   bit 3:    link type byte present (absent ⇒ hard)
//!                   bit 4:    charset byte present
//! [type]   1 byte   0 = hard, 1 = soft, 64 = external
//! [order]  8 bytes  creation order
//! [cset]   1 byte   0 = ASCII, 1 = UTF-8
//! namelen  1/2/4/8  length of the link name, in bytes
//! name     N bytes  UTF-8, no NUL terminator
//! value             hard:     8-byte object header address
//!                   soft:     u16 length + UTF-8 path
//!                   external: u16 length + 1 flags byte + file NUL + path NUL
//! ```
//!
//! Every field width and the field *order* are pinned against messages libhdf5
//! 2.0.0 wrote, and are exactly what `oxih5_format::link_msg::parse_link`
//! expects to read back.

use oxih5_core::OxiH5Error;

use super::format::{fill_zero, write_u16_le, write_u64_le};

/// Link message version.
const LINK_MSG_VERSION: u8 = 1;

/// Flag bit 2 — an 8-byte creation order field follows the link type.
const FLAG_CREATION_ORDER: u8 = 0b0000_0100;
/// Flag bit 3 — an explicit link type byte is present.
const FLAG_LINK_TYPE: u8 = 0b0000_1000;

/// On-disk link type of a soft link.
const LINK_TYPE_SOFT: u8 = 1;
/// On-disk link type of an external link.
const LINK_TYPE_EXTERNAL: u8 = 64;

/// Version/flags byte that leads an external link's value payload.
const EXTERNAL_VALUE_VERSION: u8 = 0x00;

/// Body size of a Group Info message: version and flags, nothing optional.
pub(super) const GROUP_INFO_BODY: usize = 2;

/// What a new-style link points at, with everything the encoder needs.
///
/// A hard link carries a resolved address rather than a path: link targets are
/// resolved once, against the finished layout, exactly like object-reference
/// attributes — see [`super::plan::GroupPlan::resolve_link_targets`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum LinkValue {
    /// Hard link to an object header in this file.
    Hard(u64),
    /// Soft link: an in-file path, absolute or relative to the holding group.
    Soft(String),
    /// External link: a path inside another file.
    External {
        /// File name, as it will be resolved relative to this file's directory.
        file: String,
        /// Path of the object within that file.
        path: String,
    },
}

impl LinkValue {
    /// The link type byte, or `None` for a hard link (whose type is implied).
    const fn type_byte(&self) -> Option<u8> {
        match self {
            LinkValue::Hard(_) => None,
            LinkValue::Soft(_) => Some(LINK_TYPE_SOFT),
            LinkValue::External { .. } => Some(LINK_TYPE_EXTERNAL),
        }
    }

    /// Byte length of the value that follows the link name.
    fn encoded_len(&self) -> usize {
        match self {
            LinkValue::Hard(_) => 8,
            // u16 length + the path bytes.
            LinkValue::Soft(path) => 2 + path.len(),
            // u16 length + { flags byte, file NUL-terminated, path NUL-terminated }.
            LinkValue::External { file, path } => 2 + external_payload_len(file, path),
        }
    }
}

/// Byte length of an external link's value payload (the part the `u16` counts).
fn external_payload_len(file: &str, path: &str) -> usize {
    1 + file.len() + 1 + path.len() + 1
}

/// One new-style link, ready to be sized and emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LinkMsg {
    /// Link name within its group; UTF-8, never empty, never contains `'/'`.
    pub(super) name: String,
    /// Where the link points.
    pub(super) value: LinkValue,
    /// Creation order, when the group tracks it; `None` leaves the field out.
    pub(super) creation_order: Option<u64>,
}

/// Width of the on-disk name-length field, and the flag bits that declare it.
///
/// libhdf5 picks the narrowest width that fits, so a one-character link name
/// costs one byte rather than eight; matching that keeps the emitted bytes
/// identical to h5py's for the common case.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` if the name is longer than a `u32` can count,
/// which no HDF5 reader will accept.
fn name_len_width(name_len: usize) -> Result<(usize, u8), OxiH5Error> {
    match name_len {
        0..=0xFF => Ok((1, 0)),
        0x100..=0xFFFF => Ok((2, 1)),
        0x1_0000..=0xFFFF_FFFF => Ok((4, 2)),
        _ => Err(OxiH5Error::Format(format!(
            "link name of {name_len} bytes is too long for an HDF5 link message"
        ))),
    }
}

impl LinkMsg {
    /// Unpadded body size of this link's object-header message.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if the name is too long to be counted.
    pub(super) fn body_size(&self) -> Result<usize, OxiH5Error> {
        let (width, _) = name_len_width(self.name.len())?;
        Ok(2 // version + flags
            + usize::from(self.value.type_byte().is_some())
            + if self.creation_order.is_some() { 8 } else { 0 }
            + width
            + self.name.len()
            + self.value.encoded_len())
    }

    /// Write this link's message body at `start`; returns bytes written.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if the name is too long to be counted, or
    /// if a soft/external value does not fit its 16-bit length field.
    pub(super) fn write_body(&self, buf: &mut [u8], start: usize) -> Result<usize, OxiH5Error> {
        let body = self.body_size()?;
        fill_zero(buf, start, body);

        let (width, width_bits) = name_len_width(self.name.len())?;
        let type_byte = self.value.type_byte();
        let mut flags = width_bits;
        if type_byte.is_some() {
            flags |= FLAG_LINK_TYPE;
        }
        if self.creation_order.is_some() {
            flags |= FLAG_CREATION_ORDER;
        }

        buf[start] = LINK_MSG_VERSION;
        buf[start + 1] = flags;
        let mut pos = start + 2;

        if let Some(link_type) = type_byte {
            buf[pos] = link_type;
            pos += 1;
        }
        if let Some(order) = self.creation_order {
            write_u64_le(buf, pos, order);
            pos += 8;
        }
        // The charset field is deliberately absent: leaving bit 4 clear means
        // ASCII, and libhdf5 reads a name with no charset field as ASCII bytes,
        // which for pure-ASCII names is byte-identical to what h5py writes.
        // Non-ASCII names get an explicit UTF-8 declaration instead.
        let name_bytes = self.name.as_bytes();
        buf[pos..pos + width].copy_from_slice(&(name_bytes.len() as u64).to_le_bytes()[..width]);
        pos += width;
        buf[pos..pos + name_bytes.len()].copy_from_slice(name_bytes);
        pos += name_bytes.len();

        match &self.value {
            LinkValue::Hard(address) => {
                write_u64_le(buf, pos, *address);
                pos += 8;
            }
            LinkValue::Soft(path) => {
                write_u16_le(buf, pos, narrow_u16("soft link target path", path.len())?);
                pos += 2;
                buf[pos..pos + path.len()].copy_from_slice(path.as_bytes());
                pos += path.len();
            }
            LinkValue::External { file, path } => {
                let payload = external_payload_len(file, path);
                write_u16_le(buf, pos, narrow_u16("external link value", payload)?);
                pos += 2;
                buf[pos] = EXTERNAL_VALUE_VERSION;
                pos += 1;
                buf[pos..pos + file.len()].copy_from_slice(file.as_bytes());
                pos += file.len();
                buf[pos] = 0;
                pos += 1;
                buf[pos..pos + path.len()].copy_from_slice(path.as_bytes());
                pos += path.len();
                buf[pos] = 0;
                pos += 1;
            }
        }

        Ok(pos - start)
    }

    /// This link's message body as a standalone byte vector.
    ///
    /// Used by the dense (fractal-heap) path, where a link is not an
    /// object-header message but a heap object holding the very same bytes.
    ///
    /// # Errors
    ///
    /// As [`Self::write_body`].
    pub(super) fn to_bytes(&self) -> Result<Vec<u8>, OxiH5Error> {
        let body = self.body_size()?;
        let mut bytes = vec![0u8; body];
        let wrote = self.write_body(&mut bytes, 0)?;
        super::check_size("link message", wrote, body)?;
        Ok(bytes)
    }
}

/// Narrow a length into the 16-bit field a link value counts it in.
fn narrow_u16(what: &str, value: usize) -> Result<u16, OxiH5Error> {
    u16::try_from(value).map_err(|_| {
        OxiH5Error::Format(format!(
            "{what} is {value} bytes, over the 65535-byte limit of its on-disk field"
        ))
    })
}

// ---------------------------------------------------------------------------
// Link Info (0x0002)
// ---------------------------------------------------------------------------

/// Where a new-style group keeps its links, and whether it tracks their order.
///
/// A *compact* group leaves both addresses undefined and carries its links as
/// object-header messages; a *dense* group names a fractal heap and a version-2
/// B-tree index and carries none.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct LinkInfo {
    /// When set, links carry creation order and the message records the next
    /// order value to hand out.
    pub(super) creation_order_tracked: bool,
    /// One past the highest creation order in use; only written when tracked.
    pub(super) max_creation_order: u64,
}

impl LinkInfo {
    /// Unpadded body size of the Link Info message.
    pub(super) const fn body_size(&self) -> usize {
        // version + flags + [8-byte max creation order] + heap + name index
        2 + if self.creation_order_tracked { 8 } else { 0 } + 8 + 8
    }

    /// Write the Link Info body at `start`; returns bytes written.
    ///
    /// `fractal_heap` and `name_index` are `u64::MAX` for a compact group.
    /// Neither affects the message's length, which is why they arrive as
    /// arguments rather than fields — see [`super::oh::OhAddrs`].
    pub(super) fn write_body(
        &self,
        buf: &mut [u8],
        start: usize,
        fractal_heap: u64,
        name_index: u64,
    ) -> usize {
        let body = self.body_size();
        fill_zero(buf, start, body);
        buf[start] = 0x00; // version 0
                           // Bit 0 tracks creation order.  Bit 1 (a *creation-order index*, a
                           // second version-2 B-tree of type 6) is deliberately left clear: the
                           // order is recorded per link and preserved, without a second index
                           // that nothing here writes.
        buf[start + 1] = u8::from(self.creation_order_tracked);
        let mut pos = start + 2;
        if self.creation_order_tracked {
            write_u64_le(buf, pos, self.max_creation_order);
            pos += 8;
        }
        // Per the format specification the fractal heap address precedes the
        // name index; `parse_link_info` reads them in that order too.
        write_u64_le(buf, pos, fractal_heap);
        write_u64_le(buf, pos + 8, name_index);
        pos + 16 - start
    }
}

/// Write the Group Info (0x000A) body at `start`; returns bytes written.
///
/// Version 0 with no flags: the message exists to mark the group as new-style,
/// and every optional field (maximum compact links, minimum dense links,
/// estimated entry counts) is left out — exactly as libhdf5 writes it for a
/// group created with default creation properties.
pub(super) fn write_group_info_body(buf: &mut [u8], start: usize) -> usize {
    fill_zero(buf, start, GROUP_INFO_BODY);
    // [0] version = 0, [1] flags = 0
    GROUP_INFO_BODY
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxih5_format::context::ParseContext;
    use oxih5_format::link_msg::{parse_link, parse_link_info};

    fn encode(msg: &LinkMsg) -> Vec<u8> {
        msg.to_bytes().expect("encode")
    }

    /// Byte-pinned against the four Link messages libhdf5 2.0.0 wrote into a
    /// superblock-v0 file (`f['soft'] = SoftLink('/d')`, `f['hard'] = f['d']`,
    /// `f['ext'] = ExternalLink('other.h5', '/ds')`, plus the dataset itself).
    #[test]
    fn link_messages_match_libhdf5_byte_for_byte() {
        let hard = LinkMsg {
            name: "d".to_string(),
            value: LinkValue::Hard(0x0320),
            creation_order: None,
        };
        assert_eq!(
            encode(&hard),
            vec![0x01, 0x00, 0x01, b'd', 0x20, 0x03, 0, 0, 0, 0, 0, 0]
        );

        let alias = LinkMsg {
            name: "hard".to_string(),
            value: LinkValue::Hard(0x0320),
            creation_order: None,
        };
        assert_eq!(
            encode(&alias),
            vec![0x01, 0x00, 0x04, b'h', b'a', b'r', b'd', 0x20, 0x03, 0, 0, 0, 0, 0, 0]
        );

        let soft = LinkMsg {
            name: "soft".to_string(),
            value: LinkValue::Soft("/d".to_string()),
            creation_order: None,
        };
        assert_eq!(
            encode(&soft),
            vec![0x01, 0x08, 0x01, 0x04, b's', b'o', b'f', b't', 0x02, 0x00, b'/', b'd']
        );

        let external = LinkMsg {
            name: "ext".to_string(),
            value: LinkValue::External {
                file: "other.h5".to_string(),
                path: "/ds".to_string(),
            },
            creation_order: None,
        };
        let want: Vec<u8> = [
            &[0x01u8, 0x08, 0x40, 0x03][..],
            b"ext",
            &[0x0e, 0x00, 0x00][..],
            b"other.h5",
            &[0x00][..],
            b"/ds",
            &[0x00][..],
        ]
        .concat();
        assert_eq!(encode(&external), want);
    }

    /// A tracked creation order rides between the (absent) type byte and the
    /// name length, exactly where libhdf5 puts it — pinned against the Link
    /// message of a `track_order=True` group.
    #[test]
    fn creation_order_sits_before_the_name_length() {
        let msg = LinkMsg {
            name: "g".to_string(),
            value: LinkValue::Hard(0xef),
            creation_order: Some(0),
        };
        let bytes = encode(&msg);
        assert_eq!(bytes[0], 0x01, "version");
        assert_eq!(bytes[1], 0x04, "flags: creation order present, 1-byte name");
        assert_eq!(&bytes[2..10], &[0u8; 8], "creation order 0");
        assert_eq!(bytes[10], 0x01, "name length");
        assert_eq!(bytes[11], b'g');
        assert_eq!(
            u64::from_le_bytes(bytes[12..20].try_into().expect("8 bytes")),
            0xef
        );
    }

    /// Everything this module writes must come back out of the reader that
    /// parses the format, for every link kind and both creation-order settings.
    #[test]
    fn every_link_kind_round_trips_through_the_reader() {
        let ctx = ParseContext::default_v0();
        for creation_order in [None, Some(7u64)] {
            for value in [
                LinkValue::Hard(0x1234_5678),
                LinkValue::Soft("/a/b/c".to_string()),
                LinkValue::External {
                    file: "peer.h5".to_string(),
                    path: "/deep/target".to_string(),
                },
            ] {
                let msg = LinkMsg {
                    name: "member".to_string(),
                    value: value.clone(),
                    creation_order,
                };
                let bytes = encode(&msg);
                let parsed = parse_link(&bytes, &ctx).expect("parse");
                assert_eq!(parsed.name, "member");
                let want = match &value {
                    LinkValue::Hard(a) => oxih5_core::Link::Hard { address: *a },
                    LinkValue::Soft(p) => oxih5_core::Link::Soft { path: p.clone() },
                    LinkValue::External { file, path } => oxih5_core::Link::External {
                        file: file.clone(),
                        path: path.clone(),
                    },
                };
                assert_eq!(parsed.link, want, "{value:?} order={creation_order:?}");
            }
        }
    }

    /// A name over 255 bytes widens the length field rather than truncating.
    #[test]
    fn a_long_name_widens_its_length_field() {
        let name = "n".repeat(300);
        let msg = LinkMsg {
            name: name.clone(),
            value: LinkValue::Hard(8),
            creation_order: None,
        };
        let bytes = encode(&msg);
        assert_eq!(bytes[1] & 0b11, 1, "2-byte name length");
        assert_eq!(u16::from_le_bytes([bytes[2], bytes[3]]), 300);
        let parsed = parse_link(&bytes, &ParseContext::default_v0()).expect("parse");
        assert_eq!(parsed.name, name);
    }

    /// A soft link whose path does not fit the 16-bit length field is a typed
    /// error, not a truncated file.
    #[test]
    fn an_oversized_soft_target_is_rejected() {
        let msg = LinkMsg {
            name: "x".to_string(),
            value: LinkValue::Soft("p".repeat(70_000)),
            creation_order: None,
        };
        let err = msg.to_bytes().expect_err("must reject");
        assert!(format!("{err}").contains("65535-byte limit"), "{err}");
    }

    /// The Link Info message frames a compact group with two undefined
    /// addresses and a dense one with real ones; both must parse back.
    #[test]
    fn link_info_round_trips_for_compact_and_dense() {
        let ctx = ParseContext::default_v0();

        let compact = LinkInfo::default();
        let mut buf = vec![0u8; compact.body_size()];
        let wrote = compact.write_body(&mut buf, 0, u64::MAX, u64::MAX);
        assert_eq!(wrote, 18);
        // Byte-pinned against libhdf5: version 0, flags 0, two undefined addresses.
        assert_eq!(buf[0..2], [0x00, 0x00]);
        assert!(buf[2..18].iter().all(|&b| b == 0xFF));
        let info = parse_link_info(&buf, &ctx).expect("parse");
        assert!(!info.creation_order_tracked);
        assert_eq!(info.fractal_heap_address, None);
        assert_eq!(info.name_index_address, None);

        let dense = LinkInfo {
            creation_order_tracked: true,
            max_creation_order: 13,
        };
        let mut buf = vec![0u8; dense.body_size()];
        let wrote = dense.write_body(&mut buf, 0, 0x1950, 0x19e2);
        assert_eq!(wrote, 26);
        assert_eq!(buf[1], 0x01, "creation order tracked, not indexed");
        let info = parse_link_info(&buf, &ctx).expect("parse");
        assert!(info.creation_order_tracked);
        assert_eq!(info.creation_order_index_address, None);
        assert_eq!(info.fractal_heap_address, Some(0x1950));
        assert_eq!(info.name_index_address, Some(0x19e2));
    }

    /// The Group Info body is the two zero bytes libhdf5 writes.
    #[test]
    fn group_info_body_is_version_zero_no_flags() {
        let mut buf = vec![0xAAu8; GROUP_INFO_BODY];
        assert_eq!(write_group_info_body(&mut buf, 0), 2);
        assert_eq!(buf, vec![0x00, 0x00]);
    }
}
