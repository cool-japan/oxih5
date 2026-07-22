//! Wave-2 LANE READER — v2 object-header NIL handling regression.
//!
//! A version-2 object header (`OHDR`) stores its messages consecutively and is
//! scanned to the full chunk size; a NIL message (type `0x00`) is *padding*, not
//! a terminator. netCDF-C (and libhdf5 after an object is modified) reserves
//! space with a NIL message and then appends real Link (`0x0006`) / Attribute
//! (`0x000C`) messages *after* it. A reader that stops at the first NIL silently
//! drops every object declared past that gap — the "root group lists only the
//! first dataset" enumeration bug (e.g. `ref2.nc` → only `['lat']`).
//!
//! These tests build synthetic OHDR bytes with a real message following a NIL
//! and assert `parse_messages` returns the post-NIL messages, for both the
//! 4-byte (no creation order) and 6-byte (creation-order tracked, as netCDF-C
//! writes) message-record layouts. They run without any external toolchain.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use oxih5_format::header::{self, Message};

/// One message to place in the synthetic header: `(type, body)`.
struct Msg {
    ty: u8,
    body: Vec<u8>,
}

/// Build a minimal version-2 object header holding `msgs` verbatim.
///
/// `track_creation_order` sets OHDR flag bit 2, which makes every message record
/// carry a 2-byte creation-order field (6-byte record header instead of 4).
/// A 1-byte chunk-0 size field is used (flag bits 0-1 = 0), so the whole message
/// region must be < 256 bytes — ample for these tests.
fn build_ohdr(msgs: &[Msg], track_creation_order: bool) -> Vec<u8> {
    // First lay out the message region so we can size chunk 0.
    let mut region: Vec<u8> = Vec::new();
    for (i, m) in msgs.iter().enumerate() {
        region.push(m.ty);
        let size = u16::try_from(m.body.len()).unwrap();
        region.extend_from_slice(&size.to_le_bytes());
        region.push(0); // message flags
        if track_creation_order {
            region.extend_from_slice(&(i as u16).to_le_bytes());
        }
        region.extend_from_slice(&m.body);
    }
    let chunk0 = u8::try_from(region.len()).unwrap();

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"OHDR");
    out.push(2); // version
    out.push(if track_creation_order { 0x04 } else { 0x00 }); // flags
    out.push(chunk0); // chunk #0 size (1-byte width)
    out.extend_from_slice(&region);
    out.extend_from_slice(&[0u8; 4]); // checksum (not verified by the reader)
    out
}

fn types(msgs: &[Message]) -> Vec<u16> {
    msgs.iter().map(|m| m.msg_type).collect()
}

/// Core regression: a Link message after a NIL must be returned, not dropped.
#[test]
fn v2_header_returns_messages_after_a_nil() {
    for track_co in [false, true] {
        let msgs = [
            Msg {
                ty: 0x03,
                body: vec![0x11, 0x22, 0x33, 0x44],
            }, // datatype-like
            Msg {
                ty: 0x00,
                body: vec![],
            }, // NIL (reserved gap)
            Msg {
                ty: 0x06,
                body: vec![0x55; 5],
            }, // link after the NIL
            Msg {
                ty: 0x0C,
                body: vec![0x66; 3],
            }, // attribute after the NIL
            Msg {
                ty: 0x00,
                body: vec![0, 0],
            }, // trailing NIL padding
        ];
        let bytes = build_ohdr(&msgs, track_co);
        let parsed = header::parse_messages(&bytes, 0).unwrap();
        assert_eq!(
            types(&parsed),
            vec![0x03, 0x06, 0x0C],
            "track_creation_order={track_co}: messages after the NIL were dropped"
        );
        // Bodies are preserved verbatim.
        assert_eq!(parsed[1].data, vec![0x55; 5]);
        assert_eq!(parsed[2].data, vec![0x66; 3]);
    }
}

/// Multiple interior NILs, each followed by real messages.
#[test]
fn v2_header_skips_multiple_interior_nils() {
    let msgs = [
        Msg {
            ty: 0x00,
            body: vec![0, 0, 0],
        }, // leading NIL
        Msg {
            ty: 0x01,
            body: vec![1, 2, 3, 4],
        },
        Msg {
            ty: 0x00,
            body: vec![],
        }, // interior NIL
        Msg {
            ty: 0x06,
            body: vec![7],
        },
        Msg {
            ty: 0x00,
            body: vec![9, 9],
        }, // interior NIL
        Msg {
            ty: 0x06,
            body: vec![8],
        },
    ];
    let bytes = build_ohdr(&msgs, true);
    let parsed = header::parse_messages(&bytes, 0).unwrap();
    assert_eq!(types(&parsed), vec![0x01, 0x06, 0x06]);
}
