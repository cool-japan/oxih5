//! Panic-free fuzzing: feed random and mutated byte sequences to all parsers.
//! Asserts that parsers NEVER panic — returning `Err` is always acceptable.
//!
//! This test uses a deterministic xorshift64 PRNG so results are reproducible
//! without any external fuzzing infrastructure or nightly Rust.

use oxih5_format::{
    btree, btree_v1_chunk, btree_v2, context::ParseContext, datatype, ea_index, fa_index,
    fractal_heap, global_heap, header, heap, message, snod, superblock,
};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Deterministic xorshift64 PRNG
// ---------------------------------------------------------------------------

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a pseudo-random `Vec<u8>` of `len` bytes driven by `state`.
fn random_bytes(state: &mut u64, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let v = xorshift64(state);
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.truncate(len);
    out
}

// ---------------------------------------------------------------------------
// Core helpers: invoke every parser, assert no panic
// ---------------------------------------------------------------------------

/// Feed `data` to all body-based parsers (those taking only `&[u8]`).
fn fuzz_body_parsers(data: &[u8]) {
    let _ = superblock::parse(data);
    let _ = message::parse_dataspace(data);
    let _ = message::parse_dataspace_rich(data);
    let _ = message::parse_datatype(data);
    let _ = message::parse_layout(data);
    let _ = message::parse_filter_pipeline(data);
    let _ = message::parse_attribute(data);
    let _ = message::parse_fill_value(data);
    let _ = message::parse_symbol_table(data);
    let _ = message::parse_modification_time(data);
    let _ = datatype::parse_datatype(data);
    let _ = datatype::parse_datatype_consuming(data, 0);
    // Depth clamped to 1 to avoid stack blow-up on recursive compound types
    let _ = datatype::parse_datatype_consuming(data, 1);
}

/// Feed `data` + address-based parsers (those taking `(file_data, address)`).
fn fuzz_address_parsers(data: &[u8], addr: u64) {
    let _ = header::parse_messages(data, addr);
    let _ = heap::parse(data, addr);
    let _ = snod::parse(data, addr);
    let _ = btree::parse(data, addr);
    let _ = global_heap::GlobalHeap::parse(data, addr);
}

/// Feed `data` to parsers that require a `ParseContext`.
fn fuzz_context_parsers(data: &[u8]) {
    // Use a variety of ParseContext configurations that mirror real files.
    let ctxs = [
        ParseContext::new(8, 8, 0),
        ParseContext::new(4, 4, 0),
        ParseContext::new(8, 8, 512),
    ];
    for ctx in &ctxs {
        let _ = oxih5_format::link_msg::parse_link_info(data, ctx);
        let _ = oxih5_format::link_msg::parse_link(data, ctx);
    }
}

/// Feed `data` to parsers that take `(file_data, address, extra)` arguments.
fn fuzz_extra_parsers(data: &[u8], addr: u64, state: &mut u64) {
    // btree_v2::parse_name_index — takes (file_data, header_address, heap_id_len)
    for heap_id_len in [0u8, 1, 7, 8, 16] {
        let _ = btree_v2::parse_name_index(data, addr, heap_id_len);
    }

    // btree_v1_chunk::parse — takes (file_data, btree_address, ndims)
    for ndims in [0usize, 1, 2, 4] {
        let _ = btree_v1_chunk::parse(data, addr, ndims);
    }

    // ea_index::parse_extensible_array — takes (file_data, header_address, ndims)
    for ndims in [0usize, 1, 2, 4] {
        let _ = ea_index::parse_extensible_array(data, addr, ndims);
    }

    // fa_index::parse_fixed_array — takes (file_data, header_address, ndims)
    for ndims in [0usize, 1, 2] {
        let _ = fa_index::parse_fixed_array(data, addr, ndims);
        let _ = fa_index::parse_fixed_array_v4(data, addr, ndims, &[], 0);
    }

    // fractal_heap::FractalHeap::parse — takes (Arc<Vec<u8>>, header_address, size_of_offsets)
    let arc_data = Arc::new(data.to_vec());
    for soo in [4u8, 8] {
        if let Ok(fh) = fractal_heap::FractalHeap::parse(Arc::clone(&arc_data), addr, soo) {
            // If the header parsed successfully, try reading with garbage heap IDs
            let heap_id_len = fh.heap_id_len() as usize;
            if heap_id_len > 0 {
                let fake_id = random_bytes(state, heap_id_len);
                let _ = fh.parse_heap_id(&fake_id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 1 — Random byte sequences, all lengths
// ---------------------------------------------------------------------------

#[test]
fn fuzz_random_bytes_no_panic() {
    let mut state = 0xDEAD_BEEF_CAFE_1337u64;

    // Sweep a representative set of lengths
    let lengths = [0usize, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096];

    for &len in &lengths {
        for _ in 0..50 {
            let data = random_bytes(&mut state, len);
            fuzz_body_parsers(&data);
            fuzz_context_parsers(&data);
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2 — Address-based parsers with varied OOB/in-bound addresses
// ---------------------------------------------------------------------------

#[test]
fn fuzz_random_with_address_parsers_no_panic() {
    let mut state = 0x1234_5678_9ABC_DEF0u64;

    for _ in 0..200 {
        let len = (xorshift64(&mut state) % 2048 + 16) as usize;
        let data = random_bytes(&mut state, len);

        let addrs = [
            0u64,
            (len / 2) as u64,
            len.saturating_sub(8) as u64,
            len as u64,   // exactly at boundary
            u64::MAX / 2, // far out of bounds
        ];

        for &addr in &addrs {
            fuzz_address_parsers(&data, addr);
            fuzz_extra_parsers(&data, addr, &mut state);
        }
    }
}

// ---------------------------------------------------------------------------
// Test 3 — Bit-flip and byte-replacement mutations on a real fixture
// ---------------------------------------------------------------------------

#[test]
fn fuzz_bit_flips_on_real_fixture() {
    let fixture_dir = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../oxih5/tests/fixtures"
    ));

    // Read the first available .h5 fixture file
    let fixture = std::fs::read_dir(fixture_dir)
        .ok()
        .and_then(|d| {
            d.filter_map(|e| e.ok())
                .find(|e| e.path().extension().is_some_and(|ext| ext == "h5"))
        })
        .and_then(|e| std::fs::read(e.path()).ok());

    let original = match fixture {
        Some(data) => data,
        None => {
            eprintln!("no .h5 fixture found; skipping bit-flip fuzz");
            return;
        }
    };

    let mut state = 0xF0F0_F0F0_0F0F_0F0Fu64;

    for _ in 0..200 {
        let mutation = xorshift64(&mut state) % 3;

        let mutated: Vec<u8> = match mutation {
            0 => {
                // Flip a random bit
                let mut v = original.clone();
                let byte_idx = (xorshift64(&mut state) as usize) % v.len().max(1);
                let bit = (xorshift64(&mut state) % 8) as u8;
                if byte_idx < v.len() {
                    v[byte_idx] ^= 1 << bit;
                }
                v
            }
            1 => {
                // Replace a random byte with a random value
                let mut v = original.clone();
                let idx = (xorshift64(&mut state) as usize) % v.len().max(1);
                if idx < v.len() {
                    v[idx] = (xorshift64(&mut state) & 0xFF) as u8;
                }
                v
            }
            _ => {
                // Truncate to a random shorter length
                let new_len = ((xorshift64(&mut state) as usize) % original.len().max(1)) + 1;
                original[..new_len.min(original.len())].to_vec()
            }
        };

        fuzz_body_parsers(&mutated);
        fuzz_context_parsers(&mutated);
        fuzz_address_parsers(&mutated, 0);
        fuzz_extra_parsers(&mutated, 0, &mut state);
    }
}

// ---------------------------------------------------------------------------
// Test 4 — Empty input never panics
// ---------------------------------------------------------------------------

#[test]
fn fuzz_empty_input_no_panic() {
    fuzz_body_parsers(&[]);
    fuzz_context_parsers(&[]);
    fuzz_address_parsers(&[], 0);
    fuzz_address_parsers(&[], u64::MAX);
    let mut state = 0xCAFEu64;
    fuzz_extra_parsers(&[], 0, &mut state);
    fuzz_extra_parsers(&[], u64::MAX, &mut state);
}

// ---------------------------------------------------------------------------
// Test 5 — Uniform byte patterns (all-zeros, all-ones)
// ---------------------------------------------------------------------------

#[test]
fn fuzz_uniform_bytes_no_panic() {
    let zeros = vec![0u8; 4096];
    let ones = vec![0xFFu8; 4096];

    for &size in &[1usize, 8, 32, 256, 4096] {
        fuzz_body_parsers(&zeros[..size]);
        fuzz_body_parsers(&ones[..size]);

        fuzz_context_parsers(&zeros[..size]);
        fuzz_context_parsers(&ones[..size]);

        fuzz_address_parsers(&zeros[..size], 0);
        fuzz_address_parsers(&ones[..size], 0);

        let mut state = 0x5A5A_5A5Au64;
        fuzz_extra_parsers(&zeros[..size], 0, &mut state);
        fuzz_extra_parsers(&ones[..size], 0, &mut state);
    }
}
