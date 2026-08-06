#![no_main]
use libfuzzer_sys::fuzz_target;

// Drives the B-tree v1 raw-data-chunk index parser directly. This is the
// chunk index `libver='earliest'` files use by default, and its recursive
// internal-node traversal is exactly where the confirmed
// `chunked.rs` divide-by-zero (a zero chunk dimension reaching an
// unconditional division deep in a chunk reader) and its sibling in
// `chunked_hyperslab.rs` were rooted, one layer downstream of this parser.
// `MAX_DEPTH` bounds recursion inside `parse` itself, so this target only
// needs to bound the address/dimensionality inputs.
fuzz_target!(|data: &[u8]| {
    if data.len() < 9 {
        return;
    }
    let ndims = (data[0] % 4) as usize + 1;
    let mut addr_bytes = [0u8; 8];
    addr_bytes.copy_from_slice(&data[1..9]);
    let file_data = &data[9..];
    if file_data.is_empty() {
        return;
    }
    let addr = u64::from_le_bytes(addr_bytes) % file_data.len() as u64;

    let _ = oxih5_format::btree_v1_chunk::parse(file_data, addr, ndims);
});
