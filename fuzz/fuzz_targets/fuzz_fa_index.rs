#![no_main]
use libfuzzer_sys::fuzz_target;

// Drives the Fixed Array (FA) chunk-index parser directly with raw bytes,
// rather than only through a structurally valid file via `fuzz_file_open`
// (which random mutation almost never produces). This is one of the readers
// that had a confirmed capacity-overflow bug (FA "Number of Elements" used
// unchecked as a `Vec::with_capacity` argument), so it deserves direct
// coverage rather than relying on the whole-file fuzzer to stumble into it.
fuzz_target!(|data: &[u8]| {
    if data.len() < 9 {
        return;
    }
    // First byte selects `ndims` (dataset rank); kept small so mutations
    // spend their budget on the header/data-block bytes instead of just
    // tripping an "ndims too large" guard.
    let ndims = (data[0] % 4) as usize + 1;
    // Next 8 bytes select the header address, reduced modulo the remaining
    // buffer length so it usually lands inside `file_data` where the parser
    // can actually walk into the FA header/index-block/data-block chain.
    let mut addr_bytes = [0u8; 8];
    addr_bytes.copy_from_slice(&data[1..9]);
    let file_data = &data[9..];
    if file_data.is_empty() {
        return;
    }
    let addr = u64::from_le_bytes(addr_bytes) % file_data.len() as u64;

    let _ = oxih5_format::fa_index::parse_fixed_array(file_data, addr, ndims);
    let _ = oxih5_format::fa_index::parse_fixed_array_v4(file_data, addr, ndims, &[4, 4], 64);
});
