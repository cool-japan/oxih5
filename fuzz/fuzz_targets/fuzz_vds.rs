#![no_main]
use libfuzzer_sys::fuzz_target;

// Drives the Virtual Dataset (VDS) mapping-block parser directly with the
// raw contents of a global-heap object. `size_of_lengths` is normally the
// superblock's "Size of Lengths" field, which is always 4 or 8 in practice —
// covering both keeps the fuzzer inside the interesting branches of
// `Reader::uint` instead of spending its whole budget on an unsupported
// width.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let size_of_lengths: usize = if data[0] % 2 == 0 { 4 } else { 8 };
    let _ = oxih5_format::vds::parse_vds_block(&data[1..], size_of_lengths);
});
