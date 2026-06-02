#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // File::open_from_bytes — should never panic, only return Ok/Err
    let _ = oxih5::File::open_from_bytes(data);
});
