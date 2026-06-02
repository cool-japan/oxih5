#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Object header parsing — should never panic, only return Ok/Err
    let _ = oxih5_format::header::parse_messages(data, 0);
});
