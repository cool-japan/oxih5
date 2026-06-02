#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Should never panic — only return Ok/Err
    let _ = oxih5_format::superblock::parse(data);
});
