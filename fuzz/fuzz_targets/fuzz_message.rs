#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 9 {
        return;
    }
    // Try parsing as a superblock, then follow the root object header address
    if let Ok(sb) = oxih5_format::superblock::parse(data) {
        let addr = sb.root_object_header_address;
        // Only attempt if address is within the data bounds
        if (addr as usize) < data.len() {
            let _ = oxih5_format::header::parse_messages(data, addr);
        }
    }
});
