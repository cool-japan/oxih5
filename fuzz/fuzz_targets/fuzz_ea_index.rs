#![no_main]
use libfuzzer_sys::fuzz_target;

// Drives the Extensible Array (EA) chunk-index parser directly. EA is the
// index libhdf5 selects for a chunked dataset with a single unlimited
// dimension, and its header/index-block/data-block/secondary-block traversal
// is exactly the kind of length-prefixed, offset-chasing format that benefits
// from direct fuzzing rather than only reaching it through a structurally
// valid whole file (`fuzz_file_open`).
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

    // The element→chunk mapping is driven by the dataset geometry, so vary the
    // rank and exercise both the "maximum dimensions known" and "absent" paths.
    let chunk_dims = vec![4u64; ndims];
    let dataset_dims = vec![64u64; ndims];
    let mut max_dims = vec![64u64; ndims];
    max_dims[ndims - 1] = u64::MAX;
    for max in [None, Some(max_dims.as_slice())] {
        let _ = oxih5_format::ea_index::parse_extensible_array(
            file_data,
            addr,
            &oxih5_format::ea_index::EaGeometry {
                chunk_dims: &chunk_dims,
                dataset_dims: &dataset_dims,
                max_dims: max,
                chunk_bytes: 32,
            },
        );
    }
});
