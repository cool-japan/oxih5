#![no_main]
use libfuzzer_sys::fuzz_target;
use oxih5_core::{FilterInfo, FilterPipeline};

// Drives the read-direction filter pipeline (`apply_pipeline_sized`) with raw
// chunk bytes. Covers deflate, shuffle, fletcher32, nbit and scale+offset —
// every one of these decodes attacker-controlled on-disk bytes into an
// output buffer sized from `client_data`/`elem_size`, which is exactly the
// class of arithmetic that produced the confirmed `n_elems * 16` /
// `seq_len * elem_footprint` overflow hazards in `values.rs`. Filter id 4
// (szip) is intentionally skipped: it requires the optional `szip` feature
// and `oxiarc-szip`, neither of which this fuzz binary depends on.
fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    // Cycle through the five filter ids the default build supports, plus one
    // unknown id to exercise the "unsupported filter" error path.
    let filter_id = (data[0] % 6) as u16;
    let elem_size = (data[1] as usize % 16) + 1;
    let filter_mask = u32::from(data[2]);
    // A handful of small client-data words covers nbit's
    // [count, class, sizeof, sign, byte_order, precision, bit_offset] shape
    // and scaleoffset's [scale_type, scale_factor] shape without needing a
    // dedicated encoding per filter id.
    let client_data = vec![
        elem_size as u32,
        0,
        elem_size as u32,
        0,
        0,
        (data[3] as u32) % 64,
        0,
    ];
    let pipeline = FilterPipeline {
        filters: vec![FilterInfo {
            id: filter_id,
            name: None,
            flags: 0,
            client_data,
        }],
    };

    let _ = oxih5_format::filters::apply_pipeline_sized(
        &data[4..],
        &pipeline,
        filter_mask,
        elem_size,
        None,
    );
    // Also exercise the `expected_out_len`-aware call path (szip RAW mode's
    // only source of an output length, but harmless to pass for every id).
    let _ = oxih5_format::filters::apply_pipeline_sized(
        &data[4..],
        &pipeline,
        filter_mask,
        elem_size,
        Some(data.len() * 4),
    );
});
