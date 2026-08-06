//! Write one file per newer writer capability into a directory given on the
//! command line, so a **third-party** reader can check them.
//!
//! The crate's own tests already round-trip every one of these through the
//! oxih5 reader; this example exists for the other half of the contract, which
//! no Rust test can assert: that h5py / libhdf5 reads the same files.  Run it,
//! then open the output with `h5py` (or `h5dump`) and compare.
//!
//! ```text
//! cargo run --example interop_fixtures -- /tmp/oxih5-interop
//! ```

use oxih5::FileWriter;
use oxih5_core::{ByteOrder, CompoundField, Dtype};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let out = std::path::Path::new(&dir);

    // G011 old-style: soft link + hard alias in a symbol-table group.
    let mut w = FileWriter::new();
    w.write_dataset_i32("/d", &[1, 2, 3], &[3])?;
    w.create_soft_link("/soft", "/d")?;
    w.create_hard_link("/hard", "/d")?;
    w.build(out.join("links_old.h5"))?;

    // G011 + G013 compact: an external link converts the group.
    let mut w = FileWriter::new();
    w.write_dataset_i32("/d", &[1, 2, 3], &[3])?;
    w.create_soft_link("/soft", "/d")?;
    w.create_hard_link("/hard", "/d")?;
    w.create_external_link("/ext", "peer.h5", "/ds")?;
    w.build(out.join("links_compact.h5"))?;

    // G013 dense: more than max_compact links in a sub-group.
    let mut w = FileWriter::new();
    w.create_group("/g")?;
    w.set_link_storage("/g")?;
    for i in 0..12i32 {
        w.write_dataset_i32(&format!("/g/ds{i:02}"), &[i], &[1])?;
    }
    w.create_external_link("/g/ext", "peer.h5", "/ds")?;
    w.build(out.join("links_dense.h5"))?;

    // G013 track_order.
    let mut w = FileWriter::new();
    w.create_group("/t")?;
    w.set_track_order("/t")?;
    for name in ["zeta", "alpha", "mid"] {
        w.write_dataset_i32(&format!("/t/{name}"), &[1], &[1])?;
    }
    w.build(out.join("track_order.h5"))?;

    // G004 compound.
    let fields = vec![
        CompoundField {
            name: "id".to_string(),
            offset: 0,
            dtype: Dtype::Int {
                size: 4,
                signed: true,
                order: ByteOrder::Little,
            },
        },
        CompoundField {
            name: "value".to_string(),
            offset: 4,
            dtype: Dtype::Float {
                size: 8,
                order: ByteOrder::Little,
            },
        },
    ];
    let mut rows = Vec::new();
    for (id, value) in [(1i32, 2.5f64), (3, 4.5)] {
        rows.extend_from_slice(&id.to_le_bytes());
        rows.extend_from_slice(&value.to_le_bytes());
    }
    FileWriter::new()
        .create_compound_dataset("/events", &fields, 12, &rows, &[2])?
        .build(out.join("compound.h5"))?;

    // G015 ragged sequences.
    FileWriter::new()
        .create_vlen_i32_dataset("/rows", &[vec![1, 2, 3], vec![], vec![10]])?
        .create_vlen_f64_dataset("/reals", &[vec![1.5], vec![2.5, 3.5]])?
        .build(out.join("ragged.h5"))?;

    // G016 array / opaque / bitfield.
    let mut w = FileWriter::new();
    let arr: Vec<u8> = (0..12i32).flat_map(i32::to_le_bytes).collect();
    w.create_array_dataset(
        "/arr",
        &Dtype::Int {
            size: 4,
            signed: true,
            order: ByteOrder::Little,
        },
        &[2, 3],
        &arr,
        &[2],
    )?;
    w.create_opaque_dataset("/opq", 7, "NUMPY:|V7", &[0xABu8; 14], &[2])?;
    w.create_bitfield_dataset("/bits", 1, ByteOrder::Little, &[0x0F, 0xF0], &[2])?;
    w.build(out.join("dtypes.h5"))?;

    // G013 dense *root* group: the superblock's root symbol table entry still
    // caches the (now empty) structures, which is what libhdf5 leaves behind
    // when it converts a group — this file proves libhdf5 reads it back.
    let mut w = FileWriter::new();
    w.set_link_storage("/")?;
    for i in 0..10i32 {
        w.write_dataset_i32(&format!("/ds{i:02}"), &[i], &[1])?;
    }
    w.create_soft_link("/first", "/ds00")?;
    w.create_hard_link("/alias", "/ds03")?;
    w.build(out.join("dense_root.h5"))?;

    // G013 track_order on the root group.
    let mut w = FileWriter::new();
    w.set_track_order("/")?;
    for name in ["zeta", "alpha", "mid"] {
        w.write_dataset_i32(&format!("/{name}"), &[1], &[1])?;
    }
    w.build(out.join("track_root.h5"))?;

    // Cross-feature: a dense group holding a filtered dataset, vlen strings, a
    // compact dataset and a custom fill value, plus links to a *group* in both
    // storage styles and an old/new-style nesting.
    let mut w = FileWriter::new();
    w.create_group("/d")?;
    w.set_link_storage("/d")?;
    for i in 0..8i32 {
        w.write_dataset_i32(&format!("/d/pad{i}"), &[i], &[1])?;
    }
    let big: Vec<f64> = (0..256).map(f64::from).collect();
    w.write_dataset_f64("/d/packed", &big, &[256])?;
    w.set_shuffle("d/packed")?;
    w.set_deflate("d/packed", 6)?;
    w.set_fletcher32("d/packed")?;
    w.create_vlen_string_dataset("/d/names", &["alpha", "beta"])?;
    w.write_dataset_i32("/d/inline", &[7, 8, 9], &[3])?;
    w.set_compact("d/inline")?;
    w.write_string_attr("/d/packed", "units", "km")?;
    w.write_dataset_i32("/g/inner", &[5], &[1])?;
    w.create_hard_link("/galias", "/g")?;
    w.create_soft_link("/gsoft", "/g")?;
    w.build(out.join("dense_features.h5"))?;

    println!("wrote 10 files into {}", out.display());
    Ok(())
}
