//! Build a new HDF5 file with [`oxih5::FileWriter`]: nested groups, a
//! chunked + gzip-compressed dataset, and attributes on both a dataset and a
//! group.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p oxih5 --example write_dataset
//! ```

use oxih5::FileWriter;

fn main() -> Result<(), oxih5::OxiH5Error> {
    // A fresh temporary path — never a hardcoded absolute path — so this
    // example is runnable as-is on any machine.
    let path = std::env::temp_dir().join(format!("oxih5_example_write_{}.h5", std::process::id()));

    let mut writer = FileWriter::new();

    // A 1-D float64 "readings" dataset, chunked into groups of 256 elements
    // and gzip-compressed at level 6. `set_deflate`/`set_chunking` operate
    // on a dataset already added to the writer, so the dataset write comes
    // first.
    let readings: Vec<f64> = (0..1024).map(|i| i as f64 * 0.5).collect();
    writer.write_dataset_f64("readings", &readings, &[readings.len()])?;
    writer.set_chunking("readings", &[256])?;
    writer.set_deflate("readings", 6)?;

    // A scalar string attribute and a scalar float64 attribute on that
    // dataset.
    writer.write_string_attr("readings", "units", "volts")?;
    writer.write_f64_attr("readings", "sample_rate_hz", 44_100.0)?;

    // A 2-D int32 dataset living inside a nested group. Writing to
    // "instrument/status/flags" creates the intermediate groups
    // `instrument` and `instrument/status` along the way — there is no
    // separate "create group" call needed.
    let flags: Vec<i32> = (0..12).collect();
    writer.write_dataset_i32("instrument/status/flags", &flags, &[3, 4])?;

    // A string attribute on the group itself (not on a dataset inside it).
    writer.write_string_attr("instrument/status", "firmware_version", "2.3.1")?;

    writer.build(&path)?;

    // Re-open and spot-check what was written, so running this example
    // prints a concrete confirmation rather than just "no error".
    let file = oxih5::open(&path)?;
    let ds = file.dataset("readings")?;
    println!(
        "wrote {} elements to /readings (shape {:?}, gzip+chunked)",
        ds.as_f64()?.len(),
        ds.shape
    );
    for attr in ds.attrs() {
        println!(
            "  attribute {} (string {:?}, f64 {:?})",
            attr.name,
            attr.as_str_fixed(),
            attr.as_f64()
        );
    }

    let flags_ds = file.dataset("instrument/status/flags")?;
    println!(
        "wrote {:?} elements to /instrument/status/flags (shape {:?})",
        flags_ds.as_i32()?,
        flags_ds.shape
    );

    std::fs::remove_file(&path).ok();
    Ok(())
}
