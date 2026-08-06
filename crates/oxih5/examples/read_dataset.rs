//! Open an HDF5 file with [`oxih5::open`] and read a dataset in three ways:
//! whole, via a contiguous slice, and via a strided hyperslab selection.
//!
//! The example first writes a small fixture file with [`oxih5::FileWriter`]
//! so it is runnable standalone, without depending on any file shipped
//! elsewhere in the repository — the interesting part is everything after
//! "reading starts here".
//!
//! Run with:
//!
//! ```sh
//! cargo run -p oxih5 --example read_dataset
//! ```

use oxih5::{DimSelection, FileWriter};

fn main() -> Result<(), oxih5::OxiH5Error> {
    let path = std::env::temp_dir().join(format!("oxih5_example_read_{}.h5", std::process::id()));

    // --- Fixture setup -----------------------------------------------------
    // A 4x6 int32 dataset, chunked 2x3, so both the slice and the hyperslab
    // reads below cross a chunk boundary.
    let data: Vec<i32> = (0..24).collect();
    let mut writer = FileWriter::new();
    writer.write_dataset_i32("grid", &data, &[4, 6])?;
    writer.set_chunking("grid", &[2, 3])?;
    writer.build(&path)?;

    // --- Reading starts here -------------------------------------------
    let file = oxih5::open(&path)?;

    // 1. Read the whole dataset.
    let ds = file.dataset("grid")?;
    println!(
        "full dataset: shape {:?}, {} elements",
        ds.shape,
        ds.as_i32()?.len()
    );

    // 2. Read a contiguous slice: rows 1..3 (all columns), via
    //    `Range<usize>` per dimension.
    let ranges: Vec<std::ops::Range<usize>> = vec![1..3, 0..6];
    let slice = file.dataset_slice("grid", &ranges)?;
    println!(
        "rows 1..3 slice: shape {:?}, values {:?}",
        slice.shape,
        slice.as_i32()?
    );

    // 3. Read a strided hyperslab: every other column (columns 0, 2, 4),
    //    all rows. `DimSelection { start, stride, count, block }` mirrors
    //    HDF5's own hyperslab parameters directly.
    let selection = [
        DimSelection::contiguous(0..4), // all 4 rows
        DimSelection {
            start: 0,
            stride: 2,
            count: 3,
            block: 1,
        }, // columns 0, 2, 4
    ];
    let hyperslab = file.dataset_hyperslab("grid", &selection)?;
    println!(
        "every-other-column hyperslab: shape {:?}, values {:?}",
        hyperslab.shape,
        hyperslab.as_i32()?
    );

    std::fs::remove_file(&path).ok();
    Ok(())
}
