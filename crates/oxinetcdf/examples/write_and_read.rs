//! Write a small NetCDF-4 file with [`oxinetcdf::write::NcFileWriter`], then
//! read it back with [`oxinetcdf::NcFile`] — dimensions, a float64 variable,
//! and both a global and a per-variable string attribute.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p oxinetcdf --example write_and_read
//! ```

use oxinetcdf::write::{NcFileWriter, VarOrGroup};
use oxinetcdf::{NcFile, NcType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join(format!("oxinetcdf_example_{}.nc", std::process::id()));

    // --- Write ---------------------------------------------------------
    let mut writer = NcFileWriter::new();
    let lat = writer.def_dim("lat", 3)?;
    let lon = writer.def_dim("lon", 4)?;
    let temp = writer.def_var("temperature", &[lat, lon], NcType::Float64)?;

    let data: Vec<f64> = (0..12).map(|i| 15.0 + i as f64 * 0.25).collect();
    writer.put_var_f64(temp, &data)?;
    writer.put_att_str(VarOrGroup::Var(temp), "units", "degC")?;
    writer.put_att_str(
        VarOrGroup::Root,
        "title",
        "oxinetcdf write_and_read example",
    )?;

    writer.close(&path)?;

    // --- Read ------------------------------------------------------------
    let nc = NcFile::open(&path)?;
    let root = nc.root_group()?;

    for attr in &root.attrs {
        println!("global attribute {} = {:?}", attr.name, attr.as_text()?);
    }

    println!("dimensions:");
    for dim in &root.dimensions {
        println!("  {} = {}", dim.name, dim.len);
    }

    println!("variables:");
    for var in &root.variables {
        let values = var.read_f64(&nc)?;
        println!("  {} : shape {:?} = {:?}", var.name, var.shape, values);
        for attr in &var.attrs {
            println!("    attribute {} = {:?}", attr.name, attr.as_text()?);
        }
    }

    std::fs::remove_file(&path).ok();
    Ok(())
}
