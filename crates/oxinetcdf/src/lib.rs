#![forbid(unsafe_code)]
//! Pure-Rust NetCDF-4 conventions reader.
//!
//! Reads NetCDF-4 files (which are HDF5 files with NetCDF-4 conventions encoded
//! in attributes and object references) atop the OxiH5 Pure-Rust HDF5 reader.
//! No libnetcdf, no libhdf5, no FFI.
//!
//! # Quick start
//!
//! ```no_run
//! use oxinetcdf::NcFile;
//!
//! let nc = NcFile::open("my_data.nc").unwrap();
//! let root = nc.root_group().unwrap();
//! for var in &root.variables {
//!     println!("{}: shape {:?}", var.name, var.shape);
//! }
//! ```

mod conventions;
mod error;
mod file;
mod model;

pub use error::NcError;
pub use file::NcFile;
pub use model::{NcAttribute, NcAxis, NcDimension, NcGroup, NcVariable};

pub use oxih5::{ByteOrder, Dataset, Dtype};
