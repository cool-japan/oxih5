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

pub mod cf;
mod conventions;
mod error;
mod file;
mod model;
mod resolver;
pub mod types;
pub mod write;

pub use error::NcError;
pub use file::NcFile;
pub use model::{
    apply_fill_mask, apply_fill_mask_f32, apply_fill_mask_f64, NcAttribute, NcAxis, NcDimension,
    NcGroup, NcVariable,
};
pub use resolver::{collect_global_dims, GlobalDim, MAX_GROUP_DEPTH};
pub use types::NcType;
pub use write::{NcDimId, NcFileWriter, NcVarId, VarOrGroup};

pub use oxih5::{ByteOrder, Dataset, Dtype};
