//! NetCDF-4 file writer — C9 implementation.
//!
//! [`NcFileWriter`] wraps [`oxih5::FileWriter`] and encodes NetCDF-4
//! conventions (dimension scales, DIMENSION_LIST, _Netcdf4Dimid) so that the
//! resulting HDF5 file is readable by `NcFile::open`.
//!
//! # Supported features (C9)
//! - `def_dim` — define a named dimension (always fixed-length for now)
//! - `def_var` — define a variable with given dimensions and type
//! - `put_var_f64` / `put_var_i32` — store variable data
//! - `put_att_str` — store a string attribute on a variable (root attrs silently deferred)
//! - `close` — materialise the HDF5 file on disk
//!
//! # NetCDF-4 conventions written
//! For each dimension `d` at index `i`:
//! - A 1-D i32 coordinate dataset named `d` containing `[0, 1, …, size-1]`.
//! - Attribute `CLASS = "DIMENSION_SCALE"` on the coord dataset.
//! - Attribute `NAME = d` on the coord dataset (for cross-tool compatibility).
//! - Attribute `_Netcdf4Dimid = i` (i32) on the coord dataset.
//!
//! For each variable `v`:
//! - A dataset with the variable's data.
//! - Attribute `DIMENSION_LIST` (object-ref list) referencing each coordinate dataset.

use crate::error::NcError;
use crate::types::NcType;
use oxih5::FileWriter;
use oxih5_core::OxiH5Error;
use std::path::Path;

// ---------------------------------------------------------------------------
// Id types
// ---------------------------------------------------------------------------

/// Opaque handle for a defined dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NcDimId(pub(crate) usize);

/// Opaque handle for a defined variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NcVarId(pub(crate) usize);

/// Target for a `put_att_str` call — either the root group or a variable.
#[derive(Debug, Clone, Copy)]
pub enum VarOrGroup {
    /// Root group attribute (written as file-global metadata — silently deferred,
    /// not currently written to the HDF5 file).
    Root,
    /// A specific variable identified by its [`NcVarId`].
    Var(NcVarId),
}

// ---------------------------------------------------------------------------
// Internal dimension / variable descriptors
// ---------------------------------------------------------------------------

struct NcDimDef {
    name: String,
    /// Current size.  For unlimited dims this grows as `put_vara_*` is called.
    size: usize,
    /// C10: if true, dim 0 is unlimited (chunked HDF5 storage on close).
    unlimited: bool,
}

struct NcVarDef {
    name: String,
    dim_ids: Vec<usize>,
    nc_type: NcType,
    data: Option<NcData>,
    str_attrs: Vec<(String, String)>,
    /// C10: extra data appended via `put_vara_f64`.
    appended_f64: Vec<f64>,
    /// C10: extra data appended via `put_vara_i32`.
    appended_i32: Vec<i32>,
}

enum NcData {
    F64(Vec<f64>),
    I32(Vec<i32>),
    /// NC_STRING variable data: one UTF-8 string per element.
    Str(Vec<String>),
}

// ---------------------------------------------------------------------------
// NcFileWriter — public API
// ---------------------------------------------------------------------------

/// Writes a NetCDF-4 (HDF5) file using the NetCDF-4 dimension-scale conventions.
///
/// # Example
///
/// ```no_run
/// use oxinetcdf::{NcFileWriter, NcType, VarOrGroup};
///
/// let path = std::env::temp_dir().join("out.nc");
/// let mut nc = NcFileWriter::new();
/// let lat = nc.def_dim("lat", 4).unwrap();
/// let lon = nc.def_dim("lon", 8).unwrap();
/// let temp = nc.def_var("temp", &[lat, lon], NcType::Float64).unwrap();
/// let data: Vec<f64> = (0..32).map(|i| i as f64 * 0.5).collect();
/// nc.put_var_f64(temp, &data).unwrap();
/// nc.close(&path).unwrap();
/// ```
pub struct NcFileWriter {
    dims: Vec<NcDimDef>,
    vars: Vec<NcVarDef>,
    /// C11: when true, write `_nc3_strict = ""` attribute on the root group.
    classic_mode: bool,
    /// Root-group string attributes (forwarded from `put_att_str(Root, …)`).
    root_str_attrs: Vec<(String, String)>,
    /// W0b: sub-group names to be created during `close()`.
    pending_groups: Vec<String>,
}

impl Default for NcFileWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl NcFileWriter {
    /// Create a new, empty `NcFileWriter`.
    pub fn new() -> Self {
        Self {
            dims: Vec::new(),
            vars: Vec::new(),
            classic_mode: false,
            root_str_attrs: Vec::new(),
            pending_groups: Vec::new(),
        }
    }

    /// Define a new dimension.
    ///
    /// Returns an [`NcDimId`] handle that can be passed to [`NcFileWriter::def_var`].
    pub fn def_dim(&mut self, name: &str, size: usize) -> Result<NcDimId, NcError> {
        if name.is_empty() {
            return Err(NcError::H5(OxiH5Error::Format(
                "dimension name must not be empty".to_string(),
            )));
        }
        if self.dims.iter().any(|d| d.name == name) {
            return Err(NcError::H5(OxiH5Error::Format(format!(
                "duplicate dimension name '{name}'"
            ))));
        }
        let id = self.dims.len();
        self.dims.push(NcDimDef {
            name: name.to_string(),
            size,
            unlimited: false,
        });
        Ok(NcDimId(id))
    }

    /// Define a new **unlimited** dimension (C10).
    ///
    /// An unlimited dimension has no fixed size: data is appended via
    /// [`NcFileWriter::put_vara_f64`] / [`NcFileWriter::put_vara_i32`].  The HDF5 file will use chunked
    /// storage for any variable whose first dimension is unlimited.
    ///
    /// `initial_size` is the pre-allocated size hint (use 0 if unknown).
    pub fn def_dim_unlimited(
        &mut self,
        name: &str,
        initial_size: usize,
    ) -> Result<NcDimId, NcError> {
        if name.is_empty() {
            return Err(NcError::H5(OxiH5Error::Format(
                "dimension name must not be empty".to_string(),
            )));
        }
        if self.dims.iter().any(|d| d.name == name) {
            return Err(NcError::H5(OxiH5Error::Format(format!(
                "duplicate dimension name '{name}'"
            ))));
        }
        let id = self.dims.len();
        self.dims.push(NcDimDef {
            name: name.to_string(),
            size: initial_size,
            unlimited: true,
        });
        Ok(NcDimId(id))
    }

    /// Define a new variable with the given dimensions and type.
    ///
    /// Returns an [`NcVarId`] handle that can be passed to `put_var_*` and
    /// `put_att_str`.
    pub fn def_var(
        &mut self,
        name: &str,
        dims: &[NcDimId],
        nc_type: NcType,
    ) -> Result<NcVarId, NcError> {
        if name.is_empty() {
            return Err(NcError::H5(OxiH5Error::Format(
                "variable name must not be empty".to_string(),
            )));
        }
        if self.vars.iter().any(|v| v.name == name) {
            return Err(NcError::H5(OxiH5Error::Format(format!(
                "duplicate variable name '{name}'"
            ))));
        }
        for dim_id in dims {
            if dim_id.0 >= self.dims.len() {
                return Err(NcError::H5(OxiH5Error::Format(format!(
                    "NcDimId({}) out of range (only {} dims defined)",
                    dim_id.0,
                    self.dims.len()
                ))));
            }
        }
        let id = self.vars.len();
        self.vars.push(NcVarDef {
            name: name.to_string(),
            dim_ids: dims.iter().map(|d| d.0).collect(),
            nc_type,
            data: None,
            str_attrs: Vec::new(),
            appended_f64: Vec::new(),
            appended_i32: Vec::new(),
        });
        Ok(NcVarId(id))
    }

    /// Store float64 data for the given variable.
    ///
    /// `data.len()` must equal the product of all dimension sizes for `var`.
    pub fn put_var_f64(&mut self, var: NcVarId, data: &[f64]) -> Result<(), NcError> {
        let expected = self.var_elem_count(var)?;
        if data.len() != expected {
            return Err(NcError::H5(OxiH5Error::Format(format!(
                "put_var_f64: expected {expected} elements, got {}",
                data.len()
            ))));
        }
        self.vars[var.0].data = Some(NcData::F64(data.to_vec()));
        Ok(())
    }

    /// Store int32 data for the given variable.
    ///
    /// `data.len()` must equal the product of all dimension sizes for `var`.
    pub fn put_var_i32(&mut self, var: NcVarId, data: &[i32]) -> Result<(), NcError> {
        let expected = self.var_elem_count(var)?;
        if data.len() != expected {
            return Err(NcError::H5(OxiH5Error::Format(format!(
                "put_var_i32: expected {expected} elements, got {}",
                data.len()
            ))));
        }
        self.vars[var.0].data = Some(NcData::I32(data.to_vec()));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // W0d: NC_STRING variable support
    // -----------------------------------------------------------------------

    /// Define a new NC_STRING variable with the given dimensions.
    ///
    /// Equivalent to `def_var(name, dims, NcType::String)`.  The resulting
    /// HDF5 dataset uses the variable-length string (vlen) encoding backed by
    /// a Global Heap Collection (GCOL).
    ///
    /// # Errors
    ///
    /// Returns `NcError::H5(OxiH5Error::Format)` if the name is empty, a
    /// dimension ID is out of range, or the name is already in use.
    pub fn def_var_strings(&mut self, name: &str, dims: &[NcDimId]) -> Result<NcVarId, NcError> {
        self.def_var(name, dims, NcType::String)
    }

    /// Write string data for a previously defined NC_STRING variable.
    ///
    /// `data.len()` must equal the product of the variable's dimension sizes.
    ///
    /// # Errors
    ///
    /// Returns `NcError::H5(OxiH5Error::Format)` if the element count does
    /// not match, or if `var` is out of range.
    pub fn put_var_strings(&mut self, var: NcVarId, data: &[&str]) -> Result<(), NcError> {
        let expected = self.var_elem_count(var)?;
        if data.len() != expected {
            return Err(NcError::H5(OxiH5Error::Format(format!(
                "put_var_strings: expected {expected} elements, got {}",
                data.len()
            ))));
        }
        self.vars[var.0].data = Some(NcData::Str(data.iter().map(|s| s.to_string()).collect()));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // C10: Unlimited dimension append
    // -----------------------------------------------------------------------

    /// Append float64 data along the unlimited (first) axis of `var` (C10).
    ///
    /// `data` is a flat slice of elements representing one or more new records
    /// along dimension 0.  The number of elements must be a multiple of the
    /// product of all non-unlimited dimensions.
    ///
    /// The unlimited dimension's current size is grown by
    /// `data.len() / stride`, where `stride = product of dim[1..]`.
    ///
    /// On `close()`, the full accumulated data is written using chunked storage.
    pub fn put_vara_f64(&mut self, var: NcVarId, data: &[f64]) -> Result<(), NcError> {
        if var.0 >= self.vars.len() {
            return Err(NcError::H5(OxiH5Error::NotFound(format!(
                "NcVarId({}) out of range",
                var.0
            ))));
        }
        let stride = self.var_trailing_stride(var)?;
        if stride > 0 && data.len() % stride != 0 {
            return Err(NcError::H5(OxiH5Error::Format(format!(
                "put_vara_f64: data length {} is not a multiple of trailing stride {stride}",
                data.len()
            ))));
        }
        let new_records = data.len().checked_div(stride).unwrap_or(0);
        self.vars[var.0].appended_f64.extend_from_slice(data);
        // Grow the unlimited dim (dim_ids[0])
        if let Some(&dim0) = self.vars[var.0].dim_ids.first() {
            self.dims[dim0].size += new_records;
        }
        Ok(())
    }

    /// Append int32 data along the unlimited (first) axis of `var` (C10).
    pub fn put_vara_i32(&mut self, var: NcVarId, data: &[i32]) -> Result<(), NcError> {
        if var.0 >= self.vars.len() {
            return Err(NcError::H5(OxiH5Error::NotFound(format!(
                "NcVarId({}) out of range",
                var.0
            ))));
        }
        let stride = self.var_trailing_stride(var)?;
        if stride > 0 && data.len() % stride != 0 {
            return Err(NcError::H5(OxiH5Error::Format(format!(
                "put_vara_i32: data length {} is not a multiple of trailing stride {stride}",
                data.len()
            ))));
        }
        let new_records = data.len().checked_div(stride).unwrap_or(0);
        self.vars[var.0].appended_i32.extend_from_slice(data);
        if let Some(&dim0) = self.vars[var.0].dim_ids.first() {
            self.dims[dim0].size += new_records;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // C11: NETCDF4_CLASSIC strict mode
    // -----------------------------------------------------------------------

    /// Enable NETCDF4_CLASSIC mode (C11).
    ///
    /// When [`NcFileWriter::close`] is called, writes `_nc3_strict = ""` as a string attribute
    /// on the root group, making the file recognizable as a NetCDF-4 classic-model
    /// file.
    pub fn set_classic_mode(&mut self) {
        self.classic_mode = true;
    }

    // -----------------------------------------------------------------------
    // W0b: Sub-group creation
    // -----------------------------------------------------------------------

    /// Create an empty sub-group named `name` directly under the root group.
    ///
    /// This is the NcFileWriter wrapper for `FileWriter::create_group`.
    /// The underlying HDF5 writer handles the binary encoding.
    ///
    /// Returns the group name for use in build logic (currently no-op beyond
    /// validation; `NcFileWriter` does not yet support variables inside groups).
    pub fn create_group(&mut self, name: &str) -> Result<(), NcError> {
        // Validate the name; the actual group is created during build_bytes.
        if name.is_empty() || name.contains('/') {
            return Err(NcError::H5(OxiH5Error::Format(format!(
                "invalid group name '{name}'"
            ))));
        }
        // Store the name for use during build.  We use a dedicated field
        // (groups list) on the NcFileWriter.
        self.pending_groups.push(name.to_string());
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Attribute writing
    // -----------------------------------------------------------------------

    /// Attach a string attribute to a variable or the root group.
    ///
    /// For `VarOrGroup::Root`, the attribute is now properly forwarded to the
    /// HDF5 root group object header.
    pub fn put_att_str(
        &mut self,
        target: VarOrGroup,
        name: &str,
        value: &str,
    ) -> Result<(), NcError> {
        match target {
            VarOrGroup::Root => {
                // Store root group attributes for writing during close().
                self.root_str_attrs
                    .push((name.to_string(), value.to_string()));
                Ok(())
            }
            VarOrGroup::Var(var_id) => {
                if var_id.0 >= self.vars.len() {
                    return Err(NcError::H5(OxiH5Error::NotFound(format!(
                        "NcVarId({}) out of range",
                        var_id.0
                    ))));
                }
                self.vars[var_id.0]
                    .str_attrs
                    .push((name.to_string(), value.to_string()));
                Ok(())
            }
        }
    }

    /// Write the NetCDF-4/HDF5 file to `path`, consuming the writer.
    ///
    /// This finalises the file: all dimensions, coordinate variables, data
    /// variables, and their NetCDF-4 convention attributes are written.
    pub fn close(self, path: impl AsRef<Path>) -> Result<(), NcError> {
        let bytes = self.build_bytes()?;
        std::fs::write(path, &bytes)
            .map_err(OxiH5Error::Io)
            .map_err(NcError::H5)
    }

    /// Build the HDF5/NetCDF-4 byte vector without writing to disk.
    ///
    /// Useful for in-memory testing.
    pub fn close_to_vec(self) -> Result<Vec<u8>, NcError> {
        self.build_bytes()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn var_elem_count(&self, var: NcVarId) -> Result<usize, NcError> {
        if var.0 >= self.vars.len() {
            return Err(NcError::H5(OxiH5Error::NotFound(format!(
                "NcVarId({}) out of range",
                var.0
            ))));
        }
        let v = &self.vars[var.0];
        let n: usize = v
            .dim_ids
            .iter()
            .fold(1usize, |acc, &d| acc.saturating_mul(self.dims[d].size));
        Ok(n)
    }

    /// Return the product of all trailing (non-first) dimension sizes for `var`.
    ///
    /// For a variable with dims [time, lat, lon], returns lat_size * lon_size.
    /// For a 1-D variable (single dim), returns 1.
    /// For a scalar variable (no dims), returns 0 (appending not applicable).
    fn var_trailing_stride(&self, var: NcVarId) -> Result<usize, NcError> {
        if var.0 >= self.vars.len() {
            return Err(NcError::H5(OxiH5Error::NotFound(format!(
                "NcVarId({}) out of range",
                var.0
            ))));
        }
        let v = &self.vars[var.0];
        if v.dim_ids.is_empty() {
            return Ok(0); // scalar: no trailing dims
        }
        let stride = v.dim_ids[1..]
            .iter()
            .fold(1usize, |acc, &d| acc.saturating_mul(self.dims[d].size));
        Ok(stride)
    }

    /// Check whether the first dimension of `var` is an unlimited dimension.
    fn var_has_unlimited_dim0(&self, var: &NcVarDef) -> bool {
        var.dim_ids
            .first()
            .map(|&d| self.dims[d].unlimited)
            .unwrap_or(false)
    }

    fn build_bytes(self) -> Result<Vec<u8>, NcError> {
        let mut writer = FileWriter::new();

        // ------------------------------------------------------------------
        // C11: classic mode — write _nc3_strict on root group.
        // ------------------------------------------------------------------
        if self.classic_mode {
            writer.write_root_str_attr("_nc3_strict", "");
        }

        // Forward any user-supplied root group attributes.
        for (name, val) in &self.root_str_attrs {
            writer.write_root_str_attr(name, val);
        }

        // ------------------------------------------------------------------
        // W0b: create pending groups.
        // ------------------------------------------------------------------
        for grp_name in &self.pending_groups {
            writer.create_group(grp_name).map_err(NcError::H5)?;
        }

        // ------------------------------------------------------------------
        // 1. Write coordinate variables for each dimension.
        //    Unlimited dims use create_dataset_unlimited for their coord var.
        // ------------------------------------------------------------------
        for (dim_idx, dim) in self.dims.iter().enumerate() {
            let indices: Vec<i32> = (0..dim.size).map(|i| i as i32).collect();

            if dim.unlimited {
                // Use chunked storage for unlimited coordinate variable.
                let raw: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
                let dtype = oxih5_core::Dtype::Int {
                    size: 4,
                    signed: true,
                    order: oxih5_core::ByteOrder::Little,
                };
                writer
                    .create_dataset_unlimited(
                        &dim.name,
                        &[dim.size],
                        &[dim.size.max(1)],
                        &dtype,
                        &raw,
                    )
                    .map_err(NcError::H5)?;
            } else {
                writer
                    .write_dataset_i32(&dim.name, &indices, &[dim.size])
                    .map_err(NcError::H5)?;
            }

            writer
                .write_string_attr(&dim.name, "CLASS", "DIMENSION_SCALE")
                .map_err(NcError::H5)?;
            writer
                .write_string_attr(&dim.name, "NAME", &dim.name)
                .map_err(NcError::H5)?;
            writer
                .write_i32_attr(&dim.name, "_Netcdf4Dimid", dim_idx as i32)
                .map_err(NcError::H5)?;

            let self_name: &str = &dim.name;
            writer
                .write_obj_ref_list_attr(&dim.name, "DIMENSION_LIST", &[self_name])
                .map_err(NcError::H5)?;
        }

        // ------------------------------------------------------------------
        // 2. Write data variables.
        //    C10: for unlimited-dim variables, combine put_var + put_vara data.
        // ------------------------------------------------------------------
        for var in &self.vars {
            let shape: Vec<usize> = var.dim_ids.iter().map(|&d| self.dims[d].size).collect();
            let n_elems: usize = if shape.is_empty() {
                1
            } else {
                shape.iter().product()
            };
            let unlimited = self.var_has_unlimited_dim0(var);

            match &var.nc_type {
                NcType::Float64 => {
                    // Combine put_var data (initial) with appended data.
                    let mut data: Vec<f64> = match &var.data {
                        Some(NcData::F64(v)) => v.clone(),
                        _ => Vec::new(),
                    };
                    data.extend_from_slice(&var.appended_f64);
                    if data.is_empty() {
                        data = vec![0.0f64; n_elems];
                    }

                    if unlimited {
                        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                        let dtype = oxih5_core::Dtype::Float {
                            size: 8,
                            order: oxih5_core::ByteOrder::Little,
                        };
                        writer
                            .create_dataset_unlimited(
                                &var.name,
                                &shape,
                                &[shape[0].max(1)],
                                &dtype,
                                &raw,
                            )
                            .map_err(NcError::H5)?;
                    } else {
                        writer
                            .write_dataset_f64(&var.name, &data, &shape)
                            .map_err(NcError::H5)?;
                    }
                }
                NcType::Float32 => {
                    let data: Vec<f32> = match &var.data {
                        Some(NcData::F64(v)) => v.iter().map(|&x| x as f32).collect(),
                        _ => vec![0.0f32; n_elems],
                    };
                    writer
                        .write_dataset_f32(&var.name, &data, &shape)
                        .map_err(NcError::H5)?;
                }
                NcType::Int32 => {
                    let mut data: Vec<i32> = match &var.data {
                        Some(NcData::I32(v)) => v.clone(),
                        _ => Vec::new(),
                    };
                    data.extend_from_slice(&var.appended_i32);
                    if data.is_empty() {
                        data = vec![0i32; n_elems];
                    }

                    if unlimited {
                        let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                        let dtype = oxih5_core::Dtype::Int {
                            size: 4,
                            signed: true,
                            order: oxih5_core::ByteOrder::Little,
                        };
                        writer
                            .create_dataset_unlimited(
                                &var.name,
                                &shape,
                                &[shape[0].max(1)],
                                &dtype,
                                &raw,
                            )
                            .map_err(NcError::H5)?;
                    } else {
                        writer
                            .write_dataset_i32(&var.name, &data, &shape)
                            .map_err(NcError::H5)?;
                    }
                }
                NcType::Int64 => {
                    let data: Vec<i64> = match &var.data {
                        Some(NcData::I32(v)) => v.iter().map(|&x| x as i64).collect(),
                        Some(NcData::F64(v)) => v.iter().map(|&x| x as i64).collect(),
                        Some(NcData::Str(_)) | None => vec![0i64; n_elems],
                    };
                    writer
                        .write_dataset_i64(&var.name, &data, &shape)
                        .map_err(NcError::H5)?;
                }
                NcType::UInt8 => {
                    let data: Vec<u8> = match &var.data {
                        Some(NcData::I32(v)) => v.iter().map(|&x| x as u8).collect(),
                        _ => vec![0u8; n_elems],
                    };
                    writer
                        .write_dataset_u8(&var.name, &data, &shape)
                        .map_err(NcError::H5)?;
                }
                NcType::String => {
                    // W0d: NC_STRING variable — vlen-string HDF5 dataset.
                    let strings: Vec<String> = match &var.data {
                        Some(NcData::Str(v)) => v.clone(),
                        _ => vec![String::new(); n_elems],
                    };
                    let str_refs: Vec<&str> = strings.iter().map(String::as_str).collect();
                    writer
                        .create_vlen_string_dataset(&var.name, &str_refs)
                        .map_err(NcError::H5)?;
                }
                other => {
                    return Err(NcError::H5(OxiH5Error::Format(format!(
                        "NcFileWriter: unsupported variable type {other:?}"
                    ))));
                }
            }

            if !var.dim_ids.is_empty() {
                let dim_names: Vec<&str> = var
                    .dim_ids
                    .iter()
                    .map(|&d| self.dims[d].name.as_str())
                    .collect();
                writer
                    .write_obj_ref_list_attr(&var.name, "DIMENSION_LIST", &dim_names)
                    .map_err(NcError::H5)?;
            }

            for (attr_name, attr_val) in &var.str_attrs {
                writer
                    .write_string_attr(&var.name, attr_name, attr_val)
                    .map_err(NcError::H5)?;
            }
        }

        // ------------------------------------------------------------------
        // 3. Build the HDF5 bytes.
        // ------------------------------------------------------------------
        writer.build_to_vec().map_err(NcError::H5)
    }
}

// ---------------------------------------------------------------------------
// Tests — C9
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NcFile, NcType};

    // -----------------------------------------------------------------------
    // C9.1 — basic two-dimension, one-variable file round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn c9_write_and_read_basic() {
        let tmp = std::env::temp_dir().join("oxinetcdf_test_c9_basic.nc");

        // Write
        let mut nc = NcFileWriter::new();
        let lat = nc.def_dim("lat", 4).expect("def_dim lat");
        let lon = nc.def_dim("lon", 8).expect("def_dim lon");
        let temp = nc
            .def_var("temp", &[lat, lon], NcType::Float64)
            .expect("def_var");
        let data: Vec<f64> = (0..32).map(|i| i as f64 * 0.5).collect();
        nc.put_var_f64(temp, &data).expect("put_var_f64");
        nc.close(&tmp).expect("close");

        // Read back
        let nc2 = NcFile::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let root = nc2.root_group().expect("root_group");

        // Dimensions
        let dims = &root.dimensions;
        assert_eq!(dims.len(), 2, "expected 2 dimensions, got {}", dims.len());
        assert!(
            dims.iter().any(|d| d.name == "lat" && d.len == 4),
            "lat dim not found: {dims:?}"
        );
        assert!(
            dims.iter().any(|d| d.name == "lon" && d.len == 8),
            "lon dim not found: {dims:?}"
        );

        // Variables: lat (coord), lon (coord), temp (data)
        assert_eq!(root.variables.len(), 3, "expected 3 vars: lat, lon, temp");
        let temp_var = root
            .variables
            .iter()
            .find(|v| v.name == "temp")
            .expect("temp variable not found");
        assert_eq!(
            temp_var.shape,
            vec![4u64, 8],
            "temp shape wrong: {:?}",
            temp_var.shape
        );
        assert_eq!(temp_var.dims.len(), 2);
    }

    // -----------------------------------------------------------------------
    // C9.2 — scalar variable (0 dimensions)
    // -----------------------------------------------------------------------

    #[test]
    fn c9_scalar_variable() {
        let tmp = std::env::temp_dir().join("oxinetcdf_test_c9_scalar.nc");

        let mut nc = NcFileWriter::new();
        let scalar_var = nc.def_var("pi", &[], NcType::Float64).expect("def_var");
        nc.put_var_f64(scalar_var, &[std::f64::consts::PI])
            .expect("put_var_f64");
        nc.close(&tmp).expect("close");

        let nc2 = NcFile::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);
        let root = nc2.root_group().expect("root_group");

        assert_eq!(root.dimensions.len(), 0);
        assert_eq!(root.variables.len(), 1);
        let pi_var = &root.variables[0];
        assert_eq!(pi_var.name, "pi");
        // Scalar dataset has empty shape.
        assert!(
            pi_var.shape.is_empty(),
            "scalar shape should be empty, got {:?}",
            pi_var.shape
        );
    }

    // -----------------------------------------------------------------------
    // C9.3 — string attribute on a variable
    // -----------------------------------------------------------------------

    #[test]
    fn c9_string_attribute_on_var() {
        let tmp = std::env::temp_dir().join("oxinetcdf_test_c9_strattr.nc");

        let mut nc = NcFileWriter::new();
        let x = nc.def_dim("x", 3).expect("def_dim");
        let pressure = nc
            .def_var("pressure", &[x], NcType::Float32)
            .expect("def_var");
        nc.put_att_str(VarOrGroup::Var(pressure), "units", "hPa")
            .expect("put_att_str");
        nc.close(&tmp).expect("close");

        let nc2 = NcFile::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);
        let root = nc2.root_group().expect("root_group");

        let pvar = root
            .variables
            .iter()
            .find(|v| v.name == "pressure")
            .expect("pressure");
        let units_attr = pvar.attrs.iter().find(|a| a.name == "units");
        assert!(units_attr.is_some(), "units attr not found on pressure");
        let val = units_attr.unwrap().as_text().expect("as_text");
        assert_eq!(val, "hPa");
    }

    // -----------------------------------------------------------------------
    // C9.4 — root group attribute is written (not an error, and stored)
    // -----------------------------------------------------------------------

    #[test]
    fn c9_root_group_attr_is_no_error() {
        let tmp = std::env::temp_dir().join("oxinetcdf_test_c9_rootattr.nc");

        let mut nc = NcFileWriter::new();
        // Root attrs are now properly stored and written to the HDF5 root group.
        nc.put_att_str(VarOrGroup::Root, "Conventions", "CF-1.8")
            .expect("put_att_str root");
        nc.close(&tmp).expect("close");
        let _ = std::fs::remove_file(&tmp);
    }

    // -----------------------------------------------------------------------
    // C9.5 — int32 variable
    // -----------------------------------------------------------------------

    #[test]
    fn c9_int32_variable() {
        let tmp = std::env::temp_dir().join("oxinetcdf_test_c9_int32.nc");

        let mut nc = NcFileWriter::new();
        let t = nc.def_dim("t", 4).expect("def_dim");
        let flags = nc.def_var("flags", &[t], NcType::Int32).expect("def_var");
        nc.put_var_i32(flags, &[1i32, 0, 1, 1])
            .expect("put_var_i32");
        nc.close(&tmp).expect("close");

        let nc2 = NcFile::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);
        let root = nc2.root_group().expect("root_group");

        let fv = root
            .variables
            .iter()
            .find(|v| v.name == "flags")
            .expect("flags");
        assert_eq!(fv.shape, vec![4u64]);
        assert_eq!(fv.dims.len(), 1);
    }

    // -----------------------------------------------------------------------
    // C9.6 — duplicate dimension name returns error
    // -----------------------------------------------------------------------

    #[test]
    fn c9_duplicate_dim_error() {
        let mut nc = NcFileWriter::new();
        nc.def_dim("x", 4).expect("first def_dim");
        let result = nc.def_dim("x", 8);
        assert!(result.is_err(), "expected error for duplicate dim name");
    }

    // -----------------------------------------------------------------------
    // C9.7 — DIMENSION_LIST refs resolve correctly (two vars sharing dims)
    // -----------------------------------------------------------------------

    #[test]
    fn c9_two_vars_sharing_dims() {
        let tmp = std::env::temp_dir().join("oxinetcdf_test_c9_twovars.nc");

        let mut nc = NcFileWriter::new();
        let lat = nc.def_dim("lat", 3).expect("lat");
        let lon = nc.def_dim("lon", 5).expect("lon");
        let t1 = nc
            .def_var("u_wind", &[lat, lon], NcType::Float64)
            .expect("u");
        let t2 = nc
            .def_var("v_wind", &[lat, lon], NcType::Float64)
            .expect("v");
        let data = vec![0.0f64; 15];
        nc.put_var_f64(t1, &data).expect("u data");
        nc.put_var_f64(t2, &data).expect("v data");
        nc.close(&tmp).expect("close");

        let nc2 = NcFile::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);
        let root = nc2.root_group().expect("root");

        assert_eq!(root.dimensions.len(), 2);
        let u = root
            .variables
            .iter()
            .find(|v| v.name == "u_wind")
            .expect("u_wind");
        let v = root
            .variables
            .iter()
            .find(|v| v.name == "v_wind")
            .expect("v_wind");
        assert_eq!(u.shape, vec![3u64, 5]);
        assert_eq!(v.shape, vec![3u64, 5]);

        // Both variables must reference the same lat/lon dim ids.
        assert_eq!(u.dims[0].dim_id, v.dims[0].dim_id, "lat dim ids differ");
        assert_eq!(u.dims[1].dim_id, v.dims[1].dim_id, "lon dim ids differ");
    }

    // -----------------------------------------------------------------------
    // C10: unlimited dimension append — 5+5 time steps round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn c10_unlimited_dim_append_roundtrip() {
        let tmp = std::env::temp_dir().join("oxinetcdf_test_c10_append.nc");

        let mut nc = NcFileWriter::new();
        let time = nc.def_dim_unlimited("time", 0).expect("def_dim_unlimited");
        let temp_var = nc
            .def_var("temp", &[time], NcType::Float64)
            .expect("def_var temp");

        // Append 5 time steps
        let data1: Vec<f64> = (0..5).map(|i| i as f64).collect();
        nc.put_vara_f64(temp_var, &data1).expect("put_vara first 5");

        // Append 5 more time steps
        let data2: Vec<f64> = (5..10).map(|i| i as f64).collect();
        nc.put_vara_f64(temp_var, &data2).expect("put_vara next 5");

        nc.close(&tmp).expect("close");

        // Read back
        let nc2 = NcFile::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);
        let root = nc2.root_group().expect("root_group");

        let temp_v = root
            .variables
            .iter()
            .find(|v| v.name == "temp")
            .expect("temp variable not found");

        assert_eq!(
            temp_v.shape,
            vec![10u64],
            "expected shape [10], got {:?}",
            temp_v.shape
        );

        let values = temp_v.read_f64(&nc2).expect("read_f64");
        assert_eq!(values.len(), 10, "expected 10 values");
        for (i, &v) in values.iter().enumerate() {
            assert!(
                (v - i as f64).abs() < 1e-15,
                "value mismatch at index {i}: expected {}, got {v}",
                i as f64
            );
        }
    }

    /// C10 with 2-D data: time × lat (time unlimited, lat fixed).
    #[test]
    fn c10_unlimited_2d_append_roundtrip() {
        let tmp = std::env::temp_dir().join("oxinetcdf_test_c10_2d.nc");

        let mut nc = NcFileWriter::new();
        let time = nc.def_dim_unlimited("time", 0).expect("time");
        let lat = nc.def_dim("lat", 3).expect("lat");
        let temp_var = nc
            .def_var("temp", &[time, lat], NcType::Float64)
            .expect("def_var");

        // First batch: 2 time steps × 3 lats = 6 values
        let data1: Vec<f64> = (0..6).map(|i| i as f64).collect();
        nc.put_vara_f64(temp_var, &data1).expect("put_vara batch1");

        // Second batch: 3 more time steps × 3 lats = 9 values
        let data2: Vec<f64> = (6..15).map(|i| i as f64).collect();
        nc.put_vara_f64(temp_var, &data2).expect("put_vara batch2");

        nc.close(&tmp).expect("close");

        let nc2 = NcFile::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);
        let root = nc2.root_group().expect("root_group");

        let temp_v = root
            .variables
            .iter()
            .find(|v| v.name == "temp")
            .expect("temp");

        assert_eq!(
            temp_v.shape,
            vec![5u64, 3],
            "expected shape [5, 3], got {:?}",
            temp_v.shape
        );
    }

    // -----------------------------------------------------------------------
    // C11: NETCDF4_CLASSIC strict mode — _nc3_strict on root group
    // -----------------------------------------------------------------------

    #[test]
    fn c11_classic_mode_nc3_strict_attribute() {
        let tmp = std::env::temp_dir().join("oxinetcdf_test_c11_classic.nc");

        let mut nc = NcFileWriter::new();
        nc.set_classic_mode();
        // Write a trivial variable so the file isn't completely empty.
        let x = nc.def_dim("x", 2).expect("def_dim");
        let v = nc.def_var("vals", &[x], NcType::Int32).expect("def_var");
        nc.put_var_i32(v, &[1i32, 2]).expect("put_var_i32");
        nc.close(&tmp).expect("close");

        let nc2 = NcFile::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);
        let root = nc2.root_group().expect("root_group");

        let nc3_attr = root.attrs.iter().find(|a| a.name == "_nc3_strict");
        assert!(
            nc3_attr.is_some(),
            "_nc3_strict attribute not found on root group; present attrs: {:?}",
            root.attrs
                .iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>()
        );
    }

    /// C11: a non-classic file must NOT have _nc3_strict.
    #[test]
    fn c11_non_classic_no_nc3_strict() {
        let tmp = std::env::temp_dir().join("oxinetcdf_test_c11_nonclassic.nc");

        let mut nc = NcFileWriter::new();
        let x = nc.def_dim("x", 2).expect("def_dim");
        let v = nc.def_var("vals", &[x], NcType::Int32).expect("def_var");
        nc.put_var_i32(v, &[10i32, 20]).expect("put_var_i32");
        nc.close(&tmp).expect("close");

        let nc2 = NcFile::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);
        let root = nc2.root_group().expect("root_group");

        let nc3_attr = root.attrs.iter().find(|a| a.name == "_nc3_strict");
        assert!(
            nc3_attr.is_none(),
            "_nc3_strict should be absent on non-classic file"
        );
    }

    // -----------------------------------------------------------------------
    // W0d: NC_STRING variable round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn nc_string_variable_round_trip() {
        let tmp = std::env::temp_dir().join("oxinetcdf_test_nc_str.nc");

        let mut ncw = NcFileWriter::new();
        let nobs = ncw.def_dim("nobs", 3).expect("def_dim");
        let names = ncw
            .def_var_strings("names", &[nobs])
            .expect("def_var_strings");
        ncw.put_var_strings(names, &["alice", "bob", "carol"])
            .expect("put_var_strings");
        ncw.close(&tmp).expect("close");

        let nc = NcFile::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let root = nc.root_group().expect("root_group");
        let var = root.variable("names").expect("names variable not found");
        let strs = var.read_strings(&nc).expect("read_strings");
        assert_eq!(strs, vec!["alice", "bob", "carol"]);
    }

    #[test]
    fn nc_string_var_with_empty_strings() {
        let tmp = std::env::temp_dir().join("oxinetcdf_test_nc_str_empty.nc");

        let mut ncw = NcFileWriter::new();
        let n = ncw.def_dim("n", 4).expect("def_dim");
        let v = ncw
            .def_var_strings("labels", &[n])
            .expect("def_var_strings");
        ncw.put_var_strings(v, &["first", "", "third", ""])
            .expect("put_var_strings");
        ncw.close(&tmp).expect("close");

        let nc = NcFile::open(&tmp).expect("open");
        let _ = std::fs::remove_file(&tmp);

        let root = nc.root_group().expect("root_group");
        let var = root.variable("labels").expect("labels variable");
        let strs = var.read_strings(&nc).expect("read_strings");
        assert_eq!(strs, vec!["first", "", "third", ""]);
    }
}
