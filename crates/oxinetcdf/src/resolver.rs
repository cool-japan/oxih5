//! HDF5 group traversal (B3) and cross-group dimension resolution (B4).
//!
//! # Two-phase dimension resolution
//!
//! Phase 1 ([`collect_global_dims`]): walk the entire file tree, find every
//! dataset tagged with `CLASS = "DIMENSION_SCALE"`, and build a map from its
//! HDF5 object-header address to a [`GlobalDim`] record.
//!
//! Phase 2 ([`resolve_group_deep`]): recursively resolve each group into an
//! [`NcGroup`].  For each variable's `DIMENSION_LIST` attribute, resolve each
//! object-reference address via:
//! 1. The local addr cache (already resolved in this group).
//! 2. The `global_dims` registry (cross-group dim scales from Phase 1).
//! 3. Lazy resolution via [`oxih5::File::attrs_of`] + [`oxih5::File::dataset_at`].
//! 4. Phony fallback.

use std::collections::{HashMap, HashSet};

use oxih5::File as H5File;

use crate::conventions::{is_reserved_attr, parse_pure_dim_sentinel, phony_dim_name};
use crate::error::NcError;
use crate::model::{NcAttribute, NcAxis, NcDimension, NcGroup, NcVariable};

/// Maximum allowed group recursion depth.  Deep beyond this indicates a cycle
/// via hard links or an unusually nested file structure.
pub const MAX_GROUP_DEPTH: usize = 64;

// ---------------------------------------------------------------------------
// GlobalDim — B4 cross-group dimension record
// ---------------------------------------------------------------------------

/// Metadata for one dimension-scale dataset discovered during the Phase-1 scan.
///
/// Keyed by the HDF5 object-header address of the dimension-scale dataset.
#[derive(Debug, Clone)]
pub struct GlobalDim {
    /// NetCDF dimension name (last path segment of the dim-scale dataset).
    pub name: String,
    /// Number of elements along this dimension.
    pub size: u64,
    /// True when any axis of the dim-scale dataset is unlimited (`H5S_UNLIMITED`).
    pub is_unlimited: bool,
    /// Full HDF5 path of the group that owns this dimension scale.
    pub group_path: String,
    /// The `_Netcdf4Dimid` attribute value, or 0 if absent.
    pub dim_id: u32,
}

// ---------------------------------------------------------------------------
// Phase 1 — collect_global_dims
// ---------------------------------------------------------------------------

/// Walk the entire file tree starting at `root_path` and collect all
/// dimension-scale datasets.
///
/// Returns `HashMap<header_addr, GlobalDim>`.  A dataset is a dimension scale
/// if it carries `CLASS = "DIMENSION_SCALE"` as an HDF5 attribute.
pub fn collect_global_dims(file: &H5File, root_path: &str) -> HashMap<u64, GlobalDim> {
    let mut result = HashMap::new();
    let mut path_visited: HashSet<String> = HashSet::new();
    collect_global_dims_rec(file, root_path, &mut result, &mut path_visited, 0);
    result
}

fn collect_global_dims_rec(
    file: &H5File,
    group_path: &str,
    result: &mut HashMap<u64, GlobalDim>,
    path_visited: &mut HashSet<String>,
    depth: usize,
) {
    if depth > MAX_GROUP_DEPTH {
        return;
    }
    if !path_visited.insert(group_path.to_string()) {
        return; // already visited (soft-link cycle)
    }

    let grp = match file.group(group_path) {
        Ok(g) => g,
        Err(_) => return,
    };

    let ds_names = match grp.datasets() {
        Ok(v) => v,
        Err(_) => return,
    };

    for ds_name in &ds_names {
        let full_path = build_path(group_path, ds_name);

        let attr_views = match file.attr_views(&full_path) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let is_dim_scale = attr_views
            .iter()
            .find(|v| v.name() == "CLASS")
            .and_then(|v| v.as_str_fixed())
            .map(|s| s.trim() == "DIMENSION_SCALE")
            .unwrap_or(false);

        if !is_dim_scale {
            continue;
        }

        // Use the new header_addr_of API to get a stable cross-group key.
        let addr = match file.header_addr_of(&full_path) {
            Ok(a) => a,
            Err(_) => continue,
        };

        let dim_id = attr_views
            .iter()
            .find(|v| v.name() == "_Netcdf4Dimid")
            .and_then(|v| v.as_i64())
            .and_then(|i| u32::try_from(i).ok())
            .unwrap_or(0);

        let ds = match file.dataset(&full_path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let size = ds.shape.first().copied().unwrap_or(0) as u64;
        let is_unlimited = ds.is_unlimited();

        result.insert(
            addr,
            GlobalDim {
                name: ds_name.clone(),
                size,
                is_unlimited,
                group_path: group_path.to_string(),
                dim_id,
            },
        );
    }

    let subgroup_names = match grp.groups() {
        Ok(v) => v,
        Err(_) => return,
    };

    for sg_name in &subgroup_names {
        let sg_path = build_path(group_path, sg_name);
        collect_global_dims_rec(file, &sg_path, result, path_visited, depth + 1);
    }
}

// ---------------------------------------------------------------------------
// Phase 2 — resolve_group_deep (B3)
// ---------------------------------------------------------------------------

/// Recursively resolve an HDF5 group into an [`NcGroup`].
///
/// All sub-groups are resolved and stored in [`NcGroup::children`].  Cross-group
/// dimension references in `DIMENSION_LIST` are resolved via `global_dims`.
///
/// `visited` tracks group paths already on the current call stack to detect
/// soft-link cycles.  `depth` enforces [`MAX_GROUP_DEPTH`].
pub fn resolve_group_deep(
    file: &H5File,
    group_path: &str,
    group_name: &str,
    global_dims: &HashMap<u64, GlobalDim>,
    visited: &mut HashSet<String>,
    depth: usize,
) -> Result<NcGroup, NcError> {
    if depth > MAX_GROUP_DEPTH {
        return Err(NcError::MaxDepthExceeded);
    }
    if !visited.insert(group_path.to_string()) {
        return Err(NcError::CycleDetected);
    }

    let grp = file.group(group_path)?;

    // ------------------------------------------------------------------
    // 1. Group-level attributes (skip HDF5 convention internals).
    // ------------------------------------------------------------------
    let group_attr_views = grp.attr_views()?;
    let group_attrs: Vec<NcAttribute> = group_attr_views
        .iter()
        .filter(|v| !is_reserved_attr(v.name()))
        .map(|v| NcAttribute::new_with_view(v))
        .collect();

    // ------------------------------------------------------------------
    // 2. Enumerate datasets and subgroup names.
    // ------------------------------------------------------------------
    let ds_names = grp.datasets()?;
    let subgroup_names_raw = grp.groups()?;

    // ------------------------------------------------------------------
    // 3. Collect local dimension scales.
    // ------------------------------------------------------------------
    let LocalDimState {
        mut dim_by_id,
        coord_var_dimid,
        mut addr_map,
        next_phony,
    } = collect_local_dims(file, group_path, &ds_names)?;
    let mut phony_counter = next_phony;

    // ------------------------------------------------------------------
    // 4. Process variables.
    // ------------------------------------------------------------------
    let mut variables: Vec<NcVariable> = Vec::new();

    for ds_name in &ds_names {
        let full_path = build_path(group_path, ds_name);
        let ds = match file.dataset(&full_path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let attr_views = match file.attr_views(&full_path) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let is_coord = coord_var_dimid.contains_key(ds_name.as_str());

        let nc_attrs: Vec<NcAttribute> = attr_views
            .iter()
            .filter(|v| !is_reserved_attr(v.name()))
            .map(|v| NcAttribute::new_with_view(v))
            .collect();

        let shape: Vec<u64> = ds.shape.iter().map(|&s| s as u64).collect();
        let rank = shape.len();

        let dim_list_view = attr_views.iter().find(|v| v.name() == "DIMENSION_LIST");

        let dims: Vec<NcAxis> = match dim_list_view {
            Some(dl_view) => match dl_view.as_object_refs() {
                Ok(refs) => {
                    if refs.len() != rank {
                        return Err(NcError::DimensionListArity {
                            var: ds_name.clone(),
                            found: refs.len(),
                            rank,
                        });
                    }
                    resolve_dim_list(
                        file,
                        ds_name,
                        &refs,
                        &shape,
                        group_path,
                        global_dims,
                        &mut dim_by_id,
                        &mut addr_map,
                        &mut phony_counter,
                    )?
                }
                Err(_) => {
                    // DIMENSION_LIST present but unreadable (vlen-of-refs or other
                    // format) — fall back to phony dims.
                    create_phony_axes(&shape, &mut phony_counter, &mut dim_by_id, group_path)
                }
            },
            None if rank == 0 => vec![],
            None => create_phony_axes(&shape, &mut phony_counter, &mut dim_by_id, group_path),
        };

        variables.push(NcVariable {
            name: ds_name.clone(),
            dtype: ds.dtype.clone(),
            dims,
            shape,
            attrs: nc_attrs,
            is_coordinate: is_coord,
            h5_path: full_path,
        });
    }

    // ------------------------------------------------------------------
    // 5. Sort dimensions by id for deterministic output.
    // ------------------------------------------------------------------
    let mut dimensions: Vec<NcDimension> = dim_by_id.into_values().collect();
    dimensions.sort_by_key(|d| d.id);

    // ------------------------------------------------------------------
    // 6. Recurse into sub-groups (B3).
    // ------------------------------------------------------------------
    let mut children: Vec<NcGroup> = Vec::new();
    for sg_name in &subgroup_names_raw {
        let sg_path = build_path(group_path, sg_name);
        match resolve_group_deep(file, &sg_path, sg_name, global_dims, visited, depth + 1) {
            Ok(child) => children.push(child),
            // Cycles / depth exceeded are soft errors: skip the child but
            // don't abort the parent.
            Err(NcError::CycleDetected) | Err(NcError::MaxDepthExceeded) => {}
            Err(e) => return Err(e),
        }
    }

    Ok(NcGroup {
        name: group_name.to_string(),
        path: group_path.to_string(),
        dimensions,
        variables,
        attrs: group_attrs,
        subgroup_names: subgroup_names_raw,
        children,
    })
}

// ---------------------------------------------------------------------------
// Local dim collection helpers
// ---------------------------------------------------------------------------

struct LocalDimState {
    dim_by_id: HashMap<u32, NcDimension>,
    coord_var_dimid: HashMap<String, u32>,
    addr_map: HashMap<u64, u32>,
    /// Counter value to use for the next phony dimension created during
    /// variable processing.
    next_phony: u32,
}

/// Scan `ds_names` in `group_path`, identify dimension scales, and return
/// the local dim state.
fn collect_local_dims(
    file: &H5File,
    group_path: &str,
    ds_names: &[String],
) -> Result<LocalDimState, NcError> {
    let mut dim_by_id: HashMap<u32, NcDimension> = HashMap::new();
    let mut coord_var_dimid: HashMap<String, u32> = HashMap::new();
    let mut addr_map: HashMap<u64, u32> = HashMap::new();
    // phony_counter is used only for dim scales without _Netcdf4Dimid.
    let mut phony_counter: u32 = 0;

    for ds_name in ds_names {
        let full_path = build_path(group_path, ds_name);
        let attr_views = match file.attr_views(&full_path) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let is_dim_scale = attr_views
            .iter()
            .find(|v| v.name() == "CLASS")
            .and_then(|v| v.as_str_fixed())
            .map(|s| s.trim() == "DIMENSION_SCALE")
            .unwrap_or(false);

        if !is_dim_scale {
            continue;
        }

        let dim_id_opt = attr_views
            .iter()
            .find(|v| v.name() == "_Netcdf4Dimid")
            .and_then(|v| v.as_i64())
            .and_then(|i| u32::try_from(i).ok());

        let ds = match file.dataset(&full_path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let dim_len = ds.shape.first().copied().unwrap_or(0) as u64;
        let is_unlimited = ds.is_unlimited();

        let dim_id = if let Some(id) = dim_id_opt {
            id
        } else {
            let id = phony_counter;
            phony_counter += 1;
            id
        };

        if dim_by_id.contains_key(&dim_id) {
            return Err(NcError::DuplicateDimId(dim_id));
        }

        dim_by_id.insert(
            dim_id,
            NcDimension {
                name: ds_name.clone(),
                len: dim_len,
                id: dim_id,
                is_unlimited,
            },
        );
        coord_var_dimid.insert(ds_name.clone(), dim_id);

        // Record the header address → dim_id mapping for DIMENSION_LIST lookups.
        if let Ok(addr) = file.header_addr_of(&full_path) {
            addr_map.insert(addr, dim_id);
        }
    }

    // The next phony ID used during variable processing must not collide with
    // any actual dim IDs already assigned.
    let max_actual_id = dim_by_id.keys().copied().max();
    let next_phony = match max_actual_id {
        // If we have actual IDs, start phony IDs after the highest one.
        Some(max_id) => max_id.saturating_add(1),
        // No actual dims at all: start from where the within-phase counter left
        // off (handles the case of multiple dims without _Netcdf4Dimid).
        None => phony_counter,
    };

    Ok(LocalDimState {
        dim_by_id,
        coord_var_dimid,
        addr_map,
        next_phony,
    })
}

// ---------------------------------------------------------------------------
// DIMENSION_LIST resolution
// ---------------------------------------------------------------------------

/// Resolve a full DIMENSION_LIST (`refs`) for one variable into `Vec<NcAxis>`.
#[allow(clippy::too_many_arguments)]
fn resolve_dim_list(
    file: &H5File,
    ds_name: &str,
    refs: &[u64],
    shape: &[u64],
    group_path: &str,
    global_dims: &HashMap<u64, GlobalDim>,
    dim_by_id: &mut HashMap<u32, NcDimension>,
    addr_map: &mut HashMap<u64, u32>,
    phony_counter: &mut u32,
) -> Result<Vec<NcAxis>, NcError> {
    let mut axes = Vec::with_capacity(refs.len());

    for (axis_idx, &obj_addr) in refs.iter().enumerate() {
        let var_len = shape[axis_idx];

        let axis = if obj_addr == u64::MAX || obj_addr == 0 {
            // Null / undefined reference → phony dimension.
            make_phony_axis(var_len, group_path, dim_by_id, phony_counter)
        } else if let Some(&cached_dimid) = addr_map.get(&obj_addr) {
            // 1. Already in local addr cache.
            match dim_by_id.get(&cached_dimid) {
                Some(dim) => {
                    // Length validation (skip for unlimited dims).
                    if var_len != 0 && dim.len != 0 && dim.len != var_len && !dim.is_unlimited {
                        return Err(NcError::AxisLengthMismatch {
                            var: ds_name.to_string(),
                            axis: axis_idx,
                            var_len,
                            dim_len: dim.len,
                        });
                    }
                    NcAxis {
                        dim_id: cached_dimid,
                        name: dim.name.clone(),
                        len: dim.len,
                        is_unlimited: dim.is_unlimited,
                        group_path: group_path.to_string(),
                    }
                }
                None => make_phony_axis(var_len, group_path, dim_by_id, phony_counter),
            }
        } else if let Some(gdim) = global_dims.get(&obj_addr) {
            // 2. Cross-group dim from Phase-1 scan.
            //
            // Register it in this group's local dim table under a fresh phony
            // ID so that subsequent references to the same address within this
            // group are resolved consistently.
            let local_id = *phony_counter;
            *phony_counter += 1;
            dim_by_id.insert(
                local_id,
                NcDimension {
                    name: gdim.name.clone(),
                    len: gdim.size,
                    id: local_id,
                    is_unlimited: gdim.is_unlimited,
                },
            );
            addr_map.insert(obj_addr, local_id);
            NcAxis {
                dim_id: local_id,
                name: gdim.name.clone(),
                len: gdim.size,
                is_unlimited: gdim.is_unlimited,
                group_path: gdim.group_path.clone(),
            }
        } else {
            // 3. Lazy resolution: read attributes at the object address.
            match lazy_resolve_dim_addr(
                file,
                obj_addr,
                dim_by_id,
                addr_map,
                group_path,
                phony_counter,
            ) {
                Ok(axis) => {
                    // Validate length if the dim has a known fixed size.
                    if let Some(dim) = dim_by_id.get(&axis.dim_id) {
                        if var_len != 0 && dim.len != 0 && dim.len != var_len && !dim.is_unlimited {
                            return Err(NcError::AxisLengthMismatch {
                                var: ds_name.to_string(),
                                axis: axis_idx,
                                var_len,
                                dim_len: dim.len,
                            });
                        }
                    }
                    axis
                }
                Err(_) => {
                    // Unresolvable → phony.
                    // Record the address so we don't retry repeatedly.
                    let phony = make_phony_axis(var_len, group_path, dim_by_id, phony_counter);
                    addr_map.insert(obj_addr, phony.dim_id);
                    phony
                }
            }
        };

        axes.push(axis);
    }

    Ok(axes)
}

/// Attempt to resolve a dim-scale dataset entirely by its object-header address.
///
/// Uses [`File::attrs_of`] (attribute-only, no data load) and
/// [`File::dataset_at`] (for shape and unlimited detection).
fn lazy_resolve_dim_addr(
    file: &H5File,
    addr: u64,
    dim_by_id: &mut HashMap<u32, NcDimension>,
    addr_map: &mut HashMap<u64, u32>,
    group_path: &str,
    phony_counter: &mut u32,
) -> Result<NcAxis, NcError> {
    let attrs = file.attrs_of(addr)?;

    let dim_id_opt = attrs
        .iter()
        .find(|a| a.name == "_Netcdf4Dimid")
        .and_then(|a| a.as_i64())
        .and_then(|i| u32::try_from(i).ok());

    // Try to get the dimension name from the NAME attribute.
    let name_raw = attrs
        .iter()
        .find(|a| a.name == "NAME")
        .and_then(|a| a.as_str_fixed())
        .unwrap_or_default();
    let name = parse_dim_name_from_hdf5(&name_raw);

    // Load dataset for shape + unlimited detection.
    let ds = file.dataset_at(addr)?;
    let size = ds.shape.first().copied().unwrap_or(0) as u64;
    let is_unlimited = ds.is_unlimited();

    let dim_id = if let Some(id) = dim_id_opt {
        id
    } else {
        let id = *phony_counter;
        *phony_counter += 1;
        id
    };

    dim_by_id.entry(dim_id).or_insert_with(|| NcDimension {
        name: name.clone(),
        len: size,
        id: dim_id,
        is_unlimited,
    });
    addr_map.insert(addr, dim_id);

    Ok(NcAxis {
        dim_id,
        name,
        len: size,
        is_unlimited,
        group_path: group_path.to_string(),
    })
}

/// Extract a meaningful dimension name from the HDF5 `NAME` attribute.
///
/// The NetCDF library encodes pure (no-variable) dimensions as:
/// `"This is a netCDF dimension but not a netCDF variable.<len>"`.
/// For such strings we return `"phony_pure_dim_<len>"`.
/// For all other strings (coordinate variable names) we return the string as-is.
/// An empty string returns an empty string; callers should substitute a fallback.
fn parse_dim_name_from_hdf5(name: &str) -> String {
    if let Some(len) = parse_pure_dim_sentinel(name) {
        return format!("phony_pure_dim_{len}");
    }
    name.to_string()
}

// ---------------------------------------------------------------------------
// Phony-dimension helpers
// ---------------------------------------------------------------------------

fn make_phony_axis(
    var_len: u64,
    group_path: &str,
    dim_by_id: &mut HashMap<u32, NcDimension>,
    phony_counter: &mut u32,
) -> NcAxis {
    let id = *phony_counter;
    *phony_counter += 1;
    let name = phony_dim_name(id);
    dim_by_id.insert(
        id,
        NcDimension {
            name: name.clone(),
            len: var_len,
            id,
            is_unlimited: false,
        },
    );
    NcAxis {
        dim_id: id,
        name,
        len: var_len,
        is_unlimited: false,
        group_path: group_path.to_string(),
    }
}

fn create_phony_axes(
    shape: &[u64],
    phony_counter: &mut u32,
    dim_by_id: &mut HashMap<u32, NcDimension>,
    group_path: &str,
) -> Vec<NcAxis> {
    shape
        .iter()
        .map(|&dim_len| make_phony_axis(dim_len, group_path, dim_by_id, phony_counter))
        .collect()
}

// ---------------------------------------------------------------------------
// Path helper
// ---------------------------------------------------------------------------

/// Build the full HDF5 path for a child entry under `parent`.
pub fn build_path(parent: &str, child: &str) -> String {
    if parent == "/" {
        format!("/{child}")
    } else {
        format!("{parent}/{child}")
    }
}

// ---------------------------------------------------------------------------
// Unit tests (B3/B4)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oxih5::FileWriter;

    fn make_simple_h5(name: &str, data: &[f64]) -> Vec<u8> {
        let tmp = std::env::temp_dir().join(format!("oxinc_res_{name}.h5"));
        FileWriter::new()
            .write_dataset_f64(name, data, &[data.len()])
            .unwrap()
            .build(&tmp)
            .unwrap();
        std::fs::read(&tmp).unwrap()
    }

    #[test]
    fn test_collect_global_dims_empty_for_plain_h5() {
        let bytes = make_simple_h5("temp_gd", &[1.0, 2.0, 3.0]);
        let file = H5File::open_from_bytes(&bytes).unwrap();
        let gdims = collect_global_dims(&file, "/");
        // Plain HDF5 without DIMENSION_SCALE attrs → empty map.
        assert!(gdims.is_empty(), "expected no global dims, got {:?}", gdims);
    }

    #[test]
    fn test_resolve_group_deep_root_simple() {
        let bytes = make_simple_h5("temperature_rg", &[10.0, 20.0, 30.0]);
        let file = H5File::open_from_bytes(&bytes).unwrap();
        let gdims = collect_global_dims(&file, "/");
        let mut visited = HashSet::new();
        let root = resolve_group_deep(&file, "/", "/", &gdims, &mut visited, 0).unwrap();
        assert_eq!(root.name, "/");
        assert!(root.variable("temperature_rg").is_some());
        assert!(root.children.is_empty());
    }

    #[test]
    fn test_cycle_detection_via_visited_set() {
        // Pre-insert the target path into `visited` to simulate a back-edge.
        let bytes = make_simple_h5("temp_cycle", &[1.0]);
        let file = H5File::open_from_bytes(&bytes).unwrap();
        let gdims = HashMap::new();
        let mut visited = HashSet::new();
        visited.insert("/".to_string()); // simulate cycle

        let result = resolve_group_deep(&file, "/", "/", &gdims, &mut visited, 0);
        assert!(
            matches!(result, Err(NcError::CycleDetected)),
            "expected CycleDetected, got {:?}",
            result
        );
    }

    #[test]
    fn test_max_depth_exceeded() {
        let bytes = make_simple_h5("temp_depth", &[1.0]);
        let file = H5File::open_from_bytes(&bytes).unwrap();
        let gdims = HashMap::new();
        let mut visited = HashSet::new();

        // Pass depth = MAX_GROUP_DEPTH + 1 to trigger the guard.
        let result = resolve_group_deep(&file, "/", "/", &gdims, &mut visited, MAX_GROUP_DEPTH + 1);
        assert!(
            matches!(result, Err(NcError::MaxDepthExceeded)),
            "expected MaxDepthExceeded"
        );
    }

    #[test]
    fn test_build_path_root() {
        assert_eq!(build_path("/", "temp"), "/temp");
    }

    #[test]
    fn test_build_path_nested() {
        assert_eq!(build_path("/group1", "sub"), "/group1/sub");
    }

    #[test]
    fn test_parse_dim_name_from_hdf5_pure_dim() {
        let s = "This is a netCDF dimension but not a netCDF variable.42";
        assert_eq!(parse_dim_name_from_hdf5(s), "phony_pure_dim_42");
    }

    #[test]
    fn test_parse_dim_name_from_hdf5_coord_var() {
        assert_eq!(parse_dim_name_from_hdf5("time"), "time");
    }

    #[test]
    fn test_parse_dim_name_from_hdf5_empty() {
        assert_eq!(parse_dim_name_from_hdf5(""), "");
    }
}
