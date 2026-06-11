use crate::cf;
use crate::error::NcError;
use crate::types::NcType;
use oxih5::{AttrView, Attribute, ByteOrder, Dtype};

/// A NetCDF dimension (resolved from an HDF5 dimension-scale dataset).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NcDimension {
    pub name: String,
    pub len: u64,
    pub id: u32,
    pub is_unlimited: bool,
}

/// One axis of a variable: the dimension it maps to.
///
/// `group_path` records which group **owns** this dimension scale.  For
/// same-group dims it equals the variable's group.  For cross-group dims
/// (B4) it is the path of the group where the dimension scale was defined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NcAxis {
    pub dim_id: u32,
    pub name: String,
    pub len: u64,
    pub is_unlimited: bool,
    /// Full HDF5 path of the group that owns the dimension scale for this axis.
    pub group_path: String,
}

/// A NetCDF attribute (thin wrapper over an oxih5 Attribute).
///
/// Vlen-string attributes are eagerly decoded into `decoded_text` at
/// construction time (when file bytes are available via [`AttrView`]).
/// [`NcAttribute::as_text`] returns the pre-decoded string for both fixed-
/// length and vlen-string attributes.
#[derive(Debug, Clone)]
pub struct NcAttribute {
    pub name: String,
    inner: Attribute,
    /// Pre-decoded text for vlen-string attributes (populated when an
    /// [`AttrView`] is available at construction time).
    decoded_text: Option<String>,
}

// ---------------------------------------------------------------------------
// Float decode helpers
// ---------------------------------------------------------------------------

fn decode_f64_le(bytes: &[u8], attr: &str) -> Result<f64, NcError> {
    let arr: [u8; 8] = bytes
        .try_into()
        .map_err(|_| NcError::BadConventionAttribute {
            owner: String::new(),
            attr: attr.to_owned(),
            reason: "f64 slice wrong length".into(),
        })?;
    Ok(f64::from_le_bytes(arr))
}

fn decode_f64_be(bytes: &[u8], attr: &str) -> Result<f64, NcError> {
    let arr: [u8; 8] = bytes
        .try_into()
        .map_err(|_| NcError::BadConventionAttribute {
            owner: String::new(),
            attr: attr.to_owned(),
            reason: "f64 slice wrong length".into(),
        })?;
    Ok(f64::from_be_bytes(arr))
}

fn decode_f32_le(bytes: &[u8], attr: &str) -> Result<f32, NcError> {
    let arr: [u8; 4] = bytes
        .try_into()
        .map_err(|_| NcError::BadConventionAttribute {
            owner: String::new(),
            attr: attr.to_owned(),
            reason: "f32 slice wrong length".into(),
        })?;
    Ok(f32::from_le_bytes(arr))
}

fn decode_f32_be(bytes: &[u8], attr: &str) -> Result<f32, NcError> {
    let arr: [u8; 4] = bytes
        .try_into()
        .map_err(|_| NcError::BadConventionAttribute {
            owner: String::new(),
            attr: attr.to_owned(),
            reason: "f32 slice wrong length".into(),
        })?;
    Ok(f32::from_be_bytes(arr))
}

/// Decode a fixed-size integer chunk into an i64.
fn decode_int_chunk(
    bytes: &[u8],
    size: usize,
    order: &ByteOrder,
    attr: &str,
) -> Result<i64, NcError> {
    let bad = || NcError::BadConventionAttribute {
        owner: String::new(),
        attr: attr.to_owned(),
        reason: "int slice wrong length".into(),
    };
    match (size, order) {
        (1, _) => Ok(i64::from(bytes[0] as i8)),
        (2, ByteOrder::Little) => {
            let arr: [u8; 2] = bytes.try_into().map_err(|_| bad())?;
            Ok(i64::from(i16::from_le_bytes(arr)))
        }
        (2, ByteOrder::Big) => {
            let arr: [u8; 2] = bytes.try_into().map_err(|_| bad())?;
            Ok(i64::from(i16::from_be_bytes(arr)))
        }
        (4, ByteOrder::Little) => {
            let arr: [u8; 4] = bytes.try_into().map_err(|_| bad())?;
            Ok(i64::from(i32::from_le_bytes(arr)))
        }
        (4, ByteOrder::Big) => {
            let arr: [u8; 4] = bytes.try_into().map_err(|_| bad())?;
            Ok(i64::from(i32::from_be_bytes(arr)))
        }
        (8, ByteOrder::Little) => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| bad())?;
            Ok(i64::from_le_bytes(arr))
        }
        (8, ByteOrder::Big) => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| bad())?;
            Ok(i64::from_be_bytes(arr))
        }
        _ => Err(NcError::BadConventionAttribute {
            owner: String::new(),
            attr: attr.to_owned(),
            reason: format!("unsupported integer size {size}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// NcAttribute
// ---------------------------------------------------------------------------

impl NcAttribute {
    /// Create from a raw `Attribute` (no file bytes available — vlen strings
    /// remain undecoded; `as_text()` will return `Unsupported` for them).
    ///
    /// When file bytes are available use the internal `new_with_view` constructor
    /// which eagerly decodes vlen-string attributes.
    pub fn new(inner: Attribute) -> Self {
        Self {
            name: inner.name.clone(),
            inner,
            decoded_text: None,
        }
    }

    /// Create from an [`AttrView`], eagerly decoding vlen-string attributes
    /// while the file bytes are available (B6).
    pub(crate) fn new_with_view(view: &AttrView<'_>) -> Self {
        let decoded_text = match &view.attr.dtype {
            // Vlen string: decode all elements and join with a single space so
            // that `as_text()` returns a usable string.
            Dtype::String {
                fixed_len: None, ..
            } => view.as_strings().ok().map(|v| v.join(" ")),
            _ => None,
        };
        Self {
            name: view.attr.name.clone(),
            inner: view.attr.clone(),
            decoded_text,
        }
    }

    pub fn dtype(&self) -> &Dtype {
        &self.inner.dtype
    }

    /// Decode this attribute as a text string.
    ///
    /// Works for both fixed-length and vlen-string attributes.  For vlen
    /// strings the value must have been eagerly decoded at construction time
    /// via the internal `new_with_view` constructor; otherwise `Unsupported` is returned.
    pub fn as_text(&self) -> Result<String, NcError> {
        // Eagerly decoded vlen string (B6).
        if let Some(text) = &self.decoded_text {
            return Ok(text.clone());
        }
        match &self.inner.dtype {
            Dtype::String {
                fixed_len: Some(n), ..
            } => {
                let n = *n;
                let bytes =
                    self.inner
                        .data
                        .get(..n)
                        .ok_or_else(|| NcError::BadConventionAttribute {
                            owner: String::new(),
                            attr: self.name.clone(),
                            reason: "attribute data shorter than fixed string length".into(),
                        })?;
                let trimmed = bytes.split(|&b| b == 0).next().unwrap_or(bytes);
                String::from_utf8(trimmed.to_vec()).map_err(|e| NcError::BadConventionAttribute {
                    owner: String::new(),
                    attr: self.name.clone(),
                    reason: format!("UTF-8 decode: {e}"),
                })
            }
            Dtype::String {
                fixed_len: None, ..
            } => Err(NcError::Unsupported(
                "vlen-string attribute: open with NcFile::open to get eager decode".into(),
            )),
            _ => Err(NcError::BadConventionAttribute {
                owner: String::new(),
                attr: self.name.clone(),
                reason: "attribute dtype is not a string".into(),
            }),
        }
    }

    /// Decode this attribute as a vector of f64 values.
    pub fn as_f64(&self) -> Result<Vec<f64>, NcError> {
        match &self.inner.dtype {
            Dtype::Float { size: 8, order } => {
                let data = &self.inner.data;
                if data.len() % 8 != 0 {
                    return Err(NcError::BadConventionAttribute {
                        owner: String::new(),
                        attr: self.name.clone(),
                        reason: "f64 data length not multiple of 8".into(),
                    });
                }
                let name = self.name.as_str();
                match order {
                    ByteOrder::Little => data
                        .chunks_exact(8)
                        .map(|c| decode_f64_le(c, name))
                        .collect(),
                    ByteOrder::Big => data
                        .chunks_exact(8)
                        .map(|c| decode_f64_be(c, name))
                        .collect(),
                }
            }
            Dtype::Float { size: 4, order } => {
                let data = &self.inner.data;
                if data.len() % 4 != 0 {
                    return Err(NcError::BadConventionAttribute {
                        owner: String::new(),
                        attr: self.name.clone(),
                        reason: "f32 data length not multiple of 4".into(),
                    });
                }
                let name = self.name.as_str();
                match order {
                    ByteOrder::Little => data
                        .chunks_exact(4)
                        .map(|c| decode_f32_le(c, name).map(f64::from))
                        .collect(),
                    ByteOrder::Big => data
                        .chunks_exact(4)
                        .map(|c| decode_f32_be(c, name).map(f64::from))
                        .collect(),
                }
            }
            _ => Err(NcError::BadConventionAttribute {
                owner: String::new(),
                attr: self.name.clone(),
                reason: "attribute dtype is not a float".into(),
            }),
        }
    }

    /// Decode this attribute as a vector of i64 values.
    pub fn as_i64(&self) -> Result<Vec<i64>, NcError> {
        match &self.inner.dtype {
            Dtype::Int { size, order, .. } => {
                let sz = *size;
                let data = &self.inner.data;
                if sz == 0 || data.len() % sz != 0 {
                    return Err(NcError::BadConventionAttribute {
                        owner: String::new(),
                        attr: self.name.clone(),
                        reason: "int data length not multiple of element size".into(),
                    });
                }
                let name = self.name.as_str();
                data.chunks_exact(sz)
                    .map(|c| decode_int_chunk(c, sz, order, name))
                    .collect()
            }
            _ => Err(NcError::BadConventionAttribute {
                owner: String::new(),
                attr: self.name.clone(),
                reason: "attribute dtype is not an integer".into(),
            }),
        }
    }

    /// Return the underlying oxih5 Attribute for escape-hatch access.
    pub fn raw(&self) -> &Attribute {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// Fill-mask helpers (B7)
// ---------------------------------------------------------------------------

/// Map a flat data slice to `Vec<Option<T>>`, replacing every element equal to
/// `fill` with `None`.
///
/// Uses `T::PartialEq` — suitable for integer types.  For **floating-point**
/// types use the bit-exact variants [`apply_fill_mask_f32`] and
/// [`apply_fill_mask_f64`], which correctly handle NaN fill values
/// (`NaN != NaN` under IEEE 754 equality).
pub fn apply_fill_mask<T: PartialEq + Copy>(data: &[T], fill: T) -> Vec<Option<T>> {
    data.iter()
        .map(|&v| if v == fill { None } else { Some(v) })
        .collect()
}

/// Bit-exact fill-mask for `f32` data.
///
/// `f32::NAN` as a fill value is handled correctly: if `fill` is NaN and a
/// data element is NaN with the same bit pattern, it is mapped to `None`.
pub fn apply_fill_mask_f32(data: &[f32], fill: f32) -> Vec<Option<f32>> {
    let fill_bits = fill.to_bits();
    data.iter()
        .map(|&v| {
            if v.to_bits() == fill_bits {
                None
            } else {
                Some(v)
            }
        })
        .collect()
}

/// Bit-exact fill-mask for `f64` data.
///
/// `f64::NAN` as a fill value is handled correctly: if `fill` is NaN and a
/// data element is NaN with the same bit pattern, it is mapped to `None`.
pub fn apply_fill_mask_f64(data: &[f64], fill: f64) -> Vec<Option<f64>> {
    let fill_bits = fill.to_bits();
    data.iter()
        .map(|&v| {
            if v.to_bits() == fill_bits {
                None
            } else {
                Some(v)
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// NcVariable
// ---------------------------------------------------------------------------

/// A NetCDF variable (an HDF5 dataset with DIMENSION_LIST, or a coord var).
#[derive(Debug, Clone)]
pub struct NcVariable {
    pub name: String,
    pub dtype: Dtype,
    pub dims: Vec<NcAxis>,
    pub shape: Vec<u64>,
    pub attrs: Vec<NcAttribute>,
    pub is_coordinate: bool,
    /// HDF5 object path of the underlying dataset (e.g. `"/group/varname"`).
    pub h5_path: String,
}

impl NcVariable {
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn dim_names(&self) -> Vec<&str> {
        self.dims.iter().map(|a| a.name.as_str()).collect()
    }

    pub fn attr(&self, name: &str) -> Option<&NcAttribute> {
        self.attrs.iter().find(|a| a.name == name)
    }

    pub fn fill_value(&self) -> Option<&NcAttribute> {
        self.attr("_FillValue")
    }

    pub fn units(&self) -> Option<String> {
        self.attr("units").and_then(|a| a.as_text().ok())
    }

    /// Return the NetCDF logical type of this variable (B5).
    pub fn nc_type(&self) -> NcType {
        NcType::from(&self.dtype)
    }
}

// ---------------------------------------------------------------------------
// NcGroup
// ---------------------------------------------------------------------------

/// A NetCDF group (root or subgroup).
///
/// `children` is filled by the B3 recursive resolver: the root group's children
/// are its immediate sub-groups, and so on recursively.
#[derive(Debug, Clone)]
pub struct NcGroup {
    pub name: String,
    pub path: String,
    pub dimensions: Vec<NcDimension>,
    pub variables: Vec<NcVariable>,
    pub attrs: Vec<NcAttribute>,
    pub subgroup_names: Vec<String>,
    /// Recursively resolved sub-groups (B3).  Empty for leaf groups.
    pub children: Vec<NcGroup>,
}

impl NcGroup {
    pub fn dimension(&self, name: &str) -> Option<&NcDimension> {
        self.dimensions.iter().find(|d| d.name == name)
    }

    pub fn dimension_by_id(&self, id: u32) -> Option<&NcDimension> {
        self.dimensions.iter().find(|d| d.id == id)
    }

    pub fn variable(&self, name: &str) -> Option<&NcVariable> {
        self.variables.iter().find(|v| v.name == name)
    }

    pub fn attr(&self, name: &str) -> Option<&NcAttribute> {
        self.attrs.iter().find(|a| a.name == name)
    }

    /// Find a child group by name (one level only).
    pub fn child(&self, name: &str) -> Option<&NcGroup> {
        self.children.iter().find(|g| g.name == name)
    }

    // -----------------------------------------------------------------------
    // CF-convention attribute accessors (B8)
    // -----------------------------------------------------------------------

    /// Return the coordinate variable names listed in the `coordinates`
    /// attribute of `var_name` (CF §7.1).
    ///
    /// Each token may use the CF-1.7 `"group:varname"` form; tokens are
    /// returned verbatim.  Returns `None` if the variable doesn't exist or
    /// has no `coordinates` attribute.
    pub fn coordinates_of(&self, var_name: &str) -> Option<Vec<String>> {
        let text = self
            .variable(var_name)?
            .attr("coordinates")?
            .as_text()
            .ok()?;
        let names = cf::parse_cf_name_list(&text);
        if names.is_empty() {
            None
        } else {
            Some(names)
        }
    }

    /// Return the bounds variable name from the `bounds` attribute of `var_name`
    /// (CF §7.1).
    ///
    /// Returns `None` if the variable doesn't exist or has no `bounds` attribute.
    pub fn bounds_of(&self, var_name: &str) -> Option<String> {
        self.variable(var_name)?.attr("bounds")?.as_text().ok()
    }

    /// Return the grid-mapping variable name from the `grid_mapping` attribute
    /// of `var_name` (CF §5.6).
    ///
    /// Returns `None` if the variable doesn't exist or has no `grid_mapping`
    /// attribute.
    pub fn grid_mapping_of(&self, var_name: &str) -> Option<String> {
        self.variable(var_name)?
            .attr("grid_mapping")?
            .as_text()
            .ok()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use oxih5::ByteOrder;
    use oxih5_core::{Attribute, Charset, Dataspace, Dtype};

    fn make_text_attr(name: &str, value: &str) -> NcAttribute {
        let n = value.len();
        let mut data = value.as_bytes().to_vec();
        data.push(0); // NUL terminator included in fixed-len
        NcAttribute::new(Attribute {
            name: name.to_string(),
            dtype: Dtype::String {
                fixed_len: Some(n + 1),
                charset: Charset::Utf8,
            },
            dataspace: Dataspace::Scalar,
            data,
        })
    }

    fn make_group_with_var(var_name: &str, attrs: Vec<NcAttribute>) -> NcGroup {
        let var = NcVariable {
            name: var_name.to_string(),
            dtype: Dtype::Float {
                size: 8,
                order: ByteOrder::Little,
            },
            dims: vec![],
            shape: vec![],
            attrs,
            is_coordinate: false,
            h5_path: format!("/{var_name}"),
        };
        NcGroup {
            name: "/".to_string(),
            path: "/".to_string(),
            dimensions: vec![],
            variables: vec![var],
            attrs: vec![],
            subgroup_names: vec![],
            children: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // apply_fill_mask — B7
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_fill_mask_i32() {
        let data = vec![1i32, -9999, 3, -9999, 5];
        let result = apply_fill_mask(&data, -9999i32);
        assert_eq!(result, vec![Some(1), None, Some(3), None, Some(5)]);
    }

    #[test]
    fn test_apply_fill_mask_no_fill() {
        let data = vec![1i32, 2, 3];
        let result = apply_fill_mask(&data, -9999i32);
        assert_eq!(result, vec![Some(1), Some(2), Some(3)]);
    }

    #[test]
    fn test_apply_fill_mask_f64_exact_bits() {
        let fill = -9999.0_f64;
        let nan_fill = f64::NAN;
        let data = vec![1.0_f64, -9999.0, 3.0, f64::NAN];

        // f64 with regular fill value: elements at index 1 should be None,
        // others Some.  The NaN at index 3 does NOT match the -9999.0 fill.
        let result = apply_fill_mask_f64(&data, fill);
        assert_eq!(result.len(), 4);
        // Index 0: Some(1.0)
        assert!(matches!(result[0], Some(v) if v == 1.0));
        // Index 1: fill value → None
        assert!(result[1].is_none(), "fill element should be None");
        // Index 2: Some(3.0)
        assert!(matches!(result[2], Some(v) if v == 3.0));
        // Index 3: NaN is not the fill, so it stays Some(NaN).
        // Use bit comparison since NaN != NaN under IEEE 754.
        assert!(
            matches!(result[3], Some(v) if v.is_nan()),
            "NaN (not fill) should be Some(NaN)"
        );

        // NaN fill: bit-exact comparison — the NaN in data has same bits as nan_fill.
        let result_nan = apply_fill_mask_f64(&data, nan_fill);
        // Index 3 has NaN, which matches the NaN fill by bits.
        assert!(
            result_nan[3].is_none(),
            "NaN element should be masked as fill"
        );
        assert!(result_nan[0].is_some());
        assert!(
            result_nan[1].is_some(),
            "-9999.0 is not NaN fill, should be Some"
        );
    }

    #[test]
    fn test_apply_fill_mask_f32_exact_bits() {
        let fill = f32::NAN;
        let data = vec![1.0_f32, f32::NAN, 3.0_f32];
        let result = apply_fill_mask_f32(&data, fill);
        assert!(result[0].is_some());
        assert!(result[1].is_none(), "NaN should be masked");
        assert!(result[2].is_some());
    }

    #[test]
    fn test_apply_fill_mask_empty() {
        let data: Vec<i64> = vec![];
        let result = apply_fill_mask(&data, 0i64);
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // CF methods — B8
    // -----------------------------------------------------------------------

    #[test]
    fn test_coordinates_of_whitespace_separated() {
        let coords_attr = make_text_attr("coordinates", "lat lon");
        let group = make_group_with_var("temp", vec![coords_attr]);
        assert_eq!(
            group.coordinates_of("temp"),
            Some(vec!["lat".to_string(), "lon".to_string()])
        );
    }

    #[test]
    fn test_coordinates_of_colon_form() {
        // CF-1.7: "group:var" tokens are returned verbatim.
        let coords_attr = make_text_attr("coordinates", "grp:lat grp:lon");
        let group = make_group_with_var("temp", vec![coords_attr]);
        assert_eq!(
            group.coordinates_of("temp"),
            Some(vec!["grp:lat".to_string(), "grp:lon".to_string()])
        );
    }

    #[test]
    fn test_coordinates_of_missing_var() {
        let group = make_group_with_var("temp", vec![]);
        assert!(group.coordinates_of("nonexistent").is_none());
    }

    #[test]
    fn test_coordinates_of_no_attr() {
        let group = make_group_with_var("temp", vec![]);
        assert!(group.coordinates_of("temp").is_none());
    }

    #[test]
    fn test_bounds_of() {
        let bounds_attr = make_text_attr("bounds", "lat_bounds");
        let group = make_group_with_var("lat", vec![bounds_attr]);
        assert_eq!(group.bounds_of("lat"), Some("lat_bounds".to_string()));
    }

    #[test]
    fn test_bounds_of_missing() {
        let group = make_group_with_var("lat", vec![]);
        assert!(group.bounds_of("lat").is_none());
    }

    #[test]
    fn test_grid_mapping_of() {
        let gm_attr = make_text_attr("grid_mapping", "crs");
        let group = make_group_with_var("precip", vec![gm_attr]);
        assert_eq!(group.grid_mapping_of("precip"), Some("crs".to_string()));
    }

    #[test]
    fn test_grid_mapping_of_missing() {
        let group = make_group_with_var("precip", vec![]);
        assert!(group.grid_mapping_of("precip").is_none());
    }

    // -----------------------------------------------------------------------
    // NcType — B5 via nc_type()
    // -----------------------------------------------------------------------

    #[test]
    fn test_nc_variable_nc_type_float64() {
        let var = NcVariable {
            name: "temp".to_string(),
            dtype: Dtype::Float {
                size: 8,
                order: ByteOrder::Little,
            },
            dims: vec![],
            shape: vec![],
            attrs: vec![],
            is_coordinate: false,
            h5_path: "/temp".to_string(),
        };
        assert_eq!(var.nc_type(), NcType::Float64);
    }

    #[test]
    fn test_nc_variable_nc_type_int32() {
        let var = NcVariable {
            name: "count".to_string(),
            dtype: Dtype::Int {
                size: 4,
                signed: true,
                order: ByteOrder::Little,
            },
            dims: vec![],
            shape: vec![],
            attrs: vec![],
            is_coordinate: false,
            h5_path: "/count".to_string(),
        };
        assert_eq!(var.nc_type(), NcType::Int32);
    }
}
