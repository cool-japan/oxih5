//! `FileWriter` attribute-writing entry points.
//!
//! Every `write_*_attr` method funnels through `attach_attr`, so the rule
//! for which objects can carry attributes — and the duplicate-name overwrite
//! semantics — lives in exactly one place.  See [`super`] for the overview.

use oxih5_core::{ByteOrder, OxiH5Error};

use super::api_datasets::NumericValues;
use super::elem::{AttrDesc, AttrKind, ElemType};
use super::tree::attrs_mut;
use super::FileWriter;

impl FileWriter {
    // -----------------------------------------------------------------------
    // Attribute writing
    //
    // Every one of these resolves `path` to a dataset **or** a group, at any
    // depth, through one resolver.  `"/"` is the root group; a bare name finds
    // a root dataset first, which is what keeps pre-path callers working.
    // -----------------------------------------------------------------------

    /// Write a scalar fixed-length string attribute on a dataset or a group.
    ///
    /// Writing an attribute whose name already exists on the same object
    /// **overwrites** the previous value, exactly as h5py's
    /// `obj.attrs[name] = value` does — HDF5 requires attribute names to be
    /// unique within an object, so the writer never emits two of the same name.
    /// Every `write_*_attr` method shares this rule.
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::Format` if `path` is malformed, or
    /// `OxiH5Error::NotFound` if it names no dataset or group.
    pub fn write_string_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        value: &str,
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(path, attr_name, AttrKind::FixedStr(value.to_string()))
    }

    /// Write a scalar fixed-length string attribute declared `H5T_STR_NULLTERM`.
    ///
    /// Identical to [`Self::write_string_attr`] except the datatype's padding is
    /// null-terminated (with a spare terminator byte, on-disk width `len + 1`)
    /// rather than null-padded.  Use it for the HDF5 dimension-scale `CLASS`
    /// attribute: libnetcdf's `H5DSis_scale` recognises a `DIMENSION_SCALE`
    /// only when `CLASS` is a NULLTERM string, so a NULLPAD one makes netCDF-4
    /// fall back to `phony_dim_*` and lose the file's dimensions.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_string_attr_nullterm(
        &mut self,
        path: &str,
        attr_name: &str,
        value: &str,
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(
            path,
            attr_name,
            AttrKind::FixedStrNullTerm(value.to_string()),
        )
    }

    /// Write a scalar float64 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_f64_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        value: f64,
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(path, attr_name, AttrKind::F64(value))
    }

    /// Write a scalar signed int64 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_i64_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        value: i64,
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(path, attr_name, AttrKind::I64(value))
    }

    /// Write a scalar signed int32 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_i32_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        value: i32,
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(path, attr_name, AttrKind::I32(value))
    }

    /// Write a 1-D float64 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_f64_array_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        values: &[f64],
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(path, attr_name, AttrKind::F64Array(values.to_vec()))
    }

    /// Write a 1-D signed int64 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_i64_array_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        values: &[i64],
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(path, attr_name, AttrKind::I64Array(values.to_vec()))
    }

    /// Write a 1-D fixed-length string attribute on a dataset or a group.
    ///
    /// HDF5 gives an attribute one datatype, so every element is stored at the
    /// width of the longest, NUL-padded — which is how the reader recovers the
    /// individual lengths.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_string_array_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        values: &[&str],
    ) -> Result<(), OxiH5Error> {
        let owned: Vec<String> = values.iter().map(|s| (*s).to_string()).collect();
        self.attach_attr(path, attr_name, AttrKind::StrArray(owned))
    }

    /// Write an object-reference list attribute on a dataset or a group.
    ///
    /// Each target is named by its path from the root, with or without a
    /// leading `/`; a root-level object may be named bare.  Targets are
    /// resolved when the file is built, not here, because their addresses do
    /// not exist until the layout pass has run — so a target that names nothing
    /// is reported by [`Self::build`], naming both the attribute and the target.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_obj_ref_list_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        target_names: &[&str],
    ) -> Result<(), OxiH5Error> {
        let targets: Vec<String> = target_names.iter().map(|s| (*s).to_string()).collect();
        self.attach_attr(path, attr_name, AttrKind::ObjRefsByName(targets))
    }

    // -----------------------------------------------------------------------
    // Widened numeric scalars (G006)
    //
    // Every one of these is a thin wrapper over the generic `AttrKind::Num`
    // seam: the value is serialised little-endian here and the datatype is
    // derived from the `ElemType` in `elem.rs`, so a new numeric attribute type
    // is one method rather than a new datatype body.  `f64`/`i64`/`i32` keep
    // their own dedicated variants for byte-for-byte stability with the golden
    // oracle; the rest funnel through here.
    // -----------------------------------------------------------------------

    /// Write a scalar float32 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_f32_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        value: f32,
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(
            path,
            attr_name,
            AttrKind::Num {
                elem: ElemType::F32,
                bytes: value.to_le_bytes().to_vec(),
            },
        )
    }

    /// Write a scalar signed int8 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_i8_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        value: i8,
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(
            path,
            attr_name,
            AttrKind::Num {
                elem: ElemType::I8,
                bytes: value.to_le_bytes().to_vec(),
            },
        )
    }

    /// Write a scalar signed int16 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_i16_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        value: i16,
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(
            path,
            attr_name,
            AttrKind::Num {
                elem: ElemType::I16,
                bytes: value.to_le_bytes().to_vec(),
            },
        )
    }

    /// Write a scalar unsigned int8 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_u8_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        value: u8,
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(
            path,
            attr_name,
            AttrKind::Num {
                elem: ElemType::U8,
                bytes: value.to_le_bytes().to_vec(),
            },
        )
    }

    /// Write a scalar unsigned int16 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_u16_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        value: u16,
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(
            path,
            attr_name,
            AttrKind::Num {
                elem: ElemType::U16,
                bytes: value.to_le_bytes().to_vec(),
            },
        )
    }

    /// Write a scalar unsigned int32 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_u32_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        value: u32,
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(
            path,
            attr_name,
            AttrKind::Num {
                elem: ElemType::U32,
                bytes: value.to_le_bytes().to_vec(),
            },
        )
    }

    /// Write a scalar unsigned int64 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_u64_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        value: u64,
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(
            path,
            attr_name,
            AttrKind::Num {
                elem: ElemType::U64,
                bytes: value.to_le_bytes().to_vec(),
            },
        )
    }

    // -----------------------------------------------------------------------
    // Widened numeric arrays (G006)
    // -----------------------------------------------------------------------

    /// Write a 1-D float32 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_f32_array_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        values: &[f32],
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(
            path,
            attr_name,
            AttrKind::NumArray {
                elem: ElemType::F32,
                bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
        )
    }

    /// Write a 1-D signed int8 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_i8_array_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        values: &[i8],
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(
            path,
            attr_name,
            AttrKind::NumArray {
                elem: ElemType::I8,
                bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
        )
    }

    /// Write a 1-D signed int16 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_i16_array_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        values: &[i16],
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(
            path,
            attr_name,
            AttrKind::NumArray {
                elem: ElemType::I16,
                bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
        )
    }

    /// Write a 1-D signed int32 attribute on a dataset or a group.
    ///
    /// netCDF-4 stores `_Netcdf4Coordinates` and `_Netcdf4Dimid` as native
    /// (32-bit) integers and reads them back with `H5T_NATIVE_INT`, so a
    /// dimension-id array must be a true int32 attribute — a wider integer is
    /// misread element-for-element (an `int64` `[0, 1]` decodes as `[0, 0]`,
    /// collapsing every axis onto the first dimension).
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_i32_array_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        values: &[i32],
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(
            path,
            attr_name,
            AttrKind::NumArray {
                elem: ElemType::I32,
                bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
        )
    }

    /// Write a 1-D unsigned int8 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_u8_array_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        values: &[u8],
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(
            path,
            attr_name,
            AttrKind::NumArray {
                elem: ElemType::U8,
                bytes: values.to_vec(),
            },
        )
    }

    /// Write a 1-D unsigned int16 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_u16_array_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        values: &[u16],
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(
            path,
            attr_name,
            AttrKind::NumArray {
                elem: ElemType::U16,
                bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
        )
    }

    /// Write a 1-D unsigned int32 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_u32_array_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        values: &[u32],
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(
            path,
            attr_name,
            AttrKind::NumArray {
                elem: ElemType::U32,
                bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
        )
    }

    /// Write a 1-D unsigned int64 attribute on a dataset or a group.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_u64_array_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        values: &[u64],
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(
            path,
            attr_name,
            AttrKind::NumArray {
                elem: ElemType::U64,
                bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
        )
    }

    // -----------------------------------------------------------------------
    // Reference-bearing attributes (netCDF-4 dimension-scale contract)
    // -----------------------------------------------------------------------

    /// Write a vlen-of-object-reference attribute on a dataset or a group.
    ///
    /// This is the datatype netCDF-4 uses for `DIMENSION_LIST`: a 1-D attribute
    /// of `H5T_VLEN{ H5T_REFERENCE(object) }` whose element `i` is an
    /// independent sequence of object references, given here as `sequences[i]` —
    /// a list of target object names (paths from the root, with or without a
    /// leading `/`; a root-level object may be named bare).  Each sequence's
    /// references live in the file's global heap.
    ///
    /// Targets are resolved when the file is built, exactly like
    /// [`Self::write_obj_ref_list_attr`]; a name that matches no dataset or
    /// group is reported by [`Self::build`].
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_vlen_obj_ref_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        sequences: &[Vec<String>],
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(
            path,
            attr_name,
            AttrKind::VlenObjRefsByName(sequences.to_vec()),
        )
    }

    /// Write a `{ dataset: objref, dimension: u32 }` compound-list attribute on a
    /// dataset or a group.
    ///
    /// This is the datatype netCDF-4 uses for `REFERENCE_LIST` on a
    /// dimension-scale dataset: a 1-D attribute of a compound whose two members
    /// are `dataset` (an 8-byte object reference to a variable that attaches the
    /// scale) and `dimension` (the axis index at which it attaches).  Each entry
    /// is one `(target object name, index)` pair; the reference is resolved at
    /// build time like [`Self::write_obj_ref_list_attr`].
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`].
    pub fn write_ref_index_list_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        entries: &[(String, u32)],
    ) -> Result<(), OxiH5Error> {
        self.attach_attr(path, attr_name, AttrKind::RefIndexList(entries.to_vec()))
    }

    /// Write a string attribute on the root group.
    ///
    /// Used for NetCDF-4 global metadata such as `_nc3_strict`.  Equivalent to
    /// `write_string_attr("/", name, value)`, which is also how it is
    /// implemented; the root group is not special any more.
    pub fn write_root_str_attr(&mut self, name: &str, value: &str) {
        // `"/"` is the one path that always resolves — the root group exists
        // for the lifetime of the writer — so this cannot fail, and the
        // infallible signature is preserved for existing callers.
        let _ = self.write_string_attr("/", name, value);
    }

    /// Attach one attribute to whatever `path` names.
    ///
    /// Every `write_*_attr` method funnels through here, so "which objects can
    /// carry attributes" is one question with one answer rather than one per
    /// value type — including the duplicate-name rule.
    ///
    /// # Duplicate names
    ///
    /// HDF5 requires attribute names to be unique within an object. Writing the
    /// same `attr_name` twice therefore **overwrites** the first value, matching
    /// h5py's `attrs[name] = value`: the pending [`AttrDesc`] in the plan tree is
    /// updated in place. A plain push would have emitted two attribute messages
    /// of the same name — a spec-invalid file whose second value no reader can
    /// reach — so the replacement happens here, where the attribute is still just
    /// a planned value, not yet serialized.
    fn attach_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        kind: AttrKind,
    ) -> Result<(), OxiH5Error> {
        let attrs = attrs_mut(&mut self.root, path)?;
        if let Some(existing) = attrs.iter_mut().find(|a| a.name == attr_name) {
            existing.kind = kind;
        } else {
            attrs.push(AttrDesc {
                name: attr_name.to_string(),
                kind,
            });
        }
        Ok(())
    }
}

impl FileWriter {
    // -----------------------------------------------------------------------
    // G008 / G012: attributes in an explicit byte order, and half precision
    // -----------------------------------------------------------------------

    /// Write a 1-D numeric attribute in an explicit byte order.
    ///
    /// The counterpart of [`FileWriter::write_dataset_numeric`] for attributes:
    /// it covers the two shapes the per-type `write_*_array_attr` methods do not
    /// — **big-endian** values, and **half-precision** floats — and is
    /// equivalent to them for `ByteOrder::Little` otherwise.  The declared
    /// datatype and the payload bytes come from the same `order`.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`]; additionally `OxiH5Error::Format` if
    /// `values` is empty, since HDF5 has no zero-length attribute dataspace
    /// this writer can emit.
    pub fn write_numeric_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        values: NumericValues<'_>,
        order: ByteOrder,
    ) -> Result<(), OxiH5Error> {
        if values.is_empty() {
            return Err(OxiH5Error::Format(format!(
                "attribute '{attr_name}': a numeric array attribute needs at least one value"
            )));
        }
        self.attach_attr(
            path,
            attr_name,
            AttrKind::NumArray {
                elem: values.num_type().as_elem(order),
                bytes: values.to_bytes(order),
            },
        )
    }

    /// Write a scalar numeric attribute in an explicit byte order.
    ///
    /// Same as [`Self::write_numeric_attr`] but emits HDF5's *scalar* dataspace
    /// rather than a one-element vector, so h5py reads the value back with
    /// shape `()` instead of `(1,)`.
    ///
    /// # Errors
    ///
    /// As [`Self::write_string_attr`]; additionally `OxiH5Error::Format` unless
    /// `values` holds exactly one value.
    pub fn write_numeric_scalar_attr(
        &mut self,
        path: &str,
        attr_name: &str,
        values: NumericValues<'_>,
        order: ByteOrder,
    ) -> Result<(), OxiH5Error> {
        if values.len() != 1 {
            return Err(OxiH5Error::Format(format!(
                "attribute '{attr_name}': a scalar numeric attribute needs exactly one value, got {}",
                values.len()
            )));
        }
        self.attach_attr(
            path,
            attr_name,
            AttrKind::Num {
                elem: values.num_type().as_elem(order),
                bytes: values.to_bytes(order),
            },
        )
    }
}
