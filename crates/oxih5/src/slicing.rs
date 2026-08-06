//! Lazy sub-region readers: range slices and strided HDF5 hyperslabs.
//!
//! Extracted from `lib.rs` to keep individual source files under the 2000-line
//! limit.  For chunked layouts these paths decode only the chunks overlapping
//! the requested region; for every other layout they fall back to a full read
//! followed by an in-memory gather.

use oxih5_core::{Dataset, OxiH5Error};
use oxih5_format::{header, message, ChunkIndexCache, Hyperslab};

use crate::links::GroupRef;
use crate::reader::{
    is_vlen_dtype, navigate_to_group, on_disk_elem_footprint, read_attributes_from_header,
    read_dataset_from_object_header, reject_vlen_incompatible_filters, resolve_header_address_in,
};
use crate::FileData;

/// Resolve `path` to the object header address of the dataset it names.
///
/// Shared by the slice and hyperslab entry points: both need the header address
/// of a dataset given a full path, with every group along the way resolved in
/// whichever storage style it happens to use, and soft links followed.
fn resolve_path_header(
    file_data: &FileData,
    path: &str,
    source_dir: &std::path::Path,
) -> Result<(u64, String), OxiH5Error> {
    let normalized = path.trim_start_matches('/');
    let mut parts: Vec<&str> = normalized.split('/').filter(|s| !s.is_empty()).collect();
    let dataset_name = parts
        .pop()
        .ok_or_else(|| OxiH5Error::NotFound(path.to_string()))?;

    let parent = navigate_to_group(file_data, GroupRef::root(file_data)?, &parts)?;
    let header_addr = resolve_header_address_in(file_data, parent, dataset_name, source_dir)?;
    Ok((header_addr, dataset_name.to_string()))
}

/// Lazy slice reader for `File::dataset_slice`: resolves the path, extracts
/// messages, and for chunked layouts calls `read_chunked_slice` directly.
pub(crate) fn read_dataset_slice_lazy(
    file_data: &FileData,
    path: &str,
    ranges: &[std::ops::Range<usize>],
    source_dir: &std::path::Path,
    cache: &ChunkIndexCache,
) -> Result<Dataset, OxiH5Error> {
    let (header_addr, dataset_name) = resolve_path_header(file_data, path, source_dir)?;

    slice_dataset_at_header(
        file_data,
        header_addr,
        &dataset_name,
        ranges,
        source_dir,
        Some(cache),
    )
}

/// Lazy slice reader for `Group::dataset_slice`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_dataset_slice_lazy_from_group(
    file_data: &FileData,
    object_header_address: u64,
    btree_address: u64,
    heap_address: u64,
    new_style: bool,
    name: &str,
    ranges: &[std::ops::Range<usize>],
    source_dir: &std::path::Path,
    cache: &ChunkIndexCache,
) -> Result<Dataset, OxiH5Error> {
    let group = GroupRef::new(
        object_header_address,
        if new_style {
            None
        } else {
            Some((btree_address, heap_address))
        },
    );
    let header_addr = resolve_header_address_in(file_data, group, name, source_dir)?;
    slice_dataset_at_header(
        file_data,
        header_addr,
        name,
        ranges,
        source_dir,
        Some(cache),
    )
}

/// Extract messages at `header_addr` and perform a lazy slice.
///
/// For chunked layouts only the overlapping chunks are decompressed.
/// For other layouts the full data is loaded and then sliced in memory.
#[allow(clippy::too_many_arguments)]
pub(crate) fn slice_dataset_at_header(
    file_data: &[u8],
    header_addr: u64,
    name: &str,
    ranges: &[std::ops::Range<usize>],
    source_dir: &std::path::Path,
    cache: Option<&ChunkIndexCache>,
) -> Result<Dataset, OxiH5Error> {
    let ds_messages = header::parse_messages(file_data, header_addr)?;

    let mut dataspace = None;
    let mut datatype = None;
    let mut layout = None;
    let mut filter_pipeline = None;
    let mut fill_value: Option<Vec<u8>> = None;

    for msg in &ds_messages {
        match msg.msg_type {
            0x0001 => dataspace = Some(message::parse_dataspace(&msg.data)?),
            0x0003 => datatype = Some(message::parse_datatype(&msg.data)?),
            0x0005 => {
                if let Ok(fv) = message::parse_fill_value(&msg.data) {
                    fill_value = fv;
                }
            }
            0x0008 => layout = Some(message::parse_layout(&msg.data)?),
            0x000B => filter_pipeline = Some(message::parse_filter_pipeline(&msg.data)?),
            _ => {}
        }
    }

    let dsp = dataspace
        .ok_or_else(|| OxiH5Error::Format(format!("no dataspace message in dataset '{name}'")))?;
    let dtp = datatype
        .ok_or_else(|| OxiH5Error::Format(format!("no datatype message in dataset '{name}'")))?;
    let lay = layout
        .ok_or_else(|| OxiH5Error::Format(format!("no layout message in dataset '{name}'")))?;

    use oxih5_format::message::LayoutInfo;

    // Attempt lazy chunked slice first.
    if let LayoutInfo::Chunked { .. } = &lay {
        let ndims = dsp.dims.len();
        if ndims >= 1 && ranges.len() == ndims {
            let all_in_bounds = ranges
                .iter()
                .zip(dsp.dims.iter())
                .all(|(r, &dim)| r.end <= dim as usize);

            if all_in_bounds {
                let elem_size = on_disk_elem_footprint(&dtp.dtype, 8).ok_or_else(|| {
                    OxiH5Error::NotImplemented(format!(
                        "chunked dataset '{name}': element size not supported"
                    ))
                })?;
                let dataset_dims: Vec<u64> = dsp.dims.clone();
                let pipeline = filter_pipeline
                    .clone()
                    .unwrap_or_else(|| oxih5_core::FilterPipeline { filters: vec![] });
                if is_vlen_dtype(&dtp.dtype) {
                    reject_vlen_incompatible_filters(&pipeline, name)?;
                }
                let ranges_u64: Vec<std::ops::Range<u64>> = ranges
                    .iter()
                    .map(|r| r.start as u64..r.end as u64)
                    .collect();
                let out_shape: Vec<usize> = ranges.iter().map(|r| r.len()).collect();

                let raw = oxih5_format::chunked::read_chunked_slice(
                    file_data,
                    &lay,
                    &pipeline,
                    oxih5_format::chunked::DatasetShape {
                        dims: &dataset_dims,
                        max_dims: dsp.max_dims.as_deref(),
                    },
                    oxih5_format::chunked::ChunkSliceParams {
                        elem_size,
                        fill_value: fill_value.as_deref(),
                    },
                    &ranges_u64,
                    cache,
                )?;

                let attributes =
                    read_attributes_from_header(file_data, header_addr).unwrap_or_default();
                return Ok(Dataset {
                    data: raw,
                    shape: out_shape,
                    dtype: dtp.dtype,
                    attributes,
                    max_dims: dsp.max_dims.clone(),
                });
            }
        }
    }

    // Fallback: full read then in-memory slice.
    let full_ds = read_dataset_from_object_header(file_data, header_addr, name, source_dir, cache)?;
    full_ds.slice(ranges)
}

/// Hyperslab reader for `File::dataset_hyperslab`.
pub(crate) fn read_dataset_hyperslab_internal(
    file_data: &FileData,
    path: &str,
    selection: &Hyperslab,
    source_dir: &std::path::Path,
    cache: &ChunkIndexCache,
) -> Result<Dataset, OxiH5Error> {
    let (header_addr, dataset_name) = resolve_path_header(file_data, path, source_dir)?;

    hyperslab_dataset_at_header(
        file_data,
        header_addr,
        &dataset_name,
        selection,
        source_dir,
        Some(cache),
    )
}

/// Hyperslab reader for `Group::dataset_hyperslab`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_dataset_hyperslab_from_group_internal(
    file_data: &FileData,
    object_header_address: u64,
    btree_address: u64,
    heap_address: u64,
    new_style: bool,
    name: &str,
    selection: &Hyperslab,
    source_dir: &std::path::Path,
    cache: &ChunkIndexCache,
) -> Result<Dataset, OxiH5Error> {
    let group = GroupRef::new(
        object_header_address,
        if new_style {
            None
        } else {
            Some((btree_address, heap_address))
        },
    );
    let header_addr = resolve_header_address_in(file_data, group, name, source_dir)?;
    hyperslab_dataset_at_header(
        file_data,
        header_addr,
        name,
        selection,
        source_dir,
        Some(cache),
    )
}

/// Parse messages at `header_addr` and perform a hyperslab read.
///
/// For chunked layouts with an in-bounds bounding box, only the overlapping
/// chunks are decompressed.  For other layouts (or when out-of-bounds) the full
/// data is loaded and then sampled with `gather_hyperslab_contiguous`.
pub(crate) fn hyperslab_dataset_at_header(
    file_data: &[u8],
    header_addr: u64,
    name: &str,
    selection: &Hyperslab,
    source_dir: &std::path::Path,
    cache: Option<&ChunkIndexCache>,
) -> Result<Dataset, OxiH5Error> {
    let ds_messages = header::parse_messages(file_data, header_addr)?;

    let mut dataspace = None;
    let mut datatype = None;
    let mut layout = None;
    let mut filter_pipeline = None;
    let mut fill_value: Option<Vec<u8>> = None;

    for msg in &ds_messages {
        match msg.msg_type {
            0x0001 => dataspace = Some(message::parse_dataspace(&msg.data)?),
            0x0003 => datatype = Some(message::parse_datatype(&msg.data)?),
            0x0005 => {
                if let Ok(fv) = message::parse_fill_value(&msg.data) {
                    fill_value = fv;
                }
            }
            0x0008 => layout = Some(message::parse_layout(&msg.data)?),
            0x000B => filter_pipeline = Some(message::parse_filter_pipeline(&msg.data)?),
            _ => {}
        }
    }

    let dsp = dataspace
        .ok_or_else(|| OxiH5Error::Format(format!("no dataspace message in dataset '{name}'")))?;
    let dtp = datatype
        .ok_or_else(|| OxiH5Error::Format(format!("no datatype message in dataset '{name}'")))?;
    let lay = layout
        .ok_or_else(|| OxiH5Error::Format(format!("no layout message in dataset '{name}'")))?;

    use oxih5_format::message::LayoutInfo;

    // Attempt the lazy chunked-hyperslab path first.
    if let LayoutInfo::Chunked { .. } = &lay {
        let ndims = dsp.dims.len();
        if ndims >= 1 && selection.dims.len() == ndims {
            let bbox = selection.bounding_ranges();
            let all_in_bounds = bbox
                .iter()
                .zip(dsp.dims.iter())
                .all(|(r, &dim)| r.end <= dim);

            if all_in_bounds {
                let elem_size = on_disk_elem_footprint(&dtp.dtype, 8).ok_or_else(|| {
                    OxiH5Error::NotImplemented(format!(
                        "chunked dataset '{name}': element size not supported"
                    ))
                })?;
                let dataset_dims: Vec<u64> = dsp.dims.clone();
                let pipeline = filter_pipeline
                    .clone()
                    .unwrap_or_else(|| oxih5_core::FilterPipeline { filters: vec![] });
                if is_vlen_dtype(&dtp.dtype) {
                    reject_vlen_incompatible_filters(&pipeline, name)?;
                }
                let out_shape: Vec<usize> = selection
                    .output_shape()
                    .iter()
                    .map(|&s| s as usize)
                    .collect();

                let raw = oxih5_format::chunked_hyperslab::read_chunked_hyperslab(
                    file_data,
                    &lay,
                    &pipeline,
                    oxih5_format::chunked::DatasetShape {
                        dims: &dataset_dims,
                        max_dims: dsp.max_dims.as_deref(),
                    },
                    oxih5_format::chunked::ChunkSliceParams {
                        elem_size,
                        fill_value: fill_value.as_deref(),
                    },
                    selection,
                    cache,
                )?;

                let attributes =
                    read_attributes_from_header(file_data, header_addr).unwrap_or_default();
                return Ok(Dataset {
                    data: raw,
                    shape: out_shape,
                    dtype: dtp.dtype,
                    attributes,
                    max_dims: dsp.max_dims.clone(),
                });
            }
        }
    }

    // Fallback: full read then gather via contiguous sampler.
    let full_ds = read_dataset_from_object_header(file_data, header_addr, name, source_dir, cache)?;
    let dataset_dims_u64: Vec<u64> = full_ds.shape.iter().map(|&s| s as u64).collect();
    let elem_size = on_disk_elem_footprint(&full_ds.dtype, 8).ok_or_else(|| {
        OxiH5Error::NotImplemented(format!(
            "dataset '{name}': element size not supported for hyperslab fallback"
        ))
    })?;

    let raw = oxih5_format::chunked_hyperslab::gather_hyperslab_contiguous(
        &full_ds.data,
        &dataset_dims_u64,
        selection,
        elem_size,
    )?;
    let out_shape: Vec<usize> = selection
        .output_shape()
        .iter()
        .map(|&s| s as usize)
        .collect();
    Ok(Dataset {
        data: raw,
        shape: out_shape,
        dtype: full_ds.dtype,
        attributes: full_ds.attributes,
        max_dims: full_ds.max_dims,
    })
}
