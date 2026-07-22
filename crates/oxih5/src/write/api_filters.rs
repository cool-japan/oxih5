//! `FileWriter` filter/compression and chunking entry points.
//!
//! HDF5 can only filter chunked data, so every `set_*` filter method converts a
//! contiguous dataset to chunked storage as it composes the filter pipeline.
//! The three filters — [`FileWriter::set_shuffle`], [`FileWriter::set_deflate`]
//! and [`FileWriter::set_fletcher32`] — are independent and may be called in any
//! order; the canonical on-disk order (shuffle, deflate, fletcher32) is settled
//! by [`super::pipeline`].  [`FileWriter::set_chunking`] tiles a dataset with a
//! **fixed** maxshape, the tiled counterpart of
//! [`FileWriter::create_dataset_unlimited`].  See [`super`] for the overview.

use oxih5_core::OxiH5Error;

use super::chunked::chunk_shape_of;
use super::tree::{dataset_mut, DatasetDesc, Storage};
use super::{FileWriter, MAX_DEFLATE_LEVEL};

/// Reject a dataset that cannot carry chunked storage or a filter pipeline.
///
/// A variable-length string dataset stores global-heap references the reader
/// will not decode through a filter, and a scalar dataset has no chunked
/// dataspace for a tile or a filter to attach to — so both are refused here,
/// before any storage is converted, so a rejected call leaves the dataset
/// exactly as it was.
///
/// # Errors
///
/// Returns `OxiH5Error::Format` naming `what` and `path`.
fn ensure_filterable(ds: &DatasetDesc, what: &str, path: &str) -> Result<(), OxiH5Error> {
    if ds.vlen_strings.is_some() {
        return Err(OxiH5Error::Format(format!(
            "{what}('{path}'): a variable-length string dataset cannot be tiled or filtered — \
             its elements are global-heap references, which the reader refuses to decode \
             through a filter pipeline"
        )));
    }
    if ds.shape.is_empty() {
        // A scalar (0-dimensional) dataset has no chunked dataspace, so a filter
        // — which HDF5 records only per chunk — has nowhere to live.  Fail fast
        // here rather than convert it to an illegal chunked scalar whose only
        // symptom is a distant "chunked storage needs at least one dimension"
        // error at `build`.
        return Err(OxiH5Error::Format(format!(
            "{what}('{path}'): a scalar (0-dimensional) dataset cannot be tiled or filtered — \
             HDF5 chunked storage has no scalar form"
        )));
    }
    Ok(())
}

/// Convert a contiguous dataset to single-chunk storage so a filter can attach,
/// leaving an already-chunked dataset's geometry and unlimited flag untouched.
fn ensure_chunked(ds: &mut DatasetDesc) {
    if ds.chunked().is_none() {
        ds.storage = Storage::Chunked {
            // Empty means "one chunk, the whole dataset": the geometry is
            // completed from the shape by `chunked::chunk_shape_of`, which is
            // also what keeps a zero-length dimension from becoming a zero chunk
            // extent.
            chunk_shape: Vec::new(),
            unlimited_dim0: false,
        };
    }
}

impl FileWriter {
    // -----------------------------------------------------------------------
    // W1e / W2: the filter pipeline
    // -----------------------------------------------------------------------

    /// Compress a dataset's data with the DEFLATE filter.
    ///
    /// `level` is the zlib compression level: `0` stores, `6` is what libhdf5
    /// and h5py use by default, `9` compresses hardest.  Reading the file back
    /// needs no cooperation from the caller — the level is recorded in the
    /// dataset's filter pipeline message, and h5py reports it as
    /// `dset.compression == 'gzip'` with `dset.compression_opts == level`.
    ///
    /// HDF5 can only filter **chunked** data, because a filter changes the byte
    /// count and only a chunk index has anywhere to record the new one.  This
    /// therefore converts a contiguous dataset to chunked storage as it sets
    /// the filter, which is also what `h5py`'s `compression=` argument does.
    /// The dataset becomes one chunk covering its whole extent unless it was
    /// already tiled — by [`create_dataset_unlimited`](Self::create_dataset_unlimited)
    /// or [`set_chunking`](Self::set_chunking) — in which case each existing
    /// chunk is compressed on its own, which is what lets a reader decompress
    /// one without touching the rest.  Composes with
    /// [`set_shuffle`](Self::set_shuffle) and
    /// [`set_fletcher32`](Self::set_fletcher32) in any call order.
    ///
    /// Compression is not free below roughly 8 KB of data: a chunked dataset
    /// carries a fixed-width chunk index of `chunked::chunk_node_size` bytes
    /// — 2096 for a 1-D dataset — which a small dataset will not save back.
    ///
    /// ```no_run
    /// use oxih5::FileWriter;
    /// let path = std::env::temp_dir().join("compressed.h5");
    /// let mut w = FileWriter::new();
    /// w.write_dataset_f64("readings", &[0.0; 4096], &[4096]).unwrap();
    /// w.set_deflate("readings", 6).unwrap();
    /// w.build(&path).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::NotFound` if `path` names no dataset, and
    /// `OxiH5Error::Format` if `path` is malformed or names a group, if `level`
    /// is above 9, if the dataset is scalar (0-dimensional), or if the dataset
    /// holds variable-length strings — those last two have no filterable form.
    pub fn set_deflate(&mut self, path: &str, level: u8) -> Result<(), OxiH5Error> {
        if level > MAX_DEFLATE_LEVEL {
            return Err(OxiH5Error::Format(format!(
                "set_deflate('{path}'): compression level {level} out of range \
                 (expected 0..={MAX_DEFLATE_LEVEL})"
            )));
        }
        let ds = dataset_mut(&mut self.root, path)?;
        ensure_filterable(ds, "set_deflate", path)?;
        ensure_chunked(ds);
        let mut filter = ds.filter.unwrap_or_default();
        filter.deflate = Some(level);
        ds.filter = Some(filter);
        Ok(())
    }

    /// Apply the SHUFFLE filter (HDF5 filter id 2) to a dataset.
    ///
    /// Shuffle transposes a chunk's bytes so the *i*-th byte of every element is
    /// grouped together, clustering the high-order bytes that tend to repeat
    /// across neighbouring samples — which lets a following DEFLATE find longer
    /// runs.  On its own it changes nothing a reader sees; paired with
    /// [`set_deflate`](Self::set_deflate) it reproduces h5py's
    /// `shuffle=True, compression='gzip'`, the standard NetCDF-4 integer/float
    /// compression combo, and is encoded as two filter descriptions (shuffle
    /// then deflate) in the pipeline message regardless of the order the two
    /// methods are called.
    ///
    /// Like every filter this needs chunked storage, so it converts a contiguous
    /// dataset to a single chunk and leaves an already-tiled dataset's geometry
    /// untouched.
    ///
    /// ```no_run
    /// use oxih5::FileWriter;
    /// let path = std::env::temp_dir().join("shuffled.h5");
    /// let mut w = FileWriter::new();
    /// w.write_dataset_i32("v", &(0..4096).collect::<Vec<_>>(), &[4096]).unwrap();
    /// w.set_shuffle("v").unwrap();
    /// w.set_deflate("v", 6).unwrap();
    /// w.build(&path).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::NotFound` if `path` names no dataset, and
    /// `OxiH5Error::Format` if `path` is malformed or names a group, if the
    /// dataset is scalar (0-dimensional), or if it holds variable-length strings.
    pub fn set_shuffle(&mut self, path: &str) -> Result<(), OxiH5Error> {
        let ds = dataset_mut(&mut self.root, path)?;
        ensure_filterable(ds, "set_shuffle", path)?;
        ensure_chunked(ds);
        let mut filter = ds.filter.unwrap_or_default();
        filter.shuffle = true;
        ds.filter = Some(filter);
        Ok(())
    }

    /// Attach the FLETCHER32 filter (HDF5 filter id 3) to a dataset.
    ///
    /// Fletcher-32 appends a 4-byte checksum to every chunk, so a reader can
    /// detect a corrupted chunk before it uses the data; h5py verifies the
    /// checksum natively and errors on a mismatch.  It is placed last in the
    /// pipeline, after any shuffle and deflate, so the checksum covers the exact
    /// bytes stored on disk.  Composes with the other two filters in any order.
    ///
    /// Like every filter this needs chunked storage, so it converts a contiguous
    /// dataset to a single chunk and leaves an already-tiled dataset's geometry
    /// untouched.
    ///
    /// ```no_run
    /// use oxih5::FileWriter;
    /// let path = std::env::temp_dir().join("checksummed.h5");
    /// let mut w = FileWriter::new();
    /// w.write_dataset_f64("v", &[1.0; 1024], &[1024]).unwrap();
    /// w.set_fletcher32("v").unwrap();
    /// w.build(&path).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::NotFound` if `path` names no dataset, and
    /// `OxiH5Error::Format` if `path` is malformed or names a group, if the
    /// dataset is scalar (0-dimensional), or if it holds variable-length strings.
    pub fn set_fletcher32(&mut self, path: &str) -> Result<(), OxiH5Error> {
        let ds = dataset_mut(&mut self.root, path)?;
        ensure_filterable(ds, "set_fletcher32", path)?;
        ensure_chunked(ds);
        let mut filter = ds.filter.unwrap_or_default();
        filter.fletcher32 = true;
        ds.filter = Some(filter);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // W2 / G010: fixed-maxshape tiling
    // -----------------------------------------------------------------------

    /// Tile a dataset into chunks of `chunk_shape` with a **fixed** maxshape.
    ///
    /// This is the tiled counterpart of
    /// [`create_dataset_unlimited`](Self::create_dataset_unlimited): it cuts the
    /// dataset into `ceil(shape[d] / chunk_shape[d])` chunks along each
    /// dimension, exactly like that method, but its dimensions stay **bounded**
    /// — the dataspace records `maxshape == shape` with no unlimited dimension,
    /// so h5py reports `dset.maxshape == dset.shape` rather than an unintended
    /// `(None, …)`.  It is what lets a tiled *and* compressed dataset (call
    /// [`set_deflate`](Self::set_deflate) afterwards) keep a fixed shape, which a
    /// geo raster or any bounded array wants.
    ///
    /// A **shorter** `chunk_shape` is completed from the shape, so `&[]` means
    /// "one chunk, the whole dataset" and `&[2]` over a shape of `[6, 4]` means
    /// `[2, 4]` — the same completion `create_dataset_unlimited` uses.  Because
    /// the shape is fixed, a chunk extent may not exceed its dimension (libhdf5
    /// rejects a chunk larger than a bounded dimension); an unlimited dataset
    /// exempts only its growth dimension from that rule.
    ///
    /// ```no_run
    /// use oxih5::FileWriter;
    /// let path = std::env::temp_dir().join("tiled_fixed.h5");
    /// let mut w = FileWriter::new();
    /// w.write_dataset_i32("raster", &(0..64).collect::<Vec<_>>(), &[8, 8]).unwrap();
    /// w.set_chunking("raster", &[4, 4]).unwrap();  // 4 tiles, fixed 8×8 maxshape
    /// w.set_deflate("raster", 6).unwrap();
    /// w.build(&path).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `OxiH5Error::NotFound` if `path` names no dataset, and
    /// `OxiH5Error::Format` if `path` is malformed or names a group, if the
    /// dataset is scalar or holds variable-length strings, if `chunk_shape` has
    /// more dimensions than the dataspace, holds a 0, or carries a chunk extent
    /// larger than its (fixed) dimension.
    pub fn set_chunking(&mut self, path: &str, chunk_shape: &[usize]) -> Result<(), OxiH5Error> {
        let ds = dataset_mut(&mut self.root, path)?;
        ensure_filterable(ds, "set_chunking", path)?;
        // Commit the new tiling, then validate it against the canonical geometry
        // checker; on any complaint restore the previous storage so a refused
        // call leaves no trace, exactly as the dataset creators promise.
        let previous = std::mem::replace(
            &mut ds.storage,
            Storage::Chunked {
                chunk_shape: chunk_shape.to_vec(),
                unlimited_dim0: false,
            },
        );
        if let Err(err) = chunk_shape_of(ds) {
            ds.storage = previous;
            return Err(err);
        }
        Ok(())
    }
}
