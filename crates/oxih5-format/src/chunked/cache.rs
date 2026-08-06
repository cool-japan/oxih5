//! Thread-safe cache from chunk-index address to resolved chunk records.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::btree_v2::ChunkRecord;
use oxih5_core::OxiH5Error;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Thread-safe cache from chunk-index address → resolved chunk records.
///
/// Keyed by `(index_address, num_dims)` to handle multi-dimensional datasets
/// unambiguously.  The same index address with different ranks would (in
/// principle) produce different record sets, so the rank is part of the key.
///
/// The cache is `Clone` (cheap: it clones the `Arc`, so all clones share the
/// same underlying storage) to allow `Group` to hold a reference to the same
/// cache as the parent `File`.
#[derive(Debug, Default, Clone)]
pub struct ChunkIndexCache {
    pub(super) inner: CacheMap,
}
impl ChunkIndexCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self::default()
    }
    /// Return the cached records for `key`, or compute + cache them on first access.
    ///
    /// `compute` is only called when the key is absent.  On success the result
    /// is stored and an `Arc` clone is returned; on failure the error propagates
    /// and nothing is stored.
    pub fn get_or_insert(
        &self,
        key: (u64, usize),
        compute: impl FnOnce() -> Result<Vec<ChunkRecord>, OxiH5Error>,
    ) -> Result<Arc<Vec<ChunkRecord>>, OxiH5Error> {
        {
            let guard = self
                .inner
                .read()
                .map_err(|_| OxiH5Error::Format("chunk cache read-lock poisoned".into()))?;
            if let Some(v) = guard.get(&key) {
                return Ok(Arc::clone(v));
            }
        }
        let records = compute()?;
        let arc = Arc::new(records);
        let mut guard = self
            .inner
            .write()
            .map_err(|_| OxiH5Error::Format("chunk cache write-lock poisoned".into()))?;
        let stored = guard.entry(key).or_insert_with(|| Arc::clone(&arc));
        Ok(Arc::clone(stored))
    }
}
/// The inner storage type for [`ChunkIndexCache`].
pub(super) type CacheMap = Arc<RwLock<HashMap<(u64, usize), Arc<Vec<ChunkRecord>>>>>;
