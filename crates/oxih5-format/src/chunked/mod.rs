//! Chunk assembly: scatter-to-contiguous buffer reconstruction.
//!
//! Given a list of chunk records (each with a file address, compressed size,
//! filter mask, and N-dimensional offset), this module reads and assembles
//! them into a single contiguous buffer that matches the full dataset shape.
//!
//! Split from a single ~2000-line `chunked.rs` into cohesive submodules:
//! - [`cache`] — the thread-safe chunk-index cache.
//! - [`index`] — chunk-index-type dispatch and resolution.
//! - [`read`] — whole-dataset chunk reading.
//! - [`mod@slice`] — hyperslab-range chunk reading.
//! - [`geometry`] — shared low-level I/O, filtering and coordinate helpers.
//!
//! Every item that used to be reachable as `chunked::X` before the split
//! (whether `pub` or crate-visible `pub(crate)`) is re-exported here at the
//! same flat path via `pub use <submodule>::*;`, so this split is purely an
//! internal reorganisation: no caller anywhere in the workspace needs to
//! change its `use` paths or fully-qualified references.

pub mod cache;
pub mod geometry;
pub mod index;
pub mod read;
pub mod slice;

pub use cache::*;
pub use index::*;
pub use read::*;
pub use slice::*;
// `geometry`'s items are all `pub(crate)` (crate-internal helpers shared by
// `read`/`slice`/`chunked_hyperslab.rs`) — re-export at the same ceiling
// rather than `pub use`, which would try (and fail, with a lint warning) to
// widen their visibility to fully public.
pub(crate) use geometry::*;

#[cfg(test)]
mod tests;
