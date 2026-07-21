//! # OxiH5 Format — low-level HDF5 binary-format parsers
//!
//! `oxih5-format` is the binary-parsing layer of OxiH5: it turns raw HDF5
//! file bytes — exactly as produced by h5py / libhdf5 — into the typed data
//! model defined by `oxih5-core`. It decodes the superblock (versions 0, 1,
//! 2, and 3, including the v2/v3 extension), object headers (v1 and v2) and
//! their standard header messages, local/global/fractal heaps, B-tree v1
//! and v2 nodes, the extensible- and fixed-array chunk indices, the filter
//! pipeline, full chunked-dataset assembly, and Virtual Dataset (VDS)
//! mapping-block parsing.
//!
//! This crate is 100% Pure Rust with `#![forbid(unsafe_code)]`; DEFLATE/zlib
//! decompression is delegated to the COOLJAPAN `oxiarc-deflate` crate
//! (never `flate2`/`miniz_oxide`). It exposes a flat, function-oriented API
//! — there is no `File` handle here; that abstraction lives in the `oxih5`
//! facade. Most users should depend on `oxih5` instead and reach for
//! `oxih5-format` only when building custom HDF5 tooling.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use oxih5_format::{header, message, superblock};
//!
//! let bytes: Vec<u8> = std::fs::read("data.h5")?;
//!
//! // 1. Parse the superblock to find the root object header.
//! let sb = superblock::parse(&bytes)?;
//!
//! // 2. Decode the object header's message list.
//! let messages = header::parse_messages(&bytes, sb.root_object_header_address)?;
//!
//! // 3. Inspect a message — e.g. find the symbol-table message (type 0x0011).
//! for msg in &messages {
//!     if msg.msg_type == 0x0011 {
//!         let st = message::parse_symbol_table(&msg.data)?;
//!         println!("b-tree @ {}, heap @ {}", st.btree_address, st.heap_address);
//!     }
//! }
//! # Ok::<(), oxih5_core::OxiH5Error>(())
//! ```

#![forbid(unsafe_code)]

pub mod btree;
pub mod btree_v1_chunk;
pub mod btree_v2;
pub mod chunked;
pub mod chunked_hyperslab;
pub mod context;
pub mod datatype;
pub mod ea_index;
pub mod fa_index;
pub mod filters;
pub mod fractal_heap;
pub mod global_heap;
pub mod global_heap_writer;
pub mod group;
pub mod header;
pub mod heap;
pub mod hyperslab;
pub mod link_msg;
pub mod message;
pub mod snod;
pub mod superblock;
pub mod values;
pub mod vds;

pub use chunked::ChunkIndexCache;
pub use chunked_hyperslab::{gather_hyperslab_contiguous, read_chunked_hyperslab};
pub use global_heap_writer::{GlobalHeapRef, GlobalHeapWriter};
pub use hyperslab::{DimSelection, Hyperslab};
