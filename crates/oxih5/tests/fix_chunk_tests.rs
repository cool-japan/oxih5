//! Write-path hardening: chunk index conformance and guard ordering.
//!
//! * **B016** — the terminal B-tree v1 chunk key now matches libhdf5 byte for
//!   byte: the trailing element pseudo-dimension carries `elem_size` (was 0),
//!   and the real dimensions reproduce libhdf5's chunk-insertion right key
//!   (was the rounded-up extent).  h5py still enumerates every chunk, oxih5
//!   still reads files written by 0.2.1, and the exact bytes now agree with
//!   what h5py (`libver='earliest'`) writes for the same geometry.
//! * **B024** — a chunk extent wider than a *fixed* (non-dimension-0) dimension
//!   is rejected, the way libhdf5 rejects it at `H5Dcreate`; dimension 0 (the
//!   growth dimension) may still be tiled wider than its current extent.
//! * **R003** — an over-cap chunk count is refused from the geometry alone,
//!   before a single tile is cut or compressed, so a pathological request errors
//!   in microseconds instead of after gigabytes of work.
//! * **R007** — a chunk shape with more dimensions than the dataspace is
//!   rejected rather than silently truncated to a lower-rank chunking.
//!
//! The h5py-facing tests skip gracefully when python3/h5py is unavailable,
//! following the same three-way guard as `write_tests.rs`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Instant;

use oxih5::{ByteOrder, Dtype, File, FileWriter, OxiH5Error};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dt_i32() -> Dtype {
    Dtype::Int {
        size: 4,
        signed: true,
        order: ByteOrder::Little,
    }
}

fn dt_f64() -> Dtype {
    Dtype::Float {
        size: 8,
        order: ByteOrder::Little,
    }
}

fn i32_raw(n: i32) -> Vec<u8> {
    (0..n).flat_map(i32::to_le_bytes).collect()
}

fn f64_raw(n: i32) -> Vec<u8> {
    (0..n).flat_map(|i| f64::from(i).to_le_bytes()).collect()
}

/// Build one chunked dataset in memory and return the whole file image.
fn build_chunked(shape: &[usize], chunk: &[usize], dtype: &Dtype, raw: &[u8]) -> Vec<u8> {
    let mut w = FileWriter::new();
    w.create_dataset_unlimited("d", shape, chunk, dtype, raw)
        .expect("create");
    w.build_to_vec().expect("build")
}

/// The terminal (rightmost) chunk-B-tree key of `bytes`, as its `ndims + 1`
/// on-disk offset values — the last is the element pseudo-dimension.
///
/// Scans for the type-1 `TREE` node directly rather than walking the layout
/// message, so the check does not lean on the encoder it is verifying.
fn terminal_key_offsets(bytes: &[u8], ndims: usize) -> Vec<u64> {
    let node = (0..bytes.len().saturating_sub(8))
        .find(|&i| &bytes[i..i + 4] == b"TREE" && bytes[i + 4] == 1)
        .expect("a type-1 chunk B-tree node");
    let nchild = u16::from_le_bytes([bytes[node + 6], bytes[node + 7]]) as usize;
    let key_size = 8 + (ndims + 1) * 8;
    let stride = key_size + 8;
    let term = node + 24 + nchild * stride;
    (0..=ndims)
        .map(|d| {
            let at = term + 8 + d * 8;
            u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8"))
        })
        .collect()
}

/// Run `script` through python3, deleting `path` first; skip on a missing
/// interpreter or module, fail on any other non-zero exit.
fn h5py_check(path: &std::path::Path, script: &str, what: &str) {
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .output();
    let _ = std::fs::remove_file(path);
    match output {
        Ok(out) if out.status.success() => {
            eprintln!("{what}: {}", String::from_utf8_lossy(&out.stdout).trim());
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let stdout = String::from_utf8_lossy(&out.stdout);
            if stderr.contains("ModuleNotFoundError")
                || stderr.contains("ImportError")
                || stderr.contains("No module named")
            {
                eprintln!("h5py not available — skipping {what}");
            } else {
                panic!("{what} FAILED:\nstdout: {stdout}\nstderr: {stderr}");
            }
        }
        Err(_) => eprintln!("python3 not found — skipping {what}"),
    }
}

// ---------------------------------------------------------------------------
// B016 — terminal key matches libhdf5 byte for byte
// ---------------------------------------------------------------------------

/// Every terminal key oxih5 writes equals the one h5py (`libver='earliest'`)
/// writes for the same geometry.  The expected values were dumped from real
/// h5py 3.16 / libhdf5 files: the trailing element offset is always `elem_size`
/// (never 0), and the real dimensions follow libhdf5's insertion right key —
/// the rounded-up extent only for 1-D and single-chunk grids, the corner
/// chunk's own offsets for a divisible multi-D grid.
#[test]
fn b016_terminal_key_matches_libhdf5_bytes() {
    // (shape, chunk, dtype, elem_count, ndims, expected terminal incl. elem dim)
    // 1-D [7] chunk (3): i32 -> [9, 4], f64 -> [9, 8].
    assert_eq!(
        terminal_key_offsets(&build_chunked(&[7], &[3], &dt_i32(), &i32_raw(7)), 1),
        vec![9, 4],
        "1-D i32: real extent 9, element dimension 4"
    );
    assert_eq!(
        terminal_key_offsets(&build_chunked(&[7], &[3], &dt_f64(), &f64_raw(7)), 1),
        vec![9, 8],
        "1-D f64: the element dimension tracks the element size"
    );
    // 2-D [5,3] single chunk (5,3) -> [5, 3, 4].
    assert_eq!(
        terminal_key_offsets(&build_chunked(&[5, 3], &[5, 3], &dt_i32(), &i32_raw(15)), 2),
        vec![5, 3, 4]
    );
    // 2-D [5,3] chunk (2,2), ragged both ways -> [6, 2, 4] (NOT the rounded
    // extent [6, 4]): dimension 1's right key never rises past one chunk.
    assert_eq!(
        terminal_key_offsets(&build_chunked(&[5, 3], &[2, 2], &dt_i32(), &i32_raw(15)), 2),
        vec![6, 2, 4]
    );
    // 2-D [4,4] chunk (2,2), divisible -> [2, 2, 4] (the corner chunk's own
    // real offsets; only the element dimension makes the bound strict).
    assert_eq!(
        terminal_key_offsets(&build_chunked(&[4, 4], &[2, 2], &dt_i32(), &i32_raw(16)), 2),
        vec![2, 2, 4]
    );
    // 3-D [4,3,2] chunk (2,2,2) -> [2, 2, 2, 4].
    assert_eq!(
        terminal_key_offsets(
            &build_chunked(&[4, 3, 2], &[2, 2, 2], &dt_i32(), &i32_raw(24)),
            3
        ),
        vec![2, 2, 2, 4]
    );
}

/// A data chunk's own key keeps the trailing element offset at 0 — only the
/// terminal boundary carries `elem_size`.  A regression here would corrupt the
/// chunk's decoded coordinate for a strict reader.
#[test]
fn b016_data_chunk_keys_keep_a_zero_element_offset() {
    let bytes = build_chunked(&[5, 3], &[2, 2], &dt_i32(), &i32_raw(15));
    let node = (0..bytes.len() - 8)
        .find(|&i| &bytes[i..i + 4] == b"TREE" && bytes[i + 4] == 1)
        .expect("chunk node");
    // key[0]: the first data chunk. Its element pseudo-dimension is at
    // key_off + 8 + ndims*8 = node + 24 + 8 + 2*8.
    let elem_at = node + 24 + 8 + 2 * 8;
    assert_eq!(
        u64::from_le_bytes(bytes[elem_at..elem_at + 8].try_into().unwrap()),
        0,
        "a data chunk key's trailing element offset stays 0"
    );
}

/// oxih5 reads a chunked file written by its own **0.2.1** writer — whose
/// terminal key is the old rounded extent with a zero element offset — byte for
/// byte.  The fixture was produced by the pre-fix writer; a reader that had come
/// to depend on the new terminal would fail here.
#[test]
fn b016_reads_old_writer_terminal_fixture() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/chunk_old_terminal_5x3_2x2.h5");
    let f = File::open(&path).expect("open pre-fix fixture");
    let ds = f.dataset("d").expect("dataset d");
    assert_eq!(ds.shape, vec![5usize, 3]);
    assert_eq!(
        ds.as_i32().expect("as_i32"),
        (0..15i32).collect::<Vec<_>>(),
        "the 0.2.1 rounded-extent terminal must still read every chunk"
    );
}

/// h5py must still enumerate every chunk of an oxih5-written tiled dataset — the
/// property the terminal key exists to serve.  `iter_chunks()` asks libhdf5 for
/// the origins it decoded out of the B-tree, so it pins the count and placement.
#[test]
fn b016_h5py_enumerates_every_chunk() {
    let path = std::env::temp_dir().join("oxih5_fix_chunk_b016_enumerate.h5");
    let mut w = FileWriter::new();
    // Ragged both ways: six chunks, four of them hanging over an edge.
    w.create_dataset_unlimited("tiled", &[5, 3], &[2, 2], &dt_i32(), &i32_raw(15))
        .expect("tiled");
    // And a 3-D divisible grid, whose terminal is the corner chunk's offsets.
    w.create_dataset_unlimited("cube", &[4, 3, 2], &[2, 2, 2], &dt_i32(), &i32_raw(24))
        .expect("cube");
    w.build(&path).expect("build");

    let path_lit = path.display();
    let script = format!(
        "import h5py, numpy as np\n\
         f = h5py.File({path_lit:?}, 'r')\n\
         t = f['tiled']\n\
         assert t.shape == (5, 3), repr(t.shape)\n\
         assert t.chunks == (2, 2), repr(t.chunks)\n\
         assert np.array_equal(t[...], np.arange(15).reshape(5, 3)), repr(t[...])\n\
         ch = sorted((s[0].start, s[1].start) for s in t.iter_chunks())\n\
         assert ch == [(0, 0), (0, 2), (2, 0), (2, 2), (4, 0), (4, 2)], repr(ch)\n\
         c = f['cube']\n\
         assert c.shape == (4, 3, 2), repr(c.shape)\n\
         assert np.array_equal(c[...], np.arange(24).reshape(4, 3, 2)), repr(c[...])\n\
         cc = sorted((s[0].start, s[1].start, s[2].start) for s in c.iter_chunks())\n\
         assert cc == [(0, 0, 0), (0, 2, 0), (2, 0, 0), (2, 2, 0)], repr(cc)\n\
         print('h5py enumerated', len(ch) + len(cc), 'chunks')\n"
    );
    h5py_check(&path, &script, "B016 h5py chunk enumeration");
}

/// The exact terminal bytes oxih5 writes equal the bytes h5py writes for the
/// same geometry — compared directly against a freshly h5py-authored file, so
/// the oracle is libhdf5 itself, not a remembered constant.
#[test]
fn b016_terminal_bytes_equal_h5py_output() {
    let ox = std::env::temp_dir().join("oxih5_fix_chunk_b016_ox.h5");
    let hx = std::env::temp_dir().join("oxih5_fix_chunk_b016_hx.h5");
    let mut w = FileWriter::new();
    w.create_dataset_unlimited("d", &[5, 3], &[2, 2], &dt_i32(), &i32_raw(15))
        .expect("d");
    w.build(&ox).expect("build");
    let ox_term = terminal_key_offsets(&std::fs::read(&ox).expect("read ox"), 2);
    let _ = std::fs::remove_file(&ox);

    let hx_lit = hx.display();
    let script = format!(
        "import h5py, numpy as np, struct\n\
         with h5py.File({hx_lit:?}, 'w', libver='earliest') as f:\n\
         \x20   f.create_dataset('d', data=np.arange(15, dtype='int32').reshape(5, 3), chunks=(2, 2))\n\
         b = open({hx_lit:?}, 'rb').read()\n\
         off = 0\n\
         node = -1\n\
         while True:\n\
         \x20   i = b.find(b'TREE', off)\n\
         \x20   if i < 0: break\n\
         \x20   if b[i+4] == 1: node = i; break\n\
         \x20   off = i + 4\n\
         assert node >= 0, 'no chunk tree'\n\
         nchild = struct.unpack_from('<H', b, node+6)[0]\n\
         key_size = 8 + 3*8\n\
         term = node + 24 + nchild*(key_size+8)\n\
         offs = list(struct.unpack_from('<3Q', b, term+8))\n\
         assert offs == [6, 2, 4], 'h5py terminal ' + repr(offs)\n\
         print('h5py terminal', offs)\n"
    );
    // The Rust side already knows oxih5's terminal; assert it equals libhdf5's
    // known value, and let the python side confirm h5py writes the same.
    assert_eq!(ox_term, vec![6, 2, 4], "oxih5 terminal must equal h5py's");
    h5py_check(&hx, &script, "B016 h5py terminal-byte agreement");
}

// ---------------------------------------------------------------------------
// B024 — a chunk wider than a fixed dimension is rejected
// ---------------------------------------------------------------------------

/// `create_dataset_unlimited([4,3], chunk [2,8])` emits `chunk[1] = 8` over a
/// fixed dimension of size 3 — a file libhdf5 refuses to create.  oxih5 now
/// rejects it at build with a typed error naming the dimension.
#[test]
fn b024_rejects_chunk_larger_than_fixed_dimension() {
    let mut w = FileWriter::new();
    w.create_dataset_unlimited("g", &[4, 3], &[2, 8], &dt_i32(), &i32_raw(12))
        .expect("create is accepted; the geometry is checked at build");
    let err = w
        .build_to_vec()
        .expect_err("a chunk wider than a fixed dimension must be rejected");
    let msg = format!("{err}");
    assert!(matches!(err, OxiH5Error::Format(_)), "{msg}");
    assert!(
        msg.contains("fixed dimension") && msg.contains("exceeds"),
        "the error must say why: {msg}"
    );
}

/// Dimension 0 is exempt: it is the growth dimension, so a chunk wider than its
/// current extent is a whole-dataset tile with fill, exactly as before — and it
/// still builds and reads back.
#[test]
fn b024_dimension_zero_may_exceed_its_extent() {
    let mut w = FileWriter::new();
    w.create_dataset_unlimited("g", &[4], &[8], &dt_i32(), &i32_raw(4))
        .expect("g");
    let path = std::env::temp_dir().join("oxih5_fix_chunk_b024_dim0.h5");
    w.build(&path).expect("dim0 oversize still builds");
    let f = File::open(&path).expect("open");
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        f.dataset("g").expect("g").as_i32().expect("as_i32"),
        vec![0, 1, 2, 3]
    );
}

/// A chunk *equal* to a fixed dimension is fine — only strictly greater is a
/// violation — so an ordinary tiling is never caught by the new guard.
#[test]
fn b024_chunk_equal_to_fixed_dimension_is_allowed() {
    let bytes = build_chunked(&[5, 3], &[2, 3], &dt_i32(), &i32_raw(15));
    assert!(!bytes.is_empty());
}

// ---------------------------------------------------------------------------
// R003 — the chunk-count cap fires before any tile is materialised
// ---------------------------------------------------------------------------

/// A request for more than `2^24` chunks is refused from the geometry alone,
/// before `payload::build` cuts and compresses a single tile.  The pre-fix path
/// materialised every chunk first, taking seconds and gigabytes for a dataset
/// whose data is a few megabytes; the guarded path errors effectively
/// instantly.
#[test]
fn r003_over_cap_chunk_count_errors_before_materialising() {
    // 2^24 + 1 chunks of one byte each: over the cap by exactly one.
    let n = (1usize << 24) + 1;
    let data = vec![0u8; n];
    let dt = Dtype::Int {
        size: 1,
        signed: false,
        order: ByteOrder::Little,
    };
    let mut w = FileWriter::new();
    w.create_dataset_unlimited("d", &[n], &[1], &dt, &data)
        .expect("create stores the descriptor");

    let start = Instant::now();
    let err = w
        .build_to_vec()
        .expect_err("an over-cap chunk count must be refused");
    let elapsed = start.elapsed();

    let msg = format!("{err}");
    assert!(
        msg.contains("over the writer's limit"),
        "the cap must be the reason: {msg}"
    );
    // The guard is O(rank); materialising 2^24 one-byte chunks would take far
    // longer and allocate gigabytes.  A generous ceiling still distinguishes the
    // two: the guarded path returns in milliseconds.
    assert!(
        elapsed.as_secs() < 5,
        "the cap must fire before materialisation, not after: took {elapsed:?}"
    );
}

/// A legitimately large chunk count just under the cap still builds — the guard
/// refuses only the pathological case, never a real dataset.
#[test]
fn r003_under_cap_chunk_count_still_builds() {
    let values = i32_raw(4096);
    let mut w = FileWriter::new();
    // 2048 chunks of two elements: a real multi-level index, well under the cap.
    w.create_dataset_unlimited("d", &[4096], &[2], &dt_i32(), &values)
        .expect("d");
    let path = std::env::temp_dir().join("oxih5_fix_chunk_r003_under.h5");
    w.build(&path).expect("a sub-cap dataset builds");
    let f = File::open(&path).expect("open");
    let _ = std::fs::remove_file(&path);
    assert_eq!(
        f.dataset("d").expect("d").as_i32().expect("as_i32"),
        (0..4096i32).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// R007 — a chunk shape with too many dimensions is rejected
// ---------------------------------------------------------------------------

/// `create_dataset_unlimited([4], chunk [2, 2])` used to silently drop the
/// second chunk dimension and store a 1-D `(2,)` chunking.  It is now a typed
/// error: a chunk cannot tile more dimensions than the dataset has.
#[test]
fn r007_rejects_chunk_rank_over_dataset_rank() {
    let mut w = FileWriter::new();
    w.create_dataset_unlimited("g", &[4], &[2, 2], &dt_i32(), &i32_raw(4))
        .expect("create is accepted; the rank seam is checked at build");
    let err = w
        .build_to_vec()
        .expect_err("a chunk with more dimensions than the dataset must be rejected");
    let msg = format!("{err}");
    assert!(matches!(err, OxiH5Error::Format(_)), "{msg}");
    assert!(
        msg.contains("more dimensions") || msg.contains("cannot tile"),
        "the error must explain the rank mismatch: {msg}"
    );
}

/// The *too-few* completion rule is untouched: a short chunk shape is still
/// completed from the dataset shape, so `oxinetcdf`'s single-extent request for
/// a 2-D variable keeps working.
#[test]
fn r007_too_few_chunk_dims_still_complete() {
    let mut w = FileWriter::new();
    // One chunk extent for a 2-D dataset: completed to (5, 3), the whole shape.
    w.create_dataset_unlimited("grid", &[5, 3], &[5], &dt_i32(), &i32_raw(15))
        .expect("grid");
    let path = std::env::temp_dir().join("oxih5_fix_chunk_r007_short.h5");
    w.build(&path).expect("a short chunk shape still builds");
    let f = File::open(&path).expect("open");
    let _ = std::fs::remove_file(&path);
    let ds = f.dataset("grid").expect("grid");
    assert_eq!(ds.shape, vec![5usize, 3]);
    assert_eq!(ds.as_i32().expect("as_i32"), (0..15i32).collect::<Vec<_>>());
}
