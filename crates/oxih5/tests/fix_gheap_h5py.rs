//! Global-heap (vlen-string) write-path conformance — lane GHEAP.
//!
//! Regression coverage for B001/B002/B008/B009: the global heap collection(s)
//! oxih5 writes must be readable by libhdf5/h5py, not just by oxih5's own
//! tolerant reader.  Every finding gets a check that fails on the pre-fix bytes
//! and passes after:
//!
//! * **B001** — every collection is padded to at least `H5HG_MINSIZE` (4096);
//!   below that libhdf5 raises `OSError: global heap size is too small`.
//! * **B002** — the leftover space is a real free-space object (index 0, size =
//!   collection_size − offset), never a size-0 terminator that hangs libhdf5.
//! * **B008** — objects spill into multiple collections rather than truncating
//!   the 16-bit heap index past 65535.
//! * **B009** — heap object size and reference sequence length are `strlen`
//!   (no NUL terminator), matching `H5T__vlen_disk_write`.
//!
//! The h5py-facing checks shell out to `python3` and skip gracefully when
//! python3 or h5py is absent (mirroring `write_tests.rs`).  The structural
//! checks parse the raw bytes in Rust and need no python at all.  Backward
//! compatibility is proved against 0.2.1-era fixtures under `tests/fixtures/`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

use oxih5::{File, FileWriter};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn tmp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("oxih5_fix_gheap_{name}"))
}

fn cleanup(path: &Path) {
    let _ = std::fs::remove_file(path);
}

/// One parsed global heap collection.
struct Gcol {
    /// File offset of the collection.
    offset: usize,
    /// Declared collection size (header field).
    declared: usize,
    /// (1-based index, object size, data) for each real object.
    objects: Vec<(u16, usize, Vec<u8>)>,
    /// (offset-within-collection, size) of the free-space object, if present.
    free: Option<(usize, usize)>,
}

/// Locate and parse every `GCOL` collection in a file image.
fn scan_gcols(data: &[u8]) -> Vec<Gcol> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while let Some(rel) = data[off..].windows(4).position(|w| w == b"GCOL") {
        let g = off + rel;
        if g + 16 > data.len() || data[g + 4] != 1 {
            off = g + 4;
            continue;
        }
        let declared = u64::from_le_bytes(data[g + 8..g + 16].try_into().unwrap()) as usize;
        if declared < 32 || g + declared > data.len() {
            off = g + 4;
            continue;
        }
        let end = g + declared;
        let mut pos = g + 16;
        let mut objects = Vec::new();
        let mut free = None;
        while pos + 16 <= end {
            let idx = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
            let osize = u64::from_le_bytes(data[pos + 8..pos + 16].try_into().unwrap()) as usize;
            if idx == 0 {
                free = Some((pos - g, osize));
                break;
            }
            let dstart = pos + 16;
            let dend = (dstart + osize).min(data.len());
            objects.push((idx, osize, data[dstart..dend].to_vec()));
            pos = (dstart + osize + 7) & !7;
        }
        out.push(Gcol {
            offset: g,
            declared,
            objects,
            free,
        });
        off = end;
    }
    out
}

/// Find on-disk vlen references pointing into `gcol_offsets`, as
/// `(seq_len, heap_addr, obj_idx)`.  Only used on tiny controlled files.
fn find_vlen_refs(data: &[u8], gcol_offsets: &[u64]) -> Vec<(u32, u64, u32)> {
    let mut hits = Vec::new();
    let mut i = 0usize;
    while i + 16 <= data.len() {
        let seq = u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
        let addr = u64::from_le_bytes(data[i + 4..i + 12].try_into().unwrap());
        let idx = u32::from_le_bytes(data[i + 12..i + 16].try_into().unwrap());
        if gcol_offsets.contains(&addr) && (1..=4).contains(&idx) && seq <= 64 {
            hits.push((seq, addr, idx));
        }
        i += 1;
    }
    hits
}

/// Run a python3/h5py `script`; skip gracefully if python3/h5py are missing,
/// fail loudly on any other non-zero exit.  Does NOT delete any file.
fn h5py_check(script: &str, what: &str) {
    // Guard against a regression *hang* (B002): SIGALRM terminates a wedged
    // interpreter so the test fails instead of blocking forever.
    let guarded =
        format!("import signal\ntry:\n signal.alarm(60)\nexcept Exception:\n pass\n{script}");
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(&guarded)
        .output();

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
        Err(_) => {
            eprintln!("python3 not found — skipping {what}");
        }
    }
}

fn write_vlen(path: &Path, name: &str, strings: &[&str]) {
    let mut w = FileWriter::new();
    w.create_vlen_string_dataset(name, strings)
        .expect("create_vlen_string_dataset");
    w.build(path).expect("build");
}

// ---------------------------------------------------------------------------
// B001 — collection padded to H5HG_MINSIZE
// ---------------------------------------------------------------------------

#[test]
fn b001_small_collection_padded_to_minsize() {
    let path = tmp_path("b001_pad.h5");
    write_vlen(&path, "v", &["alpha", "beta", "gamma"]);
    let data = std::fs::read(&path).unwrap();
    cleanup(&path);

    let gcols = scan_gcols(&data);
    assert_eq!(gcols.len(), 1, "one collection");
    // Pre-fix this was 104 bytes and h5py rejected it outright.
    assert!(
        gcols[0].declared >= 4096,
        "collection declared size {} must reach H5HG_MINSIZE (4096)",
        gcols[0].declared
    );
    assert_eq!(gcols[0].objects.len(), 3);
}

#[test]
fn b001_h5py_reads_small_vlen_dataset() {
    let path = tmp_path("b001_h5py.h5");
    write_vlen(&path, "v", &["alpha", "beta", "gamma"]);
    let script = format!(
        "import h5py\n\
         f=h5py.File(r'{p}','r')\n\
         v=[x.decode() for x in f['v'][()]]\n\
         assert v==['alpha','beta','gamma'], v\n\
         print('ok', v)",
        p = path.display()
    );
    h5py_check(&script, "B001 small vlen readable by h5py");
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// B002 — real free-space object, never size-0 terminator
// ---------------------------------------------------------------------------

#[test]
fn b002_free_space_object_has_real_size() {
    let path = tmp_path("b002_free.h5");
    // 200 strings > 4096 payload, so the collection has real free space.
    let strings: Vec<String> = (0..200).map(|i| format!("label-{i:04}")).collect();
    let refs: Vec<&str> = strings.iter().map(String::as_str).collect();
    write_vlen(&path, "v", &refs);
    let data = std::fs::read(&path).unwrap();
    cleanup(&path);

    let gcols = scan_gcols(&data);
    assert_eq!(gcols.len(), 1);
    let g = &gcols[0];
    let (free_off, free_size) = g
        .free
        .expect("a padded collection must carry a free-space object");
    // The B002 bug wrote a size-0 free object (16 zero bytes) that hangs libhdf5.
    assert!(
        free_size >= 16,
        "free object size {free_size} must be >= 16, never 0"
    );
    assert_eq!(
        free_size,
        g.declared - free_off,
        "free object size must equal collection_size - its offset"
    );
}

#[test]
fn b002_h5py_reads_2000_strings_without_hanging() {
    let path = tmp_path("b002_2000.h5");
    let strings: Vec<String> = (0..2000).map(|i| format!("item-{i:04}")).collect();
    let refs: Vec<&str> = strings.iter().map(String::as_str).collect();
    write_vlen(&path, "v", &refs);
    // Pre-fix, h5py wedged forever here; the SIGALRM guard would trip.
    let script = format!(
        "import h5py\n\
         f=h5py.File(r'{p}','r')\n\
         v=f['v'][()]\n\
         assert len(v)==2000, len(v)\n\
         assert v[0]==b'item-0000' and v[1999]==b'item-1999', (v[0],v[1999])\n\
         print('read', len(v), 'strings')",
        p = path.display()
    );
    h5py_check(&script, "B002 2000 strings readable without hang");
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// B008 — spill across multiple collections past the 16-bit index limit
// ---------------------------------------------------------------------------

#[test]
fn b008_many_strings_split_into_multiple_collections() {
    let path = tmp_path("b008_70000.h5");
    // 70000 > 65535: a single collection could not index them all with a 16-bit
    // heap index, so the writer must emit multiple collections.
    let strings: Vec<String> = (0..70_000).map(|i| (i % 97).to_string()).collect();
    let refs: Vec<&str> = strings.iter().map(String::as_str).collect();
    write_vlen(&path, "v", &refs);
    let data = std::fs::read(&path).unwrap();

    let gcols = scan_gcols(&data);
    assert!(
        gcols.len() >= 2,
        "70000 strings must split across collections, got {}",
        gcols.len()
    );
    // No collection may declare more objects than a 16-bit index can hold.
    for g in &gcols {
        assert!(g.objects.len() <= 65535, "collection over-indexed");
        assert!(g.declared >= 4096);
    }
    cleanup(&path);
}

#[test]
fn b008_h5py_reads_70000_strings_exactly() {
    let path = tmp_path("b008_h5py_70000.h5");
    let strings: Vec<String> = (0..70_000).map(|i| (i % 97).to_string()).collect();
    let refs: Vec<&str> = strings.iter().map(String::as_str).collect();
    write_vlen(&path, "v", &refs);
    let script = format!(
        "import h5py\n\
         f=h5py.File(r'{p}','r')\n\
         v=f['v'][()]\n\
         assert len(v)==70000, len(v)\n\
         assert v[0]==b'0' and v[96]==b'96' and v[69999]==str(69999%97).encode(), (v[0],v[96],v[69999])\n\
         print('read', len(v), 'strings across collections')",
        p = path.display()
    );
    h5py_check(&script, "B008 70000 strings readable by h5py");
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// B009 — strlen sizing, no NUL terminator
// ---------------------------------------------------------------------------

#[test]
fn b009_heap_object_and_reference_use_strlen_no_nul() {
    let path = tmp_path("b009_strlen.h5");
    write_vlen(&path, "v", &["hello"]);
    let data = std::fs::read(&path).unwrap();
    cleanup(&path);

    let gcols = scan_gcols(&data);
    assert_eq!(gcols.len(), 1);
    let g = &gcols[0];
    assert_eq!(g.objects.len(), 1);
    let (idx, osize, obj_data) = &g.objects[0];
    assert_eq!(*idx, 1);
    // Pre-fix: osize == 6 and data == b"hello\0".  Post-fix: strlen, no NUL.
    assert_eq!(*osize, 5, "heap object size must be strlen, not strlen+1");
    assert_eq!(obj_data.as_slice(), b"hello");
    assert!(
        !obj_data.contains(&0u8),
        "no NUL terminator in the heap object"
    );

    // The 16-byte vlen reference's sequence length must also be strlen.
    let refs = find_vlen_refs(&data, &[g.offset as u64]);
    assert_eq!(refs.len(), 1, "exactly one vlen reference: {refs:?}");
    assert_eq!(refs[0].0, 5, "reference seq_len must be strlen (5), not 6");
    assert_eq!(refs[0].2, 1, "reference obj_idx");
}

#[test]
fn b009_h5py_reads_empty_and_multibyte_utf8() {
    let path = tmp_path("b009_utf8.h5");
    // Empty strings (seq_len 0) and multi-byte UTF-8 must round-trip exactly.
    write_vlen(&path, "v", &["", "a", "héllo", "日本語", ""]);
    let script = format!(
        "import h5py\n\
         f=h5py.File(r'{p}','r')\n\
         v=[x.decode('utf-8') for x in f['v'][()]]\n\
         assert v==['','a','héllo','日本語',''], v\n\
         print('utf8 ok', v)",
        p = path.display()
    );
    h5py_check(&script, "B009 empty + UTF-8 readable by h5py");
    cleanup(&path);
}

// ---------------------------------------------------------------------------
// Backward compatibility — oxih5 still reads 0.2.1-era files
// ---------------------------------------------------------------------------

#[test]
fn backward_compat_reads_old_021_vlen_files() {
    // These fixtures were written by oxih5 0.2.1: undersized collection,
    // NUL-terminated heap objects (strlen+1), and a size-0 NIL terminator.
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");

    let f = File::open(base.join("vlen_old021_three.h5")).expect("open old three");
    assert_eq!(
        f.dataset_strings("words").expect("read words"),
        vec!["alpha", "beta", "gamma"]
    );

    let f = File::open(base.join("vlen_old021_single.h5")).expect("open old single");
    assert_eq!(f.dataset_strings("vl").expect("read vl"), vec!["hello"]);
}

// ---------------------------------------------------------------------------
// Cross-direction — oxih5 reads an h5py-written vlen file
// ---------------------------------------------------------------------------

#[test]
fn oxih5_reads_h5py_written_vlen_file() {
    let path = tmp_path("h5py_written.h5");
    cleanup(&path);
    let gen = format!(
        "import h5py, numpy as np\n\
         dt=h5py.string_dtype(encoding='utf-8')\n\
         with h5py.File(r'{p}','w') as f:\n\
         \x20f.create_dataset('v', data=np.array(['','one','héllo','日本語','three'],dtype=object), dtype=dt)\n\
         print('written')",
        p = path.display()
    );
    let out = std::process::Command::new("python3")
        .arg("-c")
        .arg(&gen)
        .output();
    match out {
        Ok(o) if o.status.success() && path.exists() => {
            let f = File::open(&path).expect("oxih5 open h5py file");
            let v = f.dataset_strings("v").expect("read h5py vlen");
            cleanup(&path);
            assert_eq!(v, vec!["", "one", "héllo", "日本語", "three"]);
        }
        _ => {
            cleanup(&path);
            eprintln!("python3/h5py unavailable — skipping h5py->oxih5 read");
        }
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust round trip across every size class (always runs)
// ---------------------------------------------------------------------------

#[test]
fn pure_rust_roundtrip_all_size_classes() {
    let cases: Vec<(&str, Vec<String>)> = vec![
        ("small", vec!["alpha".into(), "beta".into(), "gamma".into()]),
        (
            "empty_utf8",
            vec![
                "".into(),
                "a".into(),
                "héllo".into(),
                "日本語".into(),
                "".into(),
            ],
        ),
        (
            "big2000",
            (0..2000).map(|i| format!("item-{i:04}")).collect(),
        ),
        (
            "huge70000",
            (0..70_000).map(|i| (i % 97).to_string()).collect(),
        ),
    ];
    for (label, strings) in cases {
        let path = tmp_path(&format!("rt_{label}.h5"));
        let refs: Vec<&str> = strings.iter().map(String::as_str).collect();
        write_vlen(&path, "v", &refs);
        let f = File::open(&path).expect("open");
        let got = f.dataset_strings("v").expect("read");
        cleanup(&path);
        assert_eq!(got.len(), strings.len(), "{label}: count");
        assert_eq!(got, strings, "{label}: values");
    }
}
