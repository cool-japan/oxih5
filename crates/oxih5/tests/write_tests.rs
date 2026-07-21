// Phase 8 write support integration tests.
//
// Each test writes a file to a unique temp path (process-id + atomic counter),
// reads it back via oxih5::File::open(), and validates the round-trip.

use oxih5::FileWriter;
use std::sync::atomic::{AtomicU64, Ordering};

static WT_COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_path(tag: &str) -> std::path::PathBuf {
    let n = WT_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("oxih5_write_{}_{n}_{tag}.h5", std::process::id()))
}

fn cleanup(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}

// ---------------------------------------------------------------------------
// Test 1: round-trip 1-D float32
// ---------------------------------------------------------------------------
#[test]
fn test_write_read_f32_1d() {
    let path = tmp_path("f32_1d");
    FileWriter::new()
        .write_dataset_f32("data", &[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5])
        .unwrap()
        .build(&path)
        .unwrap();

    let file = oxih5::open(&path).unwrap();
    cleanup(&path);

    let ds = file.dataset("data").unwrap();
    assert_eq!(ds.shape, vec![5]);
    let vals = ds.as_f32().unwrap();
    assert_eq!(vals, vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
}

// ---------------------------------------------------------------------------
// Test 2: round-trip 1-D float64
// ---------------------------------------------------------------------------
#[test]
fn test_write_read_f64_1d() {
    let path = tmp_path("f64_1d");
    FileWriter::new()
        .write_dataset_f64("temps", &[0.1f64, 0.2, 0.3], &[3])
        .unwrap()
        .build(&path)
        .unwrap();

    let file = oxih5::open(&path).unwrap();
    cleanup(&path);

    let vals = file.dataset("temps").unwrap().as_f64().unwrap();
    assert_eq!(vals.len(), 3);
    assert!((vals[0] - 0.1).abs() < 1e-10);
    assert!((vals[2] - 0.3).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Test 3: round-trip 1-D int32
// ---------------------------------------------------------------------------
#[test]
fn test_write_read_i32_1d() {
    let path = tmp_path("i32_1d");
    FileWriter::new()
        .write_dataset_i32("ints", &[-1i32, 0, 1, 100, -100], &[5])
        .unwrap()
        .build(&path)
        .unwrap();

    let file = oxih5::open(&path).unwrap();
    cleanup(&path);

    let vals = file.dataset("ints").unwrap().as_i32().unwrap();
    assert_eq!(vals, vec![-1i32, 0, 1, 100, -100]);
}

// ---------------------------------------------------------------------------
// Test 4: round-trip 1-D int64
// ---------------------------------------------------------------------------
#[test]
fn test_write_read_i64_1d() {
    let path = tmp_path("i64_1d");
    FileWriter::new()
        .write_dataset_i64("longs", &[i64::MIN, -1, 0, 1, i64::MAX], &[5])
        .unwrap()
        .build(&path)
        .unwrap();

    let file = oxih5::open(&path).unwrap();
    cleanup(&path);

    let vals = file.dataset("longs").unwrap().as_i64().unwrap();
    assert_eq!(vals, vec![i64::MIN, -1i64, 0, 1, i64::MAX]);
}

// ---------------------------------------------------------------------------
// Test 5: round-trip 1-D uint8
// ---------------------------------------------------------------------------
#[test]
fn test_write_read_u8_1d() {
    let path = tmp_path("u8_1d");
    FileWriter::new()
        .write_dataset_u8("bytes", &[0u8, 127, 255], &[3])
        .unwrap()
        .build(&path)
        .unwrap();

    let file = oxih5::open(&path).unwrap();
    cleanup(&path);

    let vals = file.dataset("bytes").unwrap().as_u8().unwrap();
    assert_eq!(vals, vec![0u8, 127, 255]);
}

// ---------------------------------------------------------------------------
// Test 6: multiple datasets in one file
// ---------------------------------------------------------------------------
#[test]
fn test_write_read_multiple_datasets() {
    let path = tmp_path("multi");
    FileWriter::new()
        .write_dataset_f32("x", &[1.0f32, 2.0], &[2])
        .unwrap()
        .write_dataset_i32("y", &[10i32, 20], &[2])
        .unwrap()
        .build(&path)
        .unwrap();

    let file = oxih5::open(&path).unwrap();
    cleanup(&path);

    assert_eq!(
        file.dataset("x").unwrap().as_f32().unwrap(),
        vec![1.0f32, 2.0]
    );
    assert_eq!(
        file.dataset("y").unwrap().as_i32().unwrap(),
        vec![10i32, 20]
    );
}

// ---------------------------------------------------------------------------
// Test 7: 2-D float64 — shape and values
// ---------------------------------------------------------------------------
#[test]
fn test_write_read_2d_f64() {
    let path = tmp_path("2d_f64");
    let data: Vec<f64> = (0..6).map(|x| x as f64).collect();
    FileWriter::new()
        .write_dataset_f64("matrix", &data, &[2, 3])
        .unwrap()
        .build(&path)
        .unwrap();

    let file = oxih5::open(&path).unwrap();
    cleanup(&path);

    let ds = file.dataset("matrix").unwrap();
    assert_eq!(ds.shape, vec![2, 3]);
    let vals = ds.as_f64().unwrap();
    assert_eq!(vals.len(), 6);
    assert_eq!(vals[3], 3.0);
}

// ---------------------------------------------------------------------------
// Test 8: no root capacity limit — the symbol-table B-tree grows instead
//
// This test used to assert the opposite: that the 65th dataset was rejected,
// because the writer emitted exactly one symbol table node per group and had
// to cap the file at whatever fitted in it.  Links now chain across SNODs and
// the B-tree grows a level once 32 of them fill one node, so 300 datasets are
// an ordinary file — and one that crosses into a level-1 tree.
// ---------------------------------------------------------------------------
#[test]
fn test_write_three_hundred_datasets_roundtrip() {
    let path = tmp_path("three_hundred");
    let mut writer = FileWriter::new();
    for i in 0..300usize {
        writer
            .write_dataset_f32(&format!("d{i:03}"), &[i as f32], &[1])
            .unwrap();
    }
    writer.build(&path).unwrap();

    let file = oxih5::open(&path).unwrap();
    cleanup(&path);

    let names = file.dataset_names().unwrap();
    assert_eq!(names.len(), 300, "every dataset must be listed");
    for i in 0..300usize {
        let name = format!("d{i:03}");
        let ds = file
            .dataset(&name)
            .unwrap_or_else(|e| panic!("{name} not readable: {e}"));
        assert_eq!(ds.as_f32().unwrap(), vec![i as f32], "{name}");
    }
}

// ---------------------------------------------------------------------------
// Test 9: dataset_names lists all written datasets
// ---------------------------------------------------------------------------
#[test]
fn test_write_dataset_names() {
    let path = tmp_path("names");
    FileWriter::new()
        .write_dataset_f32("alpha", &[1.0f32], &[1])
        .unwrap()
        .write_dataset_i32("beta", &[2i32], &[1])
        .unwrap()
        .build(&path)
        .unwrap();

    let file = oxih5::open(&path).unwrap();
    cleanup(&path);

    let names = file.dataset_names().unwrap();
    assert!(
        names.contains(&"alpha".to_string()),
        "expected 'alpha', got {names:?}"
    );
    assert!(
        names.contains(&"beta".to_string()),
        "expected 'beta', got {names:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 10: validation — empty name
// ---------------------------------------------------------------------------
#[test]
fn test_write_empty_name_rejected() {
    let mut writer = FileWriter::new();
    let result = writer.write_dataset_f32("", &[1.0f32], &[1]);
    assert!(result.is_err(), "empty name should be rejected");
}

// ---------------------------------------------------------------------------
// Test 11: validation — a slash separates path components, but the path still
//          has to be well formed
// ---------------------------------------------------------------------------

/// `'/'` is a separator now, not a forbidden character.
///
/// This test used to assert that `"a/b"` was rejected, back when a name was a
/// single symbol-table link and nothing else.  It now names a dataset `b` in a
/// group `a`; what is still rejected is a path that cannot mean anything.
#[test]
fn test_write_malformed_path_rejected() {
    let mut writer = FileWriter::new();

    // A well-formed path is accepted and creates the group above it.
    writer
        .write_dataset_f32("a/b", &[1.0f32], &[1])
        .expect("'a/b' names a dataset in group 'a'");

    for path in ["", "/", "a//b", "b/", "//b", "a/./b", "a/../b"] {
        assert!(
            writer.write_dataset_f32(path, &[1.0f32], &[1]).is_err(),
            "malformed path {path:?} should be rejected"
        );
    }

    // A path deeper than the writer's limit is refused rather than recursed.
    let too_deep = vec!["g"; 65].join("/");
    assert!(
        writer
            .write_dataset_f32(&too_deep, &[1.0f32], &[1])
            .is_err(),
        "a 65-component path should be rejected"
    );
}

// ---------------------------------------------------------------------------
// Validation — data length must match declared shape (H5 write-path robustness).
// A mismatch must return a typed error, not silently corrupt the file or panic.
// ---------------------------------------------------------------------------
#[test]
fn test_write_shape_data_mismatch_rejected() {
    let mut writer = FileWriter::new();
    // 2 data elements but shape declares 5.
    let too_few = writer.write_dataset_f64("bad", &[1.0, 2.0], &[5]);
    assert!(
        too_few.is_err(),
        "data shorter than shape should be rejected"
    );

    // 6 data elements but shape declares [2, 2] = 4.
    let too_many = writer.write_dataset_i32("bad2", &[1, 2, 3, 4, 5, 6], &[2, 2]);
    assert!(
        too_many.is_err(),
        "data longer than shape should be rejected"
    );

    // Matching length still succeeds.
    let ok = writer.write_dataset_f64("good", &[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert!(ok.is_ok(), "matching data/shape should be accepted");
}

// ---------------------------------------------------------------------------
// Test 12: validation — duplicate name
// ---------------------------------------------------------------------------
#[test]
fn test_write_duplicate_name_rejected() {
    let mut writer = FileWriter::new();
    writer.write_dataset_f32("same", &[1.0f32], &[1]).unwrap();
    let result = writer.write_dataset_f32("same", &[2.0f32], &[1]);
    assert!(result.is_err(), "duplicate dataset name should be rejected");
}

// ---------------------------------------------------------------------------
// Test 13: Default trait works
// ---------------------------------------------------------------------------
#[test]
fn test_filewriter_default() {
    let path = tmp_path("default");
    let mut writer = FileWriter::default();
    writer
        .write_dataset_u8("pixels", &[10u8, 20, 30], &[3])
        .unwrap();
    writer.build(&path).unwrap();

    let file = oxih5::open(&path).unwrap();
    cleanup(&path);

    assert_eq!(
        file.dataset("pixels").unwrap().as_u8().unwrap(),
        vec![10u8, 20, 30]
    );
}

// ---------------------------------------------------------------------------
// Test 14: empty file (zero datasets) writes and re-opens cleanly
// ---------------------------------------------------------------------------
#[test]
fn test_write_empty_file() {
    let path = tmp_path("empty");
    FileWriter::new().build(&path).unwrap();

    let file = oxih5::open(&path).unwrap();
    cleanup(&path);

    let names = file.dataset_names().unwrap();
    assert!(names.is_empty(), "expected no datasets, got {names:?}");
}

// ---------------------------------------------------------------------------
// Test 15: 8 datasets in one file (full capacity)
// ---------------------------------------------------------------------------
#[test]
fn test_write_eight_datasets() {
    let path = tmp_path("eight");
    let mut writer = FileWriter::new();
    for i in 0..8usize {
        let val = i as f32;
        writer
            .write_dataset_f32(&format!("ds{i:02}"), &[val], &[1])
            .unwrap();
    }
    writer.build(&path).unwrap();

    let file = oxih5::open(&path).unwrap();
    cleanup(&path);

    for i in 0..8usize {
        let ds = file.dataset(&format!("ds{i:02}")).unwrap();
        let vals = ds.as_f32().unwrap();
        assert_eq!(vals.len(), 1);
        assert!(
            (vals[0] - i as f32).abs() < 1e-6,
            "ds{i:02}: expected {}, got {}",
            i as f32,
            vals[0]
        );
    }
}

// ---------------------------------------------------------------------------
// Test 16: h5py interoperability verification (skipped if h5py not installed)
// ---------------------------------------------------------------------------
#[test]
fn test_write_h5py_verification() {
    let path = tmp_path("h5py_check");
    FileWriter::new()
        .write_dataset_f32("values", &[10.0f32, 20.0, 30.0], &[3])
        .unwrap()
        .build(&path)
        .unwrap();

    let script = format!(
        "import h5py, numpy as np; \
         f=h5py.File({path:?},'r'); \
         d=f['values'][:]; \
         assert list(d)==[10.0,20.0,30.0], f'got {{list(d)}}'; \
         print('h5py OK')",
        path = path.display()
    );

    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(&script)
        .output();

    cleanup(&path);

    match output {
        Ok(out) if out.status.success() => {
            eprintln!(
                "h5py verification passed: {}",
                String::from_utf8_lossy(&out.stdout).trim()
            );
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let stdout = String::from_utf8_lossy(&out.stdout);
            // h5py available but verification failed — treat as test failure.
            if stderr.contains("ModuleNotFoundError")
                || stderr.contains("ImportError")
                || stderr.contains("No module named")
            {
                // h5py/numpy not installed — skip gracefully.
                eprintln!("h5py not available — skipping interop test");
            } else {
                panic!("h5py verification FAILED:\nstdout: {stdout}\nstderr: {stderr}");
            }
        }
        Err(_) => {
            // python3 not found — skip gracefully.
            eprintln!("python3 not found — skipping h5py interop test");
        }
    }
}

// ---------------------------------------------------------------------------
// Test 17: h5py reads every writable fixed-width element type
// ---------------------------------------------------------------------------

/// Run an h5py check over `path`, then delete it.
///
/// Applies the same three-way guard as `test_write_h5py_verification`:
/// python3 missing → skip, h5py/numpy missing → skip, any other non-zero exit
/// → fail.  A silent skip on a *real* failure would make this test worthless,
/// so only the two "not installed" shapes are tolerated.
fn h5py_check(path: &std::path::Path, script: &str, what: &str) {
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .output();

    cleanup(path);

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

/// Build the Python source that checks one file against `cases`.
///
/// Each case is `(dataset name, numpy dtype name, values as a Python list
/// literal)`.  The dtype assertion is the interesting half: it pins the class,
/// the element size, and the signedness — exactly the three fields the writer
/// puts in the class-0/class-1 datatype message — so a wrong sign bit is caught
/// even where the values themselves happen to round-trip.
fn h5py_dtype_script(path: &std::path::Path, cases: &[(&str, &str, &str)]) -> String {
    let case_list: String = cases
        .iter()
        .map(|(name, dtype, values)| format!("({name:?},{dtype:?},{values}),"))
        .collect();
    let path_lit = path.display();
    format!(
        "import h5py, numpy as np\n\
         f = h5py.File({path_lit:?}, 'r')\n\
         cases = [{case_list}]\n\
         for name, dt, want in cases:\n\
         \x20   d = f[name]\n\
         \x20   want_dt = np.dtype(dt)\n\
         \x20   assert d.dtype == want_dt, name + ': dtype ' + str(d.dtype) + ' != ' + dt\n\
         \x20   assert d.dtype.kind == want_dt.kind, name + ': kind ' + d.dtype.kind\n\
         \x20   assert d.dtype.itemsize == want_dt.itemsize, name + ': itemsize'\n\
         \x20   got = [x.item() for x in d[:]]\n\
         \x20   assert got == want, name + ': ' + repr(got) + ' != ' + repr(want)\n\
         print('h5py verified', len(cases), 'datatypes')\n"
    )
}

/// h5py must read back every fixed-width element type the writer can emit,
/// with the exact values and the exact numpy dtype.
///
/// Our own reader parses the datatype messages this writer produces, so a
/// self-consistent mistake — a sign bit misplaced in both codecs, say — is
/// invisible to a pure Rust round-trip.  libhdf5 is the independent authority,
/// and the values are the extremes of each range because that is where a wrong
/// width or sign flag stops being a silent no-op.
///
/// All ten live in one file, so this also covers a ten-entry root symbol table
/// end to end under real libhdf5.
#[test]
fn test_write_h5py_all_element_types() {
    let path = tmp_path("h5py_all_types");
    FileWriter::new()
        .write_dataset_f32("a_f32", &[1.5f32, -2.25, 3.0], &[3])
        .unwrap()
        .write_dataset_f64("b_f64", &[0.5f64, -1.25, 2.0], &[3])
        .unwrap()
        .write_dataset_i8("c_i8", &[i8::MIN, -1, 0, i8::MAX], &[4])
        .unwrap()
        .write_dataset_i16("d_i16", &[i16::MIN, -1, 0, i16::MAX], &[4])
        .unwrap()
        .write_dataset_i32("e_i32", &[i32::MIN, 0, i32::MAX], &[3])
        .unwrap()
        .write_dataset_i64("f_i64", &[i64::MIN, 0, i64::MAX], &[3])
        .unwrap()
        .write_dataset_u8("g_u8", &[0u8, 128, u8::MAX], &[3])
        .unwrap()
        .write_dataset_u16("h_u16", &[0u16, 32_768, u16::MAX], &[3])
        .unwrap()
        .write_dataset_u32("i_u32", &[0u32, 2_147_483_648, u32::MAX], &[3])
        .unwrap()
        .write_dataset_u64("j_u64", &[0u64, 9_223_372_036_854_775_808, u64::MAX], &[3])
        .unwrap()
        .build(&path)
        .unwrap();

    let cases = [
        ("a_f32", "float32", "[1.5, -2.25, 3.0]"),
        ("b_f64", "float64", "[0.5, -1.25, 2.0]"),
        ("c_i8", "int8", "[-128, -1, 0, 127]"),
        ("d_i16", "int16", "[-32768, -1, 0, 32767]"),
        ("e_i32", "int32", "[-2147483648, 0, 2147483647]"),
        (
            "f_i64",
            "int64",
            "[-9223372036854775808, 0, 9223372036854775807]",
        ),
        ("g_u8", "uint8", "[0, 128, 255]"),
        ("h_u16", "uint16", "[0, 32768, 65535]"),
        ("i_u32", "uint32", "[0, 2147483648, 4294967295]"),
        (
            "j_u64",
            "uint64",
            "[0, 9223372036854775808, 18446744073709551615]",
        ),
    ];
    h5py_check(
        &path,
        &h5py_dtype_script(&path, &cases),
        "h5py all-element-type interop",
    );
}

// ---------------------------------------------------------------------------
// Test 18: h5py must find every link, however many there are and however they
//          were declared
// ---------------------------------------------------------------------------

/// Build the Python source that looks up every dataset of a numbered file
/// **by name**.
///
/// `f[name]` is the operation that matters.  It runs `H5G__node_found`, which
/// binary-searches the symbol table node with the names resolved through the
/// local heap — the code path that an unsorted or oversized node breaks.
/// `sorted(f.keys())` is checked too, but only as a cross-check: iteration is
/// linear and kept reporting links that lookup could no longer find, which is
/// precisely why the defect read as "a missing dataset" rather than a corrupt
/// file.
fn h5py_by_name_script(path: &std::path::Path, n: usize) -> String {
    let path_lit = path.display();
    format!(
        "import h5py\n\
         f = h5py.File({path_lit:?}, 'r')\n\
         names = ['ds%03d' % i for i in range({n})]\n\
         keys = sorted(f.keys())\n\
         assert keys == sorted(names), 'keys: ' + repr(keys)\n\
         for i, name in enumerate(names):\n\
         \x20   assert name in f, name + ' not found by H5Lexists'\n\
         \x20   got = int(f[name][0])\n\
         \x20   assert got == i, name + ': ' + repr(got)\n\
         print('h5py looked up', len(names), 'links by name')\n"
    )
}

/// Write `n` numbered datasets and have libhdf5 read every one back by name.
fn h5py_numbered_roundtrip(n: usize, what: &str) {
    let path = tmp_path(&format!("h5py_symtab_{n}"));
    let mut writer = FileWriter::new();
    for i in 0..n {
        writer
            .write_dataset_i32(&format!("ds{i:03}"), &[i as i32], &[1])
            .unwrap();
    }
    writer.build(&path).unwrap();
    h5py_check(&path, &h5py_by_name_script(&path, n), what);
}

/// Nine root links spill into a second symbol table node.
///
/// Eight or fewer used to work and nine did not, because libhdf5 sizes a SNOD
/// image from the superblock's `leaf_node_K` (`8 + 2*4*40` = 328 bytes) and
/// then decodes `nsyms` entries out of it:
///
/// ```text
/// #012: H5Gcache.c line 188 in H5G__cache_node_deserialize(): unable to decode symbol table entries
/// #013: H5Gent.c line 86 in H5G__ent_decode_vec(): ran off the end of the image buffer
/// ```
#[test]
fn test_write_h5py_nine_root_datasets() {
    h5py_numbered_roundtrip(9, "h5py nine-link symbol table");
}

/// Forty root links span five symbol table nodes under one B-tree node.
#[test]
fn test_write_h5py_forty_root_datasets() {
    h5py_numbered_roundtrip(40, "h5py forty-link symbol table");
}

/// Three hundred root links no longer fit under a single B-tree node, so the
/// tree grows a level — and libhdf5 has to descend it.
///
/// 256 links is exactly `2 * internal_node_K` SNODs' worth; the 257th forces a
/// level-1 root.  Nothing below this size exercises `H5B__find`'s recursive
/// descent, so without this the multi-level path would only ever have been
/// read back by oxih5's own reader.
#[test]
fn test_write_h5py_three_hundred_root_datasets() {
    h5py_numbered_roundtrip(300, "h5py level-1 symbol table B-tree");
}

/// Links declared out of order must all still be reachable.
///
/// Two files differing only in declaration order used to behave differently:
/// `["aaa","bbb"]` opened fine, `["bbb","aaa"]` raised `KeyError: object 'bbb'
/// doesn't exist` — while `list(f.keys())` still listed both.  A realistic
/// NetCDF-style file that declared `values` before `lat` lost `values`.
#[test]
fn test_write_h5py_reverse_declaration_order() {
    let path = tmp_path("h5py_reverse_order");
    let mut writer = FileWriter::new();
    // Strictly descending, plus the NetCDF-style pair that first exposed this.
    writer.write_dataset_f64("zzz", &[3.0], &[1]).unwrap();
    writer.write_dataset_f64("mmm", &[2.0], &[1]).unwrap();
    writer.write_dataset_f64("aaa", &[1.0], &[1]).unwrap();
    writer.write_dataset_f64("values", &[9.0], &[1]).unwrap();
    writer.write_dataset_f64("lat", &[8.0], &[1]).unwrap();
    writer.build(&path).unwrap();

    let path_lit = path.display();
    let script = format!(
        "import h5py\n\
         f = h5py.File({path_lit:?}, 'r')\n\
         want = {{'zzz': 3.0, 'mmm': 2.0, 'aaa': 1.0, 'values': 9.0, 'lat': 8.0}}\n\
         keys = sorted(f.keys())\n\
         assert keys == sorted(want), 'keys: ' + repr(keys)\n\
         for name, value in want.items():\n\
         \x20   assert name in f, name + ' not found by H5Lexists'\n\
         \x20   got = float(f[name][0])\n\
         \x20   assert got == value, name + ': ' + repr(got)\n\
         print('h5py looked up', len(want), 'reverse-declared links')\n"
    );
    h5py_check(&path, &script, "h5py reverse-declaration-order interop");
}

/// A sub-group's symbol table obeys the same one `leaf_node_K` as the root's.
///
/// The root and sub-groups once declared different node capacities — 32 and 16
/// entries against a superblock that said 4 — which cannot be represented,
/// because `leaf_node_K` is a single per-file field.  Ten links in each of two
/// symbol tables exercises both.
#[test]
fn test_write_h5py_subgroup_symbol_table() {
    let path = tmp_path("h5py_subgroup_symtab");
    let mut writer = FileWriter::new();
    writer.create_group("grp").unwrap();
    for i in 0..10usize {
        writer
            .write_dataset_i32(&format!("root{i:02}"), &[i as i32], &[1])
            .unwrap();
        writer
            .write_group_dataset_i32("grp", &format!("inner{i:02}"), &[i as i32 * 10], &[1])
            .unwrap();
    }
    writer.build(&path).unwrap();

    let path_lit = path.display();
    let script = format!(
        "import h5py\n\
         f = h5py.File({path_lit:?}, 'r')\n\
         g = f['grp']\n\
         for i in range(10):\n\
         \x20   root = 'root%02d' % i\n\
         \x20   inner = 'inner%02d' % i\n\
         \x20   assert int(f[root][0]) == i, root\n\
         \x20   assert int(g[inner][0]) == i * 10, inner\n\
         \x20   assert int(f['/grp/' + inner][0]) == i * 10, inner\n\
         assert sorted(f.keys()) == sorted(['grp'] + ['root%02d' % i for i in range(10)])\n\
         assert sorted(g.keys()) == sorted(['inner%02d' % i for i in range(10)])\n\
         print('h5py looked up 10 root links and 10 sub-group links')\n"
    );
    h5py_check(&path, &script, "h5py sub-group symbol table interop");
}

// ---------------------------------------------------------------------------
// Test 22: h5py writes, oxih5 overwrites in place, h5py verifies
// ---------------------------------------------------------------------------

/// Run `script` under python3.
///
/// Returns `Ok(stdout)` on success, `Err(None)` when python3 or h5py/numpy is
/// unavailable (skip), and `Err(Some(detail))` on a genuine failure.  Same
/// three-way guard as [`h5py_check`], but the caller keeps control of the file
/// so it can be used for both halves of a write-then-verify round trip.
fn h5py_try(script: &str) -> Result<String, Option<String>> {
    match std::process::Command::new("python3")
        .arg("-c")
        .arg(script)
        .output()
    {
        Ok(out) if out.status.success() => {
            Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            if stderr.contains("ModuleNotFoundError")
                || stderr.contains("ImportError")
                || stderr.contains("No module named")
            {
                Err(None)
            } else {
                Err(Some(format!(
                    "stdout: {}\nstderr: {stderr}",
                    String::from_utf8_lossy(&out.stdout)
                )))
            }
        }
        Err(_) => Err(None),
    }
}

/// The scenario this feature exists for, end to end.
///
/// **h5py authors the file**, so it contains real libhdf5 structures — and, in
/// particular, the things a `FileWriter` rebuild would destroy: a
/// `MATLAB_class` attribute, an object-reference attribute, and a `#refs#`
/// group, exactly as a MATLAB v7.3 `.mat` file has.  oxih5 then overwrites one
/// dataset in place, and h5py reads the file back and checks both the new
/// values and that everything else survived.
#[test]
fn test_h5py_authored_file_overwritten_in_place() {
    let path = tmp_path("h5py_inplace");
    let path_lit = path.display();

    // -- Phase 1: h5py writes a MATLAB-shaped file ---------------------------
    let create = format!(
        "import h5py, numpy as np\n\
         f = h5py.File({path_lit:?}, 'w')\n\
         g = f.create_group('results')\n\
         g.create_dataset('values', data=np.array([1.0, 2.0, 3.0, 4.0], dtype='<f8'))\n\
         g.create_dataset('keep', data=np.array([10, 20, 30], dtype='<i4'))\n\
         f.create_dataset('root_keep', data=np.array([0.5, 1.5], dtype='<f4'))\n\
         refs = f.create_group('#refs#')\n\
         refs.create_dataset('a', data=np.array([7, 8], dtype='<i8'))\n\
         g.attrs['MATLAB_class'] = np.bytes_(b'double')\n\
         g.attrs['target'] = refs['a'].ref\n\
         f.attrs['note'] = 'preserve me'\n\
         f.close()\n\
         print('h5py wrote the fixture')\n"
    );
    if h5py_try(&create).is_err() {
        cleanup(&path);
        eprintln!("python3/h5py not available — skipping h5py in-place interop test");
        return;
    }

    // -- Phase 2: oxih5 overwrites one dataset in place ----------------------
    let extent = oxih5::dataset_data_extent(&path, "/results/values");
    let before = std::fs::read(&path).unwrap_or_default();
    let overwrite =
        oxih5::write_dataset_in_place_f64(&path, "/results/values", &[9.0, 8.0, 7.0, 6.0]);
    let after = std::fs::read(&path).unwrap_or_default();

    // -- Phase 3: h5py reads it back -----------------------------------------
    let verify = format!(
        "import h5py, numpy as np\n\
         def s(x):\n\
         \x20   return x.decode() if isinstance(x, bytes) else str(x)\n\
         f = h5py.File({path_lit:?}, 'r')\n\
         v = [float(x) for x in f['results/values'][:]]\n\
         assert v == [9.0, 8.0, 7.0, 6.0], 'values=' + repr(v)\n\
         k = [int(x) for x in f['results/keep'][:]]\n\
         assert k == [10, 20, 30], 'sibling dataset changed: ' + repr(k)\n\
         r = [float(x) for x in f['root_keep'][:]]\n\
         assert r == [0.5, 1.5], 'root dataset changed: ' + repr(r)\n\
         assert s(f['results'].attrs['MATLAB_class']) == 'double', 'MATLAB_class lost'\n\
         assert s(f.attrs['note']) == 'preserve me', 'root attribute lost'\n\
         assert '#refs#' in f, '#refs# group lost'\n\
         d = f[f['results'].attrs['target']]\n\
         assert [int(x) for x in d[:]] == [7, 8], 'object reference broken'\n\
         print('h5py verified in-place overwrite; MATLAB-shaped structures intact')\n"
    );
    let verified = h5py_try(&verify);

    cleanup(&path);

    overwrite.expect("oxih5 in-place overwrite of an h5py-written file");
    let extent = extent.expect("data extent of an h5py-written contiguous dataset");
    assert_eq!(extent.size, 32, "4 x f64");
    assert_eq!(before.len(), after.len(), "file size must not change");

    let lo = extent.address as usize;
    let hi = lo + extent.size as usize;
    let outside: Vec<usize> = (0..before.len())
        .filter(|&i| before[i] != after[i])
        .filter(|&i| i < lo || i >= hi)
        .collect();
    assert!(
        outside.is_empty(),
        "oxih5 changed bytes outside the data range at offsets {outside:?}"
    );

    match verified {
        Ok(stdout) => eprintln!("h5py in-place interop: {stdout}"),
        Err(None) => eprintln!("h5py became unavailable mid-test — skipping verification"),
        Err(Some(detail)) => panic!("h5py in-place verification FAILED:\n{detail}"),
    }
}

// ---------------------------------------------------------------------------
// Test 23: h5py must walk a nested hierarchy the writer built from paths
// ---------------------------------------------------------------------------

/// libhdf5 has to reach a dataset three groups down, by full path.
///
/// Nesting used to be one level deep and a sub-group's object header a fixed
/// 40 bytes, so this hierarchy could not be written at all.  Each level is a
/// symbol table of its own, and the parent's entry for it must carry
/// `cache_type = 1` with the child's B-tree root and local heap in the scratch
/// pad — `H5G__stab_find` reads exactly those two addresses on the way down.
#[test]
fn test_write_h5py_three_level_nesting() {
    let path = tmp_path("h5py_nested");
    let mut writer = FileWriter::new();
    // Only the leaf paths are ever mentioned; a, a/b and a/b/c are implied.
    writer
        .write_dataset_f64("/a/b/c/ds", &[1.0, 2.0, 3.0], &[3])
        .unwrap();
    writer
        .write_dataset_i32("/a/b/sibling", &[7i32], &[1])
        .unwrap();
    writer.write_dataset_i32("top", &[42i32], &[1]).unwrap();
    writer.build(&path).unwrap();

    let path_lit = path.display();
    let script = format!(
        "import h5py\n\
         f = h5py.File({path_lit:?}, 'r')\n\
         assert sorted(f.keys()) == ['a', 'top'], repr(sorted(f.keys()))\n\
         assert sorted(f['a'].keys()) == ['b'], repr(sorted(f['a'].keys()))\n\
         assert sorted(f['/a/b'].keys()) == ['c', 'sibling'], repr(sorted(f['/a/b'].keys()))\n\
         assert sorted(f['/a/b/c'].keys()) == ['ds'], repr(sorted(f['/a/b/c'].keys()))\n\
         v = [float(x) for x in f['/a/b/c/ds'][:]]\n\
         assert v == [1.0, 2.0, 3.0], repr(v)\n\
         chained = [float(x) for x in f['a']['b']['c']['ds'][:]]\n\
         assert chained == [1.0, 2.0, 3.0], repr(chained)\n\
         assert f['/a/b/c/ds'].name == '/a/b/c/ds', f['/a/b/c/ds'].name\n\
         assert int(f['/a/b/sibling'][0]) == 7\n\
         assert int(f['top'][0]) == 42\n\
         names = []\n\
         f.visit(names.append)\n\
         assert sorted(names) == ['a', 'a/b', 'a/b/c', 'a/b/c/ds', 'a/b/sibling', 'top'], \
         repr(sorted(names))\n\
         print('h5py walked a three-level hierarchy of', len(names), 'objects')\n"
    );
    h5py_check(&path, &script, "h5py three-level nesting interop");
}

// ---------------------------------------------------------------------------
// Test 24: h5py must read attributes attached to a sub-group
// ---------------------------------------------------------------------------

/// Every attribute kind, on a group two levels down.
///
/// A sub-group's object header was sized by a `sub_group_oh_size()` that took
/// no attributes, so a group could not carry any and only the root could —
/// and the root only strings.  The group's symbol table message still has to
/// come first, or `H5G__stab_find` stops recognising it as a group at all.
#[test]
fn test_write_h5py_group_attributes() {
    let path = tmp_path("h5py_group_attrs");
    let mut writer = FileWriter::new();
    writer
        .write_dataset_f64("/a/b/inner", &[1.0], &[1])
        .unwrap();
    writer
        .write_string_attr("/a/b", "title", "nested group")
        .unwrap();
    writer.write_f64_attr("/a/b", "scale", 0.25).unwrap();
    writer.write_i64_attr("/a/b", "count", i64::MAX).unwrap();
    writer.write_i32_attr("/a/b", "dimid", -3).unwrap();
    writer
        .write_i64_array_attr("/a/b", "valid_range", &[i64::MIN, 0, i64::MAX])
        .unwrap();
    writer
        .write_f64_array_attr("/a/b", "bounds", &[-1.5, 0.0, 2.25])
        .unwrap();
    writer
        .write_string_array_attr(
            "/a/b",
            "flag_meanings",
            &["low", "medium", "high-and-longest"],
        )
        .unwrap();
    // The group above carries one too, so a mix-up between the two would show.
    writer
        .write_string_attr("a", "title", "outer group")
        .unwrap();
    writer.build(&path).unwrap();

    let path_lit = path.display();
    let script = format!(
        "import h5py\n\
         def s(x):\n\
         \x20   return x.decode() if isinstance(x, bytes) else str(x)\n\
         f = h5py.File({path_lit:?}, 'r')\n\
         g = f['/a/b']\n\
         assert sorted(g.attrs.keys()) == \
         ['bounds', 'count', 'dimid', 'flag_meanings', 'scale', 'title', 'valid_range'], \
         repr(sorted(g.attrs.keys()))\n\
         assert s(g.attrs['title']) == 'nested group', repr(g.attrs['title'])\n\
         assert float(g.attrs['scale']) == 0.25, repr(g.attrs['scale'])\n\
         assert int(g.attrs['count']) == 2**63 - 1, repr(g.attrs['count'])\n\
         assert int(g.attrs['dimid']) == -3, repr(g.attrs['dimid'])\n\
         vr = [int(x) for x in g.attrs['valid_range']]\n\
         assert vr == [-2**63, 0, 2**63 - 1], repr(vr)\n\
         b = [float(x) for x in g.attrs['bounds']]\n\
         assert b == [-1.5, 0.0, 2.25], repr(b)\n\
         fm = [s(x) for x in g.attrs['flag_meanings']]\n\
         assert fm == ['low', 'medium', 'high-and-longest'], repr(fm)\n\
         assert s(f['a'].attrs['title']) == 'outer group', repr(f['a'].attrs['title'])\n\
         assert list(f['/a/b'].keys()) == ['inner'], repr(list(f['/a/b'].keys()))\n\
         print('h5py read', len(g.attrs), 'sub-group attributes of every kind')\n"
    );
    h5py_check(&path, &script, "h5py sub-group attribute interop");
}

// ---------------------------------------------------------------------------
// Test 25: h5py must read non-string and array attributes on the root group
// ---------------------------------------------------------------------------

/// `write_root_str_attr` took a `&str`, so an integer or float global
/// attribute — `geospatial_lat_min`, a `_FillValue`, a CF `valid_range` — could
/// not be attached to the root group at all.
#[test]
fn test_write_h5py_root_group_attributes() {
    let path = tmp_path("h5py_root_attrs");
    let mut writer = FileWriter::new();
    writer.write_dataset_f64("data", &[1.0, 2.0], &[2]).unwrap();
    writer.write_root_str_attr("Conventions", "CF-1.8");
    writer
        .write_i64_attr("/", "total_records", 9_000_000_000)
        .unwrap();
    writer.write_f64_attr("/", "resolution", 0.125).unwrap();
    writer.write_i32_attr("/", "revision", -7).unwrap();
    writer
        .write_f64_array_attr("/", "geospatial_bounds", &[-90.0, -180.0, 90.0, 180.0])
        .unwrap();
    writer
        .write_string_array_attr("/", "sources", &["buoy", "satellite"])
        .unwrap();
    writer.build(&path).unwrap();

    let path_lit = path.display();
    let script = format!(
        "import h5py\n\
         def s(x):\n\
         \x20   return x.decode() if isinstance(x, bytes) else str(x)\n\
         f = h5py.File({path_lit:?}, 'r')\n\
         a = f.attrs\n\
         assert sorted(a.keys()) == ['Conventions', 'geospatial_bounds', 'resolution', \
         'revision', 'sources', 'total_records'], repr(sorted(a.keys()))\n\
         assert s(a['Conventions']) == 'CF-1.8', repr(a['Conventions'])\n\
         assert int(a['total_records']) == 9000000000, repr(a['total_records'])\n\
         assert float(a['resolution']) == 0.125, repr(a['resolution'])\n\
         assert int(a['revision']) == -7, repr(a['revision'])\n\
         gb = [float(x) for x in a['geospatial_bounds']]\n\
         assert gb == [-90.0, -180.0, 90.0, 180.0], repr(gb)\n\
         assert [s(x) for x in a['sources']] == ['buoy', 'satellite'], repr(a['sources'])\n\
         assert f['/'].attrs.keys() == a.keys()\n\
         d = [float(x) for x in f['data'][:]]\n\
         assert d == [1.0, 2.0], repr(d)\n\
         print('h5py read', len(a), 'root-group attributes')\n"
    );
    h5py_check(&path, &script, "h5py root-group attribute interop");
}

// ---------------------------------------------------------------------------
// Test 26: a sub-group whose own links span several symbol table nodes
// ---------------------------------------------------------------------------

/// Nesting must compose with the symbol-table geometry at every level.
///
/// A group's links are chunked eight to a 328-byte SNOD and indexed by a
/// B-tree; the address a parent caches for a child is that tree's *root*, not
/// its first node.  With one SNOD per group the two coincide, so nothing below
/// nine links in a sub-group can tell them apart.  Twenty datasets and twenty
/// sub-groups, at two different depths, do.
#[test]
fn test_write_h5py_subgroup_spanning_multiple_snods() {
    let path = tmp_path("h5py_nested_symtab");
    let mut writer = FileWriter::new();
    for i in 0..20usize {
        writer
            .write_dataset_i32(&format!("outer/leaf{i:02}"), &[i as i32], &[1])
            .unwrap();
        writer
            .write_dataset_i32(&format!("outer/deep/g{i:02}/value"), &[i as i32 * 10], &[1])
            .unwrap();
    }
    writer.build(&path).unwrap();

    let path_lit = path.display();
    let script = format!(
        "import h5py\n\
         f = h5py.File({path_lit:?}, 'r')\n\
         outer = f['outer']\n\
         assert sorted(outer.keys()) == sorted(['deep'] + ['leaf%02d' % i for i in range(20)]), \
         repr(sorted(outer.keys()))\n\
         deep = f['/outer/deep']\n\
         assert sorted(deep.keys()) == sorted(['g%02d' % i for i in range(20)]), \
         repr(sorted(deep.keys()))\n\
         for i in range(20):\n\
         \x20   leaf = 'leaf%02d' % i\n\
         \x20   assert leaf in outer, leaf + ' not found by H5Lexists'\n\
         \x20   assert int(outer[leaf][0]) == i, leaf\n\
         \x20   sub = 'g%02d' % i\n\
         \x20   assert sub in deep, sub + ' not found by H5Lexists'\n\
         \x20   assert int(f['/outer/deep/' + sub + '/value'][0]) == i * 10, sub\n\
         print('h5py looked up 20 links in each of two nested symbol tables')\n"
    );
    h5py_check(
        &path,
        &script,
        "h5py nested multi-SNOD symbol table interop",
    );
}

// ---------------------------------------------------------------------------
// Test 27: h5py must read a DEFLATE-compressed dataset, filter and all
// ---------------------------------------------------------------------------

/// libhdf5 must decompress what we compressed, and agree about how.
///
/// Our own reader inverts the same filter pipeline our writer encodes, so a
/// self-consistent mistake — a level recorded in the wrong client-data slot, a
/// pipeline message version libhdf5 does not accept — round-trips perfectly
/// through oxih5 and produces a file no other tool can open.  `dset.compression`
/// and `dset.compression_opts` are read out of the pipeline message by libhdf5
/// itself, so they pin the encoding and not just the bytes.
#[test]
fn test_write_h5py_deflate_roundtrip() {
    let path = tmp_path("h5py_deflate");
    // 32 KiB of repetitive f64: comfortably past the point where compression
    // beats the fixed-width chunk index it costs.
    let values: Vec<f64> = (0..4096).map(|i| f64::from(i % 17) * 0.25).collect();
    let grid: Vec<i32> = (0..24).collect();

    let mut writer = FileWriter::new();
    writer
        .write_dataset_f64("readings", &values, &[4096])
        .unwrap();
    writer.set_deflate("readings", 6).unwrap();
    // A 2-D dataset at a different level, so a hard-coded level or a rank
    // assumption in the pipeline encoder would show.
    writer.write_dataset_i32("grid", &grid, &[4, 6]).unwrap();
    writer.set_deflate("grid", 9).unwrap();
    // And an uncompressed neighbour, which must stay uncompressed.
    writer
        .write_dataset_f64("plain", &[1.5, 2.5], &[2])
        .unwrap();
    writer.build(&path).unwrap();

    let path_lit = path.display();
    let script = format!(
        "import h5py, numpy as np\n\
         f = h5py.File({path_lit:?}, 'r')\n\
         r = f['readings']\n\
         assert r.compression == 'gzip', repr(r.compression)\n\
         assert r.compression_opts == 6, repr(r.compression_opts)\n\
         assert r.chunks == (4096,), repr(r.chunks)\n\
         assert r.shape == (4096,), repr(r.shape)\n\
         want = [(i % 17) * 0.25 for i in range(4096)]\n\
         assert list(r[:]) == want, 'readings differ'\n\
         g = f['grid']\n\
         assert g.compression == 'gzip', repr(g.compression)\n\
         assert g.compression_opts == 9, repr(g.compression_opts)\n\
         assert g.chunks == (4, 6), repr(g.chunks)\n\
         assert g[...].tolist() == np.arange(24, dtype='int32').reshape(4, 6).tolist()\n\
         p = f['plain']\n\
         assert p.compression is None, repr(p.compression)\n\
         assert p.chunks is None, repr(p.chunks)\n\
         assert list(p[:]) == [1.5, 2.5], repr(list(p[:]))\n\
         print('h5py decompressed', r.size + g.size, 'gzip elements')\n"
    );
    h5py_check(&path, &script, "h5py deflate interop");
}

// ---------------------------------------------------------------------------
// Test 28: h5py must read a chunked dataset at all
// ---------------------------------------------------------------------------

/// The chunk B-tree, which libhdf5 refused outright until W1e.
///
/// A node was sized for the one chunk it held — 80 bytes for a 1-D dataset —
/// while libhdf5 computes the image size it reads from a *compile-time*
/// `K` of 32 and asked for 2096, running past the end of the file:
/// `addr overflow, addr = 3000, size = 2096, eoa = 3112`.  Every unlimited
/// dataset this writer had ever produced was unopenable by anything but oxih5.
///
/// The 2-D case additionally pins the chunk-shape completion: `oxinetcdf` passes
/// one chunk extent for a variable of any rank, which used to declare chunk
/// dimensions `[5, 1]` — five elements — over a chunk holding all fifteen.
/// h5py reads the declaration, so `chunks == (5, 3)` is the assertion that
/// catches it; the values then confirm nothing was lost.
#[test]
fn test_write_h5py_chunked_unlimited() {
    let path = tmp_path("h5py_chunked");
    let dtype = oxih5::Dtype::Float {
        size: 8,
        order: oxih5::ByteOrder::Little,
    };
    let dt_i32 = oxih5::Dtype::Int {
        size: 4,
        signed: true,
        order: oxih5::ByteOrder::Little,
    };
    let raw: Vec<u8> = (0..10)
        .flat_map(|i| (f64::from(i) * 0.5).to_le_bytes())
        .collect();
    let raw2: Vec<u8> = (0..15i32).flat_map(i32::to_le_bytes).collect();

    let mut writer = FileWriter::new();
    writer
        .create_dataset_unlimited("time", &[10], &[10], &dtype, &raw)
        .unwrap();
    // Deliberately short, exactly as `oxinetcdf` passes it.
    writer
        .create_dataset_unlimited("grid", &[5, 3], &[5], &dt_i32, &raw2)
        .unwrap();
    writer.build(&path).unwrap();

    let path_lit = path.display();
    let script = format!(
        "import h5py, numpy as np\n\
         f = h5py.File({path_lit:?}, 'r')\n\
         t = f['time']\n\
         assert t.shape == (10,), repr(t.shape)\n\
         assert t.chunks == (10,), repr(t.chunks)\n\
         assert t.maxshape == (None,), repr(t.maxshape)\n\
         assert list(t[:]) == [i * 0.5 for i in range(10)], repr(list(t[:]))\n\
         g = f['grid']\n\
         assert g.shape == (5, 3), repr(g.shape)\n\
         assert g.chunks == (5, 3), 'declared chunk shape is ' + repr(g.chunks)\n\
         assert g.maxshape == (None, 3), repr(g.maxshape)\n\
         assert g[...].tolist() == np.arange(15, dtype='int32').reshape(5, 3).tolist(), \\\n\
         'grid values: ' + repr(g[...].tolist())\n\
         print('h5py read', t.size + g.size, 'chunked elements')\n"
    );
    h5py_check(&path, &script, "h5py chunked B-tree interop");
}

// ---------------------------------------------------------------------------
// Test 29: h5py must read a genuinely tiled dataset, edge chunks and all
// ---------------------------------------------------------------------------

/// Real tiling: N chunks, an N-entry index, and edge chunks that hang over.
///
/// Every shape here is chosen so that the chunk size does **not** divide the
/// extent, because that is the only case where the two hard parts show:
/// an edge chunk is stored at its full declared volume with fill in the
/// overhang, and the chunks are ordered row-major over the chunk grid with
/// dimension 0 most significant.  Getting the order backwards yields a file
/// that still holds every value, just in transposed positions — which oxih5's
/// own reader reproduces faithfully, because it assembles chunks by the same
/// rule the writer placed them by.  `iter_chunks()` asks libhdf5 for the chunk
/// origins it decoded out of the B-tree keys, so it pins the order directly.
///
/// `multi_level` has 150 chunks against a node capacity of 64, so its index is
/// two levels deep and the layout message must point at the root rather than at
/// the first leaf — an error that would return the first 64 chunks and fill for
/// the rest.
#[test]
fn test_write_h5py_multi_chunk_tiling() {
    let path = tmp_path("h5py_tiling");
    let dt_i32 = oxih5::Dtype::Int {
        size: 4,
        signed: true,
        order: oxih5::ByteOrder::Little,
    };

    let mut writer = FileWriter::new();
    let mut tiled = |name: &str, n: i32, shape: &[usize], chunk: &[usize], deflate: bool| {
        let raw: Vec<u8> = (0..n).flat_map(i32::to_le_bytes).collect();
        writer
            .create_dataset_unlimited(name, shape, chunk, &dt_i32, &raw)
            .unwrap();
        if deflate {
            writer.set_deflate(name, 6).unwrap();
        }
    };
    tiled("ragged_1d", 7, &[7], &[3], true);
    tiled("ragged_2d", 15, &[5, 3], &[2, 2], true);
    tiled("plain_2d", 15, &[5, 3], &[2, 2], false);
    tiled("cube", 60, &[3, 4, 5], &[2, 3, 2], true);
    tiled("multi_level", 300, &[300], &[2], true);
    writer.build(&path).unwrap();

    let path_lit = path.display();
    let script = format!(
        "import h5py, numpy as np\n\
         f = h5py.File({path_lit:?}, 'r')\n\
         cases = [('ragged_1d', 7, (7,), (3,)),\n\
         \x20        ('ragged_2d', 15, (5, 3), (2, 2)),\n\
         \x20        ('plain_2d', 15, (5, 3), (2, 2)),\n\
         \x20        ('cube', 60, (3, 4, 5), (2, 3, 2)),\n\
         \x20        ('multi_level', 300, (300,), (2,))]\n\
         for name, n, shape, chunks in cases:\n\
         \x20   d = f[name]\n\
         \x20   assert d.shape == shape, name + ': shape ' + repr(d.shape)\n\
         \x20   assert d.chunks == chunks, name + ': chunks ' + repr(d.chunks)\n\
         \x20   want = np.arange(n, dtype='int32').reshape(shape)\n\
         \x20   assert np.array_equal(d[...], want), name + ': ' + repr(d[...])\n\
         \x20   seen = sum(1 for _ in d.iter_chunks())\n\
         \x20   want_n = int(np.prod([-(-s // c) for s, c in zip(shape, chunks)]))\n\
         \x20   assert seen == want_n, name + ': libhdf5 found ' + str(seen) + ' chunks, not ' + str(want_n)\n\
         origins = [tuple(s.start for s in sl) for sl in f['ragged_2d'].iter_chunks()]\n\
         assert origins == [(0, 0), (0, 2), (2, 0), (2, 2), (4, 0), (4, 2)], repr(origins)\n\
         assert sum(1 for _ in f['multi_level'].iter_chunks()) == 150\n\
         print('h5py walked', sum(sum(1 for _ in f[c[0]].iter_chunks()) for c in cases), 'chunks')\n"
    );
    h5py_check(&path, &script, "h5py multi-chunk tiling interop");
}
