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
// Test 8: capacity limit — 9th dataset must fail
// ---------------------------------------------------------------------------
#[test]
fn test_write_capacity_exceeded() {
    let mut writer = FileWriter::new();
    for i in 0..8usize {
        writer
            .write_dataset_f32(&format!("d{i}"), &[1.0f32], &[1])
            .unwrap();
    }
    assert!(
        writer
            .write_dataset_f32("overflow", &[1.0f32], &[1])
            .is_err(),
        "9th dataset should return an error"
    );
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
// Test 11: validation — name contains slash
// ---------------------------------------------------------------------------
#[test]
fn test_write_slash_in_name_rejected() {
    let mut writer = FileWriter::new();
    let result = writer.write_dataset_f32("a/b", &[1.0f32], &[1]);
    assert!(result.is_err(), "name containing '/' should be rejected");
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
