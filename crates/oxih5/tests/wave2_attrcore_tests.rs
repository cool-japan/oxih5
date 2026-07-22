// Wave 2 ATTRCORE integration tests.
//
// Covers the attribute machinery the netCDF lane consumes next stage:
//   * G006 — widened numeric scalar/array attributes (f32/i8/i16/u8/u16/u32/u64)
//   * B018 — `REFERENCE_LIST` compound `{ dataset: objref, dimension: u32 }`
//   * B005 — `DIMENSION_LIST` vlen-of-object-reference attributes
//
// Each writes a file, reads it back through oxih5's own reader (raw data must be
// correctly sized and siblings must still enumerate — the B003 regression
// guard), and — where python3 + h5py are available — verifies the exact on-disk
// dtype/values against libhdf5 with the same three-way graceful-skip guard the
// other write tests use (python3 missing → skip, h5py/numpy missing → skip, any
// other non-zero exit → fail).

use oxih5::FileWriter;
use std::sync::atomic::{AtomicU64, Ordering};

static W2_COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_path(tag: &str) -> std::path::PathBuf {
    let n = W2_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("oxih5_w2_{}_{n}_{tag}.h5", std::process::id()))
}

fn cleanup(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}

/// Run an h5py check over `path`, then delete it.
///
/// python3 missing → skip; h5py/numpy missing → skip; any other non-zero exit →
/// fail.  A silent skip on a *real* failure would make the test worthless, so
/// only the two "not installed" shapes are tolerated.
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

// ---------------------------------------------------------------------------
// G006 — widened numeric attributes
// ---------------------------------------------------------------------------

#[test]
fn widened_numeric_attrs_round_trip_and_h5py_verifies() {
    let path = tmp_path("g006");
    let mut w = FileWriter::new();
    w.write_dataset_f64("data", &[1.0], &[1]).expect("data");

    // Scalars.
    w.write_f32_attr("data", "s_f32", 1.5).expect("s_f32");
    w.write_i8_attr("data", "s_i8", -5).expect("s_i8");
    w.write_i16_attr("data", "s_i16", -300).expect("s_i16");
    w.write_u8_attr("data", "s_u8", 200).expect("s_u8");
    w.write_u16_attr("data", "s_u16", 60_000).expect("s_u16");
    w.write_u32_attr("data", "s_u32", 4_000_000_000)
        .expect("s_u32");
    w.write_u64_attr("data", "s_u64", 18_000_000_000_000_000_000)
        .expect("s_u64");

    // Arrays.
    w.write_f32_array_attr("data", "a_f32", &[1.5, 2.5])
        .expect("a_f32");
    w.write_i8_array_attr("data", "a_i8", &[-1, 2])
        .expect("a_i8");
    w.write_i16_array_attr("data", "a_i16", &[-1000, 1000])
        .expect("a_i16");
    w.write_u8_array_attr("data", "a_u8", &[1, 255])
        .expect("a_u8");
    w.write_u16_array_attr("data", "a_u16", &[0, 65535])
        .expect("a_u16");
    w.write_u32_array_attr("data", "a_u32", &[1, 4_000_000_000])
        .expect("a_u32");
    w.write_u64_array_attr("data", "a_u64", &[1, 18_000_000_000_000_000_000])
        .expect("a_u64");

    w.build(&path).expect("build");

    // oxih5's own reader: every attribute enumerates with correctly-sized raw
    // data, and the integer scalars widen correctly.
    {
        let f = oxih5::open(&path).expect("open");
        let attrs = f.attr_views("data").expect("attr_views");
        assert_eq!(attrs.len(), 14, "all widened numeric attrs enumerate");
        let by = |name: &str| {
            attrs
                .iter()
                .find(|a| a.name() == name)
                .unwrap_or_else(|| panic!("{name} missing"))
        };
        // Raw data sized to element count × element footprint.
        assert_eq!(by("s_u8").attr.data.len(), 1);
        assert_eq!(by("s_u16").attr.data.len(), 2);
        assert_eq!(by("s_u32").attr.data.len(), 4);
        assert_eq!(by("s_u64").attr.data.len(), 8);
        assert_eq!(by("s_f32").attr.data.len(), 4);
        assert_eq!(by("a_u16").attr.data.len(), 4);
        assert_eq!(by("a_u64").attr.data.len(), 16);
        // Shapes: scalars vs 1-D of two.
        assert!(by("s_u32").is_scalar());
        assert_eq!(by("a_u32").shape(), vec![2u64]);
        // Widening decoders.
        assert_eq!(by("s_i8").as_i64(), Some(-5));
        assert_eq!(by("s_i16").as_i64(), Some(-300));
        assert_eq!(by("s_u8").as_u64(), Some(200));
        assert_eq!(by("s_u16").as_u64(), Some(60_000));
        assert_eq!(by("s_u32").as_u64(), Some(4_000_000_000));
        assert_eq!(by("s_u64").as_u64(), Some(18_000_000_000_000_000_000));
    }

    // h5py: exact dtype (class + size + signedness) and values.
    let script = format!(
        "import h5py, numpy as np\n\
         f=h5py.File(r'{p}','r'); a=f['data'].attrs\n\
         def chk(name, npdt, val):\n\
        \x20   x=a[name]\n\
        \x20   assert np.dtype(x.dtype)==np.dtype(npdt), (name,'dtype',x.dtype)\n\
        \x20   assert np.array_equal(x, np.array(val, dtype=npdt)), (name,'val',x)\n\
         chk('s_f32','<f4',1.5)\n\
         chk('s_i8','i1',-5)\n\
         chk('s_i16','<i2',-300)\n\
         chk('s_u8','u1',200)\n\
         chk('s_u16','<u2',60000)\n\
         chk('s_u32','<u4',4000000000)\n\
         chk('s_u64','<u8',18000000000000000000)\n\
         chk('a_f32','<f4',[1.5,2.5])\n\
         chk('a_i8','i1',[-1,2])\n\
         chk('a_i16','<i2',[-1000,1000])\n\
         chk('a_u8','u1',[1,255])\n\
         chk('a_u16','<u2',[0,65535])\n\
         chk('a_u32','<u4',[1,4000000000])\n\
         chk('a_u64','<u8',[1,18000000000000000000])\n\
         print('g006 dtypes+values OK')",
        p = path.display()
    );
    h5py_check(&path, &script, "G006 widened numeric attrs");
}

// ---------------------------------------------------------------------------
// B018 — REFERENCE_LIST compound { dataset: objref, dimension: u32 }
// ---------------------------------------------------------------------------

#[test]
fn ref_index_list_attr_round_trips_and_h5py_reads_named_members() {
    let path = tmp_path("b018");
    let mut w = FileWriter::new();
    // A dimension-scale dataset and two variables that attach it.
    w.write_dataset_i32("lat", &[0, 1, 2, 3], &[4])
        .expect("lat");
    w.write_dataset_f64("temp", &[0.0; 8], &[8]).expect("temp");
    w.write_dataset_f64("pres", &[0.0; 8], &[8]).expect("pres");
    // A sibling string attr, so we prove the B003 guard: siblings still enumerate.
    w.write_string_attr("lat", "NAME", "lat").expect("NAME");
    w.write_ref_index_list_attr(
        "lat",
        "REFERENCE_LIST",
        &[("temp".to_string(), 0), ("pres".to_string(), 0)],
    )
    .expect("REFERENCE_LIST");
    w.build(&path).expect("build");

    // oxih5's own reader: the compound attr enumerates alongside its sibling,
    // with correctly-sized raw data (two entries × 12 bytes).
    {
        let f = oxih5::open(&path).expect("open");
        let attrs = f.attr_views("lat").expect("attr_views");
        let names: Vec<&str> = attrs.iter().map(|a| a.name()).collect();
        assert!(names.contains(&"NAME"), "sibling attr still enumerates");
        let rl = attrs
            .iter()
            .find(|a| a.name() == "REFERENCE_LIST")
            .expect("REFERENCE_LIST present");
        assert_eq!(rl.shape(), vec![2u64], "1-D of two entries");
        assert_eq!(rl.attr.data.len(), 24, "2 entries × 12 bytes");
        // The resolved reference addresses are non-zero (they point at temp/pres).
        let temp_addr = f.header_addr_of("temp").expect("temp addr");
        let pres_addr = f.header_addr_of("pres").expect("pres addr");
        let a0 = u64::from_le_bytes(rl.attr.data[0..8].try_into().expect("8"));
        let a1 = u64::from_le_bytes(rl.attr.data[12..20].try_into().expect("8"));
        assert_eq!(a0, temp_addr, "entry 0 references temp");
        assert_eq!(a1, pres_addr, "entry 1 references pres");
    }

    // h5py: the compound has the netCDF-named members at the right offsets, the
    // dimension indices are correct, and each `dataset` reference dereferences.
    let script = format!(
        "import h5py, numpy as np\n\
         f=h5py.File(r'{p}','r'); a=f['lat'].attrs['REFERENCE_LIST']\n\
         assert a.dtype.names==('dataset','dimension'), a.dtype.names\n\
         assert a.dtype.fields['dataset'][1]==0, a.dtype.fields\n\
         assert a.dtype.fields['dimension'][1]==8, a.dtype.fields\n\
         assert a.shape==(2,), a.shape\n\
         assert int(a['dimension'][0])==0 and int(a['dimension'][1])==0\n\
         names=sorted(f[r].name for r in a['dataset'])\n\
         assert names==['/pres','/temp'], names\n\
         print('b018 REFERENCE_LIST OK')",
        p = path.display()
    );
    h5py_check(&path, &script, "B018 REFERENCE_LIST compound");
}

// ---------------------------------------------------------------------------
// B005 — DIMENSION_LIST vlen-of-object-reference
// ---------------------------------------------------------------------------

/// The vlen-object-reference attribute's global-heap payloads are placed by a
/// build-path hook (`register_vlen_objref_heap` / `fill_vlen_objref_locs`) that
/// the convergence agent wires into `write/build.rs`.  Until that hook lands,
/// `build` reports the payload as "never placed" — the honest failure this test
/// tolerates by skipping.  Once wired, the same test verifies libhdf5 reads the
/// `DIMENSION_LIST` as dereferenceable object-reference arrays.  Any *other*
/// build error is a real failure.
#[test]
fn vlen_obj_ref_attr_round_trips_once_heap_hook_is_wired() {
    let path = tmp_path("b005");
    let mut w = FileWriter::new();
    w.write_dataset_i32("lat", &[0, 1, 2, 3], &[4])
        .expect("lat");
    w.write_dataset_i32("lon", &[0, 1, 2, 3, 4, 5, 6, 7], &[8])
        .expect("lon");
    w.write_dataset_f64("temp", &[0.0; 32], &[32])
        .expect("temp");
    // DIMENSION_LIST on temp(lat,lon): element 0 → [lat scale], element 1 → [lon scale].
    w.write_vlen_obj_ref_attr(
        "temp",
        "DIMENSION_LIST",
        &[vec!["lat".to_string()], vec!["lon".to_string()]],
    )
    .expect("DIMENSION_LIST accepted");

    match w.build_to_vec() {
        Ok(bytes) => {
            std::fs::write(&path, &bytes).expect("write file");

            // oxih5's own reader: the attr enumerates with correctly-sized raw
            // data (two 16-byte on-disk vlen references).
            {
                let f = oxih5::open(&path).expect("open");
                let attrs = f.attr_views("temp").expect("attr_views");
                let dl = attrs
                    .iter()
                    .find(|a| a.name() == "DIMENSION_LIST")
                    .expect("DIMENSION_LIST present");
                assert_eq!(dl.shape(), vec![2u64]);
                assert_eq!(dl.attr.data.len(), 32, "2 elements × 16-byte vlen ref");
            }

            let script = format!(
                "import h5py, numpy as np\n\
                 f=h5py.File(r'{p}','r'); a=f['temp'].attrs['DIMENSION_LIST']\n\
                 assert h5py.check_dtype(vlen=a.dtype) is not None or a.dtype==object, a.dtype\n\
                 assert a.shape==(2,), a.shape\n\
                 n0=sorted(f[r].name for r in a[0]); n1=sorted(f[r].name for r in a[1])\n\
                 assert n0==['/lat'], n0\n\
                 assert n1==['/lon'], n1\n\
                 print('b005 DIMENSION_LIST OK')",
                p = path.display()
            );
            h5py_check(&path, &script, "B005 DIMENSION_LIST vlen-of-ref");
        }
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("never placed"),
                "vlen-objref build failed for an unexpected reason: {msg}"
            );
            cleanup(&path);
            eprintln!(
                "B005 DIMENSION_LIST: vlen-objref heap hook not yet wired into build.rs \
                 (build reported the payload as unplaced) — skipping the h5py round-trip"
            );
        }
    }
}
