# oxinetcdf

**oxinetcdf** is a Pure-Rust NetCDF-4 conventions reader/writer built atop OxiH5.
It implements the full NetCDF-4/HDF5 conventions layer — no `libnetcdf`, no `libhdf5`,
no C FFI — and is a member of the [OxiH5](https://github.com/cool-japan/oxih5) workspace.

---

## Features

- `NcFile::open(path)` / `open_from_bytes(&[u8])` — open any NetCDF-4 file (HDF5 backend)
- Full `NcGroup` / `NcVariable` / `NcDimension` / `NcAxis` / `NcAttribute` model
- NetCDF-4 convention resolution: `DIMENSION_SCALE`, `_Netcdf4Dimid`, `DIMENSION_LIST`
  object-reference axis linkage
- Deep group hierarchy with cycle detection (`resolve_group_deep`, `MAX_GROUP_DEPTH=64`)
- Cross-group shared dimensions (two-phase scan: global dim registry)
- CF conventions: `coordinates_of`, `bounds_of`, `grid_mapping_of`; CF-1.7 `group:var` form
- `_FillValue`-aware masked reads: `read_f64_masked` (NaN for fill), `read_i64_masked`
  (Option for fill)
- NC_STRING variable support: `read_strings` for vlen string variables
- `NcType` enum covering all 11 HDF5 datatype classes
- `NcFileWriter` — creates NetCDF-4-compliant HDF5 files:
  - `def_dim` / `def_dim_unlimited` / `def_var` / `put_var_f64/i32`
  - `put_vara_f64/i32` (unlimited append)
  - `def_var_strings` / `put_var_strings` (vlen string datasets)
  - `put_att_str` / `set_classic_mode` (`_nc3_strict`)

---

## Usage

```rust
use oxinetcdf::{NcFile, NcFileWriter};

// Reading
let nc = NcFile::open("data.nc")?;
let root = nc.root_group()?;
for var in root.variables() {
    println!("{}: {:?}", var.name(), var.shape());
}
let lat = root.variable("lat").unwrap();
let values = lat.read_f64(&nc)?;

// Writing
let mut w = NcFileWriter::create("out.nc")?;
w.def_dim("time", 0)?;   // 0 = unlimited
w.def_var("temp", &["time"], oxinetcdf::NcType::Float64)?;
w.put_var_f64("temp", &[20.5, 21.0, 19.8])?;
w.close()?;
```

---

## Feature Flags

| Flag | Default | Description |
|---|---|---|
| `ndarray` | off | Enable `ndarray::ArrayD` bridge (delegates to `oxih5/ndarray`) |
| `e2e` | off | Enable tests requiring real `.nc` fixtures or external CLI tools |

---

## Policy Compliance

- Pure Rust: no `libnetcdf`, no `libhdf5`, no C/C++ dependencies in default features.
- All compression via `oxiarc-deflate` / `oxiarc-szip` (COOLJAPAN policy).
- No `unwrap()` in production code paths.

---

## License

Apache-2.0 — Copyright COOLJAPAN OU (Team Kitasan)
