//! Lane NCR — backward compatibility.
//!
//! Files written by oxih5/oxinetcdf 0.2.x encode `DIMENSION_LIST` as a flat
//! `H5T_REFERENCE` array (not the netCDF-C `H5T_VLEN{H5T_REFERENCE}` form), give
//! their dimension scales real `NAME`s (so they are coordinate variables), and
//! carry no `_Netcdf4Coordinates`.  The reader changes for B012 add new
//! resolution paths but MUST keep reading these older bytes exactly as before.
//!
//! `prefix_021_basic.nc` is such a fixture, produced by the 0.2.x writer before
//! this wave's changes.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use oxinetcdf::NcFile;

#[test]
fn reads_oxih5_021_flat_dimension_list_fixture() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("prefix_021_basic.nc");
    if !path.exists() {
        eprintln!("fixture {} missing — skipping", path.display());
        return;
    }

    let nc = NcFile::open(&path).expect("open 0.2.x fixture");
    let root = nc.root_group().expect("root_group");

    // Two dimensions with the original lengths — no phantoms.
    assert_eq!(
        root.dimensions.len(),
        2,
        "expected 2 dims, got {:?}",
        root.dimensions
    );
    assert!(!root.dimensions.iter().any(|d| d.name.starts_with("phony")));
    assert!(root
        .dimensions
        .iter()
        .any(|d| d.name == "lat" && d.len == 4));
    assert!(root
        .dimensions
        .iter()
        .any(|d| d.name == "lon" && d.len == 8));

    // The 0.2.x writer emits real-NAME coordinate variables for its dims, so
    // lat/lon ARE variables here (unlike netCDF-C pure-dimension placeholders).
    let lat = root.variable("lat").expect("lat coordinate variable");
    assert!(lat.is_coordinate);
    assert_eq!(lat.dim_names(), vec!["lat"]);

    let lon = root.variable("lon").expect("lon coordinate variable");
    assert!(lon.is_coordinate);
    assert_eq!(lon.dim_names(), vec!["lon"]);

    // The 2-D data variable binds to [lat, lon] via the flat DIMENSION_LIST.
    let temp = root.variable("temp").expect("temp variable");
    assert_eq!(temp.shape, vec![4, 8]);
    assert_eq!(
        temp.dim_names(),
        vec!["lat", "lon"],
        "temp must still bind to [lat, lon] from the flat DIMENSION_LIST"
    );

    // The 1-D data variable binds to [lat].
    let flags = root.variable("flags").expect("flags variable");
    assert_eq!(flags.dim_names(), vec!["lat"]);
}
