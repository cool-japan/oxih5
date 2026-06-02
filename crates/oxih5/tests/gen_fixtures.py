"""
Generate HDF5 fixture files for OxiH5 integration tests.

Requires: pip install h5py numpy

Usage:
    python3 crates/oxih5/tests/gen_fixtures.py

Generates:
    crates/oxih5/tests/fixtures/nested_groups.h5
    crates/oxih5/tests/fixtures/with_attrs.h5

IMPORTANT: libver='earliest' is required so h5py writes old-style B-tree v1
group indices and inline attribute messages (0x000C).  Without it, h5py
defaults to new-style fractal-heap + B-tree v2 groups which the current parser
does not yet fully support.
"""

import os
import sys

try:
    import h5py
    import numpy as np
except ImportError:
    print("ERROR: h5py and numpy are required.  Install with: pip install h5py numpy")
    sys.exit(1)

fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(fixtures_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Fixture 1: nested_groups.h5
# Structure: /sensors/imu/accel (float32 [3]), /sensors/gps/coords (float64 [2])
# ---------------------------------------------------------------------------
nested_path = os.path.join(fixtures_dir, "nested_groups.h5")
with h5py.File(nested_path, "w", libver="earliest") as f:
    sensors = f.create_group("sensors")
    imu = sensors.create_group("imu")
    imu.create_dataset("accel", data=np.array([1.0, 2.0, 3.0], dtype="float32"))
    gps = sensors.create_group("gps")
    gps.create_dataset("coords", data=np.array([48.123, 11.456], dtype="float64"))
print(f"Generated {nested_path}")

# ---------------------------------------------------------------------------
# Fixture 2: with_attrs.h5
# /temperature (float32 [3]) with 'units' and 'scale_factor' attributes
# /metadata (group) with 'version' attribute
# ---------------------------------------------------------------------------
attrs_path = os.path.join(fixtures_dir, "with_attrs.h5")
with h5py.File(attrs_path, "w", libver="earliest") as f:
    ds = f.create_dataset("temperature", data=np.array([20.0, 21.0, 22.0], dtype="float32"))
    ds.attrs["units"] = "Celsius"
    ds.attrs["scale_factor"] = np.float32(1.0)
    g = f.create_group("metadata")
    g.attrs["version"] = np.int32(1)
print(f"Generated {attrs_path}")

print("Done.")
