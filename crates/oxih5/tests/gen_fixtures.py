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

# ---------------------------------------------------------------------------
# Fixture 3: Virtual Dataset (VDS) fixtures — require libver='latest' so h5py
# emits the version-4 "virtual" data-layout message + global-heap mapping.
#   vds_source.h5 : source datasets referenced by the virtual datasets
#   vds_simple.h5 : /virt  (float64 [6])  full "all" mapping of source/source
#   vds_concat.h5 : /cat   (int32   [8])  srcA -> [0:4], srcB -> [4:8]
# ---------------------------------------------------------------------------
source_path = os.path.join(fixtures_dir, "vds_source.h5")
with h5py.File(source_path, "w", libver="latest") as f:
    f.create_dataset("source", data=np.arange(1, 7, dtype="float64"))
    f.create_dataset("srcA", data=np.array([10, 11, 12, 13], dtype="int32"))
    f.create_dataset("srcB", data=np.array([20, 21, 22, 23], dtype="int32"))
print(f"Generated {source_path}")

simple_layout = h5py.VirtualLayout(shape=(6,), dtype="float64")
simple_layout[:] = h5py.VirtualSource("vds_source.h5", "source", shape=(6,))
vds_simple_path = os.path.join(fixtures_dir, "vds_simple.h5")
with h5py.File(vds_simple_path, "w", libver="latest") as f:
    f.create_virtual_dataset("virt", simple_layout, fillvalue=0.0)
print(f"Generated {vds_simple_path}")

concat_layout = h5py.VirtualLayout(shape=(8,), dtype="int32")
concat_layout[0:4] = h5py.VirtualSource("vds_source.h5", "srcA", shape=(4,))
concat_layout[4:8] = h5py.VirtualSource("vds_source.h5", "srcB", shape=(4,))
vds_concat_path = os.path.join(fixtures_dir, "vds_concat.h5")
with h5py.File(vds_concat_path, "w", libver="latest") as f:
    f.create_virtual_dataset("cat", concat_layout, fillvalue=-1)
print(f"Generated {vds_concat_path}")

# ---------------------------------------------------------------------------
# Fixture 4: vlen_str_chunked.h5 — chunked variable-length UTF-8 string dataset.
# chunks=(3,) over 10 elements so several elements straddle chunk boundaries.
# ---------------------------------------------------------------------------
vlen_chunked_path = os.path.join(fixtures_dir, "vlen_str_chunked.h5")
vlen_dt = h5py.string_dtype(encoding="utf-8")
with h5py.File(vlen_chunked_path, "w", libver="earliest") as f:
    ds = f.create_dataset("words", (10,), dtype=vlen_dt, chunks=(3,))
    ds[...] = [f"item-{i:02d}-value" for i in range(10)]
print(f"Generated {vlen_chunked_path}")

# ---------------------------------------------------------------------------
# Fixture 5: soft link that points through to an external link.
#   soft_ext_target.h5 : /payload (int32 [4])
#   soft_ext_main.h5   : /ext  = ExternalLink(soft_ext_target.h5, /payload)
#                        /soft = SoftLink(/ext)
# ---------------------------------------------------------------------------
soft_target_path = os.path.join(fixtures_dir, "soft_ext_target.h5")
with h5py.File(soft_target_path, "w", libver="latest") as f:
    f.create_dataset("payload", data=np.array([7, 8, 9, 10], dtype="int32"))
print(f"Generated {soft_target_path}")

soft_main_path = os.path.join(fixtures_dir, "soft_ext_main.h5")
with h5py.File(soft_main_path, "w", libver="latest") as f:
    f["ext"] = h5py.ExternalLink("soft_ext_target.h5", "/payload")
    f["soft"] = h5py.SoftLink("/ext")
print(f"Generated {soft_main_path}")

print("Done.")
