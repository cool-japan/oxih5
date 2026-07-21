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

# ---------------------------------------------------------------------------
# Fixture 6: soft_links_old.h5 — soft links inside *old-style* (symbol-table)
# groups.  Written with h5py's DEFAULT libver, which is what ordinary h5py
# code produces: superblock v0, root group indexed by a version-1 B-tree with
# SNOD symbol-table nodes and a local heap.
#
# A soft link in such a group is a symbol-table entry with cache type 2: its
# object-header address field is the "undefined address" sentinel (u64::MAX)
# and the real link value is a NUL-terminated path stored in the group's local
# heap, at the offset held in the first 4 bytes of the entry's scratch pad.
#
#   /target_ds        int32[4]   the hard-linked dataset
#   /grp/inner        int32[3]   dataset one level down
#   /grp/rel_alias -> "inner"        relative soft link (resolved from /grp)
#   /alias         -> "/target_ds"   soft link to a dataset
#   /galias        -> "/grp"         soft link to a group
#   /deep_alias    -> "/grp/inner"   soft link to a nested dataset
#   /chain         -> "/alias"       soft -> soft -> dataset
#   /dangling      -> "/no_such"     dangling soft link (clean typed error)
#   /cycle_a       -> "/cycle_b"     mutually recursive pair (cycle guard)
#   /cycle_b       -> "/cycle_a"
# ---------------------------------------------------------------------------
soft_old_path = os.path.join(fixtures_dir, "soft_links_old.h5")
with h5py.File(soft_old_path, "w") as f:  # NOTE: default libver on purpose
    f.create_dataset("target_ds", data=np.array([11, 22, 33, 44], dtype="int32"))
    g = f.create_group("grp")
    g.create_dataset("inner", data=np.array([1, 2, 3], dtype="int32"))
    g["rel_alias"] = h5py.SoftLink("inner")
    f["alias"] = h5py.SoftLink("/target_ds")
    f["galias"] = h5py.SoftLink("/grp")
    f["deep_alias"] = h5py.SoftLink("/grp/inner")
    f["chain"] = h5py.SoftLink("/alias")
    f["dangling"] = h5py.SoftLink("/no_such")
    f["cycle_a"] = h5py.SoftLink("/cycle_b")
    f["cycle_b"] = h5py.SoftLink("/cycle_a")
print(f"Generated {soft_old_path}")

# ---------------------------------------------------------------------------
# Fixture 7: layout_v4.h5 — every data-layout *version 4* variant libhdf5
# emits under libver='latest'.
#
#   class 1 (contiguous): /contig, /contig2d, /scalar
#   class 0 (compact)   : /compact                       (low-level DCPL)
#   class 2 (chunked), one dataset per chunk-index type:
#       /chunk_single       index type 1 (single chunk, unfiltered)
#       /chunk_single_gzip  index type 1 (single chunk, filtered: carries the
#                           stored size + filter mask inline in the layout msg)
#       /chunk_implicit     index type 2 (implicit; needs early allocation)
#       /chunk_fa           index type 3 (fixed array)
#       /chunk_ea           index type 4 (extensible array; 1 unlimited dim)
#       /chunk_bt2          index type 5 (v2 B-tree; 2 unlimited dims)
#       /chunk_wide         index type 3 but with a chunk dimension > 255, so
#                           libhdf5 raises "dimension size encoded length" to
#                           2 bytes instead of 1
# ---------------------------------------------------------------------------
layout_v4_path = os.path.join(fixtures_dir, "layout_v4.h5")
with h5py.File(layout_v4_path, "w", libver="latest") as f:
    ds = f.create_dataset("contig", data=np.arange(5, dtype="float64"))
    ds.attrs["units"] = "meters"
    f.create_dataset("contig2d", data=np.arange(12, dtype="int32").reshape(3, 4))
    f.create_dataset("scalar", data=np.float64(3.5))

    # Compact layout is not reachable through the high-level API.
    compact_space = h5py.h5s.create_simple((4,))
    compact_dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    compact_dcpl.set_layout(h5py.h5d.COMPACT)
    compact_id = h5py.h5d.create(
        f.id, b"compact", h5py.h5t.NATIVE_INT32, compact_space, compact_dcpl
    )
    h5py.Dataset(compact_id)[...] = np.array([5, 6, 7, 8], dtype="int32")

    f.create_dataset("chunk_single", data=np.arange(6, dtype="int32"), chunks=(6,))
    f.create_dataset(
        "chunk_single_gzip",
        data=np.arange(64, dtype="int32"),
        chunks=(64,),
        compression="gzip",
    )

    # Implicit index: early allocation, no filters, no fill.
    implicit_space = h5py.h5s.create_simple((8,))
    implicit_dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    implicit_dcpl.set_chunk((4,))
    implicit_dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
    implicit_dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
    implicit_id = h5py.h5d.create(
        f.id, b"chunk_implicit", h5py.h5t.NATIVE_INT32, implicit_space, implicit_dcpl
    )
    h5py.Dataset(implicit_id)[...] = np.arange(8, dtype="int32")

    f.create_dataset("chunk_fa", data=np.arange(20, dtype="int32"), chunks=(4,))
    f.create_dataset(
        "chunk_ea", data=np.arange(10, dtype="int32"), chunks=(3,), maxshape=(None,)
    )
    f.create_dataset(
        "chunk_bt2",
        data=np.arange(12, dtype="int32").reshape(3, 4),
        chunks=(2, 2),
        maxshape=(None, None),
    )
    f.create_dataset("chunk_wide", data=np.arange(2000, dtype="int32"), chunks=(1000,))
print(f"Generated {layout_v4_path}")

# ---------------------------------------------------------------------------
# Fixture 7: deflate_chunked.h5 — DEFLATE-compressed chunked datasets written by
# h5py, so the READ side is verified against a file our own writer did not make.
#
# libver='earliest' pins the combination oxih5's FileWriter emits: superblock v0,
# object header v1, layout v3 and a *version-1* filter pipeline message indexed
# by a B-tree v1.  Without it h5py writes layout v4 with a fixed-array index and
# a v2 pipeline, which exercises a different reader path entirely.
#
#   /whole      float64 [4096]   one chunk, level 6 — what set_deflate produces
#   /tiled      int32   [7]      chunks=(3,) — an edge chunk that does not divide
#   /grid       int32   [5, 3]   chunks=(2, 2) — 2-D, both dimensions ragged
#   /max_level  uint8   [1024]   level 9
#   /shuffled   float32 [512]    shuffle + gzip, so the pipeline has two filters
#                                in an order the reader has to invert
# ---------------------------------------------------------------------------
deflate_path = os.path.join(fixtures_dir, "deflate_chunked.h5")
with h5py.File(deflate_path, "w", libver="earliest") as f:
    f.create_dataset(
        "whole",
        data=np.array([(i % 17) * 0.25 for i in range(4096)], dtype="float64"),
        chunks=(4096,),
        compression="gzip",
        compression_opts=6,
    )
    f.create_dataset(
        "tiled",
        data=np.arange(7, dtype="int32"),
        chunks=(3,),
        compression="gzip",
        compression_opts=6,
    )
    f.create_dataset(
        "grid",
        data=np.arange(15, dtype="int32").reshape(5, 3),
        chunks=(2, 2),
        compression="gzip",
        compression_opts=6,
    )
    f.create_dataset(
        "max_level",
        data=np.arange(1024, dtype="uint8") % 7,
        chunks=(1024,),
        compression="gzip",
        compression_opts=9,
    )
    f.create_dataset(
        "shuffled",
        data=np.arange(512, dtype="float32"),
        chunks=(128,),
        shuffle=True,
        compression="gzip",
        compression_opts=4,
    )
print(f"Generated {deflate_path}")

print("Done.")
