#!/usr/bin/env bash
set -euo pipefail
RESULT=$(cargo tree --target x86_64-unknown-linux-gnu --edges normal 2>/dev/null | grep -E '(hdf5-sys v|libhdf5-sys v|hdf5-metno-sys v|hdf5-src v|netcdf-sys v|hdf5 v|hdf5-metno v)' || true)
if [ -n "$RESULT" ]; then
  echo "FFI AUDIT FAILED: forbidden crates in tree:"
  echo "$RESULT"
  exit 1
fi
echo "FFI audit PASSED"
