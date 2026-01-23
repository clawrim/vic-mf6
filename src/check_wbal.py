#!/usr/bin/env python3

import os
from netCDF4 import Dataset
import numpy as np

# edit these as needed
wbal_dir = "/home/abd/projects/nmhydro/VIC/outputs"
wbal_prefix = "wbal"          # wbal.YYYY-MM-DD.nc
var_name = "OUT_BASEFLOW"
dates = [
    "1990-01-01",
    "1990-01-02",
    "1990-01-03",
    "1990-01-04",
    "1990-01-05",
    "1990-01-06",
    "1990-01-07",
]

def read_stats(path, var_name):
    if not os.path.exists(path):
        print(f"missing: {path}")
        return
    with Dataset(path, "r") as ds:
        if var_name not in ds.variables:
            print(f"{var_name} not in {path}")
            return
        var = ds.variables[var_name]
        raw = var[:]

        # handle masked and fill values
        if np.ma.isMaskedArray(raw):
            arr = raw.filled(np.nan)
        else:
            arr = np.array(raw, dtype=float)

        fill = None
        for attr in ("_FillValue", "missing_value"):
            if hasattr(var, attr):
                try:
                    fill = float(getattr(var, attr))
                    break
                except Exception:
                    pass
        if fill is not None:
            arr[arr == fill] = np.nan

        # squeeze possible time dim
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        mean_val = np.nanmean(arr)
        min_val = np.nanmin(arr)
        max_val = np.nanmax(arr)
        print(f"{os.path.basename(path)}: mean={mean_val:.6f}, min={min_val:.6f}, max={max_val:.6f}")

def main():
    for d in dates:
        fname = f"{wbal_prefix}.{d}.nc"
        path = os.path.join(wbal_dir, fname)
        read_stats(path, var_name)

if __name__ == "__main__":
    main()
