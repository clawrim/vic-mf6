#!/usr/bin/env python

"""
create_exchange_table.py

build a vic–mf6 recharge exchange table for coupling via the rch package.

usage:
    edit the CONFIG section below, then run:

        python create_exchange_table.py

it will:
  - read the vic global file to find the parameters nc (if needed)
  - read the vic domain file (lon, lat, mask)
  - read the mf6 simulation (via mfsim.nam) to get the gwf dis grid
  - handle vic (epsg 4269) and mf6 (epsg 5070) crs with gdal/osr
  - map each active mf6 cell on rch_layer to the nearest active vic cell

output csv columns:

    mf6_layer,mf6_i,mf6_j,mf6_id,vic_i,vic_j,vic_id,mf6_area_ratio

for now mf6_area_ratio is 1.0 (1:1 mapping of each mf6 cell to a single vic cell).
"""

import os
import csv

import numpy as np
from netCDF4 import Dataset
from osgeo import gdal, osr
import flopy

gdal.UseExceptions()

# ---------------------------------------------------------------------
# config: edit these values
# ---------------------------------------------------------------------

# vic image-driver global parameter file (used only if domain path is empty)
vic_global_path = "../VIC/global_param_nc4.txt"

# vic domain nc with lon, lat, mask, area, frac
vic_domain_nc_path = "../VIC/nc4_domain.nc"

# mf6 simulation directory (must contain mfsim.nam)
mf6_sim_dir = "../MF6/nm/mfnm/"

# gwf model name inside the mf6 simulation
gwf_model_name = "mfnm"

# mf6 layer (0-based) where recharge is applied
rch_layer = 0

# output exchange table csv path
output_csv = "vic_mf6_exchange_rch.csv"

# vic and mf6 crs epsg codes
vic_epsg = 4269
mf6_epsg = 5070

# names of lon/lat vars in vic domain/params files
vic_lon_var = "lon"
vic_lat_var = "lat"
vic_mask_var = "mask"

# optional: override vic parameters nc path; used only if we fall back to params
vic_param_nc_override = ""

# ---------------------------------------------------------------------
# vic helpers
# ---------------------------------------------------------------------


def read_vic_param_path(global_path):
    """get vic parameters netcdf from global file (PARAMETERS line)."""
    if vic_param_nc_override:
        return vic_param_nc_override

    if not os.path.isfile(global_path):
        raise RuntimeError(f"vic global file not found: {global_path}")

    basedir = os.path.dirname(os.path.abspath(global_path))
    param_path = None

    with open(global_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0].upper()
            if key == "PARAMETERS":
                candidate = parts[1]
                if not os.path.isabs(candidate):
                    candidate = os.path.join(basedir, candidate)
                param_path = candidate
                break

    if param_path is None:
        raise RuntimeError("could not find PARAMETERS entry in vic global file")

    if not os.path.isfile(param_path):
        raise RuntimeError(f"vic parameters file not found: {param_path}")

    return param_path


def read_vic_grid_from_domain(domain_nc_path):
    """
    read vic lon, lat and mask from domain netcdf.

    returns:
        lon_1d, lat_1d, nrow, ncol, active_mask
    """
    if not os.path.isfile(domain_nc_path):
        raise RuntimeError(f"vic domain file not found: {domain_nc_path}")

    print(f"reading vic domain nc: {domain_nc_path}")
    ds = Dataset(domain_nc_path)

    if vic_lon_var not in ds.variables or vic_lat_var not in ds.variables:
        raise RuntimeError(
            f"could not find lon/lat variables '{vic_lon_var}', '{vic_lat_var}'"
        )

    lon = ds.variables[vic_lon_var][:]
    lat = ds.variables[vic_lat_var][:]

    if lon.ndim != 1 or lat.ndim != 1:
        raise RuntimeError("expected 1d lon/lat in vic domain file")

    lon = np.array(lon, dtype=float)
    lat = np.array(lat, dtype=float)

    ncol = lon.size
    nrow = lat.size

    print(f"  vic grid: nrow={nrow}, ncol={ncol}")
    print(f"  vic lon range: {lon[0]:.6f} .. {lon[-1]:.6f}")
    print(f"  vic lat range: {lat[0]:.6f} .. {lat[-1]:.6f}")

    if vic_mask_var in ds.variables:
        mask = ds.variables[vic_mask_var][:]
        if mask.shape != (nrow, ncol):
            raise RuntimeError("vic mask shape does not match lat/lon dimensions")
        # mask: 0 = inactive; >0 = active
        active_mask = mask != 0
    else:
        active_mask = np.ones((nrow, ncol), dtype=bool)

    ds.close()
    return lon, lat, nrow, ncol, active_mask


def read_vic_grid(global_path, domain_path):
    """
    wrapper to read vic grid from domain file (preferred) or fall back to params.

    returns:
        lon_1d, lat_1d, nrow, ncol, active_mask
    """
    if domain_path:
        return read_vic_grid_from_domain(domain_path)

    # fallback: params only (no mask)
    param_nc_path = read_vic_param_path(global_path)
    print(f"reading vic parameters nc (fallback): {param_nc_path}")
    ds = Dataset(param_nc_path)

    lon = ds.variables[vic_lon_var][:]
    lat = ds.variables[vic_lat_var][:]
    lon = np.array(lon, dtype=float)
    lat = np.array(lat, dtype=float)
    ncol = lon.size
    nrow = lat.size
    active_mask = np.ones((nrow, ncol), dtype=bool)

    ds.close()
    return lon, lat, nrow, ncol, active_mask


# ---------------------------------------------------------------------
# mf6 helpers
# ---------------------------------------------------------------------


def read_mf6_grid(sim_dir, gwf_name):
    """
    use flopy to read mf6 simulation and extract dis grid info.

    returns:
        nlay, nrow, ncol, delr, delc, idomain, x_edges, y_edges
    """
    print(f"loading mf6 simulation from: {sim_dir}")
    sim = flopy.mf6.MFSimulation.load(
        sim_ws=sim_dir,
        exe_name="mf6",
        strict=False,
    )

    gwf = sim.get_model(gwf_name)
    if gwf is None:
        raise RuntimeError(f"could not find gwf model '{gwf_name}' in simulation")

    dis = gwf.dis

    nlay = int(dis.nlay.data)
    nrow = int(dis.nrow.data)
    ncol = int(dis.ncol.data)

    delr = np.array(dis.delr.array, dtype=float)
    delc = np.array(dis.delc.array, dtype=float)

    if dis.idomain.array is None:
        idomain = np.ones((nlay, nrow, ncol), dtype=int)
    else:
        idomain = np.array(dis.idomain.array, dtype=int)

    # xorigin / yorigin are mfscalar objects; use .data
    try:
        xorigin = float(dis.xorigin.data)
        yorigin = float(dis.yorigin.data)
    except Exception as exc:
        raise RuntimeError(
            "dis.xorigin / dis.yorigin not available; ensure XORIGIN/YORIGIN "
            "are set in the dis options block"
        ) from exc

    print(f"  mf6 grid: nlay={nlay}, nrow={nrow}, ncol={ncol}")
    print(f"  mf6 cell size (first col/row): delr[0]={delr[0]:.3f}, delc[0]={delc[0]:.3f}")
    print(f"  mf6 origin (upper-left): xorigin={xorigin:.3f}, yorigin={yorigin:.3f}")

    # build cell edge arrays (north-up, rows increase downward)
    x_edges = np.zeros(ncol + 1, dtype=float)
    y_edges = np.zeros(nrow + 1, dtype=float)

    x_edges[0] = xorigin
    for j in range(ncol):
        x_edges[j + 1] = x_edges[j] + delr[j]

    y_edges[0] = yorigin
    for i in range(nrow):
        y_edges[i + 1] = y_edges[i] - delc[i]

    return nlay, nrow, ncol, delr, delc, idomain, x_edges, y_edges


## confirm cells 

    # vic side (domain)
    vic_lon, vic_lat, nrow_vic, ncol_vic, active_vic = read_vic_grid(
        vic_global_path, vic_domain_nc_path
    )

    n_active_vic = int(active_vic.sum())
    print(f"active vic cells (mask==1): {n_active_vic}")

    # mf6 side
    (
        nlay,
        nrow_mf6,
        ncol_mf6,
        delr,
        delc,
        idomain,
        x_edges_mf6,
        y_edges_mf6,
    ) = read_mf6_grid(mf6_sim_dir, gwf_model_name)

    n_active_mf6 = int((idomain[rch_layer] > 0).sum())
    print(f"active mf6 cells in rch_layer={rch_layer}: {n_active_mf6}")

# ---------------------------------------------------------------------
# crs + mapping helpers
# ---------------------------------------------------------------------


def build_transform(src_epsg, dst_epsg):
    """
    build gdal/osr coordinate transformation from src_epsg to dst_epsg.

    uses traditional gis axis order (x=lon, y=lat for geographic crs)
    to avoid gdal3 axis-order surprises.
    """
    src = osr.SpatialReference()
    src.ImportFromEPSG(src_epsg)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    dst = osr.SpatialReference()
    dst.ImportFromEPSG(dst_epsg)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    return osr.CoordinateTransformation(src, dst)


def nearest_index_1d(arr, value):
    """return index of element in arr closest to value."""
    arr = np.asarray(arr)
    idx = int(np.argmin(np.abs(arr - value)))
    return idx


# ---------------------------------------------------------------------
# exchange table builder
# ---------------------------------------------------------------------


def build_exchange_table_nearest(
    vic_lon,
    vic_lat,
    nrow_vic,
    ncol_vic,
    active_vic,
    nlay,
    nrow_mf6,
    ncol_mf6,
    idomain,
    x_edges,
    y_edges,
    rch_layer,
    mf6_to_vic_transform,
):
    """
    map each active mf6 cell on rch_layer to the nearest active vic cell
    in lon/lat space.

    mf6 cell centers are transformed from mf6 crs -> vic crs; then we
    find the closest vic lon and lat index separately.
    """
    rows = []

    # diag: compute min/max transformed lon/lat for sanity
    lon_list = []
    lat_list = []

    for i in range(nrow_mf6):
        for j in range(ncol_mf6):
            if idomain[rch_layer, i, j] <= 0:
                continue

            x_center = 0.5 * (x_edges[j] + x_edges[j + 1])
            y_center = 0.5 * (y_edges[i] + y_edges[i + 1])

            lon, lat, _ = mf6_to_vic_transform.TransformPoint(x_center, y_center, 0.0)
            lon_list.append(lon)
            lat_list.append(lat)

            j_v = nearest_index_1d(vic_lon, lon)
            i_v = nearest_index_1d(vic_lat, lat)

            if i_v < 0 or i_v >= nrow_vic or j_v < 0 or j_v >= ncol_vic:
                continue
            if not active_vic[i_v, j_v]:
                continue

            mf6_id = f"{i:03d}{j:03d}"
            vic_id = f"{i_v:03d}{j_v:03d}"

            rows.append((rch_layer, i, j, mf6_id, i_v, j_v, vic_id, 1.0))

    if lon_list and lat_list:
        print(
            f"  mf6 transformed lon range: {min(lon_list):.6f} .. {max(lon_list):.6f}"
        )
        print(
            f"  mf6 transformed lat range: {min(lat_list):.6f} .. {max(lat_list):.6f}"
        )

    return rows


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main():
    # vic side (domain)
    vic_lon, vic_lat, nrow_vic, ncol_vic, active_vic = read_vic_grid(
        vic_global_path, vic_domain_nc_path
    )

    # mf6 side
    (
        nlay,
        nrow_mf6,
        ncol_mf6,
        delr,
        delc,
        idomain,
        x_edges_mf6,
        y_edges_mf6,
    ) = read_mf6_grid(mf6_sim_dir, gwf_model_name)

    if rch_layer < 0 or rch_layer >= nlay:
        raise RuntimeError(f"rch_layer={rch_layer} is out of range (0..{nlay-1})")

    print(
        f"vic epsg={vic_epsg}, mf6 epsg={mf6_epsg}; using nearest-neighbour "
        "mapping in vic lon/lat space"
    )

    mf6_to_vic = build_transform(mf6_epsg, vic_epsg)

    rows = build_exchange_table_nearest(
        vic_lon,
        vic_lat,
        nrow_vic,
        ncol_vic,
        active_vic,
        nlay,
        nrow_mf6,
        ncol_mf6,
        idomain,
        x_edges_mf6,
        y_edges_mf6,
        rch_layer,
        mf6_to_vic,
    )

    n_rows = len(rows)
    print(f"mapped mf6→vic pairs (rows in table): {n_rows}")


    if not rows:
        raise RuntimeError(
            "no mapped vic–mf6 cells found; check grids, crs, and domain mask"
        )

    header = [
        "mf6_layer",
        "mf6_i",
        "mf6_j",
        "mf6_id",
        "vic_i",
        "vic_j",
        "vic_id",
        "mf6_area_ratio",
    ]

    print(f"writing exchange table: {output_csv}")
    write_csv(output_csv, header, rows)
    print("done.")


if __name__ == "__main__":
    main()
