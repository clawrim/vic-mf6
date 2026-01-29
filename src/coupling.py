###############################################################################
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 Abdullah Azzam and Huidae Cho
#
# License
# This source code is licensed under the GNU General Public License v3.0 or
# later (GPL-3.0-or-later). You may use, study, modify, and redistribute this
# code under the same license terms. Any derivative work must be released under
# GPL-compatible terms with source code disclosure. This software is provided
# “as is,” without warranty of any kind. See the COPYING file for details.
#
# Contact
#   Abdullah Azzam <abdazzam@nmsu.edu>
#   Department of Civil Engineering, New Mexico State University
###############################################################################
"""
vic to mf6 one-way coupling using rch recharge.

key idea:
- vic provides daily OUT_BASEFLOW as a depth per timestep (mm/day for daily runs)
- convert that to a recharge rate (model length units / day)
- aggregate vic cells to mf6 cells using a precomputed join table
- set mf6 rch recharge, then advance mf6 by one stress period

this does not do two-way feedback
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from mf6 import MF6Model
from vic import VICModel


class CouplingManager:
    def __init__(
        self,
        mf6_model: MF6Model,
        vic_model: VICModel,
        coupling_table_csv: str,
        params_file: str,
        log_file: str,
        logger,
    ) -> None:
        self.mf6 = mf6_model
        self.vic = vic_model
        self.coupling_table_csv = os.path.expanduser(coupling_table_csv)

        # resolve params file relative to vic dir if not absolute
        pf = os.path.expanduser(params_file)
        if not os.path.isabs(pf):
            pf = os.path.join(self.vic.vic_dir, pf)
        self.params_file = pf

        self.log_file = os.path.expanduser(log_file)
        self.logger = logger

        self.coupling_table: Optional[pd.DataFrame] = None
        self.vic_lat: Optional[np.ndarray] = None
        self.vic_lon: Optional[np.ndarray] = None
        self.vic_grid_shape: Optional[tuple[int, int]] = None
        self.vic_id_to_indices: dict[int, tuple[int, int]] = {}

        # mf6 cell (i,j) to list of (vic_id, ratio)
        self.mf6_cell_to_contrib: dict[tuple[int, int], list[tuple[int, float]]] = {}

        # unit conversion
        self.mm_to_model = self._mm_to_model_length(self.mf6.length_units)

        # optional recharge sanity controls (can be overwritten from yaml via attributes)
        self.recharge_scale = 1.0
        self.recharge_min_mm_day: Optional[float] = None
        self.recharge_max_mm_day: Optional[float] = None


    # init
    def initialize(self) -> None:
        try:
            if not os.path.exists(self.coupling_table_csv):
                raise FileNotFoundError(f"coupling table not found: {self.coupling_table_csv}")

            self.coupling_table = pd.read_csv(self.coupling_table_csv)
            self._validate_coupling_table()
            self._initialize_vic_id_mapping()
            self._build_mf6_cell_mapping()

            self._init_log_file()

            self.logger.info(f"coupling table rows={len(self.coupling_table)}")
            self.logger.info(f"mapped vic ids={len(self.vic_id_to_indices)}")
            self.logger.info(f"unique mf6 cells mapped={len(self.mf6_cell_to_contrib)}")
            self.logger.info(f"mm_to_model={self.mm_to_model}")

        except Exception as e:
            self.logger.error(f"coupling initialize failed: {e}")
            raise

    def _validate_coupling_table(self) -> None:
        assert self.coupling_table is not None
        required = {"mf6_id", "vic_id", "mf6_area_ratio"}
        missing = required - set(self.coupling_table.columns)
        if missing:
            raise ValueError(f"coupling table missing columns: {sorted(missing)}")

        # allow either b_lat/b_lon or explicit vic_i/vic_j for vic mapping
        have_blat = "b_lat" in self.coupling_table.columns and "b_lon" in self.coupling_table.columns
        have_vic_ij = "vic_i" in self.coupling_table.columns and "vic_j" in self.coupling_table.columns
        if not (have_blat or have_vic_ij):
            self.logger.warning(
                "coupling table has no b_lat/b_lon or vic_i/vic_j; vic_id mapping may be incomplete"
            )

    def _initialize_vic_id_mapping(self) -> None:
        ds: Optional[Dataset] = None
        try:
            ds = Dataset(self.params_file, "r")
            lat_var = ds.variables.get("lat")
            lon_var = ds.variables.get("lon")
            if lat_var is None or lon_var is None:
                raise RuntimeError("lat/lon not found in params file")
            lat = lat_var[:]
            lon = lon_var[:]

            if lat.ndim == 2:
                self.vic_lat = lat[:, 0]
            else:
                self.vic_lat = lat

            if lon.ndim == 2:
                self.vic_lon = lon[0, :]
            else:
                self.vic_lon = lon

            self.vic_grid_shape = (int(self.vic_lat.size), int(self.vic_lon.size))
            self.logger.info(f"vic grid: lat={self.vic_lat.shape}, lon={self.vic_lon.shape}")

            assert self.coupling_table is not None

            # if b_lat/b_lon exist: nearest-index mapping
            if "b_lat" in self.coupling_table.columns and "b_lon" in self.coupling_table.columns:
                for _, row in self.coupling_table.iterrows():
                    vic_id = int(row["vic_id"])
                    b_lat = float(row["b_lat"])
                    b_lon = float(row["b_lon"])
                    lat_idx = int(np.abs(self.vic_lat - b_lat).argmin())
                    lon_idx = int(np.abs(self.vic_lon - b_lon).argmin())
                    self.vic_id_to_indices[vic_id] = (lat_idx, lon_idx)
                return

            # if we have explicit vic_i/vic_j, use those directly
            if "vic_i" in self.coupling_table.columns and "vic_j" in self.coupling_table.columns:
                nlat, nlon = self.vic_grid_shape
                mapped = 0
                for _, row in self.coupling_table.iterrows():
                    vic_id = int(row["vic_id"])
                    lat_idx = int(row["vic_i"])
                    lon_idx = int(row["vic_j"])
                    if 0 <= lat_idx < nlat and 0 <= lon_idx < nlon:
                        self.vic_id_to_indices[vic_id] = (lat_idx, lon_idx)
                        mapped += 1
                self.logger.info(f"initialized vic_id mapping from vic_i/vic_j for {mapped} rows")
                return

            # fallback: assume vic_id is a 1d linear index over (lat,lon), row-major
            nlat, nlon = self.vic_grid_shape
            for _, row in self.coupling_table.iterrows():
                vic_id = int(row["vic_id"])
                if vic_id < 0 or vic_id >= nlat * nlon:
                    continue
                lat_idx = int(vic_id // nlon)
                lon_idx = int(vic_id % nlon)
                self.vic_id_to_indices[vic_id] = (lat_idx, lon_idx)

        except Exception as e:
            self.logger.error(f"init vic_id mapping failed: {e}")
            raise
        finally:
            if ds is not None:
                ds.close()

    def _build_mf6_cell_mapping(self) -> None:
        """
        build mapping from join table.
        mf6_id is expected to be like 'RRRCCC' (row/col), possibly 1-based.
        """
        assert self.coupling_table is not None
        assert self.mf6.nrow is not None and self.mf6.ncol is not None

        # decode mf6_id (row, col) and infer indexing base
        rows: list[int] = []
        cols: list[int] = []
        for mf6_id in self.coupling_table["mf6_id"].astype(str).tolist():
            r, c = self._decode_mf6_id(mf6_id)
            rows.append(r)
            cols.append(c)

        base = self._infer_mf6_id_base(rows, cols, self.mf6.nrow, self.mf6.ncol)

        for _, row in self.coupling_table.iterrows():
            mf6_id = str(row["mf6_id"])
            r_raw, c_raw = self._decode_mf6_id(mf6_id)
            i = int(r_raw - base)
            j = int(c_raw - base)

            if not (0 <= i < self.mf6.nrow and 0 <= j < self.mf6.ncol):
                continue

            vic_id = int(row["vic_id"])
            ratio = float(row["mf6_area_ratio"])
            if ratio <= 0.0:
                continue

            key = (i, j)
            self.mf6_cell_to_contrib.setdefault(key, []).append((vic_id, ratio))

        # normalize ratios per mf6 cell (helps when the join table is slightly off)
        for key, lst in self.mf6_cell_to_contrib.items():
            s = sum(r for _, r in lst)
            if s <= 0.0:
                continue
            self.mf6_cell_to_contrib[key] = [(vid, r / s) for vid, r in lst]

    @staticmethod
    def _decode_mf6_id(mf6_id: str) -> tuple[int, int]:
        s = mf6_id.strip()
        # allow ints and strings like 113106
        s = f"{int(float(s)):06d}"
        r = int(s[:3])
        c = int(s[3:])
        return r, c

    @staticmethod
    def _infer_mf6_id_base(rows: list[int], cols: list[int], nrow: int, ncol: int) -> int:
        """
        return 0 for 0-based ids, 1 for 1-based ids.
        """
        rmin, rmax = min(rows), max(rows)
        cmin, cmax = min(cols), max(cols)

        # common cases
        if rmin == 0 or cmin == 0:
            return 0
        if rmax == nrow and cmax == ncol:
            return 1
        if rmax == nrow - 1 and cmax == ncol - 1:
            return 0

        # fallback: choose base that gives most in-range cells
        score0 = sum(0 <= (r - 0) < nrow and 0 <= (c - 0) < ncol for r, c in zip(rows, cols))
        score1 = sum(0 <= (r - 1) < nrow and 0 <= (c - 1) < ncol for r, c in zip(rows, cols))
        return 0 if score0 >= score1 else 1

    @staticmethod
    def _mm_to_model_length(length_units: str) -> float:
        u = str(length_units).lower().strip()
        if u.startswith("ft") or u.startswith("foot") or u.startswith("feet"):
            return 0.0032808398950131233  # mm → ft
        return 1.0e-3  # mm → m

    def _init_log_file(self) -> None:
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write(
                    "sp_index,sp_start,sp_end,perlen_days,"
                    "mean_baseflow_mm_day,mean_recharge_mm_day,total_recharge_volume_m3\n"
                )

    # coupling loop
    def run(self, start_date: datetime, end_date: datetime) -> None:
        """
        run coupling from start_date to end_date (inclusive).

        stepping is controlled by mf6 tdis perlen if available; otherwise defaults to 1-day steps.
        """
        try:
            if end_date < start_date:
                return

            # align mf6 to coupling start
            if start_date > self.mf6.start_date:
                self.mf6.step_to_date(start_date)

            sp_start = start_date
            sp_index = 0
            prev_state_tag: Optional[str] = None

            perlen_list = self.mf6.perlen_days or []
            use_mf6_periods = len(perlen_list) > 0

            while sp_start <= end_date:
                if use_mf6_periods and sp_index < len(perlen_list):
                    perlen = float(perlen_list[sp_index])
                    ndays = max(1, int(round(perlen)))
                else:
                    ndays = 1

                sp_end = min(sp_start + timedelta(days=ndays - 1), end_date)

                self.logger.info(f"coupling sp={sp_index} {sp_start.date()} -> {sp_end.date()} (ndays={ndays})")

                # VIC: run for this period
                wbal_tag = sp_start.strftime("%Y-%m-%d")
                state_tag = sp_end.strftime("%Y%m%d")
                date_tag = f"{sp_start.year:04d}_{sp_start.month:02d}_{sp_start.day:02d}"

                sp_param = self.vic.update_global_param(
                    date_tag=date_tag,
                    sp_start=sp_start,
                    sp_end=sp_end,
                    prev_date=prev_state_tag,
                    first=(sp_index == 0),
                )
                ok = self.vic.run(sp_param)
                if not ok:
                    raise RuntimeError("vic failed")

                # carry over vic state into next stress period
                prev_state_tag = state_tag

                baseflow = self.vic.read_vic_wb(wbal_tag)  # mm per timestep (daily)
                bf_mm_day = self._period_mean_mm_day(baseflow, expected_days=ndays)

                # compute mf6 recharge array in model length/day
                recharge = self.compute_recharge_array(bf_mm_day) * self.recharge_scale

                # optional clipping in mm/day space
                if self.recharge_min_mm_day is not None or self.recharge_max_mm_day is not None:
                    r_mm_day = recharge / self.mm_to_model
                    if self.recharge_min_mm_day is not None:
                        r_mm_day = np.maximum(r_mm_day, float(self.recharge_min_mm_day))
                    if self.recharge_max_mm_day is not None:
                        r_mm_day = np.minimum(r_mm_day, float(self.recharge_max_mm_day))
                    recharge = r_mm_day * self.mm_to_model

                self.mf6.set_recharge(recharge)

                # advance mf6 through the same period
                self.mf6.step_to_date(sp_end)

                # log summary
                mean_bf = float(np.nanmean(bf_mm_day))
                mean_rch_mm = float(np.nanmean(recharge / self.mm_to_model))
                vol_m3 = float(self._recharge_volume_m3(recharge, ndays))

                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"{sp_index},{sp_start.date()},{sp_end.date()},{ndays},"
                        f"{mean_bf:.6g},{mean_rch_mm:.6g},{vol_m3:.6g}\n"
                    )

                sp_index += 1
                sp_start = sp_end + timedelta(days=1)

        except Exception as e:
            self.logger.error(f"coupling run failed: {e}")
            raise

    def _period_mean_mm_day(self, arr: np.ndarray, expected_days: int) -> np.ndarray:
        """
        vic output can be:
          - (time, lat, lon)
          - (lat, lon)

        this returns a 2d array (lat, lon) of mean mm/day over the coupling period.
        """
        a = np.asarray(arr, dtype=float)
        if a.ndim == 3:
            nt = a.shape[0]
            if expected_days > 0 and nt != expected_days:
                self.logger.warning(f"vic wbal time dim nt={nt} but expected_days={expected_days}; using mean over nt")
            return np.nanmean(a, axis=0)
        if a.ndim == 2:
            return a
        raise ValueError(f"unexpected vic wbal array shape: {a.shape}")

    def compute_recharge_array(self, bf_mm_day_2d: np.ndarray) -> np.ndarray:
        """
        aggregate vic baseflow (mm/day) to mf6 recharge (model length/day) on (nrow,ncol).
        """
        assert self.mf6.nrow is not None and self.mf6.ncol is not None
        assert self.mf6.surface_active is not None

        rch = np.zeros((self.mf6.nrow, self.mf6.ncol), dtype=float)

        skipped = 0
        for (i, j), lst in self.mf6_cell_to_contrib.items():
            if not self.mf6.surface_active[i, j]:
                continue

            s_mm = 0.0
            for vic_id, ratio in lst:
                ij = self.vic_id_to_indices.get(int(vic_id))
                if ij is None:
                    skipped += 1
                    continue
                lat_idx, lon_idx = ij
                try:
                    v = float(bf_mm_day_2d[lat_idx, lon_idx])
                except Exception:
                    skipped += 1
                    continue
                if not np.isfinite(v) or v <= 0.0:
                    continue
                s_mm += v * ratio

            rch[i, j] = s_mm * self.mm_to_model

        if skipped > 0:
            self.logger.warning(f"skipped vic contributions: {skipped}")

        return rch

    def _recharge_volume_m3(self, rch_len_day: np.ndarray, ndays: int) -> float:
        """
        compute total recharge volume over the period if delr/delc exist.
        """
        areas = self.mf6.cell_areas()
        if areas is None:
            return float("nan")
        r = np.asarray(rch_len_day, dtype=float)
        if self.mf6.length_units.startswith("ft"):
            # convert ft to m for volume
            r = r * 0.3048
        return float(np.nansum(r * areas) * float(ndays))
