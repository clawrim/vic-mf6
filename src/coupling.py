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
This module provides the monthly VIC–MF6 coupling loop to run VIC, step MF6,
exchange FINF/GWD, and log results.
"""

from __future__ import annotations

import os
import glob
import calendar
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
from netCDF4 import Dataset


class CouplingManager:
    def __init__(
        self,
        mf6_model,
        vic_model,
        coupling_table_csv: str,
        params_file: str,
        log_file: str,
        logger,
        *,
        vic_grid_shape: tuple[int, int],
        model_name: str,
    ) -> None:
        """Coordinate the VIC–MF6 monthly exchange using the provided models and inputs."""
        self.mf6 = mf6_model
        self.vic = vic_model
        self.coupling_table = pd.read_csv(os.path.expanduser(coupling_table_csv))
        self.params_file = os.path.expanduser(params_file)
        self.log_file = os.path.expanduser(log_file)
        self.logger = logger

        # single source of truth for units
        self.mm_to_ft = 1.0 / 304.8
        self.ft_to_mm = 304.8

        self.n_cells = self.mf6.nuzfcells
        self.vic_grid_shape = tuple(vic_grid_shape)
        self.vic_id_to_indices: dict[int, tuple[int, int]] = {}
        self.vic_lat: Optional[np.ndarray] = None
        self.vic_lon: Optional[np.ndarray] = None

        pattern = os.path.join(self.mf6.workspace, f"{model_name}*.uzf")
        matches = glob.glob(pattern)
        self.uzf_file = (
            matches[0]
            if matches
            else os.path.join(self.mf6.workspace, f"{model_name}.uzf")
        )

        self.skipped_ids_file = os.path.join(
            self.vic.exchange_dir, "skipped_mf6_ids.txt"
        )
        self.uzf_mapping = self._load_uzf_mapping()

    def _load_uzf_mapping(self) -> dict[str, int]:
        """Parse packagedata from the UZF file and build mf6_id→iuzno mapping."""
        try:
            if not os.path.exists(self.uzf_file):
                self.logger.error(f"uzf file not found: {self.uzf_file}")
                return {}
            with open(self.uzf_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            start_idx = (
                next(i for i, ln in enumerate(lines) if "BEGIN packagedata" in ln) + 1
            )
            end_idx = next(i for i, ln in enumerate(lines) if "END packagedata" in ln)
            uzf_data = [ln.split()[:4] for ln in lines[start_idx:end_idx]]
            uzf_df = pd.DataFrame(
                uzf_data, columns=["iuzno", "nlay", "nrow", "ncol"]
            ).astype(int)
            uzf_df["mf6_id"] = uzf_df.apply(
                lambda x: f"{x['nrow']:03d}{x['ncol']:03d}", axis=1
            )
            self.logger.info(f"loaded {len(uzf_df)} uzf rows")
            return dict(zip(uzf_df["mf6_id"], uzf_df["iuzno"]))
        except Exception as e:
            self.logger.error(f"load uzf mapping failed: {e}")
            return {}

    def _mf6_id_to_iuzno(self, mf6_id: Any) -> int:
        """Translate numeric/string mf6_id to iuzno; record missing ids to a text file."""
        try:
            k = str(int(mf6_id)).zfill(6)
            if k in self.uzf_mapping:
                return int(self.uzf_mapping[k])
            self.logger.warning(f"mf6_id not in uzf: {mf6_id}")
            with open(self.skipped_ids_file, "a", encoding="utf-8") as f:
                f.write(f"{mf6_id}\n")
            return -1
        except (ValueError, TypeError) as e:
            self.logger.warning(f"bad mf6_id: {mf6_id} ({e})")
            with open(self.skipped_ids_file, "a", encoding="utf-8") as f:
                f.write(f"{mf6_id}\n")
            return -1

    def initialize(self) -> None:
        """Validate inputs, map mf6_id→iuzno, and build vic_id→(lat_idx, lon_idx) mapping."""
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write(
                    "stress_period,mf6_id,vic_id,finf_ft_per_day,uzf_gwd_ft_per_day,init_moist_mm\n"
                )

            self.coupling_table["iuzno"] = self.coupling_table["mf6_id"].apply(
                self._mf6_id_to_iuzno
            )
            valid = (self.coupling_table["iuzno"] >= 0) & (
                self.coupling_table["iuzno"] < self.n_cells
            )
            self.coupling_table = self.coupling_table.loc[valid].copy()
            if self.coupling_table.empty:
                raise ValueError(
                    "no valid iuzno mappings; check join csv and uzf package"
                )

            ratio_sum = self.coupling_table.groupby("mf6_id")["mf6_area_ratio"].sum()
            bad = ratio_sum[~np.isclose(ratio_sum, 1.0, atol=1e-6)]
            if not bad.empty:
                self.logger.warning(
                    f"area ratios not summing to 1 for {len(bad)} mf6_id samples: {bad.head().to_dict()}"
                )

            self._initialize_vic_id_mapping()
        except Exception as e:
            self.logger.error(f"coupling initialize failed: {e}")
            raise

    def _initialize_vic_id_mapping(self) -> None:
        """Create vic_id→(lat_idx, lon_idx) map by nearest lat/lon index; keep per-row argmin behavior."""
        try:
            ds = Dataset(self.params_file, "r")
            lat = ds.variables.get("lat")
            lon = ds.variables.get("lon")
            if lat is None or lon is None:
                ds.close()
                raise Exception("lat/lon not found in params file")
            lat = lat[:]
            lon = lon[:]
            ds.close()

            if lat.ndim == 2:
                self.vic_lat = lat[:, 0]
            else:
                self.vic_lat = lat
            if lon.ndim == 2:
                self.vic_lon = lon[0, :]
            else:
                self.vic_lon = lon

            self.logger.info(
                f"vic grid: lat={self.vic_lat.shape}, lon={self.vic_lon.shape}"
            )

            for _, row in self.coupling_table.iterrows():
                vic_id = int(row["vic_id"])
                b_lat = float(row["b_lat"])
                b_lon = float(row["b_lon"])
                lat_idx = np.abs(self.vic_lat - b_lat).argmin()
                lon_idx = np.abs(self.vic_lon - b_lon).argmin()
                if not (
                    np.isclose(self.vic_lat[lat_idx], b_lat, atol=1e-5)
                    and np.isclose(self.vic_lon[lon_idx], b_lon, atol=1e-5)
                ):
                    self.logger.warning(
                        f"vic_id {vic_id} not close to grid (b_lat={b_lat}, b_lon={b_lon})"
                    )
                else:
                    self.vic_id_to_indices[vic_id] = (int(lat_idx), int(lon_idx))

            self.logger.info(f"mapped {len(self.vic_id_to_indices)} vic ids")
        except Exception as e:
            self.logger.error(f"init vic_id mapping failed: {e}")
            try:
                ds.close()
            except Exception:
                pass
            raise

    def compute_finf(self, baseflow: Optional[np.ndarray]) -> np.ndarray:
        """Aggregate VIC baseflow (mm) to UZF cells (ft/day) using the join table and area ratios."""
        try:
            finf = np.zeros(self.n_cells, dtype=float)
            skipped = []
            if baseflow is None:
                self.logger.warning("baseflow is None; finf remains zeros")
                return finf

            if baseflow.ndim == 3:
                bf2d = baseflow[-1, :, :]
            else:
                bf2d = baseflow

            for _, row in self.coupling_table.iterrows():
                vic_id = int(row["vic_id"])
                iuzno = int(row["iuzno"])
                ratio = float(row["mf6_area_ratio"])
                if iuzno < 0 or iuzno >= self.n_cells:
                    skipped.append(row["mf6_id"])
                    continue
                if vic_id not in self.vic_id_to_indices:
                    skipped.append(row["mf6_id"])
                    continue
                lat_idx, lon_idx = self.vic_id_to_indices[vic_id]
                try:
                    bf_mm = float(bf2d[lat_idx, lon_idx])
                except Exception:
                    skipped.append(row["mf6_id"])
                    continue
                finf[iuzno] += bf_mm * self.mm_to_ft * ratio

            np.savetxt(os.path.join(self.vic.exchange_dir, "computed_finf.txt"), finf)
            self.logger.info(
                f"finf>0 count={int((finf > 0).sum())}, mean={float(finf.mean()):.6f} ft/day"
            )
            if skipped:
                self.logger.info(
                    f"skipped {len(skipped)} mf6 ids (mapping/baseflow issues)"
                )
            return finf
        except Exception as e:
            self.logger.error(f"compute_finf failed: {e}")
            return np.zeros(self.n_cells, dtype=float)

    def update_vic_params(
        self, uzf_gwd: np.ndarray, baseflow: Optional[np.ndarray]
    ) -> bool:
        """Update VIC init_moist at layer index 2 using MF6 UZF GWD (ft/day) converted to mm and baseflow mm."""
        try:
            self.logger.info(f"reading {self.params_file}")
            ds = Dataset(self.params_file, "r+", format="NETCDF4")
            if "init_moist" not in ds.variables:
                self.logger.warning("init_moist not found")
                ds.close()
                return False
            init_moist = ds.variables["init_moist"]

            vic_gwd: dict[int, float] = {}
            for _, row in self.coupling_table.iterrows():
                vic_id = int(row["vic_id"])
                iuzno = int(row["iuzno"])
                vic_ratio = float(row.get("vic_area_ratio", 1.0))
                if iuzno < 0 or iuzno >= len(uzf_gwd):
                    continue
                gwd_mm = float(uzf_gwd[iuzno]) * self.ft_to_mm * vic_ratio
                vic_gwd[vic_id] = vic_gwd.get(vic_id, 0.0) + gwd_mm

            for vic_id, gwd_mm in vic_gwd.items():
                if vic_id not in self.vic_id_to_indices:
                    continue
                lat_idx, lon_idx = self.vic_id_to_indices[vic_id]
                bf_mm = 0.0
                if baseflow is not None:
                    try:
                        bf_mm = float(baseflow[-1, lat_idx, lon_idx])
                    except Exception:
                        bf_mm = 0.0
                new_moist = bf_mm + (gwd_mm if gwd_mm > 0 else 0.0)
                init_moist[2, lat_idx, lon_idx] = new_moist
            ds.close()
            self.logger.info("vic params updated (netcdf4)")
            return True
        except Exception as e:
            self.logger.error(f"update vic params failed with NETCDF4: {e}")
            # keep the original fallback pattern; vic_gwd scope intentionally unchanged
            try:
                ds = Dataset(self.params_file, "r+", format="NETCDF3_64BIT")
                init_moist = ds.variables["init_moist"]
                for vic_id, gwd_mm in vic_gwd.items():  # noqa: F821
                    if vic_id not in self.vic_id_to_indices:
                        continue
                    lat_idx, lon_idx = self.vic_id_to_indices[vic_id]
                    bf_mm = 0.0
                    if baseflow is not None:
                        try:
                            bf_mm = float(baseflow[-1, lat_idx, lon_idx])
                        except Exception:
                            bf_mm = 0.0
                    new_moist = max(
                        min(bf_mm + (gwd_mm if gwd_mm > 0 else 0.0), 100.0), 0.1
                    )
                    init_moist[2, lat_idx, lon_idx] = new_moist
                ds.close()
                self.logger.info("vic params updated with NETCDF3_64BIT")
                return True
            except Exception as e2:
                self.logger.error(f"fallback update failed: {e2}")
                try:
                    ds.close()
                except Exception:
                    pass
                return False

    def log_results(
        self,
        stress_period: int,
        finf: np.ndarray,
        uzf_gwd: np.ndarray,
        init_moist_last: np.ndarray,
    ) -> None:
        """Append a single line per mapping to the CSV summary for the current stress period."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                for _, row in self.coupling_table.iterrows():
                    mf6_id = int(row["mf6_id"])
                    iuzno = int(row["iuzno"])
                    vic_id = int(row["vic_id"])
                    lat_idx, lon_idx = self.vic_id_to_indices.get(vic_id, (-1, -1))
                    im = (
                        float(init_moist_last[lat_idx, lon_idx])
                        if (lat_idx >= 0 and lon_idx >= 0)
                        else np.nan
                    )
                    finf_val = float(finf[iuzno]) if 0 <= iuzno < len(finf) else np.nan
                    gwd_val = (
                        float(uzf_gwd[iuzno]) if 0 <= iuzno < len(uzf_gwd) else np.nan
                    )
                    f.write(
                        f"{stress_period},{mf6_id},{vic_id},{finf_val},{gwd_val},{im}\n"
                    )
        except Exception as e:
            self.logger.error(f"log write failed: {e}")

    def run(self, vic_start_date: datetime, coupling_end_date: datetime) -> None:
        """Run the monthly loop: VIC run, MF6 step, UZF I/O, VIC param update, and CSV logging."""
        self.logger.info("starting vic–mf6 coupling")
        first = True
        prev_date: Optional[str] = None
        mf6_current_date = self.mf6.start_date

        vic_start_minus_one = vic_start_date - timedelta(days=1)
        self.logger.info(
            f"mf6 pre-run: from {mf6_current_date} to {vic_start_minus_one} (coupling start date)"
        )
        try:
            self.mf6.run_to_date(vic_start_minus_one, mf6_current_date)  # type: ignore[attr-defined]
            mf6_current_date = vic_start_minus_one
        except Exception as e:
            self.logger.error(f"mf6 pre-run failed: {e}")
            return

        vic_current = vic_start_date
        sp_idx = 0
        while vic_current <= coupling_end_date:
            last_day = calendar.monthrange(vic_current.year, vic_current.month)[1]
            vic_period_end = datetime(vic_current.year, vic_current.month, last_day)

            # keep file-naming in line with the VIC convention
            #   wbal:   wbal.YYYY-MM-DD.nc      (first day of the month)
            #   state:  state.YYYYMMDD_00000.nc (last day of the month)
            wbal_date_tag = vic_current.strftime("%Y-%m-%d")
            state_date_tag = vic_period_end.strftime("%Y%m%d")
            date_tag = f"{vic_current.year:04d}_{vic_current.month:02d}"  # keep for global_param filename

            sp_param = self.vic.update_global_param(
                date_tag=date_tag,
                sp_start=vic_current,
                sp_end=vic_period_end,
                prev_date=prev_date,
                first=first,
            )
            ok = self.vic.run(sp_param)
            if not ok:
                self.logger.error(f"vic failed for {date_tag}")
                return
            self.vic.move_files(state_date_tag, wbal_date_tag)
            baseflow = self.vic.read_vic_wb(wbal_date_tag)

            self.logger.info(f"mf6 stepping to {vic_period_end}")
            try:
                self.mf6.step_to_end_of_month(vic_current.year, vic_current.month)
            except Exception as e:
                self.logger.error(f"mf6 step failed: {e}")
                return

            try:
                uzf_gwd = self.mf6.get_gwd_for_uzf_cells()
            except Exception as e:
                self.logger.error(f"get gwd failed: {e}")
                return

            finf = self.compute_finf(baseflow)
            try:
                self.mf6.set_finf_for_uzf_cells(finf)
                np.savetxt(os.path.join(self.vic.exchange_dir, "mf6_finf.txt"), finf)
                non_zero_iuzno = np.where(finf > 0)[0]
                np.savetxt(
                    os.path.join(self.vic.exchange_dir, "non_zero_iuzno.txt"),
                    non_zero_iuzno,
                    fmt="%d",
                )
            except Exception as e:
                self.logger.error(f"set finf failed: {e}")
                return

            try:
                if not self.update_vic_params(uzf_gwd, baseflow):
                    self.logger.warning(f"vic param update failed for {date_tag}")
            except Exception as e:
                self.logger.error(f"vic param update raised: {e}")
                return

            try:
                ds = Dataset(self.params_file, "r")
                im3 = ds.variables["init_moist"][2, :, :]
                ds.close()
            except Exception:
                im3 = np.zeros(self.vic_grid_shape, dtype=float)
            self.log_results(sp_idx, finf, uzf_gwd, im3)

            sp_idx += 1
            vic_current = (vic_period_end + timedelta(days=1)).replace(day=1)
            first = False
            prev_date = state_date_tag

        self.logger.info("coupling complete")
