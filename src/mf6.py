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
mf6 bmi wrapper.

this wrapper is intentionally thin:
- discover grid and key bmi variables
- set recharge (rch/rcha) arrays
- advance the model in time

it does not assume any particular mf6 package set beyond dis/tdis and rch.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import xmipy


class MF6Model:
    def __init__(
        self,
        workspace: str,
        mf6_dll: str,
        logger,
        start_date: Optional[datetime] = None,
        length_units: str = "meters",
    ) -> None:
        self.workspace = os.path.expanduser(workspace)
        self.mf6_dll = os.path.expanduser(mf6_dll)
        self.logger = logger
        self.start_date = start_date or datetime(1940, 3, 1)
        self.length_units = str(length_units).lower().strip()

        self.mf6: Optional[xmipy.XmiWrapper] = None

        self.model_name: str = "gwf"

        self.bmi_vars: set[str] = set()
        self.bmi_prefix: Optional[str] = None

        self.nlay: Optional[int] = None
        self.nrow: Optional[int] = None
        self.ncol: Optional[int] = None

        self.idomain: Optional[np.ndarray] = None
        self.surface_active: Optional[np.ndarray] = None

        self.delr: Optional[np.ndarray] = None
        self.delc: Optional[np.ndarray] = None

        self.nper: int = 1
        self.nstp: int = 1
        self.perlen_days: Optional[list[float]] = None

        # rch
        self.var_recharge: Optional[str] = None
        self.recharge_size: Optional[int] = None

    # one complete cycle
    def initialize(self) -> None:
        try:
            if not os.path.exists(self.mf6_dll):
                raise FileNotFoundError(f"mf6 dll not found: {self.mf6_dll}")
            if not os.path.exists(self.workspace):
                raise FileNotFoundError(f"mf6 workspace not found: {self.workspace}")

            self._parse_name_file()

            self.mf6 = xmipy.XmiWrapper(
                self.mf6_dll,
                working_directory=self.workspace,
            )
            self.mf6.initialize()
            self.logger.info("mf6 initialized")

            self._load_bmi_vars()
            self._resolve_bmi_prefix()
            self._parse_grid_info()
            self._parse_tdis_info()
            self._parse_tdis_file_periods()
            self._identify_rch_vars()

            self.logger.info(
                f"grid: nlay={self.nlay}, nrow={self.nrow}, ncol={self.ncol}"
            )
            self.logger.info(f"tdis: nper={self.nper}, nstp={self.nstp}")
            if self.perlen_days is not None:
                self.logger.info(
                    f"tdis perlen(days): first={self.perlen_days[0]}, last={self.perlen_days[-1]}"
                )
            self.logger.info(f"rch var: {self.var_recharge}")

        except Exception as e:
            self.logger.error(f"mf6 init failed: {e}")
            raise

    def finalize(self) -> None:
        try:
            if self.mf6 is not None:
                self.mf6.finalize()
        except Exception as e:
            self.logger.error(f"mf6 finalize failed: {e}")
            raise

    # required functions
    def _parse_name_file(self) -> None:
        try:
            nam = next(f for f in os.listdir(self.workspace) if f.lower().endswith(".nam"))
            with open(os.path.join(self.workspace, nam), "r", encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    parts = s.split()
                    if len(parts) >= 2 and parts[0].upper() == "MODELNAME":
                        self.model_name = parts[1]
                        return
            self.model_name = os.path.splitext(nam)[0]
        except StopIteration:
            self.model_name = "gwf"

    def _load_bmi_vars(self) -> None:
        try:
            assert self.mf6 is not None
            inp = list(self.mf6.get_input_var_names())
            out = list(self.mf6.get_output_var_names())
            self.bmi_vars = set(str(v) for v in (inp + out))
        except Exception as e:
            self.logger.error(f"error loading bmi vars: {e}")
            raise

    def _resolve_bmi_prefix(self) -> None:
        try:
            self.bmi_prefix = None
            for v in self.bmi_vars:
                if v.endswith("/DIS/NROW"):
                    self.bmi_prefix = v.rsplit("/DIS/NROW", 1)[0]
                    break
            if not self.bmi_prefix:
                raise RuntimeError("bmi prefix not found (no */DIS/NROW)")
        except Exception as e:
            self.logger.error(f"error resolving bmi prefix: {e}")
            raise

    def _parse_grid_info(self) -> None:
        try:
            assert self.mf6 is not None and self.bmi_prefix is not None
            p = self.bmi_prefix

            self.nrow = int(self.mf6.get_value_ptr(f"{p}/DIS/NROW")[0])
            self.ncol = int(self.mf6.get_value_ptr(f"{p}/DIS/NCOL")[0])
            self.nlay = int(self.mf6.get_value_ptr(f"{p}/DIS/NLAY")[0])

            idom = self.mf6.get_value_ptr(f"{p}/DIS/IDOMAIN")
            self.idomain = np.asarray(idom).reshape(self.nlay, self.nrow, self.ncol)

            self.surface_active = (self.idomain[0] > 0)

            # delr/delc are optional but useful for volume checks
            try:
                delr_ptr = self.mf6.get_value_ptr(f"{p}/DIS/DELR")
                delc_ptr = self.mf6.get_value_ptr(f"{p}/DIS/DELC")
                self.delr = np.asarray(delr_ptr).reshape(self.ncol).copy()
                self.delc = np.asarray(delc_ptr).reshape(self.nrow).copy()
            except Exception:
                self.delr = None
                self.delc = None

        except Exception as e:
            self.logger.error(f"error parsing grid info: {e}")
            raise

    def _parse_tdis_info(self) -> None:
        try:
            assert self.mf6 is not None
            nper_vars = [v for v in self.bmi_vars if v.endswith("/NPER")]
            nstp_vars = [v for v in self.bmi_vars if v.endswith("/NSTP")]

            self.nper = int(self.mf6.get_value_ptr(nper_vars[0])[0]) if nper_vars else 1
            self.nstp = int(self.mf6.get_value_ptr(nstp_vars[0])[0]) if nstp_vars else 1
        except Exception as e:
            self.logger.error(f"error parsing tdis info: {e}")
            raise

    def _parse_tdis_file_periods(self) -> None:
        """
        parse perlen from the tdis file because mf6 bmi does not expose perlen reliably.
        """
        try:
            tdis_files = [f for f in os.listdir(self.workspace) if f.lower().endswith(".tdis")]
            if not tdis_files:
                self.perlen_days = None
                return
            tdis_path = os.path.join(self.workspace, tdis_files[0])
            txt = open(tdis_path, "r", encoding="utf-8").read().splitlines()

            # find perioddata block
            i0 = next(i for i, ln in enumerate(txt) if ln.strip().lower().startswith("begin perioddata"))
            i1 = next(i for i, ln in enumerate(txt) if ln.strip().lower().startswith("end perioddata"))
            perlen: list[float] = []
            for ln in txt[i0 + 1 : i1]:
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                parts = re.split(r"\s+", s)
                try:
                    perlen.append(float(parts[0]))
                except Exception:
                    continue
            self.perlen_days = perlen if perlen else None
        except Exception:
            self.perlen_days = None

    def _identify_rch_vars(self) -> None:
        """
        find the bmi recharge variable.

        mf6 usually exposes something like:
          <prefix>/RCHA/RECHARGE  or  <prefix>/RCH/RECHARGE
        """
        try:
            assert self.mf6 is not None

            # prefer input vars
            inp = [str(v) for v in self.mf6.get_input_var_names()]
            candidates = [v for v in inp if v.upper().endswith("/RECHARGE") and ("/RCH" in v.upper() or "/RCHA" in v.upper())]

            if not candidates:
                # last resort: search all vars
                candidates = [v for v in self.bmi_vars if v.upper().endswith("/RECHARGE") and ("/RCH" in v.upper() or "/RCHA" in v.upper())]

            if not candidates:
                raise RuntimeError("could not find an rch recharge bmi variable (*/RCH*/RECHARGE)")

            # choose the shortest (usually most direct) to avoid aux vars
            candidates = sorted(candidates, key=len)
            self.var_recharge = candidates[0]

            ptr = self.mf6.get_value_ptr(self.var_recharge)
            arr = np.asarray(ptr)
            self.recharge_size = int(arr.size)

        except Exception as e:
            self.logger.error(f"identify rch vars failed: {e}")
            raise

    # time stepping
    def current_time_days(self) -> float:
        assert self.mf6 is not None
        return float(self.mf6.get_current_time())

    def current_date(self) -> datetime:
        return self.start_date + timedelta(days=self.current_time_days())

    def step_to_date(self, target_date: datetime) -> None:
        """advance the model to target_date (inclusive).

        target_date is interpreted in the same date system as self.start_date.
        uses the xmi time-stepping interface (get_time_step,
        prepare_time_step, do_time_step, finalize_time_step)
        """
        try:
            assert self.mf6 is not None

            # nothing to do if target is before the mf6 start_date
            if target_date < self.start_date:
                return

            target_days = float((target_date - self.start_date).days)

            # clamp to the model end time for safety
            try:
                end_time = float(self.mf6.get_end_time())
                if target_days > end_time:
                    target_days = end_time
            except Exception:
                # get_end_time not strictly required; ignore if missing
                pass

            current = float(self.mf6.get_current_time())
            if target_days <= current:
                return

            # advance in native mf6 time steps until we reach target_days
            while current < target_days:
                # let mf6 tell us the next dt
                try:
                    dt = float(self.mf6.get_time_step())
                except Exception:
                    # fall back to a 1-day step if not provided
                    dt = 1.0

                if dt <= 0.0:
                    # defensive: avoid infinite loop on zero/negative dt
                    dt = max(1.0, target_days - current)

                # xmi sequence for a single time step
                self.mf6.prepare_time_step(dt)
                self.logger.info(f"Starting MF6 timestep")
                self.mf6.do_time_step()
                self.mf6.finalize_time_step()
                self.logger.info(f"Finalizing MF6 timestep")

                current = float(self.mf6.get_current_time())
        except Exception as e:
            self.logger.error(f"mf6 step_to_date failed: {e}")
            raise

    # set forcing / define recharge
    def set_recharge(self, recharge: np.ndarray, inactive_to_zero: bool = True) -> None:
        """
        set rch recharge for the current (and subsequent) mf6 timesteps until changed.

        expected shapes:
          - (nrow, ncol)
          - (nrow*ncol,)
        """
        try:
            assert (
                self.mf6 is not None
                and self.var_recharge is not None
                and self.nrow is not None
                and self.ncol is not None
            )

            r = np.asarray(recharge, dtype=float)
            if r.ndim == 2:
                if r.shape != (self.nrow, self.ncol):
                    raise ValueError(f"recharge 2d shape must be {(self.nrow, self.ncol)}")
                r_flat = r.reshape(self.nrow * self.ncol)
            elif r.ndim == 1:
                if r.size != self.nrow * self.ncol:
                    raise ValueError(f"recharge 1d size must be {self.nrow*self.ncol}")
                r_flat = r
            else:
                raise ValueError("recharge must be 1d or 2d")

            if inactive_to_zero and self.idomain is not None:
                mask = (self.idomain[0].reshape(self.nrow * self.ncol) > 0)
                r_flat = np.where(mask, r_flat, 0.0)

            ptr = self.mf6.get_value_ptr(self.var_recharge)
            buf = np.asarray(ptr)
            if buf.size != r_flat.size:
                raise RuntimeError(
                    f"rch recharge bmi var size mismatch: bmi={buf.size}, provided={r_flat.size}"
                )
            buf[:] = r_flat

        except Exception as e:
            self.logger.error(f"set_recharge failed: {e}")
            raise

    # diagnostics
    def cell_areas(self) -> Optional[np.ndarray]:
        """
        return (nrow, ncol) cell areas if delr/delc are available.
        """
        if self.delr is None or self.delc is None or self.nrow is None or self.ncol is None:
            return None
        return np.outer(self.delc, self.delr)
