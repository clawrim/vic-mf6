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
This module provides MF6 BMI helpers for reading grid/UZF variables and
stepping the MODFLOW 6 clock.
"""

from __future__ import annotations

import os
import calendar
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import xmipy


class MF6Model:
    def __init__(self, workspace: str, mf6_dll: str, logger, start_date: Optional[datetime] = None) -> None:
        """Holds MF6 paths, logger, and basic BMI state."""
        self.workspace = os.path.expanduser(workspace)
        self.mf6_dll = os.path.expanduser(mf6_dll)
        self.logger = logger
        self.start_date = start_date or datetime(1940, 3, 1)

        self.mf6: Optional[xmipy.XmiWrapper] = None
        self.model_name: Optional[str] = None
        self.bmi_vars: set[str] = set()
        self.bmi_prefix: Optional[str] = None

        self.nrow: Optional[int] = None
        self.ncol: Optional[int] = None
        self.nlay: Optional[int] = None
        self.nper: Optional[int] = None
        self.nstp: Optional[int] = None
        self.idomain: Optional[np.ndarray] = None
        self.nuzfcells: Optional[int] = None
        self.uzf_cell_indices: Optional[np.ndarray] = None

        self.var_finf: Optional[str] = None
        self.var_gwd: Optional[str] = None

    def initialize(self) -> None:
        """Open MF6 via BMI, read model name, variable names, grid info, TDIS, and UZF handles."""
        try:
            if not os.path.exists(self.mf6_dll):
                raise FileNotFoundError(f"MF6 DLL not found: {self.mf6_dll}")
            if not os.path.exists(self.workspace):
                raise FileNotFoundError(f"MF6 workspace not found: {self.workspace}")
            self.mf6 = xmipy.XmiWrapper(self.mf6_dll, working_directory=self.workspace)
            self.mf6.initialize()
            self.logger.info("MF6 initialized")
        except Exception as e:
            self.logger.error(f"error initializing MF6: {e}")
            raise

        self._parse_name_file()
        self._load_bmi_vars()
        self._resolve_bmi_prefix()
        self._parse_grid_info()
        self._parse_tdis_info()
        self._identify_uzf_vars()

        self.logger.info(f"Grid: nlay={self.nlay}, nrow={self.nrow}, ncol={self.ncol}")
        self.logger.info(f"TDIS: nper={self.nper}, nstp={self.nstp}")
        self.logger.info(f"UZF cells={self.nuzfcells}")

    def finalize(self) -> None:
        """Close the BMI session."""
        try:
            if self.mf6 is not None:
                self.mf6.finalize()
        except Exception as e:
            self.logger.error(f"mf6 finalize failed: {e}")
            raise

    def _parse_name_file(self) -> None:
        """Read the model name from the .nam file; fall back to filename stem."""
        try:
            nam = next(f for f in os.listdir(self.workspace) if f.lower().endswith(".nam"))
            with open(os.path.join(self.workspace, nam), "r", encoding="utf-8") as fh:
                for line in fh:
                    t = line.strip().split()
                    if len(t) >= 2 and t[0].lower() == "modelname":
                        self.model_name = t[1]
                        break
            if not self.model_name:
                self.model_name = os.path.splitext(nam)[0]
            self.logger.info(f"model name: {self.model_name}")
        except StopIteration:
            self.logger.error("no .nam file found in workspace")
            raise
        except Exception as e:
            self.logger.error(f"error parsing name file: {e}")
            raise

    def _load_bmi_vars(self) -> None:
        """Collect BMI input and output variable names into a set."""
        try:
            assert self.mf6 is not None
            inp = list(self.mf6.get_input_var_names())
            out = list(self.mf6.get_output_var_names())
            self.bmi_vars = set(str(v) for v in (inp + out))
            self.logger.info(f"bmi var list size={len(self.bmi_vars)}")
        except Exception as e:
            self.logger.error(f"error loading bmi vars: {e}")
            raise

    def _resolve_bmi_prefix(self) -> None:
        """Derive a prefix by finding a var ending with /DIS/NROW and trimming that suffix."""
        try:
            self.bmi_prefix = None
            for v in self.bmi_vars:
                if v.endswith("/DIS/NROW"):
                    self.bmi_prefix = v.rsplit("/DIS/NROW", 1)[0]
                    break
            if not self.bmi_prefix:
                raise Exception("BMI prefix not found (no */DIS/NROW)")
            self.logger.info(f"using BMI prefix: {self.bmi_prefix}")
        except Exception as e:
            self.logger.error(f"error resolving BMI prefix: {e}")
            raise

    def _parse_grid_info(self) -> None:
        """Read NROW, NCOL, NLAY, and IDOMAIN; compute top-layer active cell indices."""
        try:
            assert self.mf6 is not None and self.bmi_prefix is not None
            p = self.bmi_prefix
            self.nrow = int(self.mf6.get_value_ptr(f"{p}/DIS/NROW")[0])
            self.ncol = int(self.mf6.get_value_ptr(f"{p}/DIS/NCOL")[0])
            self.nlay = int(self.mf6.get_value_ptr(f"{p}/DIS/NLAY")[0])
            idom = self.mf6.get_value_ptr(f"{p}/DIS/IDOMAIN")
            self.idomain = np.asarray(idom).reshape(self.nlay, self.nrow, self.ncol)
            self.uzf_cell_indices = np.argwhere(self.idomain[0] > 0)
            self.nuzfcells = int((self.idomain[0] > 0).sum())
        except Exception as e:
            self.logger.error(f"error parsing grid info: {e}")
            raise

    def _parse_tdis_info(self) -> None:
        """Read NPER and NSTP if present; otherwise default to 1 and warn."""
        try:
            assert self.mf6 is not None
            nper_vars = [v for v in self.bmi_vars if v.endswith("/NPER")]
            nstp_vars = [v for v in self.bmi_vars if v.endswith("/NSTP")]
            self.nper = int(self.mf6.get_value_ptr(nper_vars[0])[0]) if nper_vars else 1
            self.nstp = int(self.mf6.get_value_ptr(nstp_vars[0])[0]) if nstp_vars else 1
            if not nper_vars:
                self.logger.warning("NPER not found; default=1")
            if not nstp_vars:
                self.logger.warning("NSTP not found; default=1")
        except Exception as e:
            self.logger.error(f"error parsing TDIS info: {e}")
            raise

    def _identify_uzf_vars(self) -> None:
        """Find UZF FINF and GWD variable names by suffix and require both."""
        try:
            self.var_finf = None
            self.var_gwd = None
            for v in self.bmi_vars:
                if v.endswith("/UZF/FINF"):
                    self.var_finf = v
                elif v.endswith("/UZF/GWD"):
                    self.var_gwd = v
            if not self.var_gwd:
                raise Exception("UZF GWD variable not found")
            if not self.var_finf:
                raise Exception("UZF FINF variable not found")
            self.logger.info(f"uzf vars: finf={self.var_finf}, gwd={self.var_gwd}")
        except Exception as e:
            self.logger.error(f"error identifying UZF variables: {e}")
            raise

    def run_to_date(self, end_date: datetime, start_date: datetime) -> None:
        """Advance MF6 with prepare/do/finalize steps until the clock reaches end_date."""
        try:
            assert self.mf6 is not None
            start_days = (start_date - self.start_date).total_seconds() / 86400.0
            end_days = (end_date - self.start_date).total_seconds() / 86400.0
            current = float(self.mf6.get_current_time())

            if current < start_days:
                self.logger.warning(f"current time {current} < start {start_days}; advancing to start first")
                while current < start_days:
                    self.mf6.prepare_time_step(0.0)
                    self.mf6.do_time_step()
                    self.mf6.finalize_time_step()
                    current = float(self.mf6.get_current_time())

            while current < end_days:
                self.mf6.prepare_time_step(0.0)
                self.mf6.do_time_step()
                self.mf6.finalize_time_step()
                current = float(self.mf6.get_current_time())
            self.logger.info(f"mf6 ran to {end_date}")
        except Exception as e:
            self.logger.error(f"error running mf6 to date: {e}")
            raise

    def run_timestep(self) -> None:
        """Execute a single MF6 time step."""
        try:
            assert self.mf6 is not None
            self.mf6.prepare_time_step(0.0)
            self.mf6.do_time_step()
            self.mf6.finalize_time_step()
        except Exception as e:
            self.logger.error(f"error running timestep: {e}")
            raise

    def step_to_end_of_month(self, year: int, month: int) -> None:
        """Advance MF6 to the last day of (year, month)."""
        try:
            assert self.mf6 is not None
            last_day = calendar.monthrange(year, month)[1]
            target = datetime(year, month, last_day)
            current_days = float(self.mf6.get_current_time())
            current_date = self.start_date + timedelta(days=current_days)
            self.run_to_date(target, current_date)
        except Exception as e:
            self.logger.error(f"error stepping to end of month: {e}")
            raise

    def set_finf_for_uzf_cells(self, finf_values: np.ndarray) -> None:
        """Write FINF (ft/day) for active UZF cells."""
        try:
            assert self.mf6 is not None and self.var_finf is not None and self.nuzfcells is not None
            if len(finf_values) != self.nuzfcells:
                raise ValueError("finf length must match active UZF cells")
            self.mf6.set_value(self.var_finf, finf_values.astype(float))
        except Exception as e:
            self.logger.error(f"error setting FINF: {e}")
            raise

    def get_gwd_for_uzf_cells(self) -> np.ndarray:
        """Read GWD (ft/day) for active UZF cells."""
        try:
            assert self.mf6 is not None and self.var_gwd is not None
            gwd = self.mf6.get_value(self.var_gwd)
            return np.asarray(gwd, dtype=float)
        except Exception as e:
            self.logger.error(f"error getting GWD: {e}")
            raise

