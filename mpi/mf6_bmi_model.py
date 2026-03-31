#!/usr/bin/env python3

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
# Abdullah Azzam <abdazzam@nmsu.edu>
# Department of Civil and Environmental Engineering,
# New Mexico State University
###############################################################################

from __future__ import annotations

"""serial mf6 bmi runtime helpers.

this wrapper stays intentionally direct. it discovers the bmi variable prefix,
reads the grid layout, resolves the recharge input variable, and advances the
simulation in time.

the wrapper does not try to hide bmi mechanics. that is deliberate. hydrologic
coupling work is easier to debug when the code still reflects the actual bmi
sequence instead of burying it under a heavy abstraction layer.
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
import xmipy


class LoggerLike(Protocol):
    def info(self, message: str) -> None: ...

    def warning(self, message: str) -> None: ...

    def error(self, message: str) -> None: ...


class MF6BmiError(RuntimeError):
    """raised when the bmi wrapper cannot initialize or apply forcing safely."""


@dataclass(slots=True)
class MF6BmiModel:
    """thin serial wrapper around the mf6 bmi or xmipy interface."""

    workspace: str
    dll_path: str
    logger: LoggerLike
    start_date: datetime = datetime(1940, 3, 1)
    length_units: str = "meters"

    bmi: xmipy.XmiWrapper | None = field(default=None, init=False)
    model_name: str = field(default="gwf", init=False)
    bmi_variable_names: set[str] = field(default_factory=set, init=False)
    bmi_prefix: str | None = field(default=None, init=False)
    recharge_variable_name: str | None = field(default=None, init=False)
    recharge_variable_size: int | None = field(default=None, init=False)
    nlay: int | None = field(default=None, init=False)
    nrow: int | None = field(default=None, init=False)
    ncol: int | None = field(default=None, init=False)
    idomain: np.ndarray | None = field(default=None, init=False)
    surface_active: np.ndarray | None = field(default=None, init=False)
    delr: np.ndarray | None = field(default=None, init=False)
    delc: np.ndarray | None = field(default=None, init=False)
    nper: int = field(default=1, init=False)
    nstp: int = field(default=1, init=False)
    perlen_days: list[float] | None = field(default=None, init=False)

    def initialize(self) -> None:
        """initialize the serial bmi runtime and discover key variables."""

        self.workspace = str(Path(self.workspace).expanduser().resolve())
        self.dll_path = str(Path(self.dll_path).expanduser().resolve())
        self.length_units = str(self.length_units).strip().lower() or "meters"

        if not Path(self.workspace).exists():
            raise FileNotFoundError(f"mf6 workspace was not found: {self.workspace}")
        if not Path(self.dll_path).exists():
            raise FileNotFoundError(f"mf6 bmi library was not found: {self.dll_path}")

        self.model_name = self._detect_model_name()
        self.bmi = xmipy.XmiWrapper(self.dll_path, working_directory=self.workspace)

        # some xmipy builds expose initialize_mpi even for serial usage.
        # calling it with a single participant keeps behavior aligned with the
        # project's parallel path when the wrapper expects that handshake.
        initialize_mpi = getattr(self.bmi, "initialize_mpi", None)
        if callable(initialize_mpi):
            initialize_mpi(1)

        self.bmi.initialize()
        self.logger.info("mf6 bmi runtime initialized")

        self.bmi_variable_names = self._load_bmi_variable_names()
        self.bmi_prefix = self._resolve_bmi_prefix()
        self._load_grid_information()
        self._load_tdis_metadata()
        self.perlen_days = self._parse_tdis_period_lengths()
        self.recharge_variable_name = self._resolve_recharge_variable_name()
        self.recharge_variable_size = self._read_recharge_variable_size()

        self.logger.info(
            f"grid nlay={self.nlay} nrow={self.nrow} ncol={self.ncol} prefix={self.bmi_prefix}"
        )
        self.logger.info(
            f"tdis nper={self.nper} nstp={self.nstp} recharge_var={self.recharge_variable_name}"
        )

    def finalize(self) -> None:
        if self.bmi is None:
            return
        self.bmi.finalize()

    def current_time_days(self) -> float:
        bmi = self._require_bmi()
        return float(bmi.get_current_time())

    def current_date(self) -> datetime:
        return self.start_date + timedelta(days=self.current_time_days())

    def do_time_step(self, time_step_days: float | None = None) -> float:
        """advance the model by one native bmi time step."""

        bmi = self._require_bmi()
        if time_step_days is None:
            time_step_days = float(bmi.get_time_step())

        bmi.prepare_time_step(float(time_step_days))
        bmi.do_time_step()
        bmi.finalize_time_step()
        return float(time_step_days)

    def step_to_date(self, target_date: datetime) -> None:
        """advance the simulation until the bmi time reaches target_date.

        the target is clamped to the model end time when that information is
        available. that avoids stepping past the declared tdis horizon.
        """

        bmi = self._require_bmi()
        if target_date < self.start_date:
            return

        target_days = float((target_date - self.start_date).days + 1)
        try:
            model_end_time = float(bmi.get_end_time())
            if target_days > model_end_time:
                target_days = model_end_time
        except Exception:
            pass

        current_time = float(bmi.get_current_time())
        if target_days <= current_time:
            return

        while current_time < target_days:
            try:
                time_step_days = float(bmi.get_time_step())
            except Exception:
                time_step_days = 1.0

            if time_step_days <= 0.0:
                time_step_days = max(1.0, target_days - current_time)

            bmi.prepare_time_step(time_step_days)
            self.logger.info("starting mf6 bmi time step")
            bmi.do_time_step()
            bmi.finalize_time_step()
            self.logger.info("finished mf6 bmi time step")
            current_time = float(bmi.get_current_time())

    def step_one_day_with_recharge(
        self,
        recharge_array: np.ndarray,
        *,
        inactive_to_zero: bool = True,
    ) -> None:
        """prepare one mf6 step, write recharge, solve, and finalize."""

        bmi = self._require_bmi()

        try:
            time_step_days = float(bmi.get_time_step())
        except Exception:
            time_step_days = 1.0

        if time_step_days <= 0.0:
            time_step_days = 1.0

        bmi.prepare_time_step(time_step_days)
        self.set_recharge(recharge_array, inactive_to_zero=inactive_to_zero)
        self.logger.info("starting mf6 bmi time step")
        bmi.do_time_step()
        bmi.finalize_time_step()
        self.logger.info("finished mf6 bmi time step")

    def set_recharge(
        self, recharge_array: np.ndarray, *, inactive_to_zero: bool = True
    ) -> None:
        """write a recharge array into the bmi input buffer.

        accepted shapes are either (nrow, ncol) or a flattened vector with
        nrow*ncol entries. the method writes directly into the bmi pointer buffer.

        why inactive cells are zeroed by default:
        the coupling table and vic field operate on the land-surface footprint,
        but mf6 may contain inactive cells in the structured grid. writing raw
        values into inactive cells usually does not change the active solution,
        but it makes diagnostics and array dumps misleading, so the safer default
        is to zero them here.
        """

        bmi = self._require_bmi()
        recharge_variable_name = self._require_recharge_variable_name()
        nrow, ncol = self._require_grid_shape()

        recharge_values = np.asarray(recharge_array, dtype=float)
        if recharge_values.ndim == 2:
            if recharge_values.shape != (nrow, ncol):
                raise MF6BmiError(
                    f"recharge array shape must be {(nrow, ncol)}, got {recharge_values.shape}"
                )
            recharge_vector = recharge_values.reshape(nrow * ncol)
        elif recharge_values.ndim == 1:
            if recharge_values.size != nrow * ncol:
                raise MF6BmiError(
                    f"recharge vector size must be {nrow * ncol}, got {recharge_values.size}"
                )
            recharge_vector = recharge_values
        else:
            raise MF6BmiError(
                "recharge input must be one-dimensional or two-dimensional"
            )

        if inactive_to_zero:
            if self.idomain is None:
                raise MF6BmiError(
                    "idomain is required when inactive_to_zero is enabled"
                )
            active_mask = self.idomain[0].reshape(nrow * ncol) > 0
            recharge_vector = np.where(active_mask, recharge_vector, 0.0)

        pointer = bmi.get_value_ptr(recharge_variable_name)
        pointer_array = np.asarray(pointer)
        self.logger.info(
                f"set_recharge var={recharge_variable_name} "
                f"size={pointer_array.size} "
                f"in_min={np.nanmin(recharge_vector):.6e} "
                f"in_max={np.nanmax(recharge_vector):.6e} "
                f"in_mean={np.nanmean(recharge_vector):.6e} "
                f"in_sum={np.nansum(recharge_vector):.6e}"
                )

        if pointer_array.size != recharge_vector.size:
            raise MF6BmiError(
                f"recharge bmi pointer size mismatch bmi={pointer_array.size} provided={recharge_vector.size}"
            )

        pointer_array[:] = recharge_vector
        self.logger.info(
                f"set_recharge readback "
                f"out_min={np.nanmin(pointer_array):.6e} "
                f"out_max={np.nanmax(pointer_array):.6e} "
                f"out_mean={np.nanmean(pointer_array):.6e} "
                f"out_sum={np.nansum(pointer_array):.6e}"
                )

    def cell_areas(self) -> np.ndarray | None:
        """return a structured-grid cell area matrix when delr and delc are available."""

        if self.delr is None or self.delc is None:
            return None
        return np.outer(self.delc, self.delr)

    def _require_bmi(self) -> xmipy.XmiWrapper:
        if self.bmi is None:
            raise MF6BmiError("the mf6 bmi runtime has not been initialized")
        return self.bmi

    def _require_recharge_variable_name(self) -> str:
        if self.recharge_variable_name is None:
            raise MF6BmiError("the recharge bmi variable has not been resolved")
        return self.recharge_variable_name

    def _require_grid_shape(self) -> tuple[int, int]:
        if self.nrow is None or self.ncol is None:
            raise MF6BmiError("the grid shape has not been loaded")
        return self.nrow, self.ncol

    def _detect_model_name(self) -> str:
        for candidate in sorted(Path(self.workspace).iterdir()):
            if candidate.suffix.lower() != ".nam":
                continue
            with candidate.open("r", encoding="utf-8") as namefile:
                for raw_line in namefile:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].upper() == "MODELNAME":
                        return parts[1]
            return candidate.stem
        return "gwf"

    def _load_bmi_variable_names(self) -> set[str]:
        bmi = self._require_bmi()
        input_names = [str(name) for name in bmi.get_input_var_names()]
        output_names = [str(name) for name in bmi.get_output_var_names()]
        return set(input_names + output_names)

    def _resolve_bmi_prefix(self) -> str:
        bmi = self._require_bmi()
        candidates = sorted(
            variable_name
            for variable_name in self.bmi_variable_names
            if variable_name.endswith("/DIS/NROW")
        )
        if not candidates:
            raise MF6BmiError(
                "failed to find a bmi prefix because no */DIS/NROW variable was exposed"
            )

        for candidate in candidates:
            try:
                value = bmi.get_value_ptr(candidate)
                _ = int(np.asarray(value).ravel()[0])
            except Exception:
                continue
            return candidate.rsplit("/DIS/NROW", 1)[0]

        return candidates[0].rsplit("/DIS/NROW", 1)[0]

    def _load_grid_information(self) -> None:
        bmi = self._require_bmi()
        prefix = self._resolve_initialized_prefix()

        self.nrow = int(bmi.get_value_ptr(f"{prefix}/DIS/NROW")[0])
        self.ncol = int(bmi.get_value_ptr(f"{prefix}/DIS/NCOL")[0])
        self.nlay = int(bmi.get_value_ptr(f"{prefix}/DIS/NLAY")[0])

        idomain_pointer = bmi.get_value_ptr(f"{prefix}/DIS/IDOMAIN")
        self.idomain = np.asarray(idomain_pointer).reshape(
            self.nlay, self.nrow, self.ncol
        )
        self.surface_active = self.idomain[0] > 0

        try:
            delr_pointer = bmi.get_value_ptr(f"{prefix}/DIS/DELR")
            delc_pointer = bmi.get_value_ptr(f"{prefix}/DIS/DELC")
            self.delr = np.asarray(delr_pointer).reshape(self.ncol).copy()
            self.delc = np.asarray(delc_pointer).reshape(self.nrow).copy()
        except Exception:
            self.delr = None
            self.delc = None

    def _load_tdis_metadata(self) -> None:
        bmi = self._require_bmi()
        nper_candidates = sorted(
            variable_name
            for variable_name in self.bmi_variable_names
            if variable_name.endswith("/NPER")
        )
        nstp_candidates = sorted(
            variable_name
            for variable_name in self.bmi_variable_names
            if variable_name.endswith("/NSTP")
        )

        if nper_candidates:
            self.nper = int(bmi.get_value_ptr(nper_candidates[0])[0])
        if nstp_candidates:
            self.nstp = int(bmi.get_value_ptr(nstp_candidates[0])[0])

    def _parse_tdis_period_lengths(self) -> list[float] | None:
        """parse perlen from the tdis file.

        bmi metadata around stress-period lengths can vary across wrappers, so the
        file remains the most explicit source of truth for the coupling schedule.
        """

        tdis_files = sorted(
            path
            for path in Path(self.workspace).iterdir()
            if path.suffix.lower() == ".tdis"
        )
        if not tdis_files:
            return None

        lines = tdis_files[0].read_text(encoding="utf-8").splitlines()
        period_start_index: int | None = None
        period_end_index: int | None = None

        for index, raw_line in enumerate(lines):
            line = raw_line.strip().lower()
            if line.startswith("begin perioddata"):
                period_start_index = index
            elif line.startswith("end perioddata"):
                period_end_index = index
                break

        if period_start_index is None or period_end_index is None:
            return None

        period_lengths: list[float] = []
        for raw_line in lines[period_start_index + 1 : period_end_index]:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            try:
                period_lengths.append(float(parts[0]))
            except Exception:
                continue

        return period_lengths or None

    def _resolve_recharge_variable_name(self) -> str:
        bmi = self._require_bmi()
        candidates = [
            variable_name
            for variable_name in [str(name) for name in bmi.get_input_var_names()]
            if variable_name.upper().endswith("/RECHARGE")
            and ("/RCH" in variable_name.upper() or "/RCHA" in variable_name.upper())
        ]

        if not candidates:
            candidates = [
                variable_name
                for variable_name in self.bmi_variable_names
                if variable_name.upper().endswith("/RECHARGE")
                and (
                    "/RCH" in variable_name.upper() or "/RCHA" in variable_name.upper()
                )
            ]

        if not candidates:
            raise MF6BmiError(
                "failed to find an rch recharge bmi variable matching */RCH*/RECHARGE"
            )

        prefix = self._resolve_initialized_prefix()
        model_identifier = prefix.split("/")[-1].upper()
        scoped_candidates = [
            variable_name
            for variable_name in candidates
            if model_identifier in variable_name.upper()
        ]
        if scoped_candidates:
            candidates = scoped_candidates

        candidates.sort(key=len)
        return candidates[0]

    def _read_recharge_variable_size(self) -> int:
        bmi = self._require_bmi()
        recharge_variable_name = self._require_recharge_variable_name()
        pointer = bmi.get_value_ptr(recharge_variable_name)
        return int(np.asarray(pointer).size)

    def _resolve_initialized_prefix(self) -> str:
        if self.bmi_prefix is None:
            raise MF6BmiError("the bmi prefix has not been resolved")
        return self.bmi_prefix


# compatibility aliases kept inside the final module.
MF6Model = MF6BmiModel
Modflow6BmiModel = MF6BmiModel
Modflow6BmiError = MF6BmiError
