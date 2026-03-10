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

"""parallel mf6 bmi runtime helpers.

this module wraps the xmipy interface for the case where one mpi rank in the
controller world corresponds to one participating mf6 rank in a sub-model run.

the wrapper keeps the contract narrow.
- initialize the rank-local bmi runtime.
- discover the rank-local model prefix and structured grid.
- write recharge into the bmi pointer buffer.
- advance the rank to a target date.

that is enough for the current vic->mf6 exchange loop without hiding the fact
that time stepping and recharge updates remain collective scientific actions.
"""

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import xmipy


class LoggerLike(Protocol):
    def info(self, message: str) -> None: ...

    def warning(self, message: str) -> None: ...

    def error(self, message: str) -> None: ...


class ParallelMF6BmiError(RuntimeError):
    """raised when the parallel bmi wrapper cannot initialize or exchange data safely."""


@dataclass(slots=True)
class ParallelMF6BmiModel:
    """thin parallel wrapper around the mf6 bmi or xmipy interface."""

    workspace: str
    dll_path: str
    logger: LoggerLike
    start_date: datetime = datetime(1940, 3, 1)
    length_units: str = "meters"

    bmi: xmipy.XmiWrapper | None = field(default=None, init=False)
    model_prefix: str | None = field(default=None, init=False)
    nlay: int | None = field(default=None, init=False)
    nrow: int | None = field(default=None, init=False)
    ncol: int | None = field(default=None, init=False)
    idomain: np.ndarray | None = field(default=None, init=False)
    surface_active: np.ndarray | None = field(default=None, init=False)
    perlen_days: list[float] | None = field(default=None, init=False)
    recharge_variable_name: str | None = field(default=None, init=False)

    def initialize_parallel(
        self, mpi_comm_handle: int, simulation_namefile_path: str
    ) -> None:
        """initialize the rank-local mf6 bmi instance for an mpi subcommunicator."""

        self.workspace = str(Path(self.workspace).expanduser().resolve())
        self.dll_path = str(Path(self.dll_path).expanduser().resolve())
        self.length_units = str(self.length_units).strip().lower() or "meters"

        simulation_namefile = Path(simulation_namefile_path).expanduser().resolve()
        if not Path(self.workspace).exists():
            raise FileNotFoundError(f"mf6 workspace was not found: {self.workspace}")
        if not Path(self.dll_path).exists():
            raise FileNotFoundError(f"mf6 bmi library was not found: {self.dll_path}")
        if not simulation_namefile.exists():
            raise FileNotFoundError(
                f"mf6 simulation name file was not found: {simulation_namefile}"
            )

        self.bmi = xmipy.XmiWrapper(self.dll_path, working_directory=self.workspace)
        self._safe_initialize_mpi_runtime(
            mpi_comm_handle=int(mpi_comm_handle),
            simulation_namefile=str(simulation_namefile),
        )

        self.model_prefix = self._resolve_model_prefix()
        self._load_grid_information()
        self.perlen_days = self._parse_tdis_period_lengths()
        self.recharge_variable_name = self._resolve_recharge_variable_name()

        self.logger.info(
            f"parallel mf6 bmi initialized prefix={self.model_prefix} nrow={self.nrow} ncol={self.ncol}"
        )

    def finalize(self) -> None:
        if self.bmi is None:
            return
        self.bmi.finalize()

    def set_recharge(self, recharge_array_2d: np.ndarray) -> None:
        """write a 2d recharge field into the rank-local bmi buffer."""

        bmi = self._require_bmi()
        recharge_variable_name = self._require_recharge_variable_name()
        nrow, ncol = self._require_grid_shape()
        if self.surface_active is None:
            raise ParallelMF6BmiError("surface_active has not been initialized")

        recharge_array = np.asarray(recharge_array_2d, dtype=float)
        if recharge_array.shape != (nrow, ncol):
            raise ParallelMF6BmiError(
                f"recharge array shape must be {(nrow, ncol)}, got {recharge_array.shape}"
            )

        recharge_vector = recharge_array.reshape(nrow * ncol)
        active_mask = self.surface_active.reshape(nrow * ncol)
        recharge_vector = np.where(active_mask, recharge_vector, 0.0)

        pointer = bmi.get_value_ptr(recharge_variable_name)
        pointer_array = np.asarray(pointer)
        if pointer_array.size != recharge_vector.size:
            raise ParallelMF6BmiError(
                f"recharge bmi pointer size mismatch bmi={pointer_array.size} provided={recharge_vector.size}"
            )

        pointer_array[:] = recharge_vector

    def step_to_date(self, target_date: datetime) -> None:
        """advance the rank-local mf6 runtime until the target date is reached."""

        bmi = self._require_bmi()
        if target_date < self.start_date:
            return

        target_days = float((target_date - self.start_date).days)
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
            bmi.do_time_step()
            bmi.finalize_time_step()
            current_time = float(bmi.get_current_time())

    def _safe_initialize_mpi_runtime(
        self, *, mpi_comm_handle: int, simulation_namefile: str
    ) -> None:
        bmi = self._require_bmi()

        if not hasattr(bmi, "initialize_mpi"):
            raise ParallelMF6BmiError(
                "the xmipy wrapper does not expose initialize_mpi"
            )

        try:
            bmi.initialize_mpi(int(mpi_comm_handle))
        except Exception as exc:
            message = str(exc).lower()
            if "already initialized" not in message:
                raise

        try:
            bmi.initialize(simulation_namefile)
            return
        except TypeError:
            bmi.initialize()
            return
        except Exception as exc:
            message = str(exc).lower()
            if "already initialized" in message:
                return

        # if initialize() raised for a nonstandard wrapper state, probe the bmi
        # interface directly. if the variable list is accessible, the model is at
        # least usable for the current project.
        try:
            _ = bmi.get_input_var_names()
        except Exception as probe_exc:
            raise ParallelMF6BmiError(
                f"mf6 bmi failed to initialize under mpi: {probe_exc}"
            ) from probe_exc

    def _resolve_model_prefix(self) -> str:
        bmi = self._require_bmi()
        input_variable_names = [str(name) for name in bmi.get_input_var_names()]
        candidates = sorted(
            variable_name
            for variable_name in input_variable_names
            if variable_name.endswith("/DIS/NROW")
        )
        if not candidates:
            raise ParallelMF6BmiError(
                "failed to resolve a model prefix because no */DIS/NROW variable was exposed"
            )

        for candidate in candidates:
            prefix = candidate.rsplit("/DIS/NROW", 1)[0]
            try:
                nrow = int(bmi.get_value_ptr(f"{prefix}/DIS/NROW")[0])
                ncol = int(bmi.get_value_ptr(f"{prefix}/DIS/NCOL")[0])
                nlay = int(bmi.get_value_ptr(f"{prefix}/DIS/NLAY")[0])
            except Exception:
                continue

            if nrow > 0 and ncol > 0 and nlay > 0:
                return prefix

        raise ParallelMF6BmiError(
            "failed to resolve a working model prefix for this rank"
        )

    def _load_grid_information(self) -> None:
        bmi = self._require_bmi()
        prefix = self._require_model_prefix()

        self.nrow = int(bmi.get_value_ptr(f"{prefix}/DIS/NROW")[0])
        self.ncol = int(bmi.get_value_ptr(f"{prefix}/DIS/NCOL")[0])
        self.nlay = int(bmi.get_value_ptr(f"{prefix}/DIS/NLAY")[0])

        idomain_pointer = bmi.get_value_ptr(f"{prefix}/DIS/IDOMAIN")
        self.idomain = np.asarray(idomain_pointer).reshape(
            self.nlay, self.nrow, self.ncol
        )
        self.surface_active = self.idomain[0] > 0

    def _parse_tdis_period_lengths(self) -> list[float] | None:
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
        prefix = self._require_model_prefix()
        input_variable_names = [str(name) for name in bmi.get_input_var_names()]

        candidates: list[str] = []
        for variable_name in input_variable_names:
            upper_name = variable_name.upper()
            if not upper_name.endswith("/RECHARGE"):
                continue
            if "/RCH" not in upper_name and "/RCHA" not in upper_name:
                continue
            if variable_name.startswith(prefix + "/") or variable_name.startswith(
                "__INPUT__/" + prefix + "/"
            ):
                candidates.append(variable_name)

        candidates.sort(key=len)
        if not candidates:
            raise ParallelMF6BmiError(
                f"failed to find a recharge bmi variable scoped to prefix={prefix}"
            )

        return candidates[0]

    def _require_bmi(self) -> xmipy.XmiWrapper:
        if self.bmi is None:
            raise ParallelMF6BmiError(
                "the parallel mf6 bmi runtime has not been initialized"
            )
        return self.bmi

    def _require_model_prefix(self) -> str:
        if self.model_prefix is None:
            raise ParallelMF6BmiError("the model prefix has not been resolved")
        return self.model_prefix

    def _require_recharge_variable_name(self) -> str:
        if self.recharge_variable_name is None:
            raise ParallelMF6BmiError("the recharge bmi variable has not been resolved")
        return self.recharge_variable_name

    def _require_grid_shape(self) -> tuple[int, int]:
        if self.nrow is None or self.ncol is None:
            raise ParallelMF6BmiError("the grid shape has not been initialized")
        return self.nrow, self.ncol


# compatibility aliases kept inside the final module.
MF6ParallelModel = ParallelMF6BmiModel
ParallelModflow6BmiModel = ParallelMF6BmiModel
ParallelModflow6BmiError = ParallelMF6BmiError
