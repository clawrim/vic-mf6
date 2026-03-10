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

"""manage vic image driver configuration files and output retrieval.

this module handles the vic side of the coupling loop.

the coupler needs three recurring operations from vic.

1. prepare a step-specific global parameter file.
2. run vic for that coupling window.
3. read the requested water-balance output and identify the latest state file.

those tasks sound simple, but there are a few project-specific pitfalls.
for example, vic can overwrite a state file if the save-state date matches the
init-state date for the next run. this module keeps that rule explicit and
close to the code that edits the parameter file so the behavior does not become
an undocumented side effect hidden in the controller.
"""

import glob
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import numpy as np
from netCDF4 import Dataset


class LoggerLike(Protocol):
    def info(self, message: str) -> None:
        ...

    def warning(self, message: str) -> None:
        ...

    def error(self, message: str) -> None:
        ...


class VicRuntimeError(RuntimeError):
    """raised when vic input preparation or output reading fails."""


@dataclass(slots=True)
class VicImageDriverRuntime:
    """manage a vic image driver working directory for sequential coupling windows."""

    working_directory: str
    executable_path: str
    global_parameter_template: str
    outputs_directory: str
    exchange_directory: str
    parameters_netcdf_path: str
    water_balance_variable: str
    initial_moisture_layer_index: int
    logger: LoggerLike

    state_file_prefix: str | None = field(default=None, init=False)
    water_balance_file_prefix: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.working_directory = str(Path(self.working_directory).expanduser().resolve())
        self.executable_path = str(Path(self.executable_path).expanduser().resolve())
        self.global_parameter_template = self._resolve_relative_to_working_directory(
            self.global_parameter_template
        )
        self.outputs_directory = self._resolve_relative_to_working_directory(
            self.outputs_directory
        )
        self.exchange_directory = self._resolve_relative_to_working_directory(
            self.exchange_directory
        )
        self.parameters_netcdf_path = self._resolve_relative_to_working_directory(
            self.parameters_netcdf_path
        )
        self.water_balance_variable = str(self.water_balance_variable).strip()
        self.initial_moisture_layer_index = int(self.initial_moisture_layer_index)

    def create_step_specific_global_parameter_file(
        self,
        *,
        step_tag: str,
        step_start: datetime,
        step_end: datetime,
        previous_state_tag: str | None = None,
        is_first_step: bool = False,
    ) -> str:
        """write a temporary global parameter file for one coupling window.

        why the save-state date can shift by one day:
        if vic reads an init state file and then writes the next state file to the
        same dated path, the new run can clobber the file that was supposed to
        represent the initial condition. that can quietly destroy restart
        reproducibility. shifting the save tag by one day avoids that collision.
        """

        template_path = Path(self.global_parameter_template)
        if not template_path.exists():
            raise FileNotFoundError(f"global parameter template was not found: {template_path}")

        step_parameter_path = Path(self.working_directory) / f"global_param_{step_tag}.txt"
        shutil.copyfile(template_path, step_parameter_path)

        template_lines = step_parameter_path.read_text(encoding="utf-8").splitlines(keepends=True)
        discovered_prefixes = self._discover_output_prefixes(template_lines)
        self.state_file_prefix = discovered_prefixes["state_prefix"]
        self.water_balance_file_prefix = discovered_prefixes["water_balance_prefix"]

        save_state_datetime = self._choose_save_state_datetime(
            step_end=step_end,
            previous_state_tag=previous_state_tag,
            is_first_step=is_first_step,
        )

        rewritten_lines = self._rewrite_global_parameter_lines(
            template_lines=template_lines,
            step_start=step_start,
            step_end=step_end,
            save_state_datetime=save_state_datetime,
            previous_state_tag=previous_state_tag,
            is_first_step=is_first_step,
        )
        step_parameter_path.write_text("".join(rewritten_lines), encoding="utf-8")

        self.logger.info(f"wrote vic step parameter file: {step_parameter_path}")
        self.logger.info(
            "vic state tags "
            f"init={previous_state_tag if previous_state_tag else 'none'} "
            f"save={save_state_datetime.strftime('%Y%m%d')}"
        )

        return str(step_parameter_path)

    def run(self, step_parameter_path: str) -> bool:
        """run vic in the configured working directory.

        the method returns a boolean instead of raising because the older code
        already used that contract. the new controller path mostly relies on mpi
        spawn, but keeping this method makes serial diagnostics straightforward.
        """

        command = [self.executable_path, "-g", Path(step_parameter_path).name]
        self.logger.info(f"running vic image driver: {' '.join(command)}")

        if not Path(self.executable_path).exists():
            self.logger.error(f"vic executable was not found: {self.executable_path}")
            return False

        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "1"
        environment["OMP_DYNAMIC"] = "FALSE"

        try:
            result = subprocess.run(
                command,
                cwd=self.working_directory,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            self.logger.error(f"vic run failed with return code {exc.returncode}")
            if exc.stdout:
                self.logger.error(f"vic stdout: {exc.stdout}")
            if exc.stderr:
                self.logger.error(f"vic stderr: {exc.stderr}")
            return False
        except Exception as exc:
            self.logger.error(f"vic run failed before completion: {exc}")
            return False

        if result.stdout:
            self.logger.info(f"vic stdout: {result.stdout}")
        if result.stderr:
            self.logger.warning(f"vic stderr: {result.stderr}")

        self.logger.info("vic image driver completed successfully")
        return True

    def latest_state_tag(self, since_epoch: float | None = None) -> str | None:
        """return the newest vic state tag that matches the current prefix.

        when since_epoch is provided, the method only accepts files that were
        updated by the most recent vic run. this prevents the coupling loop from
        silently reusing an older restart file when vic failed to write the
        expected state output for the current window.
        """

        if not self.state_file_prefix:
            return None

        state_prefix_path = Path(self._expand_state_prefix_path(self.state_file_prefix))
        candidate_files: list[str] = []
        for pattern in (
            str(state_prefix_path.parent / f"{state_prefix_path.name}.*_00000.nc"),
            str(state_prefix_path.parent / f"{state_prefix_path.name}.*.nc"),
        ):
            candidate_files.extend(glob.glob(pattern))

        if not candidate_files:
            self.logger.error(f"no vic state files were found for prefix: {state_prefix_path}")
            return None

        fresh_files = self._filter_files_by_mtime(candidate_files, since_epoch=since_epoch)
        if since_epoch is not None and not fresh_files:
            self.logger.error(
                "no fresh vic state files were written for the current run "
                f"prefix={state_prefix_path} since_epoch={float(since_epoch):.3f}"
            )
            return None

        selected_files = fresh_files if fresh_files else candidate_files
        latest_file = max(selected_files, key=os.path.getmtime)

        match = re.search(r"\.(\d{8})(?:_00000)?\.nc$", Path(latest_file).name)
        if match is None:
            self.logger.error(f"failed to parse a state tag from: {latest_file}")
            return None

        state_tag = match.group(1)
        self.logger.info(f"latest vic state tag detected: {state_tag} from {latest_file}")
        return state_tag

    def read_water_balance_near(
        self,
        target_datetime: datetime,
        *,
        since_epoch: float | None = None,
    ) -> np.ndarray | None:
        """read the nearest daily water-balance file from target-1, target, or target+1.

        when since_epoch is provided, only files updated by the current vic run
        are accepted. this prevents the coupler from silently reading a stale
        daily file left behind by an earlier standalone or failed run.
        """

        for day_shift in (0, -1, 1):
            candidate_datetime = target_datetime + timedelta(days=day_shift)
            candidate_path = self._water_balance_path_for_datetime(candidate_datetime)
            if not candidate_path.exists():
                continue
            if since_epoch is not None and os.path.getmtime(candidate_path) < float(since_epoch):
                continue

            self.logger.info(
                "selected vic water-balance file "
                f"target={target_datetime.date()} "
                f"read={candidate_datetime.date()} "
                f"shift={day_shift} "
                f"path={candidate_path}"
            )
            return self.read_water_balance(candidate_datetime.strftime("%Y-%m-%d"))

        message = f"no vic water-balance file was found near {target_datetime.date()} (tried 0, -1, +1 days)"
        if since_epoch is not None:
            message += f" with file mtime >= {float(since_epoch):.3f}"
        self.logger.error(message)
        return None

    def read_water_balance(self, date_tag: str) -> np.ndarray | None:
        """read one water-balance netcdf file and return the requested variable.

        the return value preserves the original time dimension when present.
        that matters because the controller decides whether to average over time
        or use a single daily slice.
        """

        if not self.water_balance_file_prefix:
            self.logger.error(
                "water_balance_file_prefix is not set. create_step_specific_global_parameter_file() must run first."
            )
            return None

        water_balance_path = Path(self.outputs_directory) / f"{self.water_balance_file_prefix}.{date_tag}.nc"
        if not water_balance_path.exists():
            self.logger.error(f"vic water-balance file was not found: {water_balance_path}")
            return None

        self.logger.info(f"reading vic water-balance file: {water_balance_path}")

        dataset: Dataset | None = None
        try:
            dataset = Dataset(str(water_balance_path), "r")
            if self.water_balance_variable not in dataset.variables:
                self.logger.error(
                    f"variable {self.water_balance_variable} was not found in {water_balance_path}"
                )
                return None

            variable = dataset.variables[self.water_balance_variable]
            raw_values = variable[:]

            if np.ma.isMaskedArray(raw_values):
                array = np.asarray(raw_values.filled(np.nan), dtype=float)
            else:
                array = np.asarray(raw_values, dtype=float)

            fill_values = self._collect_fill_values(variable)
            if fill_values:
                fill_mask = np.zeros(array.shape, dtype=bool)
                for fill_value in fill_values:
                    fill_mask |= array == fill_value
                array = np.where(fill_mask, np.nan, array)

            normalized_for_logging = array
            if array.ndim == 3:
                normalized_for_logging = np.nanmean(array, axis=0)

            if normalized_for_logging.ndim != 2:
                self.logger.error(
                    f"{self.water_balance_variable} has unexpected shape {array.shape} in {water_balance_path}"
                )
                return None

            self._log_array_statistics(normalized_for_logging, variable_name=self.water_balance_variable)
            return array
        except Exception as exc:
            self.logger.error(f"failed to read vic water-balance file {water_balance_path}: {exc}")
            return None
        finally:
            if dataset is not None:
                dataset.close()

    def read_water_balance_for_period(
        self,
        step_start: datetime,
        step_end: datetime,
        *,
        since_epoch: float | None = None,
    ) -> np.ndarray | None:
        """read daily water-balance arrays for a full coupling period."""

        arrays: list[np.ndarray] = []
        current_datetime = step_start

        while current_datetime <= step_end:
            daily_array = self.read_water_balance_near(current_datetime, since_epoch=since_epoch)
            if daily_array is None:
                return None

            normalized_daily_array = np.asarray(daily_array, dtype=float)
            if normalized_daily_array.ndim == 3:
                # if vic already wrote a time axis with multiple slices, the caller
                # should use that data directly rather than stacking daily means here.
                if normalized_daily_array.shape[0] > 1:
                    return normalized_daily_array
                normalized_daily_array = normalized_daily_array[0]

            if normalized_daily_array.ndim != 2:
                self.logger.error(
                    f"unexpected water-balance array shape after normalization: {normalized_daily_array.shape}"
                )
                return None

            arrays.append(normalized_daily_array)
            current_datetime += timedelta(days=1)

        return np.stack(arrays, axis=0)

    def read_baseflow_for_period(
        self,
        step_start: datetime,
        step_end: datetime,
        since_epoch: float | None = None,
    ) -> np.ndarray:
        """collect water-balance outputs between two dates.

        the name reflects the original project usage where the chosen vic variable
        often represents baseflow or a recharge-like flux. the method itself does
        not enforce a hydrologic interpretation beyond reading the configured
        variable.
        """

        if not self.water_balance_file_prefix:
            raise VicRuntimeError(
                "water_balance_file_prefix is not set. create_step_specific_global_parameter_file() must run first."
            )

        candidate_pattern = str(Path(self.outputs_directory) / f"{self.water_balance_file_prefix}.*.nc")
        candidate_files = glob.glob(candidate_pattern)
        if since_epoch is not None:
            candidate_files = [
                path for path in candidate_files if os.path.getmtime(path) >= float(since_epoch)
            ]

        tagged_files: list[tuple[datetime, str]] = []
        for path in candidate_files:
            match = re.search(r"\.(\d{4}-\d{2}-\d{2})\.nc$", Path(path).name)
            if match is None:
                continue
            file_datetime = datetime.fromisoformat(match.group(1))
            if step_start <= file_datetime <= step_end:
                tagged_files.append((file_datetime, match.group(1)))

        if tagged_files:
            tagged_files.sort(key=lambda item: item[0])
            arrays: list[np.ndarray] = []
            for _, date_tag in tagged_files:
                array = self.read_water_balance(date_tag)
                if array is None:
                    raise VicRuntimeError(f"failed to read vic water-balance file for tag {date_tag}")
                normalized_array = np.asarray(array, dtype=float)
                if normalized_array.ndim == 3 and normalized_array.shape[0] == 1:
                    normalized_array = normalized_array[0]
                arrays.append(normalized_array)
            return np.stack(arrays, axis=0)

        if candidate_files:
            latest_file = max(candidate_files, key=os.path.getmtime)
            match = re.search(r"\.(\d{4}-\d{2}-\d{2})\.nc$", Path(latest_file).name)
            if match is None:
                raise VicRuntimeError(
                    f"failed to parse the water-balance date tag from {latest_file}"
                )

            array = self.read_water_balance(match.group(1))
            if array is None:
                raise VicRuntimeError(
                    f"failed to read vic water-balance file for tag {match.group(1)}"
                )
            return np.asarray(array, dtype=float)

        raise VicRuntimeError("no vic water-balance outputs are available for this step")

    def read_latitude_longitude_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """read 1d latitude and longitude vectors from the vic parameter netcdf file."""

        parameters_path = Path(self.parameters_netcdf_path)
        if not parameters_path.exists():
            raise FileNotFoundError(f"vic parameters netcdf file was not found: {parameters_path}")

        dataset: Dataset | None = None
        try:
            dataset = Dataset(str(parameters_path), "r")
            latitude_variable = dataset.variables.get("lat")
            longitude_variable = dataset.variables.get("lon")
            if latitude_variable is None or longitude_variable is None:
                raise VicRuntimeError(
                    f"lat or lon variable was not found in vic parameters file: {parameters_path}"
                )

            latitude = np.asarray(latitude_variable[:], dtype=float)
            longitude = np.asarray(longitude_variable[:], dtype=float)

            if latitude.ndim == 2:
                latitude = latitude[:, 0]
            if longitude.ndim == 2:
                longitude = longitude[0, :]

            if latitude.ndim != 1 or longitude.ndim != 1:
                raise VicRuntimeError(
                    "vic latitude and longitude vectors must be one-dimensional after normalization"
                )

            return latitude, longitude
        finally:
            if dataset is not None:
                dataset.close()

    # legacy aliases retained for compatibility with the original module.
    update_global_param = create_step_specific_global_parameter_file
    read_vic_wb = read_water_balance
    read_vic_wb_near = read_water_balance_near
    read_vic_wb_period = read_water_balance_for_period
    _read_lat_lon_1d = read_latitude_longitude_vectors

    def _resolve_relative_to_working_directory(self, raw_path: str) -> str:
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return str(candidate.resolve())
        return str((Path(self.working_directory) / candidate).resolve())

    @staticmethod
    def _filter_files_by_mtime(candidate_files: list[str], *, since_epoch: float | None) -> list[str]:
        if since_epoch is None:
            return list(candidate_files)
        threshold = float(since_epoch)
        return [path for path in candidate_files if os.path.getmtime(path) >= threshold]

    @staticmethod
    def _normalize_global_parameter_key(raw_key: str) -> str:
        return re.sub(r"[^A-Z]", "", str(raw_key).upper())

    def _discover_output_prefixes(self, template_lines: list[str]) -> dict[str, str]:
        state_prefix: str | None = None
        water_balance_prefix: str | None = None

        for raw_line in template_lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if not parts:
                continue

            normalized_key = self._normalize_global_parameter_key(parts[0])
            if normalized_key == "STATENAME" and len(parts) >= 2:
                state_prefix = parts[1]
            if normalized_key == "OUTFILE" and len(parts) >= 2:
                water_balance_prefix = Path(parts[1]).name

        if not state_prefix:
            raise VicRuntimeError("STATENAME was not found in the vic global parameter template")
        if not water_balance_prefix:
            water_balance_prefix = "wbal"

        return {
            "state_prefix": state_prefix,
            "water_balance_prefix": water_balance_prefix,
        }

    def _choose_save_state_datetime(
        self,
        *,
        step_end: datetime,
        previous_state_tag: str | None,
        is_first_step: bool,
    ) -> datetime:
        """choose the state tag used for the restart file written by this run.

        the next coupling window starts on the calendar day after step_end.
        for daily sequential coupling, the restart file that should seed the next
        window is therefore the state at the start of that next day. using
        step_end itself can leave the controller pointing at a pre-step state or
        at no fresh file at all, depending on the vic output schedule.
        """

        del previous_state_tag
        del is_first_step
        return step_end + timedelta(days=1)

    def _rewrite_global_parameter_lines(
        self,
        *,
        template_lines: list[str],
        step_start: datetime,
        step_end: datetime,
        save_state_datetime: datetime,
        previous_state_tag: str | None,
        is_first_step: bool,
    ) -> list[str]:
        discovered_keys = self._discover_existing_parameter_keys(template_lines)
        inserted_init_state = False
        rewritten_lines: list[str] = []

        for raw_line in template_lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                rewritten_lines.append(raw_line)
                continue

            parts = line.split()
            normalized_key = self._normalize_global_parameter_key(parts[0])

            if normalized_key == "STATENAME":
                rewritten_lines.append(raw_line)
                if "STATEYEAR" not in discovered_keys:
                    rewritten_lines.append(f"STATEYEAR   {save_state_datetime.year:04d}\n")
                if "STATEMONTH" not in discovered_keys:
                    rewritten_lines.append(f"STATEMONTH  {save_state_datetime.month:02d}\n")
                if "STATEDAY" not in discovered_keys:
                    rewritten_lines.append(f"STATEDAY    {save_state_datetime.day:02d}\n")
                continue

            if normalized_key == "INITSTATE":
                rewritten_lines.append(raw_line if raw_line.startswith("#") else f"#{raw_line}")
                if not is_first_step and previous_state_tag and not inserted_init_state:
                    rewritten_lines.append(
                        f"INIT_STATE  {self.state_file_prefix}.{previous_state_tag}_00000.nc\n"
                    )
                    inserted_init_state = True
                continue

            replacement_line = self._build_parameter_replacement_line(
                normalized_key=normalized_key,
                step_start=step_start,
                step_end=step_end,
                save_state_datetime=save_state_datetime,
            )
            if replacement_line is not None:
                rewritten_lines.append(replacement_line)
                continue

            rewritten_lines.append(raw_line)

        if not is_first_step and previous_state_tag and not inserted_init_state:
            if "INITSTATE" not in discovered_keys:
                rewritten_lines.append("\n# injected by the coupling controller to preserve restart continuity.\n")
            rewritten_lines.append(
                f"INIT_STATE  {self.state_file_prefix}.{previous_state_tag}_00000.nc\n"
            )

        return rewritten_lines

    def _discover_existing_parameter_keys(self, template_lines: list[str]) -> set[str]:
        existing_keys: set[str] = set()

        for raw_line in template_lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            existing_keys.add(self._normalize_global_parameter_key(parts[0]))

        return existing_keys

    @staticmethod
    def _build_parameter_replacement_line(
        *,
        normalized_key: str,
        step_start: datetime,
        step_end: datetime,
        save_state_datetime: datetime,
    ) -> str | None:
        replacements = {
            "STARTYEAR": f"STARTYEAR   {step_start.year:04d}\n",
            "STARTMONTH": f"STARTMONTH  {step_start.month:02d}\n",
            "STARTDAY": f"STARTDAY    {step_start.day:02d}\n",
            "ENDYEAR": f"ENDYEAR     {step_end.year:04d}\n",
            "ENDMONTH": f"ENDMONTH    {step_end.month:02d}\n",
            "ENDDAY": f"ENDDAY      {step_end.day:02d}\n",
            "STATEYEAR": f"STATEYEAR   {save_state_datetime.year:04d}\n",
            "STATEMONTH": f"STATEMONTH  {save_state_datetime.month:02d}\n",
            "STATEDAY": f"STATEDAY    {save_state_datetime.day:02d}\n",
        }
        return replacements.get(normalized_key)

    def _expand_state_prefix_path(self, raw_prefix: str) -> str:
        prefix_candidate = Path(os.path.expanduser(str(raw_prefix)))
        if prefix_candidate.is_absolute():
            return str(prefix_candidate)
        return str((Path(self.working_directory) / prefix_candidate).resolve())

    def _water_balance_path_for_datetime(self, water_balance_datetime: datetime) -> Path:
        if not self.water_balance_file_prefix:
            raise VicRuntimeError(
                "water_balance_file_prefix is not set. create_step_specific_global_parameter_file() must run first."
            )
        file_name = f"{self.water_balance_file_prefix}.{water_balance_datetime.strftime('%Y-%m-%d')}.nc"
        return Path(self.outputs_directory) / file_name

    @staticmethod
    def _collect_fill_values(variable: object) -> list[float]:
        fill_values: list[float] = []
        for attribute_name in ("_FillValue", "missing_value"):
            if not hasattr(variable, attribute_name):
                continue
            try:
                fill_values.append(float(getattr(variable, attribute_name)))
            except Exception:
                continue
        return fill_values

    def _log_array_statistics(self, array_2d: np.ndarray, *, variable_name: str) -> None:
        try:
            mean_value = float(np.nanmean(array_2d))
            min_value = float(np.nanmin(array_2d))
            max_value = float(np.nanmax(array_2d))
        except Exception:
            return

        self.logger.info(
            f"{variable_name} statistics mean={mean_value:.6f} min={min_value:.6f} max={max_value:.6f}"
        )


# compatibility alias kept inside the final module.
VICModel = VicImageDriverRuntime
