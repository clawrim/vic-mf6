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

"""
map vic baseflow fields onto the modflow 6 recharge grid.

this module converts a daily vic water-balance field, expressed as a 2d array in
millimeters per day, into a modflow 6 recharge array expressed in the length
units of the target mf6 model. the spatial relationship between the two grids is
provided by a coupling table.

why this module exists:
- vic and mf6 usually do not share the same grid geometry.
- the coupler needs a deterministic way to map vic cells to mf6 cells.
- the mapping must remain explicit because hydrologic unit mistakes are easy to
  make and hard to detect later.

design notes:
- the public workflow is deliberately small: construct, initialize, compute.
- initialization validates the coupling table once and builds lookup structures
  that are reused for every coupling step.
- compute_recharge_array() stays side-effect free except for logging.
- the implementation prefers explicit validation and clear error messages over
  trying to be permissive and guessing silently.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import pandas as pd
from netCDF4 import Dataset


class RechargeMappingError(RuntimeError):
    """raised when the vic-to-mf6 mapping cannot be built or applied."""


class LoggerLike(Protocol):
    """small logger contract used by this module.

    this keeps the mapper independent from any specific logging framework while
    still documenting the methods that the caller must provide.
    """

    def info(self, message: str) -> None: ...

    def warning(self, message: str) -> None: ...

    def error(self, message: str) -> None: ...


VicGridIndex = tuple[int, int]
Mf6GridIndex = tuple[int, int]
VicContribution = tuple[int, float]
VicIndexLookup = dict[int, VicGridIndex]
Mf6ContributionLookup = dict[Mf6GridIndex, list[VicContribution]]

_REQUIRED_MAPPING_COLUMNS = frozenset({"mf6_id", "vic_id", "mf6_area_ratio"})
_MODEL_COLUMN_CANDIDATES = ("mf6_model", "model", "mname")
_FEET_UNIT_PREFIXES = ("ft", "foot", "feet")
_MILLIMETERS_TO_METERS = 1.0e-3
_MILLIMETERS_TO_FEET = 0.0032808398950131233


@dataclass(slots=True)
class VicToMf6RechargeMapper:
    """map vic recharge-like fluxes onto an mf6 recharge array.

    the mapper assumes the input field represents a vic-derived vertical flux in
    millimeters per day on the vic grid. each mf6 surface cell may receive
    contributions from one or more vic cells according to area ratios listed in
    the coupling table.

    initialization builds two lookup tables:
    - vic_id -> vic array index
    - mf6 cell -> list of vic contributions

    after initialization, repeated coupling steps can reuse the same object
    without reparsing the coupling metadata.
    """

    coupling_table_csv: str
    vic_params_file: str
    mf6_nrow: int
    mf6_ncol: int
    mf6_length_units: str
    mf6_surface_active: np.ndarray
    model_prefix: str
    logger: LoggerLike
    vic_grid_shape: Optional[tuple[int, int]] = None

    coupling_table: Optional[pd.DataFrame] = field(default=None, init=False)
    vic_id_to_indices: Optional[VicIndexLookup] = field(default=None, init=False)
    mf6_cell_to_contributions: Optional[Mf6ContributionLookup] = field(
        default=None, init=False
    )
    millimeters_to_model_length: float = field(
        default=_MILLIMETERS_TO_METERS, init=False
    )

    def initialize(self) -> None:
        """validate inputs and build cached mapping structures.

        this method is intentionally separate from __init__ so that object
        construction stays cheap and predictable. that pattern mirrors many
        scientific and bmi-style interfaces where configuration and activation
        are two distinct phases.
        """

        self._validate_static_inputs()

        coupling_table = self._read_coupling_table()
        filtered_table = self._filter_rows_for_current_model(coupling_table)
        self._validate_mapping_table_columns(filtered_table)

        self.millimeters_to_model_length = self._millimeters_to_model_length(
            self.mf6_length_units
        )
        self.vic_id_to_indices = self._build_vic_index_lookup(filtered_table)
        self.mf6_cell_to_contributions = self._build_mf6_contribution_lookup(
            filtered_table
        )
        self.coupling_table = filtered_table

        self.logger.info(
            "mapper initialized "
            f"prefix={self.model_prefix} "
            f"rows={len(filtered_table)} "
            f"vic_ids={len(self.vic_id_to_indices)} "
            f"mf6_cells={len(self.mf6_cell_to_contributions)}"
        )

    def compute_recharge_array(self, vic_flux_mm_per_day: np.ndarray) -> np.ndarray:
        """compute an mf6 recharge array from a vic 2d field.

        the input must be a 2d vic array in millimeters per day. the output is a
        2d mf6 array in the mf6 model length units per day.

        hydrologic note:
        this method performs a spatial remapping, not a groundwater-process
        transformation. it assumes the chosen vic variable is already the flux
        that should be handed to mf6 recharge. if the wrong vic variable is used,
        the code will still run but the physics will be wrong.
        """

        vic_array = self._validate_vic_flux_array(vic_flux_mm_per_day)
        vic_id_to_indices = self._require_vic_index_lookup()
        mf6_cell_to_contributions = self._require_mf6_contribution_lookup()

        recharge_array = np.zeros((self.mf6_nrow, self.mf6_ncol), dtype=float)
        skipped_contributions = 0

        vic_nrow, vic_ncol = vic_array.shape

        for (mf6_row, mf6_col), contributions in mf6_cell_to_contributions.items():
            # mf6 recharge is only assigned to active surface cells.
            # if we write recharge into inactive cells, the bmi update may still
            # succeed, but the resulting field becomes misleading for debugging
            # and can hide coupling-table mistakes.
            if not bool(self.mf6_surface_active[mf6_row, mf6_col]):
                continue

            mapped_flux_mm_per_day = 0.0

            for vic_id, area_ratio in contributions:
                vic_index = vic_id_to_indices.get(vic_id)
                if vic_index is None:
                    skipped_contributions += 1
                    continue

                vic_row, vic_col = vic_index
                if (
                    vic_row < 0
                    or vic_col < 0
                    or vic_row >= vic_nrow
                    or vic_col >= vic_ncol
                ):
                    skipped_contributions += 1
                    continue

                vic_value = float(vic_array[vic_row, vic_col])

                # non-finite values are ignored instead of propagated.
                # this keeps one bad vic cell from contaminating the full mf6
                # recharge field with nan values.
                if not np.isfinite(vic_value):
                    continue

                # negative recharge is not passed through here.
                # in this coupling design, the mapper represents downward flux
                # supplied to the mf6 recharge package. allowing negative values
                # would change the physical meaning and should be a conscious
                # design decision handled elsewhere.
                if vic_value <= 0.0:
                    continue

                mapped_flux_mm_per_day += vic_value * area_ratio

            recharge_array[mf6_row, mf6_col] = (
                mapped_flux_mm_per_day * self.millimeters_to_model_length
            )

        if skipped_contributions > 0:
            self.logger.warning(
                f"mapper skipped vic contributions: {skipped_contributions}"
            )

        return recharge_array

    def _validate_static_inputs(self) -> None:
        """validate constructor inputs that should never change during a run."""

        coupling_table_path = Path(self.coupling_table_csv).expanduser()
        vic_params_path = Path(self.vic_params_file).expanduser()

        if not coupling_table_path.exists():
            raise FileNotFoundError(f"coupling table not found: {coupling_table_path}")
        if not coupling_table_path.is_file():
            raise RechargeMappingError(
                f"coupling table path is not a file: {coupling_table_path}"
            )

        if not vic_params_path.exists():
            raise FileNotFoundError(f"vic params file not found: {vic_params_path}")
        if not vic_params_path.is_file():
            raise RechargeMappingError(
                f"vic params path is not a file: {vic_params_path}"
            )

        if self.mf6_nrow <= 0 or self.mf6_ncol <= 0:
            raise RechargeMappingError(
                f"mf6 grid shape must be positive, got nrow={self.mf6_nrow} ncol={self.mf6_ncol}"
            )

        surface_active = np.asarray(self.mf6_surface_active)
        expected_shape = (self.mf6_nrow, self.mf6_ncol)
        if surface_active.shape != expected_shape:
            raise RechargeMappingError(
                "mf6_surface_active shape does not match the mf6 grid: "
                f"expected {expected_shape}, got {surface_active.shape}"
            )

        self.coupling_table_csv = str(coupling_table_path)
        self.vic_params_file = str(vic_params_path)
        self.mf6_surface_active = surface_active.astype(bool, copy=False)

        if self.vic_grid_shape is not None:
            if len(self.vic_grid_shape) != 2:
                raise RechargeMappingError(
                    f"vic_grid_shape must contain two integers, got {self.vic_grid_shape}"
                )
            vic_nrow, vic_ncol = self.vic_grid_shape
            if vic_nrow <= 0 or vic_ncol <= 0:
                raise RechargeMappingError(
                    f"vic_grid_shape must be positive, got {self.vic_grid_shape}"
                )

    def _read_coupling_table(self) -> pd.DataFrame:
        """read the coupling table once with explicit failure handling."""

        try:
            coupling_table = pd.read_csv(self.coupling_table_csv)
        except Exception as exc:
            raise RechargeMappingError(
                f"failed to read coupling table: {self.coupling_table_csv}"
            ) from exc

        if coupling_table.empty:
            raise RechargeMappingError(
                f"coupling table is empty: {self.coupling_table_csv}"
            )

        return coupling_table

    def _filter_rows_for_current_model(
        self, coupling_table: pd.DataFrame
    ) -> pd.DataFrame:
        """keep only rows that belong to the mf6 model handled by this rank.

        mpi note:
        each mf6 rank usually works on one gwf model. the global coupling table
        may contain rows for several gwf models. filtering here prevents one rank
        from writing recharge meant for a different submodel.
        """

        model_column_name = self._find_model_column_name(coupling_table)
        if model_column_name is None:
            return coupling_table.copy()

        normalized_model_prefix = self._normalize_model_prefix(self.model_prefix)
        filtered_table = coupling_table[
            coupling_table[model_column_name].astype(str).str.strip().str.casefold()
            == normalized_model_prefix.casefold()
        ].copy()

        if not filtered_table.empty:
            return filtered_table

        self.logger.warning(
            "no coupling-table rows matched the current mf6 model prefix; "
            f"column={model_column_name} prefix={normalized_model_prefix}; "
            "using the full table"
        )
        return coupling_table.copy()

    def _find_model_column_name(self, coupling_table: pd.DataFrame) -> Optional[str]:
        """return the coupling-table column used to identify the mf6 model."""

        normalized_columns = {
            column.casefold(): column for column in coupling_table.columns
        }
        for candidate in _MODEL_COLUMN_CANDIDATES:
            column_name = normalized_columns.get(candidate.casefold())
            if column_name is not None:
                return column_name
        return None

    def _validate_mapping_table_columns(self, coupling_table: pd.DataFrame) -> None:
        """ensure the table contains the minimum information required to map data."""

        missing_columns = [
            column_name
            for column_name in sorted(_REQUIRED_MAPPING_COLUMNS)
            if column_name not in coupling_table.columns
        ]
        if missing_columns:
            raise RechargeMappingError(
                "coupling table is missing required columns: "
                + ", ".join(missing_columns)
            )

    def _build_vic_index_lookup(self, coupling_table: pd.DataFrame) -> VicIndexLookup:
        """build vic_id -> vic array index.

        supported lookup strategies, in order:
        1. explicit vic_i and vic_j columns.
        2. geographic matching using b_lat and b_lon plus vic lat/lon arrays.
        3. fallback row-major decoding from vic_id and vic_grid_shape.

        the order matters. explicit row and column indices are the least
        ambiguous and should always win when available.
        """

        if {"vic_i", "vic_j"}.issubset(coupling_table.columns):
            return self._build_vic_index_lookup_from_explicit_indices(coupling_table)

        if {"b_lat", "b_lon"}.issubset(coupling_table.columns):
            return self._build_vic_index_lookup_from_coordinates(coupling_table)

        return self._build_vic_index_lookup_from_linear_ids(coupling_table)

    def _build_vic_index_lookup_from_explicit_indices(
        self,
        coupling_table: pd.DataFrame,
    ) -> VicIndexLookup:
        """use vic_i and vic_j exactly as provided in the coupling table."""

        vic_index_lookup: VicIndexLookup = {}
        max_row = -1
        max_col = -1

        for row in coupling_table.itertuples(index=False):
            vic_id = self._coerce_int(getattr(row, "vic_id"), field_name="vic_id")
            vic_row = self._coerce_int(getattr(row, "vic_i"), field_name="vic_i")
            vic_col = self._coerce_int(getattr(row, "vic_j"), field_name="vic_j")

            if vic_row < 0 or vic_col < 0:
                raise RechargeMappingError(
                    f"vic_i and vic_j must be non-negative, got ({vic_row}, {vic_col}) for vic_id={vic_id}"
                )

            vic_index_lookup[vic_id] = (vic_row, vic_col)
            max_row = max(max_row, vic_row)
            max_col = max(max_col, vic_col)

        if self.vic_grid_shape is None:
            self.vic_grid_shape = (max_row + 1, max_col + 1)

        return vic_index_lookup

    def _build_vic_index_lookup_from_coordinates(
        self,
        coupling_table: pd.DataFrame,
    ) -> VicIndexLookup:
        """match coupling-table coordinates to the nearest vic grid coordinate.

        this fallback is more fragile than explicit vic_i and vic_j because it
        depends on coordinate consistency between the coupling table and the vic
        parameter file. we keep it because some preprocessed tables carry only
        spatial coordinates.
        """

        vic_latitudes, vic_longitudes = self._read_vic_lat_lon_1d()

        if self.vic_grid_shape is None:
            self.vic_grid_shape = (int(vic_latitudes.size), int(vic_longitudes.size))

        vic_index_lookup: VicIndexLookup = {}
        for row in coupling_table.itertuples(index=False):
            vic_id = self._coerce_int(getattr(row, "vic_id"), field_name="vic_id")
            base_latitude = self._coerce_float(
                getattr(row, "b_lat"), field_name="b_lat"
            )
            base_longitude = self._coerce_float(
                getattr(row, "b_lon"), field_name="b_lon"
            )

            vic_row = int(np.abs(vic_latitudes - base_latitude).argmin())
            vic_col = int(np.abs(vic_longitudes - base_longitude).argmin())
            vic_index_lookup[vic_id] = (vic_row, vic_col)

        return vic_index_lookup

    def _build_vic_index_lookup_from_linear_ids(
        self,
        coupling_table: pd.DataFrame,
    ) -> VicIndexLookup:
        """derive vic row and column from a row-major linear vic id."""

        vic_nrow, vic_ncol = self._require_vic_grid_shape()
        vic_cell_count = vic_nrow * vic_ncol

        vic_index_lookup: VicIndexLookup = {}
        for row in coupling_table.itertuples(index=False):
            vic_id = self._coerce_int(getattr(row, "vic_id"), field_name="vic_id")
            if vic_id < 0 or vic_id >= vic_cell_count:
                continue

            vic_row = int(vic_id // vic_ncol)
            vic_col = int(vic_id % vic_ncol)
            vic_index_lookup[vic_id] = (vic_row, vic_col)

        if not vic_index_lookup:
            raise RechargeMappingError(
                "no valid vic indices could be derived from linear vic_id values"
            )

        return vic_index_lookup

    def _read_vic_lat_lon_1d(self) -> tuple[np.ndarray, np.ndarray]:
        """read 1d vic latitude and longitude arrays from the vic parameter file."""

        dataset: Optional[Dataset] = None
        try:
            dataset = Dataset(self.vic_params_file, "r")
            latitude_variable = dataset.variables.get("lat")
            longitude_variable = dataset.variables.get("lon")

            if latitude_variable is None or longitude_variable is None:
                raise RechargeMappingError(
                    f"lat/lon variables were not found in vic params file: {self.vic_params_file}"
                )

            latitudes = np.asarray(latitude_variable[:], dtype=float)
            longitudes = np.asarray(longitude_variable[:], dtype=float)

            # some vic parameter files store 2d coordinate arrays.
            # the mapper needs 1d axes because it uses nearest-index lookup.
            if latitudes.ndim == 2:
                latitudes = latitudes[:, 0]
            if longitudes.ndim == 2:
                longitudes = longitudes[0, :]

            if latitudes.ndim != 1 or longitudes.ndim != 1:
                raise RechargeMappingError(
                    "vic lat/lon arrays must be reducible to 1d axes, got "
                    f"lat.ndim={latitudes.ndim} lon.ndim={longitudes.ndim}"
                )

            return latitudes, longitudes
        except RechargeMappingError:
            raise
        except Exception as exc:
            raise RechargeMappingError(
                f"failed to read vic coordinates from: {self.vic_params_file}"
            ) from exc
        finally:
            if dataset is not None:
                dataset.close()

    def _build_mf6_contribution_lookup(
        self,
        coupling_table: pd.DataFrame,
    ) -> Mf6ContributionLookup:
        """build mf6 cell -> normalized vic contribution list.

        hydrologic note:
        the coupling table stores mf6_area_ratio values that represent how much
        of the mf6 cell is associated with each vic cell. the code renormalizes
        ratios within each mf6 cell so the contributions sum to 1.0.

        why renormalize:
        preprocessing pipelines often produce slight floating-point drift, and
        sometimes rows with zero or invalid ratios are dropped before runtime.
        without renormalization, the effective recharge delivered to an mf6 cell
        would depend on preprocessing artifacts rather than the intended area
        weights.
        """

        decoded_rows: list[int] = []
        decoded_cols: list[int] = []

        for mf6_id_value in coupling_table["mf6_id"].tolist():
            decoded_row, decoded_col = self._decode_mf6_id(mf6_id_value)
            decoded_rows.append(decoded_row)
            decoded_cols.append(decoded_col)

        index_base = self._infer_mf6_id_base(
            decoded_rows,
            decoded_cols,
            self.mf6_nrow,
            self.mf6_ncol,
        )

        contribution_lookup: Mf6ContributionLookup = {}

        for row in coupling_table.itertuples(index=False):
            decoded_row, decoded_col = self._decode_mf6_id(getattr(row, "mf6_id"))
            mf6_row = decoded_row - index_base
            mf6_col = decoded_col - index_base

            # rows that fall outside the local mf6 grid are ignored.
            # this can happen when the table is global and this mapper is serving
            # one submodel or when the id encoding does not match the local grid.
            if not (0 <= mf6_row < self.mf6_nrow and 0 <= mf6_col < self.mf6_ncol):
                continue

            vic_id = self._coerce_int(getattr(row, "vic_id"), field_name="vic_id")
            area_ratio = self._coerce_float(
                getattr(row, "mf6_area_ratio"),
                field_name="mf6_area_ratio",
            )
            if not np.isfinite(area_ratio) or area_ratio <= 0.0:
                continue

            contribution_lookup.setdefault((mf6_row, mf6_col), []).append(
                (vic_id, area_ratio)
            )

        if not contribution_lookup:
            raise RechargeMappingError(
                "no valid mf6 cell contributions could be built from the coupling table"
            )

        normalized_lookup: Mf6ContributionLookup = {}
        for mf6_index, contributions in contribution_lookup.items():
            ratio_sum = sum(area_ratio for _, area_ratio in contributions)
            if ratio_sum <= 0.0:
                continue

            normalized_lookup[mf6_index] = [
                (vic_id, area_ratio / ratio_sum) for vic_id, area_ratio in contributions
            ]

        if not normalized_lookup:
            raise RechargeMappingError(
                "all mf6 contributions collapsed during ratio normalization"
            )

        return normalized_lookup

    def _require_vic_grid_shape(self) -> tuple[int, int]:
        """return vic_grid_shape, loading it from the vic parameter file if needed."""

        if self.vic_grid_shape is None:
            vic_latitudes, vic_longitudes = self._read_vic_lat_lon_1d()
            self.vic_grid_shape = (int(vic_latitudes.size), int(vic_longitudes.size))

        return self.vic_grid_shape

    def _require_vic_index_lookup(self) -> VicIndexLookup:
        """return the cached vic index lookup or fail fast."""

        if self.vic_id_to_indices is None:
            raise RechargeMappingError(
                "mapper is not initialized: vic index lookup is missing"
            )
        return self.vic_id_to_indices

    def _require_mf6_contribution_lookup(self) -> Mf6ContributionLookup:
        """return the cached mf6 contribution lookup or fail fast."""

        if self.mf6_cell_to_contributions is None:
            raise RechargeMappingError(
                "mapper is not initialized: mf6 contribution lookup is missing"
            )
        return self.mf6_cell_to_contributions

    def _validate_vic_flux_array(self, vic_flux_mm_per_day: np.ndarray) -> np.ndarray:
        """validate the incoming vic field before any mapping is attempted."""

        vic_array = np.asarray(vic_flux_mm_per_day, dtype=float)
        if vic_array.ndim != 2:
            raise RechargeMappingError(
                f"vic flux array must be 2d, got shape={vic_array.shape}"
            )

        expected_shape = self.vic_grid_shape
        if expected_shape is not None and vic_array.shape != expected_shape:
            raise RechargeMappingError(
                "vic flux array shape does not match the mapper vic grid: "
                f"expected {expected_shape}, got {vic_array.shape}"
            )

        return vic_array

    @staticmethod
    def _normalize_model_prefix(model_prefix: str) -> str:
        """remove bmi input prefixes so runtime and table identifiers compare cleanly."""

        normalized_prefix = str(model_prefix).strip()
        if normalized_prefix.startswith("__input__/"):
            normalized_prefix = normalized_prefix.split("/", 1)[1]
        return normalized_prefix

    @staticmethod
    def _decode_mf6_id(mf6_id_value: object) -> tuple[int, int]:
        """decode a six-digit mf6 id into row and column components.

        current assumption:
        the coupling table stores mf6 ids as rrrccc, where the first three digits
        are the row and the last three digits are the column. the input may arrive
        as a string, integer, or float-like value after csv parsing.

        if the project later adopts a different encoding, this method is the one
        place that should change.
        """

        try:
            mf6_id_as_int = int(float(str(mf6_id_value).strip()))
        except Exception as exc:
            raise RechargeMappingError(
                f"invalid mf6_id value: {mf6_id_value!r}"
            ) from exc

        if mf6_id_as_int < 0:
            raise RechargeMappingError(
                f"mf6_id must be non-negative, got {mf6_id_value!r}"
            )

        encoded_id = f"{mf6_id_as_int:06d}"
        return int(encoded_id[:3]), int(encoded_id[3:])

    @staticmethod
    def _infer_mf6_id_base(
        decoded_rows: list[int],
        decoded_cols: list[int],
        mf6_nrow: int,
        mf6_ncol: int,
    ) -> int:
        """infer whether encoded mf6 ids use zero-based or one-based indexing.

        why inference is needed:
        preprocessing tools often encode row and column ids differently. some use
        000001 style indexing, others use 001001, and csv export may strip leading
        zeros. this method keeps the runtime mapper tolerant to those variants
        while still being deterministic.
        """

        if not decoded_rows or not decoded_cols:
            raise RechargeMappingError("cannot infer mf6 id base from an empty id list")

        row_min = min(decoded_rows)
        row_max = max(decoded_rows)
        col_min = min(decoded_cols)
        col_max = max(decoded_cols)

        if row_min == 0 or col_min == 0:
            return 0
        if row_max == mf6_nrow and col_max == mf6_ncol:
            return 1
        if row_max == mf6_nrow - 1 and col_max == mf6_ncol - 1:
            return 0

        zero_based_score = sum(
            0 <= row < mf6_nrow and 0 <= col < mf6_ncol
            for row, col in zip(decoded_rows, decoded_cols)
        )
        one_based_score = sum(
            0 <= (row - 1) < mf6_nrow and 0 <= (col - 1) < mf6_ncol
            for row, col in zip(decoded_rows, decoded_cols)
        )
        return 0 if zero_based_score >= one_based_score else 1

    @staticmethod
    def _millimeters_to_model_length(length_units: str) -> float:
        """convert millimeters to the mf6 model length units.

        mf6 recharge is interpreted in model length units per time. vic fields in
        this workflow are stored in millimeters per day. if this conversion is
        skipped, recharge magnitudes will be off by exactly three orders of
        magnitude for metric models and by a different factor for foot-based
        models.
        """

        normalized_units = str(length_units).strip().lower()
        if normalized_units.startswith(_FEET_UNIT_PREFIXES):
            return _MILLIMETERS_TO_FEET
        return _MILLIMETERS_TO_METERS

    @staticmethod
    def _coerce_int(value: object, *, field_name: str) -> int:
        """coerce a value to int with a field-specific error message."""

        try:
            return int(value)
        except Exception as exc:
            raise RechargeMappingError(
                f"{field_name} must be an integer-like value, got {value!r}"
            ) from exc

    @staticmethod
    def _coerce_float(value: object, *, field_name: str) -> float:
        """coerce a value to float with a field-specific error message."""

        try:
            return float(value)
        except Exception as exc:
            raise RechargeMappingError(
                f"{field_name} must be a numeric value, got {value!r}"
            ) from exc


# compatibility alias kept inside the final module.
RechargeMapper = VicToMf6RechargeMapper
