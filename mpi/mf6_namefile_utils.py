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

"""small helpers for reading mf6 simulation name files.

these helpers stay intentionally narrow. they are used early in the mpi launch
path, before bmi initialization, to answer simple structural questions such as
how many gwf models the simulation contains.
"""

from pathlib import Path


class MF6NamefileError(RuntimeError):
    """raised when a simulation name file cannot be parsed safely."""


def list_gwf_models(simulation_namefile_path: str | Path) -> list[str]:
    """return the gwf model names declared inside the begin models block."""

    namefile = Path(simulation_namefile_path).expanduser()
    if not namefile.exists():
        raise FileNotFoundError(f"simulation name file was not found: {namefile}")
    if not namefile.is_file():
        raise MF6NamefileError(f"simulation name file path is not a file: {namefile}")

    lines = namefile.read_text(encoding="utf-8").splitlines()
    inside_models_block = False
    model_names: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        normalized = line.lower()
        if normalized.startswith("begin models"):
            inside_models_block = True
            continue
        if normalized.startswith("end models"):
            inside_models_block = False
            continue
        if not inside_models_block:
            continue

        parts = line.split()
        if len(parts) < 3:
            continue
        if not parts[0].lower().startswith("gwf"):
            continue

        # mf6 uses the third token as the model name in the models block.
        # we preserve the raw token instead of normalizing it because model
        # names can be case-sensitive in surrounding tooling.
        model_names.append(parts[2])

    return model_names


def count_gwf_models(simulation_namefile_path: str | Path) -> int:
    """return the number of gwf models declared in the simulation name file."""

    return len(list_gwf_models(simulation_namefile_path))


# compatibility aliases kept inside the final module.
list_groundwater_flow_models = list_gwf_models
count_groundwater_flow_models = count_gwf_models
ModflowNamefileError = MF6NamefileError
