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
i
from __future__ import annotations

"""public exports for the final vic-mf6 coupling package."""

from .application_config import (
    ApplicationConfig,
    ConfigError,
    ConfigurationError,
    CouplingConfig,
    MF6RuntimeConfig,
    Modflow6RuntimeConfig,
    VicImageDriverConfig,
    load_application_config,
    load_config,
)
from .mf6_bmi_model import MF6BmiModel, MF6Model, Modflow6BmiModel
from .mf6_bmi_parallel_model import (
    MF6ParallelModel,
    ParallelMF6BmiModel,
    ParallelModflow6BmiModel,
)
from .mf6_namefile_utils import (
    MF6NamefileError,
    ModflowNamefileError,
    count_gwf_models,
    count_groundwater_flow_models,
    list_gwf_models,
    list_groundwater_flow_models,
)
from .vic_image_driver_runtime import VICModel, VicImageDriverRuntime
from .vic_mpi_spawn_runtime import (
    VicMpiSpawnRequest,
    VicSpawnConfig,
    run_vic_image_driver_with_mpi_spawn,
    run_vic_spawn,
)
from .vic_to_mf6_recharge_mapper import RechargeMapper, VicToMf6RechargeMapper

__all__ = [
    "ApplicationConfig",
    "ConfigError",
    "ConfigurationError",
    "CouplingConfig",
    "MF6BmiModel",
    "MF6Model",
    "MF6NamefileError",
    "MF6ParallelModel",
    "MF6RuntimeConfig",
    "Modflow6BmiModel",
    "Modflow6RuntimeConfig",
    "ModflowNamefileError",
    "ParallelMF6BmiModel",
    "ParallelModflow6BmiModel",
    "RechargeMapper",
    "VICModel",
    "VicImageDriverConfig",
    "VicImageDriverRuntime",
    "VicMpiSpawnRequest",
    "VicSpawnConfig",
    "VicToMf6RechargeMapper",
    "count_gwf_models",
    "count_groundwater_flow_models",
    "list_gwf_models",
    "list_groundwater_flow_models",
    "load_application_config",
    "load_config",
    "run_vic_image_driver_with_mpi_spawn",
    "run_vic_spawn",
]

__version__ = "0.3.0"
