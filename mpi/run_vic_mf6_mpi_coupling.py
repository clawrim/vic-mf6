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

"""run the mpi-based vic-mf6 coupling workflow.

this script uses a controller-plus-workers layout.
- world rank 0 is the controller. it prepares vic inputs, spawns vic, reads the
  vic field, and broadcasts that field.
- world ranks 1..n are mf6 workers. each rank initializes its local bmi
  context, remaps vic fluxes to its mf6 footprint, applies recharge, and steps
  mf6 to the end of the current coupling window.

that split is not just an implementation detail. it prevents vic orchestration
and mf6 stepping from fighting over the same rank responsibilities, which makes
both debugging and future performance work much easier.
"""

import argparse
import glob
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
from mpi4py import MPI

try:
    from .application_config import ApplicationConfig, ConfigurationError, load_application_config
    from .mf6_bmi_parallel_model import ParallelMF6BmiModel
    from .mf6_namefile_utils import count_gwf_models
    from .vic_image_driver_runtime import VicImageDriverRuntime
    from .vic_mpi_spawn_runtime import VicMpiSpawnRequest, run_vic_image_driver_with_mpi_spawn
    from .vic_to_mf6_recharge_mapper import VicToMf6RechargeMapper
except ImportError:
    from application_config import ApplicationConfig, ConfigurationError, load_application_config
    from mf6_bmi_parallel_model import ParallelMF6BmiModel
    from mf6_namefile_utils import count_gwf_models
    from vic_image_driver_runtime import VicImageDriverRuntime
    from vic_mpi_spawn_runtime import VicMpiSpawnRequest, run_vic_image_driver_with_mpi_spawn
    from vic_to_mf6_recharge_mapper import VicToMf6RechargeMapper


class LoggerLike(Protocol):
    def info(self, message: str) -> None:
        ...

    def warning(self, message: str) -> None:
        ...

    def error(self, message: str) -> None:
        ...


class CouplingControllerError(RuntimeError):
    """raised when the coupling controller encounters a fatal orchestration error."""


@dataclass(frozen=True, slots=True)
class CommandLineOptions:
    """parsed command-line settings for one coupling launch."""

    config_path: str
    vic_mpi_process_count: int
    max_steps: int


@dataclass(slots=True)
class CouplingWindow:
    """one controller-selected coupling window."""

    index: int
    start_datetime: datetime
    end_datetime: datetime
    duration_days: int



def main(argv: Optional[list[str]] = None) -> int:
    options = _parse_command_line_options(argv)
    world = MPI.COMM_WORLD
    rank = int(world.Get_rank())
    logger = build_console_logger(rank)

    try:
        application_config = load_application_config(options.config_path)
    except ConfigurationError as exc:
        if rank == 0:
            print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        return _run_coupling(world=world, rank=rank, logger=logger, config=application_config, options=options)
    except Exception as exc:
        _abort_world(world, logger, f"coupling failed: {exc}")
        return 1



def _run_coupling(
    *,
    world: MPI.Comm,
    rank: int,
    logger: LoggerLike,
    config: ApplicationConfig,
    options: CommandLineOptions,
) -> int:
    simulation_namefile = Path(config.mf6.workspace) / "mfsim.nam"
    if rank == 0 and not simulation_namefile.exists():
        _abort_world(world, logger, f"missing mf6 simulation name file: {simulation_namefile}")
        return 2

    groundwater_model_count = count_gwf_models(simulation_namefile) if simulation_namefile.exists() else 0
    groundwater_model_count = int(world.bcast(groundwater_model_count, root=0))
    if groundwater_model_count <= 0:
        _abort_world(world, logger, "failed to detect groundwater flow models in mfsim.nam")
        return 2

    expected_world_size = groundwater_model_count + 1
    actual_world_size = int(world.Get_size())
    if actual_world_size != expected_world_size:
        _abort_world(
            world,
            logger,
            f"world size must equal controller + mf6 worker ranks. expected={expected_world_size} got={actual_world_size}",
        )
        return 2

    mf6_subcommunicator = _build_mf6_subcommunicator(world=world, rank=rank)
    vic_grid_shape = config.coupling.vic_grid_shape
    vic_grid_shape = world.bcast(vic_grid_shape, root=0)

    mf6_model: ParallelMF6BmiModel | None = None
    recharge_mapper: VicToMf6RechargeMapper | None = None
    period_lengths_days: list[float] | None = None

    if rank > 0:
        mf6_model, recharge_mapper, period_lengths_days = _initialize_mf6_worker(
            world=world,
            rank=rank,
            logger=logger,
            config=config,
            simulation_namefile=str(simulation_namefile),
            mf6_subcommunicator=mf6_subcommunicator,
            vic_grid_shape=vic_grid_shape,
        )

    period_lengths_days = world.bcast(period_lengths_days, root=1)
    preload_library_path = _resolve_vic_spawn_preload_library()
    if rank == 0 and preload_library_path is None:
        _abort_world(
            world,
            logger,
            "vic_spawn_preload is required but was not set and no local preload library was found",
        )
        return 2
    preload_library_path = world.bcast(preload_library_path, root=0)

    vic_runtime: VicImageDriverRuntime | None = None
    if rank == 0:
        vic_runtime = _initialize_vic_controller(logger=logger, config=config)

    selected_vic_mpi_process_count = options.vic_mpi_process_count
    if selected_vic_mpi_process_count <= 0:
        selected_vic_mpi_process_count = groundwater_model_count
    selected_vic_mpi_process_count = int(world.bcast(selected_vic_mpi_process_count, root=0))

    previous_vic_state_tag: str | None = None
    current_window_index = 0
    current_window_start = config.coupling.start_date
    current_window_end_limit = config.coupling.end_date

    stop_flag = np.zeros(1, dtype=np.int32)
    vic_array_shape = np.zeros(2, dtype=np.int32)
    vic_array_flat: np.ndarray | None = None

    try:
        while True:
            if current_window_start > current_window_end_limit:
                break
            if options.max_steps > 0 and current_window_index >= options.max_steps:
                break

            coupling_window = _build_coupling_window(
                window_index=current_window_index,
                window_start=current_window_start,
                coupling_end_limit=current_window_end_limit,
                period_lengths_days=period_lengths_days,
            )
            coupling_window = world.bcast(coupling_window if rank == 0 else None, root=0)

            if rank == 0:
                assert vic_runtime is not None
                logger.info(
                    "starting coupling window "
                    f"index={coupling_window.index} "
                    f"start={coupling_window.start_datetime.date()} "
                    f"end={coupling_window.end_datetime.date()} "
                    f"days={coupling_window.duration_days}"
                )

                vic_field_mm_per_day, next_vic_state_tag = _run_vic_and_collect_flux_field(
                    vic_runtime=vic_runtime,
                    logger=logger,
                    config=config,
                    coupling_window=coupling_window,
                    previous_vic_state_tag=previous_vic_state_tag,
                    preload_library_path=preload_library_path,
                    vic_mpi_process_count=selected_vic_mpi_process_count,
                )

                previous_vic_state_tag = next_vic_state_tag

                vic_array_shape[:] = np.asarray(vic_field_mm_per_day.shape, dtype=np.int32)
                vic_array_flat = np.ascontiguousarray(vic_field_mm_per_day.reshape(-1), dtype=np.float64)
                stop_flag[0] = 0

            world.Bcast(stop_flag, root=0)
            if int(stop_flag[0]) != 0:
                break

            world.Bcast(vic_array_shape, root=0)
            element_count = int(vic_array_shape[0]) * int(vic_array_shape[1])
            if element_count <= 0:
                _abort_world(world, logger, f"invalid vic field shape broadcast: {tuple(vic_array_shape)}")
                return 1

            if rank != 0:
                vic_array_flat = np.empty(element_count, dtype=np.float64)

            world.Bcast([vic_array_flat, MPI.DOUBLE], root=0)
            world.Barrier()

            if rank > 0:
                assert mf6_model is not None
                assert recharge_mapper is not None
                vic_field_mm_per_day = vic_array_flat.reshape(int(vic_array_shape[0]), int(vic_array_shape[1]))

                if recharge_mapper.vic_grid_shape is not None:
                    broadcast_shape = (int(vic_array_shape[0]), int(vic_array_shape[1]))
                    if tuple(recharge_mapper.vic_grid_shape) != broadcast_shape:
                        _abort_world(
                            world,
                            logger,
                            f"vic grid shape mismatch mapper={recharge_mapper.vic_grid_shape} broadcast={broadcast_shape}",
                        )
                        return 1

                recharge_array = recharge_mapper.compute_recharge_array(vic_field_mm_per_day)
                recharge_array *= float(config.coupling.recharge_scale)
                mf6_model.set_recharge(recharge_array)
                mf6_model.step_to_date(coupling_window.end_datetime)

            world.Barrier()
            current_window_index = int(world.bcast(current_window_index + 1 if rank == 0 else None, root=0))
            current_window_start = world.bcast(
                coupling_window.end_datetime + timedelta(days=1) if rank == 0 else None,
                root=0,
            )

        if rank == 0:
            stop_flag[0] = 1
        world.Bcast(stop_flag, root=0)
    finally:
        if rank > 0 and mf6_model is not None:
            try:
                mf6_model.finalize()
            except Exception:
                pass

    if rank == 0:
        logger.info("mpi vic-mf6 coupling completed successfully")
    return 0



def _parse_command_line_options(argv: Optional[list[str]]) -> CommandLineOptions:
    parser = argparse.ArgumentParser(
        prog="vicmf6-mpi",
        description="mpi vic-mf6 coupling using one controller rank and one mf6 worker rank per gwf model",
    )
    parser.add_argument("-c", "--config", required=True, help="path to the coupling yaml file")
    parser.add_argument(
        "--vic-nprocs",
        type=int,
        default=0,
        help="number of mpi ranks used when spawning vic. the default reuses the mf6 model count.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="stop after this many coupling windows. zero means no limit.",
    )
    namespace = parser.parse_args(argv)

    return CommandLineOptions(
        config_path=str(namespace.config),
        vic_mpi_process_count=int(namespace.vic_nprocs),
        max_steps=int(namespace.max_steps),
    )



def build_console_logger(rank: int) -> logging.Logger:
    logger = logging.getLogger(f"vic_mf6_mpi_rank_{rank}")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger



def _build_mf6_subcommunicator(*, world: MPI.Comm, rank: int) -> MPI.Comm | None:
    color = 1 if rank > 0 else MPI.UNDEFINED
    return world.Split(color=color, key=rank)



def _initialize_mf6_worker(
    *,
    world: MPI.Comm,
    rank: int,
    logger: LoggerLike,
    config: ApplicationConfig,
    simulation_namefile: str,
    mf6_subcommunicator: MPI.Comm | None,
    vic_grid_shape: tuple[int, int] | None,
) -> tuple[ParallelMF6BmiModel, VicToMf6RechargeMapper, list[float] | None]:
    try:
        mf6_model = ParallelMF6BmiModel(
            workspace=config.mf6.workspace,
            dll_path=config.mf6.dll,
            logger=logger,
            start_date=config.mf6.start_date,
            length_units=config.mf6.length_units,
        )
        assert mf6_subcommunicator is not None
        mf6_model.initialize_parallel(int(mf6_subcommunicator.py2f()), simulation_namefile)

        recharge_mapper = VicToMf6RechargeMapper(
            coupling_table_csv=config.coupling.coupling_table_csv,
            vic_params_file=config.vic.parameters_netcdf_path,
            mf6_nrow=int(mf6_model.nrow),
            mf6_ncol=int(mf6_model.ncol),
            mf6_length_units=str(mf6_model.length_units),
            mf6_surface_active=mf6_model.surface_active,
            model_prefix=str(mf6_model.model_prefix),
            logger=logger,
            vic_grid_shape=vic_grid_shape,
        )
        recharge_mapper.initialize()

        period_lengths_days = mf6_model.perlen_days if rank == 1 else None
        return mf6_model, recharge_mapper, period_lengths_days
    except Exception as exc:
        _abort_world(world, logger, f"mf6 worker initialization failed: {exc}")
        raise



def _initialize_vic_controller(*, logger: LoggerLike, config: ApplicationConfig) -> VicImageDriverRuntime:
    _configure_local_thread_environment(thread_count="1")

    return VicImageDriverRuntime(
        working_directory=config.vic.working_directory,
        executable_path=config.vic.executable_path,
        global_parameter_template=config.vic.global_parameter_template,
        outputs_directory=config.vic.outputs_directory,
        exchange_directory=config.vic.exchange_directory,
        parameters_netcdf_path=config.vic.parameters_netcdf_path,
        water_balance_variable=config.vic.water_balance_variable,
        initial_moisture_layer_index=config.vic.initial_moisture_layer_index,
        logger=logger,
    )



def _build_coupling_window(
    *,
    window_index: int,
    window_start: datetime,
    coupling_end_limit: datetime,
    period_lengths_days: list[float] | None,
) -> CouplingWindow:
    duration_days = _choose_coupling_window_length(period_lengths_days, window_index)
    window_end = min(window_start + timedelta(days=duration_days - 1), coupling_end_limit)

    return CouplingWindow(
        index=window_index,
        start_datetime=window_start,
        end_datetime=window_end,
        duration_days=duration_days,
    )



def _choose_coupling_window_length(period_lengths_days: list[float] | None, window_index: int) -> int:
    if not period_lengths_days:
        return 1
    if window_index < 0 or window_index >= len(period_lengths_days):
        return 1

    duration_days = int(round(float(period_lengths_days[window_index])))
    return 1 if duration_days <= 0 else duration_days



def _run_vic_and_collect_flux_field(
    *,
    vic_runtime: VicImageDriverRuntime,
    logger: LoggerLike,
    config: ApplicationConfig,
    coupling_window: CouplingWindow,
    previous_vic_state_tag: str | None,
    preload_library_path: str,
    vic_mpi_process_count: int,
) -> tuple[np.ndarray, str]:
    step_tag = f"mpi_{coupling_window.start_datetime.strftime('%Y_%m_%d')}"
    step_parameter_path = vic_runtime.create_step_specific_global_parameter_file(
        step_tag=step_tag,
        step_start=coupling_window.start_datetime,
        step_end=coupling_window.end_datetime,
        previous_state_tag=previous_vic_state_tag,
        is_first_step=(coupling_window.index == 0),
    )

    spawn_request = VicMpiSpawnRequest(
        executable_path=config.vic.executable_path,
        working_directory=config.vic.working_directory,
        global_parameter_path=step_parameter_path,
        mpi_process_count=vic_mpi_process_count,
        omp_thread_count=1,
        preload_library_path=preload_library_path,
        timeout_seconds=config.vic.spawn_timeout_seconds,
    )
    vic_spawn_started_at = time.time()
    run_vic_image_driver_with_mpi_spawn(spawn_request, logger=logger)

    vic_water_balance_array = _read_vic_water_balance_for_window(
        vic_runtime=vic_runtime,
        window_start=coupling_window.start_datetime,
        window_end=coupling_window.end_datetime,
        duration_days=coupling_window.duration_days,
        since_epoch=vic_spawn_started_at,
    )
    next_state_tag = _detect_latest_vic_state_tag(
        vic_runtime,
        logger,
        since_epoch=vic_spawn_started_at,
    )
    if next_state_tag is None:
        raise CouplingControllerError(
            "vic completed but did not produce a fresh restart state file for the current coupling window"
        )

    return _mean_water_balance_to_2d_flux(vic_water_balance_array), next_state_tag



def _resolve_vic_spawn_preload_library() -> str | None:
    configured_value = os.environ.get("vic_spawn_preload")
    if configured_value:
        configured_path = Path(os.path.expanduser(configured_value)).resolve()
        if configured_path.is_dir():
            configured_path = configured_path / "libvic_parent_disconnect.so"
        if configured_path.exists():
            return str(configured_path)

    local_default = Path.cwd() / "libvic_parent_disconnect.so"
    if local_default.exists():
        return str(local_default.resolve())

    return None



def _configure_local_thread_environment(*, thread_count: str) -> None:
    # the controller should stay single-threaded unless there is a deliberate
    # reason not to. otherwise numpy and linked math libraries can compete with
    # mpi collectives for cpu time and make the run harder to reproduce.
    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["OPENBLAS_NUM_THREADS"] = thread_count
    os.environ["MKL_NUM_THREADS"] = thread_count
    os.environ["NUMEXPR_NUM_THREADS"] = thread_count



def _abort_world(world: MPI.Comm, logger: LoggerLike, message: str) -> None:
    try:
        logger.error(message)
    except Exception:
        pass
    world.Abort(1)



def _detect_latest_vic_state_tag(
    vic_runtime: VicImageDriverRuntime,
    logger: LoggerLike,
    *,
    since_epoch: float | None = None,
) -> str | None:
    try:
        state_tag = vic_runtime.latest_state_tag(since_epoch=since_epoch)
        if state_tag:
            return str(state_tag)
    except Exception:
        pass

    outputs_directory = getattr(vic_runtime, "outputs_directory", None)
    if not outputs_directory:
        return None

    candidate_files: list[str] = []
    for pattern in (
        os.path.join(str(outputs_directory), "state.*_00000.nc"),
        os.path.join(str(outputs_directory), "state.*.nc"),
    ):
        candidate_files.extend(glob.glob(pattern))

    if not candidate_files:
        logger.error(f"no vic state files were found in: {outputs_directory}")
        return None

    if since_epoch is not None:
        candidate_files = [path for path in candidate_files if os.path.getmtime(path) >= float(since_epoch)]
        if not candidate_files:
            logger.error(
                "no fresh vic state files were written in the outputs directory "
                f"since_epoch={float(since_epoch):.3f} outputs_directory={outputs_directory}"
            )
            return None

    latest_file = max(candidate_files, key=os.path.getmtime)
    match = re.search(r"\.(\d{8})(?:_00000)?\.nc$", os.path.basename(latest_file))
    if match is None:
        logger.error(f"failed to parse a state tag from: {latest_file}")
        return None

    state_tag = match.group(1)
    logger.info(f"latest vic state tag detected: {state_tag} from {latest_file}")
    return state_tag



def _read_vic_water_balance_for_window(
    *,
    vic_runtime: VicImageDriverRuntime,
    window_start: datetime,
    window_end: datetime,
    duration_days: int,
    since_epoch: float | None = None,
) -> np.ndarray:
    if duration_days <= 1:
        array = vic_runtime.read_water_balance_near(window_start, since_epoch=since_epoch)
    else:
        array = vic_runtime.read_water_balance_for_period(
            window_start,
            window_end,
            since_epoch=since_epoch,
        )

    if array is None:
        raise CouplingControllerError(
            f"failed to read vic water-balance output for window {window_start.date()} to {window_end.date()}"
        )

    return np.asarray(array, dtype=float)



def _mean_water_balance_to_2d_flux(water_balance_array: np.ndarray) -> np.ndarray:
    """reduce a vic output array to a 2d mean flux field in millimeters per day."""

    array = np.asarray(water_balance_array, dtype=float)
    if array.ndim == 3:
        return np.nanmean(array, axis=0)
    if array.ndim == 2:
        return array
    raise CouplingControllerError(f"unexpected vic water-balance array shape: {array.shape}")


if __name__ == "__main__":
    raise SystemExit(main())
