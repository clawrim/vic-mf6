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

"""spawn the vic image driver from an mpi controller rank.

this module wraps the parent-side spawn logic used by the coupling controller.
keeping it separate from the controller improves readability because the caller
can describe *when* vic should run, while this module owns the lower-level mpi
spawn details.

the preload library support is preserved because some parent-child shutdown
patterns between mpi implementations and externally launched codes can hang on
Disconnect(). in this project that preload step is part of the runtime contract,
not an optional convenience.
"""

import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from mpi4py import MPI


class LoggerLike(Protocol):
    def info(self, message: str) -> None: ...


class VicMpiSpawnError(RuntimeError):
    """raised when the mpi spawn request is invalid or vic does not exit cleanly."""


@dataclass(frozen=True, slots=True)
class VicMpiSpawnRequest:
    """all information required to launch vic through MPI.COMM_SELF.Spawn."""

    executable_path: str
    working_directory: str
    global_parameter_path: str
    mpi_process_count: int
    omp_thread_count: int = 1
    preload_library_path: str | None = None
    timeout_seconds: int = 3600


def run_vic_image_driver_with_mpi_spawn(
    request: VicMpiSpawnRequest,
    logger: LoggerLike | None = None,
) -> None:
    """spawn vic and block until the child ranks disconnect.

    this function fails fast on invalid paths and invalid process counts because
    an mpi spawn failure deep inside the runtime is harder to diagnose than a
    direct path or environment error.
    """

    validated_request = _validate_spawn_request(request)
    _configure_local_thread_environment(validated_request.omp_thread_count)

    info = MPI.Info.Create()
    try:
        info.Set("wdir", validated_request.working_directory)
        info.Set("env", _build_spawn_environment_lines(validated_request))

        if logger is not None:
            logger.info(
                "spawning vic image driver "
                f"ranks={validated_request.mpi_process_count} "
                f"directory={validated_request.working_directory}"
            )

        _spawn_with_timeout(validated_request, info)

        if logger is not None:
            logger.info("vic image driver finished and disconnected cleanly")
    finally:
        info.Free()


# compatibility aliases kept inside the final module.
VicSpawnConfig = VicMpiSpawnRequest
run_vic_spawn = run_vic_image_driver_with_mpi_spawn


def _validate_spawn_request(request: VicMpiSpawnRequest) -> VicMpiSpawnRequest:
    if request.mpi_process_count < 1:
        raise VicMpiSpawnError(
            f"mpi_process_count must be greater than or equal to one, got {request.mpi_process_count}"
        )
    if request.omp_thread_count < 1:
        raise VicMpiSpawnError(
            f"omp_thread_count must be greater than or equal to one, got {request.omp_thread_count}"
        )
    if request.timeout_seconds < 1:
        raise VicMpiSpawnError(
            f"timeout_seconds must be greater than or equal to one, got {request.timeout_seconds}"
        )

    executable_path = Path(request.executable_path).expanduser().resolve()
    working_directory = Path(request.working_directory).expanduser().resolve()
    global_parameter_path = Path(request.global_parameter_path).expanduser().resolve()

    if not executable_path.exists():
        raise FileNotFoundError(f"vic executable was not found: {executable_path}")
    if not executable_path.is_file():
        raise VicMpiSpawnError(f"vic executable path is not a file: {executable_path}")
    if not working_directory.exists():
        raise FileNotFoundError(
            f"vic working directory was not found: {working_directory}"
        )
    if not working_directory.is_dir():
        raise VicMpiSpawnError(
            f"vic working directory is not a directory: {working_directory}"
        )
    if not global_parameter_path.exists():
        raise FileNotFoundError(
            f"vic global parameter file was not found: {global_parameter_path}"
        )
    if not global_parameter_path.is_file():
        raise VicMpiSpawnError(
            f"vic global parameter path is not a file: {global_parameter_path}"
        )

    preload_library_path: str | None = None
    if request.preload_library_path is not None:
        preload_candidate = Path(request.preload_library_path).expanduser().resolve()
        if not preload_candidate.exists():
            raise FileNotFoundError(
                f"preload library was not found: {preload_candidate}"
            )
        if not preload_candidate.is_file():
            raise VicMpiSpawnError(
                f"preload library path is not a file: {preload_candidate}"
            )
        preload_library_path = str(preload_candidate)

    return VicMpiSpawnRequest(
        executable_path=str(executable_path),
        working_directory=str(working_directory),
        global_parameter_path=str(global_parameter_path),
        mpi_process_count=int(request.mpi_process_count),
        omp_thread_count=int(request.omp_thread_count),
        preload_library_path=preload_library_path,
        timeout_seconds=int(request.timeout_seconds),
    )


def _configure_local_thread_environment(omp_thread_count: int) -> None:
    thread_count = str(omp_thread_count)

    # the controller rank should not accidentally oversubscribe the machine.
    # if numpy or a linked blas uses many threads while the same process is
    # managing mpi spawn and large broadcasts, performance becomes erratic.
    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["OPENBLAS_NUM_THREADS"] = thread_count
    os.environ["MKL_NUM_THREADS"] = thread_count
    os.environ["NUMEXPR_NUM_THREADS"] = thread_count


def _build_spawn_environment_lines(request: VicMpiSpawnRequest) -> str:
    environment_lines = [
        f"OMP_NUM_THREADS={request.omp_thread_count}",
        "OMP_DYNAMIC=FALSE",
        f"OPENBLAS_NUM_THREADS={request.omp_thread_count}",
        f"MKL_NUM_THREADS={request.omp_thread_count}",
        f"NUMEXPR_NUM_THREADS={request.omp_thread_count}",
    ]

    if request.preload_library_path:
        environment_lines.append(f"LD_PRELOAD={request.preload_library_path}")

    return "\n".join(environment_lines)


def _spawn_with_timeout(request: VicMpiSpawnRequest, info: MPI.Info) -> None:
    def _alarm_handler(signum: int, frame: object) -> None:
        raise TimeoutError(
            "timeout while waiting for vic to disconnect from the parent rank"
        )

    previous_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(int(request.timeout_seconds))

    try:
        intercommunicator = MPI.COMM_SELF.Spawn(
            request.executable_path,
            args=["-g", Path(request.global_parameter_path).name],
            maxprocs=request.mpi_process_count,
            info=info,
        )
        intercommunicator.Disconnect()
    except TimeoutError as exc:
        raise VicMpiSpawnError(str(exc)) from exc
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
