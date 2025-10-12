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
This module provides VIC image driver helpers to manage per-month runs and read
water-balance/state outputs.
"""


from __future__ import annotations

import os
import glob
import shutil
import subprocess
from datetime import datetime
from typing import Optional

from netCDF4 import Dataset


class VICModel:
    def __init__(
        self,
        vic_dir: str,
        vic_exe: str,
        global_param: str,
        outputs_dir: str,
        exchange_dir: str,
        params_file: str,
        logger,
    ) -> None:
        """Wrap the VIC image driver and related file paths."""
        self.vic_dir = os.path.expanduser(vic_dir)
        self.vic_exe = os.path.expanduser(vic_exe)
        self.global_param = os.path.expanduser(global_param)
        self.outputs_dir = os.path.expanduser(outputs_dir)
        self.exchange_dir = os.path.expanduser(exchange_dir)
        self.params_file = os.path.expanduser(params_file)
        self.state_file_prefix: Optional[str] = None
        self.wb_file_prefix: Optional[str] = None
        self.logger = logger

    def update_global_param(
        self,
        date_tag: str,
        sp_start: datetime,
        sp_end: datetime,
        prev_date: Optional[str] = None,
        first: bool = False,
    ) -> str:
        """Create a per-period global_param with start/end dates and optional INIT_STATE."""
        src_param = self.global_param
        sp_param = os.path.join(self.vic_dir, f"global_param_{date_tag}.txt")
        try:
            if not os.path.exists(src_param):
                raise FileNotFoundError(f"global param not found: {src_param}")
            shutil.copy(src_param, sp_param)
            lines = open(sp_param, "r", encoding="utf-8").readlines()

            # parse prefixes
            self.state_file_prefix = None
            self.wb_file_prefix = None
            for ln in lines:
                parts = ln.strip().split()
                if len(parts) >= 2 and parts[0] == "STATENAME":
                    self.state_file_prefix = parts[1]
                if len(parts) >= 2 and parts[0] == "OUTFILE":
                    self.wb_file_prefix = parts[1]
            if not self.state_file_prefix:
                raise RuntimeError("STATENAME not found in param file")
            if not self.wb_file_prefix:
                self.wb_file_prefix = "wbal"

            # rewrite time fields and init state
            new_lines = []
            for ln in lines:
                t = ln.strip()
                if t.startswith("INIT_STATE"):
                    new_lines.append("#" + ln)
                    if not first and prev_date:
                        new_lines.append(
                            f"INIT_STATE  {self.state_file_prefix}.{prev_date}_00000.nc\n"
                        )
                elif t.startswith("STARTYEAR"):
                    new_lines.append(f"STARTYEAR   {sp_start.year:04d}\n")
                elif t.startswith("STARTMONTH"):
                    new_lines.append(f"STARTMONTH  {sp_start.month:02d}\n")
                elif t.startswith("STARTDAY"):
                    new_lines.append(f"STARTDAY    {sp_start.day:02d}\n")
                elif t.startswith("ENDYEAR"):
                    new_lines.append(f"ENDYEAR     {sp_end.year:04d}\n")
                elif t.startswith("ENDMONTH"):
                    new_lines.append(f"ENDMONTH    {sp_end.month:02d}\n")
                elif t.startswith("ENDDAY"):
                    new_lines.append(f"ENDDAY      {sp_end.day:02d}\n")
                elif t.startswith("STATEYEAR"):
                    new_lines.append(f"STATEYEAR   {sp_end.year:04d}\n")
                elif t.startswith("STATEMONTH"):
                    new_lines.append(f"STATEMONTH  {sp_end.month:02d}\n")
                elif t.startswith("STATEDAY"):
                    new_lines.append(f"STATEDAY    {sp_end.day:02d}\n")
                else:
                    new_lines.append(ln)

            with open(sp_param, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            self.logger.info(f"wrote {sp_param}")
            return sp_param
        except Exception as e:
            self.logger.error(f"update_global_param failed: {e}")
            raise

    def run(self, sp_param: str) -> bool:
        """Execute VIC image driver for the given per-period global_param."""
        self.logger.info(f"running vic: {sp_param}")
        try:
            if not os.path.exists(self.vic_exe):
                raise FileNotFoundError(f"vic exe not found: {self.vic_exe}")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            cmd = [self.vic_exe, "-g", os.path.basename(sp_param)]
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=True,
                cwd=self.vic_dir,
            )
            self.logger.info(f"vic stdout: {result.stdout}")
            self.logger.info("vic completed")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"vic failed: {e}")
            self.logger.error(f"vic stderr: {e.stderr}")
            log_pattern = os.path.join(self.vic_dir, "logs", "vic.log.*.txt")
            log_files = glob.glob(log_pattern)
            if log_files:
                latest_log = max(log_files, key=os.path.getctime)
                try:
                    with open(latest_log, "r", encoding="utf-8") as f:
                        self.logger.error(f"vic log: {latest_log}\n{f.read()}")
                except Exception:
                    pass
            return False
        except Exception as e:
            self.logger.error(f"vic run error: {e}")
            return False

    def move_files(self, state_date_tag: str, wbal_date_tag: str) -> None:
        """Report the presence of VIC state (end-of-month) and wbal (first-of-month) files under outputs."""
        state_fn = f"{self.state_file_prefix}.{state_date_tag}_00000.nc" if self.state_file_prefix else None
        if state_fn:
            state_path = os.path.join(self.outputs_dir, state_fn)
            if os.path.exists(state_path):
                self.logger.info(f"State file exists: {state_path}")
            else:
                self.logger.error(f"Missing state: {state_path}")

        wbal_fn = f"{self.wb_file_prefix}.{wbal_date_tag}.nc" if self.wb_file_prefix else None
        if wbal_fn:
            wbal_path = os.path.join(self.outputs_dir, wbal_fn)
            if os.path.exists(wbal_path):
                self.logger.info(f"Water balance file exists: {wbal_path}")
            else:
                self.logger.error(f"Missing wbal: {wbal_path}")

    def read_vic_wb(self, wbal_date_tag: str):
        """Read OUT_BASEFLOW (mm) from wbal.<YYYY-MM-DD>.nc under outputs."""
        wbal_fn = f"{self.wb_file_prefix}.{wbal_date_tag}.nc" if self.wb_file_prefix else None
        if not wbal_fn:
            self.logger.error("wbal prefix not set")
            return None
        wbal_file = os.path.join(self.outputs_dir, wbal_fn)
        self.logger.info(f"reading wbal: {wbal_file}")
        try:
            if not os.path.exists(wbal_file):
                self.logger.warning(f"wbal file not found: {wbal_file}")
                return None
            ds = Dataset(wbal_file, "r")
            if "OUT_BASEFLOW" not in ds.variables:
                self.logger.warning("OUT_BASEFLOW not found")
                ds.close()
                return None
            baseflow = ds.variables["OUT_BASEFLOW"][:]
            ds.close()
            try:
                self.logger.info(f"OUT_BASEFLOW mean: {baseflow.mean():.6f} mm")
            except Exception:
                pass
            return baseflow
        except Exception as e:
            self.logger.error(f"read_vic_wb failed: {e}")
            try:
                ds.close()
            except Exception:
                pass
            return None
