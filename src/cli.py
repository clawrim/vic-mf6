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
This module provides the entrypoint to load config and run the monthly VIC–MF6
coupling workflow.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


# allow both:
#   python -m ems.cli -c config.yaml         
#   (when src is on PYTHONPATH or package is installed) 

#   python src/ems/cli.py -c config.yaml
#   (directscript run from repo root)
try:
    from .mf6 import MF6Model
    from .vic import VICModel
    from .coupling import CouplingManager
except Exception:
    ROOT = Path(__file__).resolve().parent
    PARENT = ROOT.parent
    if str(PARENT) not in sys.path:
        sys.path.insert(0, str(PARENT))
    from ems.mf6 import MF6Model  # type: ignore
    from ems.vic import VICModel  # type: ignore
    from ems.coupling import CouplingManager  # type: ignore


def _setup_logger() -> logging.Logger:
    """Console logger with simple formatting."""
    log = logging.getLogger("ems")
    if not log.handlers:
        log.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(levelname)s: %(message)s")
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def _load_yaml(path: str) -> dict:
    """Read a minimal YAML without adding a dependency."""
    import yaml

    p = Path(path).expanduser()
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv: list[str] | None = None) -> int:
    """Parse config and orchestrate the VIC–MF6 coupling."""
    parser = argparse.ArgumentParser(prog="ems", description="vic–mf6 coupling")
    parser.add_argument("-c", "--config", required=True, help="path to config.yaml")
    ns = parser.parse_args(argv)

    log = _setup_logger()

    try:
        cfg = _load_yaml(ns.config)
    except Exception as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    try:
        mf6_cfg = cfg["mf6"]
        vic_cfg = cfg["vic"]
        coup_cfg = cfg["coupling"]
    except Exception:
        print("config missing required sections: mf6, vic, coupling", file=sys.stderr)
        return 2

    try:
        mf6 = MF6Model(
            workspace=mf6_cfg["workspace"],
            mf6_dll=mf6_cfg["dll"],
            logger=log,
            start_date=datetime.fromisoformat(str(mf6_cfg.get("start_date", "1940-03-01"))),
        )
        mf6.initialize()
    except Exception as e:
        log.error(f"mf6 init failed: {e}")
        return 1

    vic = VICModel(
        vic_dir=vic_cfg["dir"],
        vic_exe=vic_cfg["exe"],
        global_param=vic_cfg["global_param"],
        outputs_dir=vic_cfg["outputs_dir"],
        exchange_dir=vic_cfg["exchange_dir"],
        params_file=vic_cfg["params_file"],
        logger=log,
    )

    os.makedirs(vic_cfg["exchange_dir"], exist_ok=True)
    log_file = os.path.join(vic_cfg["exchange_dir"], "coupling_log.csv")

    cm = CouplingManager(
        mf6_model=mf6,
        vic_model=vic,
        coupling_table_csv=coup_cfg["table_csv"],
        params_file=vic_cfg["params_file"],
        log_file=log_file,
        logger=log,
        vic_grid_shape=tuple(coup_cfg["vic_grid_shape"]),
        model_name=mf6_cfg["model_name"],
    )

    try:
        cm.initialize()
    except Exception as e:
        log.error(f"coupling initialize failed: {e}")
        try:
            mf6.finalize()
        except Exception:
            pass
        return 1

    try:
        start_date = datetime.fromisoformat(str(coup_cfg["start_date"]))
        end_date = datetime.fromisoformat(str(coup_cfg["end_date"]))
        cm.run(start_date, end_date)
    except Exception as e:
        log.error(f"run failed: {e}")
        try:
            mf6.finalize()
        except Exception:
            pass
        return 1

    try:
        mf6.finalize()
    except Exception as e:
        log.error(f"mf6 finalize failed: {e}")
        return 1

    log.info("done")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
