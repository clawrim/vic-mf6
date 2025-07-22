#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import subprocess
import glob
import shutil
from datetime import datetime, timedelta
from netCDF4 import Dataset
import xmipy
import logging
import calendar

class MF6Model:
    def __init__(self, workspace, mf6_dll, logger):
        self.workspace = os.path.expanduser(workspace)
        self.mf6_dll = os.path.expanduser(mf6_dll)
        self.mf6 = None
        self.model_name = None
        self.bmi_vars = []
        self.bmi_prefix = None
        self.nrow = None
        self.ncol = None
        self.nlay = None
        self.nper = None
        self.nstp = None
        self.idomain = None
        self.nuzfcells = None
        self.var_finf = None
        self.var_gwd = None
        self.logger = logger
        self.start_date = datetime(1940, 3, 1)  # Manual start date from TDIS

    def initialize(self):
        """Initialize MF6 model using BMI interface."""
        try:
            if not os.path.exists(self.mf6_dll):
                raise FileNotFoundError(f"MF6 DLL not found: {self.mf6_dll}")
            if not os.path.exists(self.workspace):
                raise FileNotFoundError(f"MF6 workspace not found: {self.workspace}")
            self.mf6 = xmipy.XmiWrapper(self.mf6_dll, working_directory=self.workspace)
            self.mf6.initialize()
            self.logger.info("MF6 initialized")
        except Exception as e:
            self.logger.error(f"Error initializing MF6: {e}")
            raise

        self._parse_name_file()
        self._load_bmi_vars()
        self._resolve_bmi_prefix()
        self._parse_grid_info()
        self._parse_tdis_info()
        self._identify_uzf_vars()

        self.logger.info(f"Grid: nlay={self.nlay}, nrow={self.nrow}, ncol={self.ncol}")
        self.logger.info(f"TDIS: nper={self.nper}, nstp={self.nstp}")
        self.logger.info(f"UZF cells={self.nuzfcells}")

    def _parse_name_file(self):
        """Parse the name file to extract model name."""
        try:
            fn = next(f for f in os.listdir(self.workspace) if f.lower().endswith(".nam"))
            with open(os.path.join(self.workspace, fn), "r") as f:
                for line in f:
                    t = line.strip().split()
                    if t and t[0].lower() == "modelname":
                        self.model_name = t[1]
                        break
            self.logger.info(f"Name file: model={self.model_name}")
        except StopIteration:
            self.logger.error("Error: No .nam file found in workspace")
            raise
        except Exception as e:
            self.logger.error(f"Error parsing name file: {e}")
            raise

    def _load_bmi_vars(self):
        """Load BMI input and output variable names."""
        try:
            inp = self.mf6.get_input_var_names()
            out = self.mf6.get_output_var_names()
            self.bmi_vars = set(inp + out)
            self.logger.info(f"Loaded {len(self.bmi_vars)} BMI variables")
        except Exception as e:
            self.logger.error(f"Error loading BMI variables: {e}")
            raise

    def _resolve_bmi_prefix(self):
        """Resolve BMI prefix for variable names."""
        try:
            for v in self.bmi_vars:
                if v.endswith("/DIS/NROW"):
                    self.bmi_prefix = v.rsplit("/DIS/NROW", 1)[0]
                    break
            if not self.bmi_prefix:
                raise Exception("BMI prefix not found")
            self.logger.info(f"Using prefix: {self.bmi_prefix}")
        except Exception as e:
            self.logger.error(f"Error resolving BMI prefix: {e}")
            raise

    def _parse_grid_info(self):
        """Parse grid information (nlay, nrow, ncol, idomain)."""
        try:
            p = self.bmi_prefix
            self.nrow = int(self.mf6.get_value_ptr(f"{p}/DIS/NROW")[0])
            self.ncol = int(self.mf6.get_value_ptr(f"{p}/DIS/NCOL")[0])
            self.nlay = int(self.mf6.get_value_ptr(f"{p}/DIS/NLAY")[0])
            arr = self.mf6.get_value_ptr(f"{p}/DIS/IDOMAIN")
            self.idomain = arr.reshape(self.nlay, self.nrow, self.ncol)
            self.nuzfcells = int((self.idomain[0] == 1).sum())
        except Exception as e:
            self.logger.error(f"Error parsing grid info: {e}")
            raise

    def _parse_tdis_info(self):
        """Parse time discretization information (nper, nstp) using manual start date."""
        try:
            nper_vars = [v for v in self.bmi_vars if v.endswith("/NPER")]
            nstp_vars = [v for v in self.bmi_vars if v.endswith("/NSTP")]
            self.nper = int(self.mf6.get_value_ptr(nper_vars[0])[0]) if nper_vars else 1
            self.nstp = int(self.mf6.get_value_ptr(nstp_vars[0])[0]) if nstp_vars else 1
            if not nper_vars:
                self.logger.warning("NPER not found; default=1")
            if not nstp_vars:
                self.logger.warning("NSTP not found; default=1")
        except Exception as e:
            self.logger.error(f"Error parsing TDIS info: {e}")
            raise

    def _identify_uzf_vars(self):
        """Identify UZF variables (finf, gwd)."""
        try:
            for v in self.bmi_vars:
                if v.endswith("/UZF/FINF"):
                    self.var_finf = v
                elif v.endswith("/UZF/GWD"):
                    self.var_gwd = v
            if not self.var_gwd:
                raise Exception("UZF GWD variable not found")
            self.logger.info(f"Variables: finf={self.var_finf}, gwd={self.var_gwd}")
        except Exception as e:
            self.logger.error(f"Error identifying UZF variables: {e}")
            raise

    def run_to_date(self, end_date, start_date):
        """Run MF6 from start_date to end_date."""
        try:
            # calculate stress periods relative to 3/1/1940
            start_time = (start_date - datetime(1940, 3, 1)).total_seconds() / 86400.0
            end_time = (end_date - datetime(1940, 3, 1)).total_seconds() / 86400.0
            # the current stress period relative to 3/1/1940
            current_time = self.mf6.get_current_time()

            if current_time < start_time:
                self.logger.warning(f"Current time {current_time} before start time {start_time}; adjusting")
                # TODO 1: I don't think this block does anything because it won't
                # affect the mf6 model and current_time is retrieved again from
                # the mf6 model within the while loop below
                current_time = start_time

            while current_time < end_time:
                self.mf6.prepare_time_step(0.0)
                self.mf6.do_time_step()
                self.mf6.finalize_time_step()
                current_time = self.mf6.get_current_time()
            self.logger.info(f"MF6 ran to {end_date}")
        except Exception as e:
            self.logger.error(f"Error running MF6 to date: {e}")
            raise

    def run_timestep(self):
        """Run a single MF6 timestep."""
        try:
            self.mf6.prepare_time_step(0.0)
            self.mf6.do_time_step()
            self.mf6.finalize_time_step()
        except Exception as e:
            self.logger.error(f"Error running MF6 timestep: {e}")
            raise

    def finalize(self):
        """Finalize MF6 model."""
        try:
            self.mf6.finalize()
            self.logger.info("MF6 finalized")
        except Exception as e:
            self.logger.error(f"Error finalizing MF6: {e}")
            raise

class VICModel:
    def __init__(self, vic_dir, vic_exe, global_param, outputs_dir, exchange_dir, params_file, logger):
        self.vic_dir = os.path.expanduser(vic_dir)
        self.vic_exe = os.path.expanduser(vic_exe)
        self.global_param = os.path.expanduser(global_param)
        self.outputs_dir = os.path.expanduser(outputs_dir)
        self.exchange_dir = os.path.expanduser(exchange_dir)
        self.params_file = os.path.expanduser(params_file)
        self.state_file_prefix = None
        self.wb_file_prefix = None
        self.logger = logger

    def update_global_param(self, date_tag, sp_start, sp_end, prev_date=None, first=False):
        """Update VIC global parameter file for a stress period."""
        src_param = self.global_param
        sp_param = os.path.join(self.vic_dir, f"global_param_{date_tag}.txt")
        self.logger.info(f"Updating global_param for period ending {sp_end}")
        try:
            if not os.path.exists(src_param):
                raise FileNotFoundError(f"Global param file not found: {src_param}")
            shutil.copy(src_param, sp_param)
            lines = open(sp_param).readlines()

            # Parse STATENAME and OUTFILE prefixes
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

            # Update parameters
            new_lines = []
            for ln in lines:
                t = ln.strip()
                if t.startswith("INIT_STATE"):
                    new_lines.append("#" + ln)  # Comment out old init
                    if not first and prev_date:
                        new_lines.append(f"INIT_STATE  {self.state_file_prefix}.{prev_date}_00000.nc\n")
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

            with open(sp_param, "w") as f:
                f.writelines(new_lines)
            self.logger.info(f"Updated global_param: {sp_param}")
            return sp_param
        except Exception as e:
            self.logger.error(f"Error updating global_param: {e}")
            raise

    def run(self, sp_param):
        """Run VIC model for a given global parameter file."""
        self.logger.info(f"Running VIC with {sp_param}")
        try:
            if not os.path.exists(self.vic_exe):
                raise FileNotFoundError(f"VIC executable not found: {self.vic_exe}")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            cmd = [self.vic_exe, "-g", os.path.basename(sp_param)]
            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True, check=True, cwd=self.vic_dir
            )
            self.logger.info(f"VIC stdout: {result.stdout}")
            self.logger.info("VIC completed")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"VIC failed: {e}")
            self.logger.error(f"VIC stderr: {e.stderr}")
            log_pattern = os.path.join(self.vic_dir, "logs/vic.log.*.txt")
            log_files = glob.glob(log_pattern)
            if log_files:
                latest_log = max(log_files, key=os.path.getctime)
                with open(latest_log, "r") as f:
                    self.logger.error(f"VIC log: {latest_log}\n{f.read()}")
            return False
        except Exception as e:
            self.logger.error(f"Error running VIC: {e}")
            return False

    def move_files(self, state_date_tag, wbal_date_tag):
        """Handle VIC state and water balance files in outputs directory."""
        state_fn = f"{self.state_file_prefix}.{state_date_tag}_00000.nc"
        state_path = os.path.join(self.vic_dir, state_fn)
        if os.path.exists(state_path):
            self.logger.info(f"State file exists: {state_path}")
        else:
            self.logger.warning(f"Missing state file: {state_path}")

        wbal_fn = f"{self.wb_file_prefix}.{wbal_date_tag}.nc"
        wbal_path = os.path.join(self.outputs_dir, wbal_fn)
        if os.path.exists(wbal_path):
            self.logger.info(f"Water balance file exists: {wbal_path}")
        else:
            self.logger.warning(f"Missing wbal file: {wbal_path}")

    def read_vic_wb(self, wbal_date_tag):
        """Read VIC water balance (OUT_BASEFLOW) from outputs directory."""
        wbal_fn = f"{self.wb_file_prefix}.{wbal_date_tag}.nc"
        wbal_file = os.path.join(self.outputs_dir, wbal_fn)
        self.logger.info(f"Reading water balance: {wbal_file}")
        try:
            if not os.path.exists(wbal_file):
                self.logger.warning(f"Water balance file not found: {wbal_file}")
                return None
            ds = Dataset(wbal_file, "r")
            if "OUT_BASEFLOW" not in ds.variables:
                self.logger.warning("OUT_BASEFLOW not found")
                ds.close()
                return None
            baseflow = ds.variables["OUT_BASEFLOW"][:]
            ds.close()
            self.logger.info(f"OUT_BASEFLOW shape: {baseflow.shape}")
            self.logger.info(f"OUT_BASEFLOW mean: {baseflow.mean():.6f} mm")
            return baseflow
        except Exception as e:
            self.logger.error(f"Error reading water balance: {e}")
            if "ds" in locals():
                ds.close()
            return None

class CouplingManager:
    def __init__(self, mf6_model, vic_model, coupling_table_csv, params_file, log_file, logger):
        self.mf6 = mf6_model
        self.vic = vic_model
        self.coupling_table = pd.read_csv(coupling_table_csv)
        self.params_file = os.path.expanduser(params_file)
        self.log_file = os.path.expanduser(log_file)
        self.logger = logger
        self.mm_to_ft = 1 / 304.8
        self.ft_to_mm = 3.28084 * 1000
        self.n_cells = self.mf6.nuzfcells
        self.vic_grid_shape = (227, 212)
        self.vic_id_to_indices = {}
        self.vic_lat = None
        self.vic_lon = None
        self.uzf_file = os.path.join(self.mf6.workspace, "rgtihm_2100.uzf")
        self.skipped_ids_file = os.path.join(self.vic.exchange_dir, "skipped_mf6_ids.txt")
        self.uzf_mapping = self._load_uzf_mapping()

    def _load_uzf_mapping(self):
        """Read UZF file and create mf6_id to iuzno mapping."""
        try:
            if not os.path.exists(self.uzf_file):
                self.logger.error(f"UZF file not found: {self.uzf_file}")
                return {}
            with open(self.uzf_file, 'r') as f:
                lines = f.readlines()
            start_idx = next(i for i, line in enumerate(lines) if 'BEGIN packagedata' in line) + 1
            end_idx = next(i for i, line in enumerate(lines) if 'END packagedata' in line)
            uzf_data = [line.split()[:4] for line in lines[start_idx:end_idx]]  # iuzno, nlay, nrow, ncol
            uzf_df = pd.DataFrame(uzf_data, columns=['iuzno', 'nlay', 'nrow', 'ncol'])
            uzf_df = uzf_df.astype({'iuzno': int, 'nlay': int, 'nrow': int, 'ncol': int})
            uzf_df['mf6_id'] = uzf_df.apply(lambda x: f"{x['nrow']:03d}{x['ncol']:03d}", axis=1)
            self.logger.info(f"Loaded {len(uzf_df)} UZF mappings")
            return dict(zip(uzf_df['mf6_id'], uzf_df['iuzno']))
        except Exception as e:
            self.logger.error(f"Error loading UZF mapping: {e}")
            return {}

    def _mf6_id_to_iuzno(self, mf6_id):
        """Convert mf6_id to iuzno using UZF mapping."""
        try:
            mf6_id_str = str(int(mf6_id)).zfill(6)  # Ensure 6 digits, e.g., '146227'
            if mf6_id_str in self.uzf_mapping:
                return self.uzf_mapping[mf6_id_str]
            self.logger.warning(f"Invalid mf6_id {mf6_id}: not found in UZF package")
            with open(self.skipped_ids_file, 'a') as f:
                f.write(f"{mf6_id}\n")
            return -1
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Invalid mf6_id format: {mf6_id}, error: {e}")
            with open(self.skipped_ids_file, 'a') as f:
                f.write(f"{mf6_id}\n")
            return -1

    def initialize(self):
        """Initialize coupling by setting up log file and validating coupling table."""
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "w") as f:
                f.write("stress_period,mf6_id,vic_id,finf_ft_per_day,uzf_gwd_ft_per_day,init_moist_mm\n")
            self.logger.info(f"Log file created: {self.log_file}")
            if not all(col in self.coupling_table for col in ["vic_id", "mf6_id", "area_ratio", "area_m2", "b_lat", "b_lon"]):
                raise ValueError("Coupling table missing required columns")
            # Validate mf6_id format
            self.coupling_table["mf6_id"] = self.coupling_table["mf6_id"].apply(lambda x: int(x) if pd.notna(x) else -1)
            invalid_mf6_id = self.coupling_table[self.coupling_table["mf6_id"] < 1000]
            if len(invalid_mf6_id) > 0:
                self.logger.warning(f"Invalid mf6_id count: {len(invalid_mf6_id)}")
                self.logger.warning(f"Sample invalid mf6_id: {invalid_mf6_id[['mf6_id', 'vic_id']].head(5).to_dict()}")
                with open(self.skipped_ids_file, 'a') as f:
                    for _, row in invalid_mf6_id.iterrows():
                        f.write(f"{row['mf6_id']},{row['vic_id']}\n")
            self.coupling_table["iuzno"] = self.coupling_table["mf6_id"].apply(self._mf6_id_to_iuzno)
            # Validate iuzno values
            invalid_iuzno = self.coupling_table[self.coupling_table["iuzno"] < 0]
            if len(invalid_iuzno) > 0:
                self.logger.info(f"Invalid iuzno count: {len(invalid_iuzno)}")
                self.logger.info(f"Sample invalid iuzno: {invalid_iuzno[['mf6_id', 'iuzno', 'vic_id']].head(5).to_dict()}")
                with open(self.skipped_ids_file, 'a') as f:
                    for _, row in invalid_iuzno.iterrows():
                        f.write(f"{row['mf6_id']},{row['vic_id']}\n")
            valid_iuzno = self.coupling_table[(self.coupling_table["iuzno"] >= 0) & (self.coupling_table["iuzno"] < self.n_cells)]
            self.logger.info(f"Valid iuzno count: {len(valid_iuzno)}")
            if len(valid_iuzno) == 0:
                raise ValueError("No valid iuzno mappings found; check mf6_vic_join.csv or UZF package")
            self.coupling_table = valid_iuzno  # Keep only valid mappings
            self.logger.info(f"Coupling table loaded with {len(self.coupling_table)} valid mappings")
            self._initialize_vic_id_mapping()
            # Verify area_ratio sums to 1 per mf6_id
            ratio_sum = self.coupling_table.groupby("mf6_id")["area_ratio"].sum()
            invalid_ratios = ratio_sum[~np.isclose(ratio_sum, 1.0, atol=1e-6)]
            if not invalid_ratios.empty:
                self.logger.warning(f"Area ratios do not sum to 1 for {len(invalid_ratios)} mf6_id: {invalid_ratios.head().to_dict()}")
        except Exception as e:
            self.logger.error(f"Error initializing coupling: {e}")
            raise

    def _initialize_vic_id_mapping(self):
        """Map vic_id to (lat_idx, lon_idx) using b_lat and b_lon."""
        try:
            ds = Dataset(self.params_file, "r")
            lat = ds.variables["lat"][:]
            lon = ds.variables["lon"][:]
            ds.close()
            # Handle 1D or 2D lat/lon arrays
            if lat.ndim == 2:
                self.vic_lat = lat[:, 0]  # Use first column for 1D lat
                self.vic_lon = lon[0, :]  # Use first row for 1D lon
            else:
                self.vic_lat = lat
                self.vic_lon = lon
            self.logger.info(f"vic_lat shape: {self.vic_lat.shape}, vic_lon shape: {self.vic_lon.shape}")
            for _, row in self.coupling_table.iterrows():
                vic_id = int(row["vic_id"])
                b_lat = row["b_lat"]
                b_lon = row["b_lon"]
                lat_idx = np.abs(self.vic_lat - b_lat).argmin()
                lon_idx = np.abs(self.vic_lon - b_lon).argmin()
                if not (np.isclose(self.vic_lat[lat_idx], b_lat, atol=1e-5) and np.isclose(self.vic_lon[lon_idx], b_lon, atol=1e-5)):
                    self.logger.warning(f"Could not map vic_id {vic_id}: b_lat={b_lat}, b_lon={b_lon} not close to grid")
                else:
                    self.vic_id_to_indices[vic_id] = (lat_idx, lon_idx)
            self.logger.info(f"Initialized vic_id mapping for {len(self.vic_id_to_indices)} IDs")
        except Exception as e:
            self.logger.error(f"Error initializing vic_id_mapping: {e}")
            if "ds" in locals():
                ds.close()
            raise

    def compute_finf(self, baseflow):
        """Compute infiltration (finf) for MF6 from VIC baseflow."""
        try:
            finf = np.zeros(self.n_cells)  # n_cells=17879 from UZF
            mm_to_ft = 0.00328084  # mm to ft conversion
            skipped_mf6_ids = []
            self.logger.info("Computing finf...")
            for idx, row in self.coupling_table.iterrows():
                if idx % 5000 == 0:
                    self.logger.info(f"Processing row {idx}/{len(self.coupling_table)}")
                if row["iuzno"] >= 0 and row["iuzno"] < self.n_cells:
                    iuzno = int(row["iuzno"])
                    lat_idx = np.abs(self.vic_lat - row["b_lat"]).argmin()
                    lon_idx = np.abs(self.vic_lon - row["b_lon"]).argmin()
                    try:
                        bf = baseflow[-1, lat_idx, lon_idx]  # Last time step
                        if bf > 0:  # Only apply positive baseflow
                            finf[iuzno] += bf * row["area_ratio"] * mm_to_ft
                        else:
                            self.logger.debug(f"Zero baseflow for vic_id {row['vic_id']}, iuzno {iuzno}")
                    except IndexError:
                        self.logger.warning(f"Index error for vic_id {row['vic_id']} at lat_idx={lat_idx}, lon_idx={lon_idx}")
                else:
                    skipped_mf6_ids.append(row["mf6_id"])
            # Save computed finf for debugging
            output_dir = self.vic.exchange_dir
            np.savetxt(os.path.join(output_dir, "computed_finf.txt"), finf)
            self.logger.info(f"Computed finf non-zero count: {(finf > 0).sum()}")
            self.logger.info(f"Skipped {len(skipped_mf6_ids)} mf6_id: {skipped_mf6_ids[:5] if skipped_mf6_ids else []}")
            self.logger.info(f"finf mean: {finf.mean():.6f} ft/day")
            self.logger.info(f"finf sample[:5]={finf[:5]}")
            return finf
        except Exception as e:
            self.logger.error(f"Error computing finf: {e}")
            return np.zeros(self.n_cells)

    def update_vic_params(self, uzf_gwd, baseflow):
        """Update VIC parameters (init_moist) using MF6 groundwater discharge."""
        try:
            self.logger.info(f"Reading {self.params_file}")
            ds = Dataset(self.params_file, "r")
            if "init_moist" not in ds.variables:
                self.logger.warning("init_moist not found")
                ds.close()
                return False
            init_moist_shape = ds.variables["init_moist"].shape
            ds.close()
            self.logger.info(f"init_moist shape: {init_moist_shape}")
            self.logger.info(f"Updating init_moist (layer 3) in {self.params_file}")
            try:
                valid_mappings = 0

                # TODO 3 hints: create a dictionary vic_gwd and initialize
                # vic_gwd[vic_id].gwd_mm to 0 for each vic_id
                for _, row in self.coupling_table.iterrows():
                    vic_id = int(row["vic_id"])
                    iuzno = int(row["iuzno"])
                    area_ratio = row["area_ratio"]
                    area_m2 = row["area_m2"]
                    if iuzno >= 0 and iuzno < self.n_cells and vic_id in self.vic_id_to_indices:
                        lat_idx, lon_idx = self.vic_id_to_indices[vic_id]
                        try:
                            # TODO 2: area_ratio (currently MF cell ratio) needs
                            # to be corrected to the VIC cell ratio
                            # TODO 2: rename area_ratio to mf6_area_ratio
                            # TODO 2: create a new column, vic_area_ratio

                            # TODO 3: accumulate gwd_mm for a VIC cell
                            # TODO 3: this calculation needs to be done per VIC
                            # cell, NOT per VIC-MF6 intersection area

                            # intersected area
                            # TODO 3: here, vic_gwd[vic_id]gwd_mm += ...
                            gwd_mm = uzf_gwd[iuzno] * self.ft_to_mm * vic_area_ratio






                            valid_mappings += 1
                        except IndexError:
                            self.logger.warning(f"Index error for vic_id {vic_id} at lat_idx={lat_idx}, lon_idx={lon_idx} or iuzno {iuzno}")
                    else:
                        self.logger.debug(f"Skipping invalid mapping: vic_id {vic_id}, iuzno {iuzno}")

                ds = Dataset(self.params_file, "r+", format="NETCDF4")
                init_moist = ds.variables["init_moist"]
                ds.close()

                # TODO 3: for each vic_id in vic_gwd
                    # extract lat_idx, lon_idx for vic_id

                    # TODO 3: cannot apply VIC-cell calculations here
                    # because next time you visit the same VIC cell
                    # from different MF6-VIC intersection area, those
                    # calculations will be overwritten!

                    # whole VIC cell
                    bf_mm = baseflow[-1, lat_idx, lon_idx] if baseflow is not None else 0.0
                    # TODO 3: here, gwd_mm should be accumulated per VIC cell
                    new_moist = vic_gwd[vic_id].gwd_mm + bf_mm if gwd_mm > 0 else 0.0
                    # whole VIC cell
                    init_moist[2, lat_idx, lon_idx] = min(max(new_moist, 0.1), 100.0)

                # TODO 4: write out init_moist

                self.logger.info(f"VIC params updated for {valid_mappings} valid mappings")
                return True
            except Exception as e:
                self.logger.error(f"Error updating VIC params with NETCDF4: {e}")
                if "ds" in locals():
                    ds.close()
                self.logger.info("Attempting update with NETCDF3_64BIT")
                try:
                    ds = Dataset(self.params_file, "r+", format="NETCDF3_64BIT")
                    init_moist = ds.variables["init_moist"]
                    valid_mappings = 0
                    for _, row in self.coupling_table.iterrows():
                        vic_id = int(row["vic_id"])
                        iuzno = int(row["iuzno"])
                        area_ratio = row["area_ratio"]
                        area_m2 = row["area_m2"]
                        if iuzno >= 0 and iuzno < self.n_cells and vic_id in self.vic_id_to_indices:
                            lat_idx, lon_idx = self.vic_id_to_indices[vic_id]
                            try:
                                gwd_mm = uzf_gwd[iuzno] * self.ft_to_mm * area_ratio * row["area_m2"] / 1e6
                                bf_mm = baseflow[-1, lat_idx, lon_idx] if baseflow is not None else 0.0
                                new_moist = gwd_mm + bf_mm if gwd_mm > 0 else 0.0
                                init_moist[2, lat_idx, lon_idx] = min(max(new_moist, 0.1), 100.0)
                                valid_mappings += 1
                            except IndexError:
                                self.logger.warning(f"Index error for vic_id {vic_id} at lat_idx={lat_idx}, lon_idx={lon_idx} or iuzno {iuzno}")
                        else:
                            self.logger.debug(f"Skipping invalid mapping: vic_id {vic_id}, iuzno {iuzno}")
                    ds.close()
                    self.logger.info(f"VIC params updated with NETCDF3_64BIT for {valid_mappings} valid mappings")
                    return True
                except Exception as e2:
                    self.logger.error(f"Error updating with NETCDF3_64BIT: {e2}")
                    if "ds" in locals():
                        ds.close()
                    return False
        except Exception as e:
            self.logger.error(f"Error accessing params file: {e}")
            if "ds" in locals():
                ds.close()
            return False

    def log_results(self, stress_period, finf, uzf_gwd, init_moist):
        """Log coupling results to CSV."""
        try:
            with open(self.log_file, "a") as f:
                for _, row in self.coupling_table.iterrows():
                    iuzno = int(row["iuzno"])
                    vic_id = int(row["vic_id"])
                    if iuzno >= 0 and iuzno < self.n_cells and vic_id in self.vic_id_to_indices:
                        try:
                            lat_idx, lon_idx = self.vic_id_to_indices[vic_id]
                            moist_val = init_moist[lat_idx, lon_idx] if init_moist is not None else 0.0
                            f.write(f"{stress_period},{row['mf6_id']},{vic_id},{finf[iuzno]:.6f},{uzf_gwd[iuzno]:.6f},{moist_val:.6f}\n")
                        except (IndexError, TypeError):
                            self.logger.warning(f"Index error logging for iuzno {iuzno}, vic_id {vic_id}")
                    else:
                        self.logger.warning(f"Skipping invalid mapping in log: mf6_id {row['mf6_id']}, vic_id {vic_id}")
            self.logger.info(f"Logged to {self.log_file}")
        except Exception as e:
            self.logger.error(f"Error logging: {e}")

    def run(self, vic_start_date, coupling_end_date):
        """Run coupled VIC-MF6 simulation with MF6 initiating from manual start to just before VIC."""
        self.logger.info("Starting full coupling with MF6 initiation")
        first = True
        prev_date = None
        mf6_current_date = self.mf6.start_date  # Manual start date from TDIS

        # Step 1: Run MF6 from its manual start date to just before VIC start date
        vic_start_minus_one_day = vic_start_date - timedelta(days=1)
        self.logger.info(f"Running MF6 from {mf6_current_date} to {vic_start_minus_one_day}")
        try:
            self.mf6.run_to_date(vic_start_minus_one_day, mf6_current_date)
            mf6_current_date = vic_start_minus_one_day
        except Exception as e:
            self.logger.error(f"Failed to run MF6 pre-VIC period: {e}")
            return

        # Main coupling loop: MF6 -> VIC -> MF6
        vic_current_date = vic_start_date
        while vic_current_date <= coupling_end_date:
            mf6_period_end = vic_current_date  # Align MF6 with VIC's current period
            vic_period_end = (vic_current_date + timedelta(days=calendar.monthrange(vic_current_date.year, vic_current_date.month)[1] - 1)) ## TODO 5: Explain this

            # Step 2: Run VIC for one stress period and stop
            self.logger.info(f"Running VIC for SP from {vic_current_date} to {vic_period_end}")
            state_date_tag = vic_period_end.strftime("%Y%m%d")
            wbal_date_tag = vic_current_date.strftime("%Y-%m-%d")
            sp_param = self.vic.update_global_param(wbal_date_tag, vic_current_date, vic_period_end, prev_date, first)
            if not self.vic.run(sp_param):
                self.logger.error(f"VIC run failed for SP starting {vic_current_date}")
                return
            self.vic.move_files(state_date_tag, wbal_date_tag)

            # Step 3: Apply VIC baseflow to MF6 finf
            self.logger.info(f"Exchanging data: Applying VIC baseflow to MF6 finf for SP starting {vic_current_date}")
            baseflow = self.vic.read_vic_wb(wbal_date_tag)
            if baseflow is None:
                self.logger.warning(f"No baseflow found for SP starting {vic_current_date}, using zeros")
                baseflow = np.zeros((1, *self.vic_grid_shape))
            finf = self.compute_finf(baseflow)
            try:
                self.mf6.mf6.update()  # Ensure model state is ready
                self.mf6.mf6.set_value(self.mf6.var_finf, finf)
                # Verify application
                mf6_finf = self.mf6.mf6.get_value(self.mf6.var_finf)
                output_dir = self.vic.exchange_dir
                np.savetxt(os.path.join(output_dir, "mf6_finf.txt"), mf6_finf)
                self.logger.info(f"Applied FINF: non-zero count = {(mf6_finf > 0).sum()}, mean = {mf6_finf.mean():.6f} ft/day")
                # Save non-zero iuzno indices
                non_zero_iuzno = np.where(mf6_finf > 0)[0]
                np.savetxt(os.path.join(output_dir, "non_zero_iuzno.txt"), non_zero_iuzno, fmt="%d")
                self.logger.info(f"Non-zero iuzno count: {len(non_zero_iuzno)}")
            except Exception as e:
                self.logger.error(f"Failed to set finf for SP starting {vic_current_date}: {e}")
                return

            # Step 4: Run MF6 for the same period and stop
            self.logger.info(f"Running MF6 for SP from {mf6_current_date} to {mf6_period_end}")
            try:
                self.mf6.run_to_date(mf6_period_end, mf6_current_date)
                mf6_current_date = mf6_period_end
            except Exception as e:
                self.logger.error(f"Failed to run MF6 for SP starting {mf6_current_date}: {e}")
                return

            # Step 5: Update VIC parameters (optional, based on MF6 output)
            self.logger.info(f"Updating VIC params with MF6 uzf_gwd for SP starting {vic_current_date}")
            try:
                uzf_gwd = self.mf6.mf6.get_value_ptr(self.mf6.var_gwd)
                self.logger.info(f"uzf_gwd mean: {uzf_gwd.mean():.6f} ft/day")
                if not self.update_vic_params(uzf_gwd, baseflow):
                    self.logger.warning(f"Failed to update VIC params for SP starting {vic_current_date}")
            except Exception as e:
                self.logger.error(f"Failed to extract uzf_gwd or update VIC for SP starting {vic_current_date}: {e}")
                return

            # Step 6: Log results
            stress_period = (vic_current_date.year - vic_start_date.year) * 12 + (vic_current_date.month - vic_start_date.month)
            self.log_results(stress_period, finf, uzf_gwd, baseflow[-1] if baseflow is not None else np.zeros(self.vic_grid_shape))

            self.logger.info(f"Completed SP {stress_period}")
            vic_current_date = (vic_period_end + timedelta(days=1)).replace(day=1)
            first = False
            prev_date = state_date_tag

        self.logger.info("Full coupling complete")

def main():
    # Configuration
    workspace = "../../MF6/rgtihm/model_2100"
    mf6_dll = "~/usr/local/src/modflow6/bin/libmf6.so"
    vic_dir = "./nm_image/"
    vic_exe = "/home/abdazzam/usr/local/src/VIC/vic/drivers/image/vic_image.exe"
    global_param = os.path.join(vic_dir, "global_param_nc4.txt")
    outputs_dir = os.path.join(vic_dir, "outputs")
    exchange_dir = os.path.join(vic_dir, "exchange_data")
    params_file = os.path.join(vic_dir, "nc4_params.nc")
    coupling_table_csv = os.path.join(exchange_dir, "mf6_vic_join.csv")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(exchange_dir, f"coupling_log_{current_time}.csv")
    debug_log_file = os.path.join(exchange_dir, f"coupling_debug_{current_time}.log")
    vic_start_date = datetime(1990, 1, 1)   # VIC starts in Jan 1990
    coupling_end_date = datetime(2091, 12, 31)

    # Create directories
    try:
        os.makedirs(exchange_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating directories: {e}")
        return

    # Setup logging
    logger = logging.getLogger('coupling')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(debug_log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Debug log file: {debug_log_file}")

    # Verify directories
    try:
        if not os.access(outputs_dir, os.W_OK):
            raise PermissionError(f"Cannot write to {outputs_dir}")
        logger.info(f"Verified directories: {exchange_dir}, {outputs_dir}")
    except Exception as e:
        logger.error(f"Error during directory setup: {e}")
        return

    # Verify params file
    try:
        if not os.path.exists(params_file):
            raise FileNotFoundError(f"Parameters file not found: {params_file}")
        with Dataset(params_file, "r") as ds:
            if "run_cell" not in ds.variables:
                raise ValueError("run_cell missing in parameters")
            active_cells = np.sum(ds.variables["run_cell"][:])
            logger.info(f"Active cells in {params_file}: {active_cells}")
    except Exception as e:
        logger.error(f"Error verifying params file: {e}")
        return

    # Initialize MF6
    try:
        mf6 = MF6Model(workspace, mf6_dll, logger)
        mf6.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize MF6: {e}")
        return

    # Initialize VIC
    try:
        vic = VICModel(vic_dir, vic_exe, global_param, outputs_dir, exchange_dir, params_file, logger)
    except Exception as e:
        logger.error(f"Failed to initialize VIC: {e}")
        mf6.finalize()
        return

    # Initialize coupling
    try:
        cpl = CouplingManager(mf6, vic, coupling_table_csv, params_file, log_file, logger)
        cpl.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize coupling: {e}")
        mf6.finalize()
        return

    # Run coupling
    try:
        cpl.run(vic_start_date, coupling_end_date)
    except Exception as e:
        logger.error(f"Error during coupling: {e}")
    finally:
        mf6.finalize()

if __name__ == "__main__":
    main()
