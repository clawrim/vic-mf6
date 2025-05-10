import numpy as np
import pandas as pd
import os
import time
import subprocess
import glob
from datetime import datetime, timedelta
from netCDF4 import Dataset

# config variables
mf6_start_date = datetime(1940, 3, 1)  # mf6 start date
vic_warmup_start = datetime(1990, 1, 1)  # vic warmup start
vic_warmup_end = datetime(1990, 1, 31)  # vic warmup end
coupling_end_date = datetime(1991, 12, 31)  # coupling end date
stress_period_days = 30.42  # avg days per month
timestep_days = [15, 16]  # alternating timesteps
vic_dir = "./nm_image/"
vic_global_param = os.path.join(vic_dir, "global_param_nc4.txt")  # global param file
vic_exe = "/home/abdazzam/usr/local/src/VIC/vic/drivers/image/vic_image.exe"
outputs_dir = os.path.join(vic_dir, "outputs")
exchange_dir = os.path.join(vic_dir, "exchange_data")
params_file = "nc4_params.nc"
params_path = os.path.join(vic_dir, params_file)
log_file = os.path.join(exchange_dir, "coupling_log.csv")
mf6_csv = os.path.join(exchange_dir, "mf6_uzf.csv")
state_file_prefix = "state"
wb_file_prefix = "wbal"
mm_to_ft = 1 / 304.8  # mm to ft for finf
ft_to_mm = 3.28084 * 1000  # ft/day to mm/day for gwd
nper = int((coupling_end_date - mf6_start_date).days / stress_period_days)  # stress periods
n_cells = 10  # cells for dummy mf6
verbose = True

# init exchange dir
def init_exchange_dir():
    os.makedirs(exchange_dir, exist_ok=True)
    print(f"exchange dir created: {exchange_dir}")

# init log file
def init_log_file():
    with open(log_file, "w") as f:
        f.write("stress_period,iuzno,finf_ft_per_day,uzf_gwd_ft_per_day,init_moist_mm\n")
    print(f"log file created: {log_file}")

# init mf6 csv
def init_mf6_csv():
    with open(mf6_csv, "w") as f:
        f.write("stress_period,iuzno,finf_ft_per_day,uzf_gwd_ft_per_day\n")
    print(f"mf6 csv created: {mf6_csv}")

# init outputs dir
def init_outputs_dir():
    os.makedirs(outputs_dir, exist_ok=True)
    print(f"outputs dir created: {outputs_dir}")

# update global_param.txt
def update_global_param(stress_period, start_date, end_date):
    global_param_file = os.path.join(vic_dir, f"global_param_sp{stress_period}.txt")  # save in vic_dir
    print(f"updating global_param for sp {stress_period+1}")
    try:
        with open(vic_global_param, "r") as f:
            lines = f.readlines()
        with open(global_param_file, "w") as f:
            for line in lines:
                if line.strip() and not line.startswith("#"):
                    parts = line.split()
                    if parts[0] == "STARTYEAR":
                        f.write(f"STARTYEAR   {start_date.year}\n")
                        continue
                    elif parts[0] == "STARTMONTH":
                        f.write(f"STARTMONTH  {start_date.month:02d}\n")
                        continue
                    elif parts[0] == "STARTDAY":
                        f.write(f"STARTDAY    {start_date.day:02d}\n")
                        continue
                    elif parts[0] == "ENDYEAR":
                        f.write(f"ENDYEAR     {end_date.year}\n")
                        continue
                    elif parts[0] == "ENDMONTH":
                        f.write(f"ENDMONTH    {end_date.month:02d}\n")
                        continue
                    elif parts[0] == "ENDDAY":
                        f.write(f"ENDDAY      {end_date.day:02d}\n")
                        continue
                f.write(line)
        print(f"updated global_param: {global_param_file}")
        return global_param_file
    except Exception as e:
        print(f"error updating global_param: {e}")
        raise

# parse vic config
def read_vic_config(global_param_file):
    start_year, start_month, start_day = None, None, None
    params_file = None
    print(f"reading global_param")
    try:
        with open(global_param_file, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    if parts[0] == "STARTYEAR":
                        start_year = int(parts[1])
                    elif parts[0] == "STARTMONTH":
                        start_month = int(parts[1])
                    elif parts[0] == "STARTDAY":
                        start_day = int(parts[1])
                    elif parts[0] == "PARAMETERS":
                        params_file = parts[1]
        if not all([start_year, start_month, start_day]):
            raise ValueError("missing startyear, startmonth, or startday")
        if not params_file:
            raise ValueError("parameters not found")
        params_path = os.path.join(vic_dir, params_file)
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"params file not found: {params_path}")
        with Dataset(params_path, "r") as ds:
            if "run_cell" not in ds.variables:
                raise ValueError("run_cell not found")
            active_cells = np.sum(ds.variables["run_cell"][:])
            if active_cells == 0:
                raise ValueError("no active grid cells")
            print(f"found {active_cells} active cells")
        start_date = datetime(start_year, start_month, start_day)
        return start_date, params_file
    except Exception as e:
        print(f"error parsing config: {e}")
        raise

# read vic state
def read_vic_state(global_param_file, variables=None, expected_date=None):
    state_prefix = None
    with open(global_param_file, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "STATENAME" in line:
                state_prefix = line.split()[1]
                break
    if not state_prefix:
        raise ValueError("statename not found")
    base_pattern = os.path.join(vic_dir, f"{state_prefix}.*")
    possible_files = glob.glob(f"{base_pattern}*.nc") + glob.glob(f"{base_pattern}.nc")
    if expected_date:
        expected_pattern = f"{state_prefix}.{expected_date.strftime('%Y%m%d')}_00000.nc"
        expected_file = os.path.join(vic_dir, expected_pattern)
        possible_files = [f for f in possible_files if expected_pattern in f] or possible_files
    if not possible_files:
        raise ValueError(f"no state file found: {base_pattern}")
    file_path = max(possible_files, key=os.path.getctime)
    print(f"detected state file: {file_path}")
    try:
        ds = Dataset(file_path, "r")
        if variables:
            data = {var: ds.variables[var][:] for var in variables if var in ds.variables}
            ds.close()
            return data
        ds.close()
        return None
    except Exception as e:
        print(f"error reading state: {e}")
        return None

# write vic state
def write_vic_state(data, file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        with Dataset(file_path, "w", format="NETCDF4") as ds:
            for var, values in data.items():
                dims = values.shape
                ds.createDimension("dim0", dims[0])
                if len(dims) > 1:
                    ds.createDimension("dim1", dims[1])
                if len(dims) > 2:
                    ds.createDimension("dim2", dims[2])
                var_obj = ds.createVariable(var, values.dtype, ("dim0", "dim1", "dim2")[:len(dims)])
                var_obj[:] = values
        print(f"wrote state: {file_path}")
    except Exception as e:
        print(f"error writing state: {e}")
        raise

# run vic
def run_vic(global_param_file, vic_dir):
    print("running vic")
    try:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        # use filename only, since cwd is vic_dir
        param_path = os.path.basename(global_param_file)
        cmd = [vic_exe, "-g", param_path]
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, check=True, cwd=vic_dir
        )
        print(f"vic stdout: {result.stdout}")
        print("vic completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"vic failed: {e}")
        print(f"vic stderr: {e.stderr}")
        log_pattern = os.path.join(vic_dir, "logs/vic.log.*.txt")
        log_files = glob.glob(log_pattern)
        if log_files:
            latest_log = max(log_files, key=os.path.getctime)
            with open(latest_log, "r") as f:
                print(f"vic log: {latest_log}\n{f.read()}")
        return False

# run dummy mf6
def run_dummy_mf6(stress_period, finf_val):
    print(f"running mf6 for sp {stress_period+1}")
    try:
        time.sleep(5)
        uzf_gwd = np.random.uniform(0.001, 0.005, n_cells)
        finf = finf_val if finf_val is not None else np.random.uniform(0.01, 0.05, n_cells)
        with open(mf6_csv, "a") as f:
            for iuzno in range(n_cells):
                finf_value = finf[iuzno] if isinstance(finf, np.ndarray) else finf
                f.write(f"{stress_period},{iuzno},{finf_value:.6f},{uzf_gwd[iuzno]:.6f}\n")
        finf_mean = finf.mean() if isinstance(finf, np.ndarray) else finf
        print(f"mf6 wrote finf: {finf_mean:.6f}, gwd: {uzf_gwd.mean():.6f} ft/day")
        print("mf6 completed")
        return uzf_gwd
    except Exception as e:
        print(f"mf6 failed: {e}")
        return None

# update vic params
def update_vic_params(params_file, uzf_gwd=None, baseflow=None):
    print("updating vic params")
    try:
        ds = Dataset(params_file, "r+")
        if "init_moist" not in ds.variables:
            print("init_moist not found")
            ds.close()
            return False
        init_moist = ds.variables["init_moist"]
        if uzf_gwd is not None and baseflow is not None:
            gwd_mm = np.mean(uzf_gwd) * ft_to_mm
            baseflow_mm = baseflow
            new_moist = gwd_mm + baseflow_mm
        elif baseflow is not None:
            new_moist = baseflow
        else:
            print("no data for init_moist")
            ds.close()
            return False
        print(f"updating init_moist (layer 3) with {new_moist:.6f} mm")
        init_moist[2, :, :] = new_moist
        ds.close()
        print("params updated")
        return True
    except Exception as e:
        print(f"error updating params: {e}")
        if "ds" in locals():
            ds.close()
        return False

# compute finf
def compute_finf(params_file):
    print("computing finf")
    try:
        with Dataset(params_file, "r") as ds:
            if "init_moist" not in ds.variables:
                print("init_moist not found")
                return None, None
            init_moist = ds.variables["init_moist"][2, :, :].mean()
        finf_mm_day = min(max(init_moist * 0.1, 0.1), 5.0)
        finf_ft_day = finf_mm_day * mm_to_ft
        print(f"finf: {finf_ft_day:.6f} ft/day, init_moist: {init_moist:.6f} mm")
        return finf_ft_day, init_moist
    except Exception as e:
        print(f"error computing finf: {e}")
        return None, None

# read vic water balance
def read_vic_wb():
    wb_pattern = os.path.join(outputs_dir, f"{wb_file_prefix}.*.nc")
    wb_files = glob.glob(wb_pattern)
    if not wb_files:
        print(f"no water balance file: {wb_pattern}")
        return None
    wb_file = max(wb_files, key=os.path.getctime)
    print(f"reading water balance: {wb_file}")
    try:
        ds = Dataset(wb_file, "r")
        if "OUT_BASEFLOW" not in ds.variables:
            print("out_baseflow not found")
            ds.close()
            return None
        baseflow = ds.variables["OUT_BASEFLOW"][:].mean()
        ds.close()
        print(f"out_baseflow: {baseflow:.6f} mm")
        return baseflow
    except Exception as e:
        print(f"error reading water balance: {e}")
        return None

# log results
def log_results(stress_period, finf_val, uzf_gwd, init_moist):
    try:
        with open(log_file, "a") as f:
            for iuzno in range(n_cells):
                f.write(f"{stress_period},{iuzno},{finf_val:.6f},{uzf_gwd[iuzno]:.6f},{init_moist:.6f}\n")
        print(f"logged to {log_file}")
    except Exception as e:
        print(f"error logging: {e}")

# get stress period dates
def get_stress_period_dates(stress_period):
    # start from vic_warmup_end + 1 day for first stress period
    base_date = vic_warmup_end + timedelta(days=1)
    period_start = base_date + timedelta(days=(stress_period - 599) * stress_period_days)
    period_end = period_start + timedelta(days=stress_period_days)
    if stress_period % 2 == 0:
        period_end = period_end.replace(day=15)
    else:
        next_month = period_end.replace(day=28) + timedelta(days=4)
        period_end = next_month - timedelta(days=next_month.day)
    return period_start, period_end

# main loop
def main():
    # check if global_param_nc4.txt exists
    if not os.path.exists(vic_global_param):
        print(f"error: global_param file not found: {vic_global_param}")
        return

    init_exchange_dir()
    init_log_file()
    init_mf6_csv()
    init_outputs_dir()

    print("parsing vic config")
    vic_start_date, params_file = read_vic_config(vic_global_param)
    print(f"vic start: {vic_start_date}, params: {params_file}")

    # warmup vic run (1 month)
    print("running vic warmup")
    print(f"using global_param: {vic_global_param}")
    if not run_vic(vic_global_param, vic_dir):
        print("vic warmup failed")
        return

    # save warmup state
    vic_state = read_vic_state(vic_global_param, variables=["SOIL_MOIST"], expected_date=vic_warmup_end)
    if vic_state is not None:
        state_date = vic_warmup_end
        state_file = os.path.join(outputs_dir, f"{state_file_prefix}.{state_date.strftime('%Y%m%d')}_00000.nc")
        write_vic_state(vic_state, state_file)
        print(f"warmup state saved: {state_file}")
    else:
        print("failed to read warmup state")

    # commented original full-year warmup
    # print("initial vic run")
    # if not run_vic(vic_global_param, vic_dir):
    #     print("initial vic run failed")
    #     return
    # vic_state = read_vic_state(vic_global_param, variables=["SOIL_MOIST"])
    # if vic_state is not None:
    #     state_date = datetime(1990, 12, 31)
    #     state_file = os.path.join(outputs_dir, f"{state_file_prefix}.{state_date.strftime('%Y%m%d')}_00000.nc")
    #     write_vic_state(vic_state, state_file)
    #     print(f"initial state saved: {state_file}")
    # else:
    #     print("failed to read initial state")

    # coupling loop
    finf_val = None
    current_date = vic_warmup_end
    stress_period = 599  # start at 599 based on previous log

    while current_date < coupling_end_date:
        print(f"stress period {stress_period+1}/{nper}")
        period_start, period_end = get_stress_period_dates(stress_period)
        current_date = period_end

        # step 1: compute finf
        print("computing finf")
        finf_val, init_moist = compute_finf(params_path)
        if finf_val is None:
            print("failed to compute finf, using default")
            finf_val = 0.01
            init_moist = 0.0

        # step 2: run mf6
        print("running mf6")
        uzf_gwd = run_dummy_mf6(stress_period, finf_val)
        if uzf_gwd is None:
            print("mf6 failed")
            break

        # step 3: update global_param and run vic
        print("updating global_param and running vic")
        global_param_file = update_global_param(stress_period, period_start, period_end)
        if not run_vic(global_param_file, vic_dir):
            print("vic failed")
            break

        # read and print water balance
        baseflow = read_vic_wb()
        if baseflow is None:
            print("failed to read out_baseflow, using default")
            finf_val = 0.01
            init_moist = 0.0
        else:
            print(f"out_baseflow: {baseflow:.6f} mm")

        # save and print state
        vic_state = read_vic_state(global_param_file, variables=["SOIL_MOIST"], expected_date=period_end)
        if vic_state is not None:
            state_file = os.path.join(outputs_dir, f"{state_file_prefix}.{period_end.strftime('%Y%m%d')}_00000.nc")
            write_vic_state(vic_state, state_file)
            print(f"state saved: {state_file}")
        else:
            print("failed to read state")

        # step 4: pause mf6
        print("pausing mf6 (not implemented)")

        # step 5: update vic params
        print("updating vic params")
        if uzf_gwd is not None:
            if not update_vic_params(params_path, uzf_gwd, baseflow):
                print("failed to update params")
        else:
            if not update_vic_params(params_path, baseflow=baseflow):
                print("failed to update params, using out_baseflow")
                finf_val = baseflow * mm_to_ft if baseflow is not None else 0.01
                init_moist = baseflow if baseflow is not None else 0.0

        # log results
        log_results(stress_period, finf_val, uzf_gwd, init_moist)
        print(f"completed sp {stress_period+1}/{nper}")
        stress_period += 1

    print("simulation complete")

if __name__ == "__main__":
    main()
