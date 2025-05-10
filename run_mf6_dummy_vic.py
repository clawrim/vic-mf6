import flopy
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from modflowapi import ModflowApi

# Load existing MF6 simulation with flopy
workspace = "../../MF6/rgtihm/model/"  # Using model directory
sim_name = "rgtihm"
sim = flopy.mf6.MFSimulation.load(
    sim_name=sim_name, version="6.7.0.dev1", exe_name="mf6", sim_ws=workspace
)
gwf = sim.get_model()
uzf = gwf.uzf
print("Simulation loaded successfully")

# Set model parameters
nrow, ncol = 912, 328  # Grid dimensions
nuzfcells = 17879  # Active UZF cells in layer 1
nper = 10  # Number of stress periods
mm_to_ft = 1 / 304.8  # Convert mm to ft for finf

# Get active UZF cells from idomain
idomain = gwf.modelgrid.idomain
active_cells = np.argwhere(idomain[0] == 1)  # Shape: (17879, 2)
print(f"Active UZF cells: {len(active_cells)}")

# Create log file for finf and uzf-gwd
log_file = "coupling_log.csv"
with open(log_file, "w") as f:
    f.write("stress_period,iuzno,finf_ft_per_day,uzf_gwd_ft_per_day\n")
print(f"Log file created: {log_file}")

# Initialize ModflowApi for timestep control
mf6_exe = os.path.expanduser("~/usr/local/src/modflow6/bin/libmf6.so")  # Path to MODFLOW 6 shared library
mf6 = ModflowApi(mf6_exe, working_directory=workspace)
mf6.initialize()
print("ModflowApi initialized")

# Dummy VIC function: generates baseflow, writes to CSV, takes 10 seconds
def run_dummy_vic(prev_gwd_mm_day=0.0, stress_period=0):
    print(f"Stress period {stress_period+1}: Running dummy VIC (10 seconds)...")
    time.sleep(10)
    # Generate random baseflow (0.1–5 mm/day), influenced by prev_gwd
    baseflow = np.random.uniform(0.1, 5.0) + (prev_gwd_mm_day * 0.1)
    baseflow = min(max(baseflow, 0.1), 5.0)
    print(f"Stress period {stress_period+1}: Dummy VIC generated baseflow: {baseflow:.6f} mm/day")
    # Write to CSV in current working directory
    csv_path = "./vic_baseflow.csv"
    df = pd.DataFrame({"stress_period": [stress_period], "baseflow_mm_day": [baseflow]})
    mode = "a" if stress_period > 0 else "w"
    df.to_csv(csv_path, mode=mode, index=False, header=(stress_period == 0))
    print(f"Stress period {stress_period+1}: Dummy VIC wrote baseflow to {csv_path}")
    return baseflow

# Run coupling loop for each stress period
prev_gwd_mm_day = 0.0  # Initial GWD for VIC
for t in range(nper):
    print(f"Stress period {t+1}/{nper}: Starting dummy VIC")
    # Run dummy VIC to get baseflow and write CSV
    baseflow_val = run_dummy_vic(prev_gwd_mm_day, t)

    print(f"Stress period {t+1}/{nper}: Starting baseflow reading")
    # Read baseflow from CSV
    try:
        baseflow_df = pd.read_csv("./vic_baseflow.csv")
        baseflow_row = baseflow_df[baseflow_df["stress_period"] == t]
        if not baseflow_row.empty:
            baseflow_val = baseflow_row["baseflow_mm_day"].iloc[0]
        else:
            print(f"Stress period {t+1}/{nper}: Warning: No baseflow data for period, using dummy value")
            baseflow_val = np.random.uniform(0.1, 5.0)
    except Exception as e:
        print(f"Stress period {t+1}/{nper}: Error reading baseflow CSV - {e}")
        print(f"Stress period {t+1}/{nper}: Using dummy baseflow")
        baseflow_val = np.random.uniform(0.1, 5.0)
    finf_val = baseflow_val * mm_to_ft
    print(f"Stress period {t+1}/{nper}: Read baseflow, finf: {finf_val:.6f} ft/day")

    print(f"Stress period {t+1}/{nper}: Starting UZF period data preparation")
    # Prepare UZF period data with single finf value for 10 cells
    period_data = []
    for idx, (i, j) in enumerate(active_cells[:10]):  # Limit to first 10 cells
        iuzno = idx
        period_data.append(
            [iuzno, finf_val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # finf, pet, extdp, extwc, ha, hroot, rootact
        )
    print(f"Stress period {t+1}/{nper}: Prepared UZF period data for {len(period_data)} cells")

    print(f"Stress period {t+1}/{nper}: Starting UZF period data update")
    # Update UZF period data
    uzf.perioddata = {t: period_data}
    uzf.write()
    print(f"Stress period {t+1}/{nper}: Updated UZF period data")

    print(f"Stress period {t+1}/{nper}: Starting MF6 stress period run")
    # Run one stress period with 2 timesteps using ModflowApi
    for ts in range(2):  # 2 timesteps
        print(f"Stress period {t+1}/{nper}: Running timestep {ts+1}/2")
        mf6.update()  # Run one timestep
    print(f"Stress period {t+1}/{nper}: Ran MF6 stress period (2 timesteps)")

    print(f"Stress period {t+1}/{nper}: Starting uzf-gwd extraction")
    # Extract uzf-gwd from observation file
    uzf_gwd = np.zeros(10)  # Size matches 10 cells
    try:
        obs_data = pd.read_csv("../../MF6/rgtihm/model/rgtihm.obs.gwf.csv")
        if t < len(obs_data):
            gwd_data = obs_data.iloc[t]["GWD"]
            uzf_gwd[:] = gwd_data if isinstance(gwd_data, float) else gwd_data[:10]
            if np.all(uzf_gwd == 0):
                print(f"Stress period {t+1}/{nper}: uzf-gwd is zero, using synthetic GWD")
                uzf_gwd[:] = finf_val * np.random.uniform(0.1, 0.5)
        else:
            print(f"Stress period {t+1}/{nper}: Warning: No obs data for period, using synthetic GWD")
            uzf_gwd[:] = finf_val * np.random.uniform(0.1, 0.5)
    except Exception as e:
        print(f"Stress period {t+1}/{nper}: Error reading obs file - {e}")
        print(f"Stress period {t+1}/{nper}: Using synthetic GWD")
        uzf_gwd[:] = finf_val * np.random.uniform(0.1, 0.5)
    print(f"Stress period {t+1}/{nper}: Extracted uzf-gwd, range (ft/day): "
          f"min={uzf_gwd.min():.6f}, max={uzf_gwd.max():.6f}, mean={uzf_gwd.mean():.6f}")

    # Convert uzf-gwd to mm/day for VIC
    prev_gwd_mm_day = uzf_gwd.mean() * 1000 / 3.28084
    print(f"Stress period {t+1}/{nper}: Passing GWD to VIC: {prev_gwd_mm_day:.6f} mm/day")

    print(f"Stress period {t+1}/{nper}: Starting logging results")
    # Log results
    with open(log_file, "a") as f:
        for idx, (i, j) in enumerate(active_cells[:10]):  # Limit to first 10 cells
            iuzno = idx
            f.write(f"{t},{iuzno},{period_data[idx][1]:.6f},{uzf_gwd[iuzno]:.6f}\n")
    print(f"Stress period {t+1}/{nper}: Logged results to {log_file}")

    print(f"Completed stress period {t+1}/{nper}")

# Finalize ModflowApi
mf6.finalize()
print("Simulation complete")
