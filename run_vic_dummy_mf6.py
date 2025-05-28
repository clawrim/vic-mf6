#!/usr/bin/env python3
import os, glob,sys, subprocess, time
from datetime import datetime, timedelta

vic_dir     = "./nm_image"
vic_exe     = "/home/abdazzam/usr/local/src/VIC/vic/drivers/image/vic_image.exe"
master_gp   = os.path.join(vic_dir, "global_param_nc4.txt")   # your full template :contentReference[oaicite:0]{index=0}
outputs_dir = os.path.join(vic_dir, "outputs")

start_date = datetime(1990,  1,  1)
end_date   = datetime(2020, 12, 31)

def make_monthly_global(master, out_gp, sim_start, sim_end, prev_state):
    """
    Copy master global_param_nc4.txt and replace only:
      STARTYEAR/STARTMONTH/STARTDAY,
      ENDYEAR/ENDMONTH/ENDDAY,
      INIT_STATE → use prev_state once available,
      STATEYEAR/STATEMONTH/STATEDAY.
    All other lines (forcing, domains, OUTPUT block, etc.) remain verbatim :contentReference[oaicite:1]{index=1}.
    """
    lines = open(master).readlines()
    date_tag = sim_end.strftime("%Y%m%d")
    with open(out_gp, "w") as f:
        for L in lines:
            if L.startswith("STARTYEAR"):
                f.write(f"STARTYEAR   {sim_start.year}\n")
            elif L.startswith("STARTMONTH"):
                f.write(f"STARTMONTH  {sim_start.month:02d}\n")
            elif L.startswith("STARTDAY"):
                f.write(f"STARTDAY    {sim_start.day:02d}\n")
            elif L.startswith("ENDYEAR"):
                f.write(f"ENDYEAR     {sim_end.year}\n")
            elif L.startswith("ENDMONTH"):
                f.write(f"ENDMONTH    {sim_end.month:02d}\n")
            elif L.startswith("ENDDAY"):
                f.write(f"ENDDAY      {sim_end.day:02d}\n")
            elif "INIT_STATE" in L:
                # first month: leave it commented; afterwards point to previous state file
                if prev_state:
                    f.write(f"INIT_STATE    outputs/state.{date_tag}_00000.nc\n")
                else:
                    f.write(L)
            elif L.startswith("STATEYEAR"):
                f.write(f"STATEYEAR     {sim_end.year}\n")
            elif L.startswith("STATEMONTH"):
                f.write(f"STATEMONTH   {sim_end.month:02d}\n")
            elif L.startswith("STATEDAY"):
                f.write(f"STATEDAY     {sim_end.day:02d}\n")
            else:
                f.write(L)

def run_monthly_vic():
    os.makedirs(outputs_dir, exist_ok=True)
    os.chmod(outputs_dir, 0o777)

    prev_state = None
    current = start_date

    while current <= end_date:
        # define first→last day of current month
        sim_start = current.replace(day=1)
        next_month = (sim_start + timedelta(days=32)).replace(day=1)
        sim_end = next_month - timedelta(days=1)
        tag = sim_end.strftime("%Y%m")      # e.g. "199001"

        # 1) build new global_param for this month
        out_gp = os.path.join(vic_dir, f"global_param_{tag}.txt")
        make_monthly_global(master_gp, out_gp, sim_start, sim_end, prev_state)

        # 2) call VIC exactly as you did by hand
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        proc = subprocess.run(
            [os.path.basename(vic_exe), "-g", os.path.basename(out_gp)],
            cwd=vic_dir, env=env,
            capture_output=True, text=True
        )
        if proc.returncode:
            print("VIC error:", proc.stderr, file=sys.stderr)
            break

        # 3) find what VIC wrote—no renaming or deletion
        #    wbal file: wbal.YYYY-MM-DD.nc
        wbal_pat = os.path.join(outputs_dir, f"wbal.{sim_end.strftime('%Y-%m-%d')}.nc")
        state_pat = os.path.join(outputs_dir, f"state.{sim_end.strftime('%Y%m%d')}_00000.nc")
        wbal_file = wbal_pat if os.path.exists(wbal_pat) else max(glob.glob(os.path.join(outputs_dir,"wbal.*.nc")), key=os.path.getctime)
        state_file = state_pat if os.path.exists(state_pat) else max(glob.glob(os.path.join(outputs_dir,"state.*_00000.nc")), key=os.path.getctime)

        print(f"→ VIC completed for {sim_start.strftime('%Y-%m')}  wbal: {os.path.basename(wbal_file)}  state: {os.path.basename(state_file)}")

        # chain into next month
        prev_state = os.path.basename(state_file)

        # advance
        current = next_month
        time.sleep(2)   # placeholder for MF6 coupling

if __name__ == "__main__":
    run_monthly_vic()
