#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import time
import xmipy

class MF6Model:
    def __init__(self, workspace, mf6_dll):
        self.workspace  = workspace
        self.mf6_dll    = os.path.expanduser(mf6_dll)
        self.mf6        = None
        self.model_name = None
        self.bmi_vars   = []
        self.bmi_prefix = None
        self.nrow       = None
        self.ncol       = None
        self.nlay       = None
        self.nper       = None
        self.nstp       = None
        self.idomain    = None
        self.nuzfcells  = None
        self.var_finf   = None
        self.var_gwd    = None

    def initialize(self):
        # init bmi api
        self.mf6 = xmipy.XmiWrapper(self.mf6_dll,
                                     working_directory=self.workspace)  # from xmi function XmiWrapper
        self.mf6.initialize()                                        # from bmi/xmi function initialize
        print("mf6 initialized")

        self._parse_name_file()
        self._load_bmi_vars()
        self._resolve_bmi_prefix()
        self._parse_grid_info()
        self._parse_tdis_info()
        self._identify_uzf_vars()

        print(f"grid: nlay={self.nlay}, nrow={self.nrow}, ncol={self.ncol}")
        print(f"tdis: nper={self.nper}, nstp={self.nstp}")
        print(f"uzf cells={self.nuzfcells}")

    def _parse_name_file(self):
        fn = next(f for f in os.listdir(self.workspace)
                  if f.lower().endswith(".nam"))
        for L in open(os.path.join(self.workspace, fn)):
            t = L.strip().split()
            if t and t[0].lower() == "modelname":
                self.model_name = t[1]
                break
        print(f"name file: model={self.model_name}")

    def _load_bmi_vars(self):
        inp = self.mf6.get_input_var_names()   # from bmi/xmi: get_input_var_names
        out = self.mf6.get_output_var_names()  # from bmi/xmi: get_output_var_names
        self.bmi_vars = set(inp + out)
        print(f"loaded {len(self.bmi_vars)} bmi vars")

    def _resolve_bmi_prefix(self):
        for v in self.bmi_vars:
            if v.endswith("/DIS/NROW"):
                self.bmi_prefix = v.rsplit("/DIS/NROW", 1)[0]
                break
        if not self.bmi_prefix:
            raise Exception("bmi prefix not found")
        print(f"using prefix: {self.bmi_prefix}")

    def _parse_grid_info(self):
        p = self.bmi_prefix
        self.nrow      = int(self.mf6.get_value_ptr(f"{p}/DIS/NROW")[0])    # from bmi/xmi: get_value_ptr
        self.ncol      = int(self.mf6.get_value_ptr(f"{p}/DIS/NCOL")[0])    # from bmi/xmi: get_value_ptr
        self.nlay      = int(self.mf6.get_value_ptr(f"{p}/DIS/NLAY")[0])    # from bmi/xmi: get_value_ptr
        arr            = self.mf6.get_value_ptr(f"{p}/DIS/IDOMAIN")         # from bmi/xmi: get_value_ptr
        self.idomain   = arr.reshape(self.nlay, self.nrow, self.ncol)
        self.nuzfcells = int((self.idomain[0] == 1).sum())

    def _parse_tdis_info(self):
        # find nper/nstp via suffix search
        nper_vars = [v for v in self.bmi_vars if v.endswith("/NPER")]
        nstp_vars = [v for v in self.bmi_vars if v.endswith("/NSTP")]
        if nper_vars:
            self.nper = int(self.mf6.get_value_ptr(nper_vars[0])[0])       # from bmi/xmi: get_value_ptr
        else:
            self.nper = 1
            print("nper not found; default=1")
        if nstp_vars:
            self.nstp = int(self.mf6.get_value_ptr(nstp_vars[0])[0])       # from bmi/xmi: get_value_ptr
        else:
            self.nstp = 1
            print("nstp not found; default=1")

    def _identify_uzf_vars(self):
        for v in self.bmi_vars:
            if   v.endswith("/UZF/FINF"):
                self.var_finf = v
            elif v.endswith("/UZF/GWD"):
                self.var_gwd  = v
        if not self.var_gwd:
            raise Exception("uzf gwd var not found")
        print(f"vars: finf={self.var_finf}, gwd={self.var_gwd}")

    def run_timestep(self):
        self.mf6.prepare_time_step(0.0)  # from bmi/xmi: prepare_time_step
        self.mf6.do_time_step()          # from bmi/xmi: do_time_step
        self.mf6.finalize_time_step()    # from bmi/xmi: finalize_time_step

    def finalize(self):
        self.mf6.finalize()              # from bmi/xmi: finalize
        print("mf6 finalized")


class VICModel:
    def __init__(self, fname="vic_baseflow.csv"):
        self.fname = fname

    def run(self, prev_mm, sp):
        time.sleep(10)
        bf = np.random.uniform(0.1, 5.0) + prev_mm * 0.1
        bf = max(0.1, min(5.0, bf))
        pd.DataFrame({
            "stress_period": [sp],
            "baseflow_mm_day": [bf]
        }).to_csv(self.fname,
                  mode="a" if sp else "w",
                  index=False,
                  header=(sp == 0))
        print(f"vic sp{sp+1}: bf={bf:.3f} mm/day")
        return bf

    def get_baseflow(self, sp):
        df  = pd.read_csv(self.fname)
        return float(df.loc[df.stress_period == sp, "baseflow_mm_day"].iloc[0])


class CouplingManager:
    def __init__(self, mf6_model, vic_model):
        self.m     = mf6_model
        self.v     = vic_model
        self.mm2ft = 1/304.8
        self.n     = self.m.nuzfcells
        self.log   = os.path.join(os.getcwd(), "coupling_log.csv")
        open(self.log, "w").write("sp,cell,finf,gwd\n")
        print(f"log will be saved to {self.log}")

    def run(self):
        prev = 0.0
        print("starting coupling")
        for sp in range(self.m.nper):
            print(f"  sp{sp+1}/{self.m.nper} start")

            # run vic
            _    = self.v.run(prev, sp)
            bf   = self.v.get_baseflow(sp)
            finf = bf * self.mm2ft
            print(f"    finf={finf:.4f}")

            # update finf via get_value_ptr only
            finf_arr = self.m.mf6.get_value_ptr(self.m.var_finf)  # from bmi/xmi: get_value_ptr
            finf_arr[:] = finf

            # verify change
            sample = self.m.mf6.get_value_ptr(self.m.var_finf)[:5]  # from bmi/xmi: get_value_ptr
            print(f"    finf sample[:5]={sample}")

            # run mf6 timesteps
            for _ in range(self.m.nstp):
                self.m.run_timestep()

            # extract gwd via get_value_ptr
            gwd = self.m.mf6.get_value_ptr(self.m.var_gwd)  # from bmi/xmi: get_value_ptr
            print(f"    gwd sample[:5]={gwd[:5]}")

            # next vic input
            prev = gwd.mean() * 1000 / 3.28084

            # log results
            with open(self.log, "a") as f:
                for i, v in enumerate(gwd):
                    f.write(f"{sp},{i},{finf:.4f},{v:.4f}\n")

            print(f"  sp{sp+1} end")
        print("coupling complete")


if __name__ == "__main__":
    workspace = "../../MF6/rgtihm/model/"
    mf6_dll   = "~/usr/local/src/modflow6/bin/libmf6.so"

    mf6 = MF6Model(workspace, mf6_dll)
    mf6.initialize()

    vic = VICModel()
    cpl = CouplingManager(mf6, vic)
    cpl.run()

    mf6.finalize()

