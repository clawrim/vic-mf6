# VIC–MF6 MPI Coupling

A controller–worker MPI coupling framework for running the VIC image driver and a split MODFLOW 6 groundwater simulation in sequential coupling windows.

This code is designed for workflows where:

- VIC is used to produce a spatially distributed surface or near-surface flux field on the VIC grid.
- A preprocessed coupling table maps that VIC field onto the surface cells of one or more MF6 groundwater submodels.
- MF6 is already split into multiple groundwater flow models and launched with one MPI worker rank per GWF model.
- VIC is spawned from the controller rank at each coupling window, rather than being launched as a long-running peer application.

The current implementation treats the configured VIC water-balance variable as the recharge-like source term passed to MF6. In the default workflow this field is read in **millimeters per day**, remapped through the coupling table, converted to the MF6 model length units, and then written into the MF6 recharge array for each worker rank.

## What this program does

At runtime, the framework uses a strict MPI rank layout:

- **world rank 0** is the **controller**.
  - Loads the YAML configuration.
  - Verifies the split MF6 simulation structure.
  - Rewrites a VIC global parameter file for the current coupling window.
  - Spawns VIC with `MPI.COMM_SELF.Spawn()`.
  - Reads the resulting VIC water-balance file(s).
  - Broadcasts the VIC field to all MF6 workers.
- **world ranks 1..N** are **MF6 workers**.
  - Build an MF6-only subcommunicator.
  - Initialize rank-local MF6 BMI contexts through `xmipy`.
  - Filter the coupling table to the local submodel when a model column is present.
  - Map the VIC field to the local MF6 recharge grid.
  - Apply recharge and step MF6 forward to the end of the current coupling window.

This design is intentional. It keeps VIC orchestration, MF6 execution, and VIC→MF6 remapping in separate modules so the code is easier to debug, reason about, and extend.

## What this program does **not** do

This repository does **not**:

- split MF6 for you,
- build VIC for you,
- build MF6 for you,
- generate the coupling table for you,
- modify the physics or numerical formulation of either VIC or MF6.

Those are treated as upstream prerequisites. This program assumes that both models and all exchange inputs already exist and are internally consistent.

## Repository layout

The final production layout should look like this:

```text
.
├── README.md
├── application_config.py
├── mf6_bmi_model.py
├── mf6_bmi_parallel_model.py
├── mf6_namefile_utils.py
├── run_vic_mf6_mpi_coupling.py
├── vic_image_driver_runtime.py
├── vic_mpi_spawn_runtime.py
├── vic_to_mf6_recharge_mapper.py
├── mpi_finalize_disconnect.c
└── compile_mpi_disconnect_c.sh
```

Module responsibilities:

- `application_config.py`  
  Parses and validates the YAML configuration file and resolves all paths.

- `mf6_namefile_utils.py`  
  Reads `mfsim.nam` and counts the number of GWF models declared in the `BEGIN MODELS` block.

- `mf6_bmi_model.py`  
  Serial MF6 BMI wrapper for diagnostics or future non-parallel workflows.

- `mf6_bmi_parallel_model.py`  
  Parallel MF6 BMI wrapper used by worker ranks through `xmipy.initialize_mpi()`.

- `vic_image_driver_runtime.py`  
  Rewrites step-specific VIC global parameter files, reads VIC water-balance files, and discovers fresh VIC restart outputs.

- `vic_mpi_spawn_runtime.py`  
  Spawns the VIC image driver through MPI and manages the parent-side spawn environment.

- `vic_to_mf6_recharge_mapper.py`  
  Reads the coupling table and converts VIC fields on the VIC grid into MF6 recharge arrays on each local submodel grid.

- `run_vic_mf6_mpi_coupling.py`  
  Main MPI entry point. Orchestrates controller behavior, worker behavior, broadcasts, time windows, and model stepping.

- `mpi_finalize_disconnect.c`  
  Small preload helper library that disconnects spawned VIC child ranks from the parent intercommunicator before `PMPI_Finalize()`.

- `compile_mpi_disconnect_c.sh`  
  Shell script that builds the preload helper into `libvic_parent_disconnect.so`.

## Prerequisites

### 1. Split MF6 simulation

MF6 must already be prepared as a **split simulation** with one groundwater flow model per worker rank.

The launcher determines the expected number of worker ranks by reading the `BEGIN MODELS` block in `mfsim.nam` and counting entries whose type begins with `gwf`. Therefore:

```text
mpirun world size = 1 controller rank + number of GWF models in mfsim.nam
```

Example:

- 4 GWF models in `mfsim.nam`
- run with `mpirun -np 5 ...`

If the world size does not match that count, the program aborts intentionally.

### 2. VIC image driver build

You need a working VIC image driver executable compiled separately.

The code assumes that VIC can be launched as:

```sh
vic_image.exe -g <global_parameter_file>
```

The VIC working directory must already contain the files referenced by the global parameter template and any restart/state files needed for the first coupling step.

### 3. Python environment

You need Python 3 and the following Python packages:

- `mpi4py`
- `numpy`
- `pandas`
- `netCDF4`
- `PyYAML`
- `xmipy`

Install them in the same environment used to launch the coupling code.

### 4. MPI runtime

You need an MPI implementation that works with both:

- `mpi4py` for the Python controller and worker ranks, and
- the VIC and MF6 binaries used in your workflow.

In practice, this means your Python environment, `mpirun`, `xmipy`, MF6 BMI library, and VIC build should all be ABI-compatible.

### 5. VIC parent-disconnect preload library

The VIC child ranks are spawned dynamically from the controller rank. In this workflow, the child process should disconnect from the parent intercommunicator before finalization.

This is handled by a tiny C interpose library, not by Python. Keep that helper as part of the runtime contract.

Build it with:

```sh
./compile_mpi_disconnect_c.sh
```

If you want the direct command, the script typically contains:

```sh
mpicc -shared -fPIC -O2 -Wall -Wextra \
    -o libvic_parent_disconnect.so mpi_finalize_disconnect.c
```

Then either:

- place `libvic_parent_disconnect.so` in the same directory from which you launch the coupling code, or
- export `vic_spawn_preload` to point to it explicitly.

Example:

```sh
export vic_spawn_preload=/absolute/path/to/libvic_parent_disconnect.so
```

## Input data and runtime requirements

### MF6 workspace

The configured MF6 workspace must contain:

- `mfsim.nam`
- the split MF6 input files for all submodels
- the TDIS file used by the simulation
- any additional MF6 package files needed by those submodels

### MF6 BMI shared library

The configured MF6 BMI library must point to the shared library loaded by `xmipy`.

Example:

```yaml
mf6:
  dll: "~/usr/local/src/modflow6/bin/libmf6.so"
```

### VIC global parameter template

The VIC global parameter template is copied and rewritten for each coupling window.

The template should include, at minimum:

- `STARTYEAR`
- `STARTMONTH`
- `STARTDAY`
- `ENDYEAR`
- `ENDMONTH`
- `ENDDAY`
- `STATEYEAR`
- `STATEMONTH`
- `STATEDAY`
- `STATENAME`
- `OUTFILE`

`INIT_STATE` may be present in the template, but it can also be injected automatically for restart continuity after the first coupling step.

### VIC outputs

The runtime expects VIC water-balance files in the configured outputs directory, typically with names like:

```text
wbal.YYYY-MM-DD.nc
```

The runtime also expects VIC restart state files produced under the configured `STATENAME` prefix, typically with names like:

```text
state.YYYYMMDD_00000.nc
```

The controller now checks file modification times to avoid silently reusing stale outputs from a previous run.

### VIC parameter NetCDF

The configured VIC parameter NetCDF file must contain latitude and longitude axes readable from `lat` and `lon`.

### Coupling table

The coupling CSV must contain these required columns:

- `mf6_id`
- `vic_id`
- `mf6_area_ratio`

It may also contain a model-name column used to filter rows to one MF6 submodel. The mapper recognizes any of these column names:

- `mf6_model`
- `model`
- `mname`

Hydrologic assumptions in the mapper:

- each MF6 surface cell can receive contributions from one or more VIC cells,
- `mf6_area_ratio` values are normalized within each MF6 cell,
- the VIC field is interpreted as a recharge-like vertical flux,
- the VIC field is read in **mm/day**,
- the mapped MF6 field is converted to the MF6 model length units per day.

## Configuration file

The main entry point reads a YAML file with three required top-level sections:

- `mf6`
- `vic`
- `coupling`

Example:

```yaml
mf6:
  workspace: "../../MF6/nm/mfnm_split"
  dll: "~/usr/local/src/modflow6/bin/libmf6.so"
  start_date: "1990-01-01"
  length_units: "meters"

vic:
  dir: "../../VIC"
  exe: "~/usr/local/src/VIC/vic/drivers/image/vic_image.exe"
  global_param: "global_param_nc4.txt"
  outputs_dir: "outputs"
  exchange_dir: "exchange_data"
  params_file: "nc4_params.nc"
  wbal_var: "OUT_BASEFLOW"
  init_moist_layer: 2
  spawn_timeout_seconds: 3600

coupling:
  table_csv: "../vic_mf6_exchange_rch.csv"
  start_date: "1990-01-01"
  end_date: "1990-12-31"
  recharge_scale: 1.0
  vic_grid_shape: [227, 212]
```

### Path resolution rules

Path handling is deliberate and important.

Top-level paths are resolved relative to the **directory that contains the YAML file**:

- `mf6.workspace`
- `mf6.dll`
- `vic.dir`
- `vic.exe`
- `coupling.table_csv`

VIC-internal subordinate paths are resolved relative to **`vic.dir`**:

- `vic.global_param`
- `vic.outputs_dir`
- `vic.exchange_dir`
- `vic.params_file`

This makes the YAML file portable and prevents launch-directory-dependent path bugs.

## How the run works

For each coupling window, the controller performs the following sequence:

1. Determine the window start and end dates.
2. Copy the VIC global parameter template to a step-specific file.
3. Rewrite VIC start, end, and state dates in that step file.
4. Inject or update `INIT_STATE` when a previous VIC restart exists.
5. Spawn the VIC image driver with MPI.
6. Wait for the VIC child ranks to disconnect cleanly.
7. Read fresh VIC water-balance output for the current window.
8. Detect the fresh VIC restart file created for the current window.
9. Broadcast the VIC field to all MF6 worker ranks.
10. On each worker, map the VIC field to the local MF6 recharge grid.
11. Apply the mapped recharge array to MF6.
12. Step MF6 to the end date of the current coupling window.
13. Advance to the next window.

By default, the coupling window length is inferred from MF6 TDIS `PERLEN` values when they are readable. If that cannot be determined, the code falls back to one-day windows.

## Running the code

Launch from the directory that contains the Python modules, unless you have installed the code as a package or set `PYTHONPATH` accordingly.

Basic command:

```sh
mpirun -np <world_size> python3 run_vic_mf6_mpi_coupling.py -c /path/to/config.yaml
```

Example with four GWF models:

```sh
mpirun -np 5 python3 run_vic_mf6_mpi_coupling.py -c config.yaml
```

Optional arguments:

```text
--vic-nprocs <n>   number of ranks used when spawning VIC
--max-steps <n>    stop after n coupling windows
```

Examples:

```sh
mpirun -np 5 python3 run_vic_mf6_mpi_coupling.py \
    -c config.yaml \
    --vic-nprocs 4
```

```sh
mpirun -np 5 python3 run_vic_mf6_mpi_coupling.py \
    -c config.yaml \
    --max-steps 2
```

If `--vic-nprocs` is not provided or is set to `0`, the launcher uses the number of MF6 groundwater models as the VIC spawn size.

## Typical setup sequence

A typical run preparation looks like this:

1. Build or obtain the VIC image driver executable.
2. Build or obtain the MF6 BMI shared library.
3. Prepare the split MF6 simulation and verify that `mfsim.nam` lists the expected number of GWF models.
4. Prepare the VIC working directory and global parameter template.
5. Prepare the VIC parameter NetCDF with `lat` and `lon`.
6. Prepare the VIC→MF6 coupling CSV.
7. Build `libvic_parent_disconnect.so`.
8. Export `vic_spawn_preload` or place the library in the launch directory.
9. Write the YAML configuration.
10. Launch the MPI job with `1 + number_of_gwf_models` ranks.

## Logging and diagnostics

The code writes console logs and emits step-level status messages such as:

- MF6 worker initialization and local grid sizes,
- coupling window start and end dates,
- generated VIC global parameter file paths,
- VIC restart tags,
- selected VIC water-balance files,
- basic statistics of the configured VIC water-balance variable,
- VIC spawn completion,
- final successful completion.

These logs are useful for diagnosing:

- path-resolution mistakes,
- stale VIC output reuse,
- missing restart files,
- mismatch between the VIC field shape and the configured coupling grid,
- wrong MPI world size,
- missing preload library,
- bad coupling table schemas.

## Common failure modes

### 1. Wrong number of MPI ranks

Symptom:

- the job aborts before stepping begins,
- the launcher reports that the world size is inconsistent with `mfsim.nam`.

Cause:

- `mpirun -np` does not equal `1 + number of GWF models`.

### 2. YAML paths resolve to the wrong files

Symptom:

- missing `mfsim.nam`, missing VIC parameters file, missing coupling CSV, or missing outputs directory.

Cause:

- relative paths were written assuming the shell working directory rather than the YAML location.

Fix:

- remember that top-level paths resolve relative to the YAML file, and VIC subordinate paths resolve relative to `vic.dir`.

### 3. VIC child ranks hang or fail to disconnect

Symptom:

- the controller blocks during or after spawn,
- timeout errors occur,
- shutdown behavior is inconsistent across windows.

Cause:

- the preload helper is missing, not exported, or not ABI-compatible.

Fix:

- rebuild `libvic_parent_disconnect.so`, verify `vic_spawn_preload`, and confirm MPI compatibility.

### 4. Repeated VIC statistics across windows

Symptom:

- the same mean/min/max values appear across multiple windows.

Possible causes:

- VIC did not write fresh outputs,
- the restart chain is broken,
- the global parameter rewrite is not producing the intended dates,
- the selected variable is constant in the current setup.

The runtime checks output modification times to reduce the risk of silently reusing old files, but you should still inspect the generated step-specific global parameter files and the actual VIC output timestamps.

### 5. Worker-side recharge shape mismatch

Symptom:

- the mapper or BMI layer aborts with a shape mismatch.

Cause:

- the VIC field shape, VIC parameter file dimensions, and coupling table assumptions are inconsistent.

## Development notes

A few design choices are worth documenting explicitly.

### Why the preload helper is in C, not Python

The parent-disconnect fix works by interposing on the child process's C-level `MPI_Finalize()` symbol and then calling `PMPI_Finalize()`. That is the correct place to do it. A Python-only replacement would not reliably intercept the VIC executable's own finalization path.

### Why the code uses one controller rank

The controller handles configuration, spawn orchestration, file I/O, and global broadcasts. Keeping those tasks on rank 0 avoids mixing runtime coordination with MF6 stepping logic and makes failures easier to localize.

### Why VIC and MF6 are coupled sequentially

This framework is designed around explicit coupling windows. For each window, VIC produces a field, the field is remapped, and MF6 advances. That explicit sequence keeps the data exchange visible and debuggable.

## License

This project is licensed under **GPL-3.0-or-later**.

See the file headers and the repository license file for details.

## Contact

**Abdullah Azzam**  
Department of Civil Engineering  
New Mexico State University  
`abdazzam@nmsu.edu`
