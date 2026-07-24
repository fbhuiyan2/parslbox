# ParslBox Applications

ParslBox ships with built-in support for a small set of HPC apps, and lets you register custom ones via `config.yaml`.

## Built-in Apps

| App | Identifier | MPI | Input Required | Default Input | Success Check |
|---|---|---|---|---|---|
| **LAMMPS (Kokkos)** | `lammps-kk` | Yes | Yes | `in.lammps` | "Total wall time:" in `log.lammps` |
| **VASP** | `vasp` | Yes | No | — | Auto-selects `vasp_gpu` or `vasp_std` based on system |
| **ORCA** | `orca` | Internal (bundled OpenMPI) | Yes | `input.inp` | "ORCA TERMINATED NORMALLY" in `.out` files |
| **Python** | `python` | Yes (1 rank/node default) | Yes | — | Reads `PBX_JOB_STATUS_REPORT` via `report_status()` |
| **Julia** | `julia` | Yes (1 rank/node default) | Yes | — | Reads `PBX_JOB_STATUS_REPORT` |

### App-specific notes

**ORCA** manages its own MPI parallelism via a bundled OpenMPI — pbx does **not** wrap it with `mpirun`. Instead, pbx generates a `.nodes` file and passes `--host` to ORCA's internal launcher. Parallelism is controlled by `%pal nprocs N end` in the ORCA input file. Ensure ORCA's directory is first on `PATH` so its bundled `mpirun` takes priority over the system MPI.

**Python / Julia** scripts are launched via `mpiexec`/`srun` with 1 rank per node by default — they run constrained to the assigned nodes whether or not they use `mpi4py`/`MPI.jl`. Scripts that want to launch nested MPI subprocesses can read the prepared MPI command from the `PBX_MPI_PREFIX` environment variable that pbx exports into their environment.

**VASP** auto-detects whether to run `vasp_gpu` or `vasp_std` based on the system's GPU configuration. No `--input` flag needed.

**LAMMPS (Kokkos)** is the only built-in LAMMPS variant. The default rank-per-node policy is 1 rank per GPU. Override with `--ranks-per-node/-rpn` if your input benefits from a different ratio.

---

## Script-Driven Status Reporting

Python and Julia jobs must explicitly report success or failure so pbx can mark the job's final status. If the report file is missing after execution, the job is marked **Failed**.

The reporting protocol is a tiny file named `PBX_JOB_STATUS_REPORT` in the job directory, containing either `Done` or `Failed`.

### Python

```python
from parslbox.apps.utils import report_status

# ... do work ...

if success:
    report_status("done")
else:
    report_status("failed")
```

`report_status()` accepts `"done"` or `"failed"` (case-insensitive) and raises `ValueError` for anything else.

### Julia

```julia
open("PBX_JOB_STATUS_REPORT", "w") do f
    write(f, "Done")   # or "Failed"
end
```

### MPI from inside Python / Julia scripts

Both apps export `PBX_MPI_PREFIX` into the job's environment so scripts can launch MPI subprocesses with the same binding pbx would have used:

```python
import os, subprocess
mpi = os.environ.get("PBX_MPI_PREFIX", "")
subprocess.run(f"{mpi} ./my_mpi_binary input.dat", shell=True, check=True)
```

---

## Custom Apps

Custom apps can be registered via `config.yaml` without modifying ParslBox source code. Two blocks are needed in the YAML — registration and per-system configuration:

```yaml
# 1) Tell pbx where to import the class from
custom_apps:
  my_app:
    module: "/path/to/my_app.py"
    class: "MyAppClass"

# 2) Per-system executable + environment setup (one entry per system you'll run on)
my_app:
  polaris:
    executable_path: "/path/to/exe"
    environment_setup: |
      module load my_module
  aurora-tile:
    executable_path: "/path/to/exe-aurora"
    environment_setup: |
      module load other_module
```

Once registered, the app is usable just like a built-in: `pbx add /path/to/job --app my_app --config polaris ...`.

### Writing the class

Inherit from `AppBase` (`parslbox/apps/appbase.py`) and provide at minimum:

| Attribute / method | Purpose |
|---|---|
| `INPUT_REQUIRED: bool` | Whether `--input/-i` is mandatory at `pbx add` time |
| `DFLT_INPUT: str \| None` | Default input filename if user omits `--input` |
| `get_command_template(**kwargs) -> str` | The shell command to run; pbx substitutes resource placeholders |

Optional overrides:

| Method | When to override |
|---|---|
| `get_default_ranks_per_node(ngpus, num_nodes, system_config)` | Custom rank policy (LAMMPS uses 1/GPU; Python uses 1/node) |
| `get_additional_setup(**kwargs)` | Extra env vars or shell setup before the command |
| `preprocess(job_id, job_path, db_path, app_config, config_name)` | One-time setup (file copies, validation) before submission |
| `check_success(job_id, job_path, db_path, error_message)` | Custom success determination beyond status file / exit code |
| `postprocess(job_id, job_path, db_path)` | Cleanup or analysis after the job finishes |
| `RUN_HOOKS_ON_COMPUTE: bool` | When `True`, `preprocess`/`postprocess` run on the assigned compute node via a subprocess wrapped with the resource launcher, instead of in-process in `pbx run` on the head node. Default `False`. Does not affect `restart()`. |

Set `RUN_HOOKS_ON_COMPUTE = True` when your hooks do non-trivial work (file staging on node-local scratch, NumPy/HDF5 post-analysis, anything that imports the heavy modules the job uses). The app class is re-instantiated on the compute node, so its `__init__` must be side-effect-free. The hook's return value (the status string from `postprocess`) is communicated back via a `PBX_HOOK_RETURN` file in the job directory, mirroring the `PBX_JOB_STATUS_REPORT` pattern.

See [`parslbox/apps/EXAMPLE_NEW_APP.py`](../parslbox/apps/EXAMPLE_NEW_APP.py) for full templates.

### Where custom apps live

You can either:
- Point `module:` at any absolute path on disk (handy for one-off scripts).
- Drop your class file into `parslbox/apps/custom_apps/` and reference it by relative module name. This directory is preserved across pbx upgrades.
