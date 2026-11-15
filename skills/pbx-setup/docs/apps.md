# ParslBox Applications

ParslBox ships with built-in support for a small set of HPC apps, and lets you register custom ones via `config.yaml`.

## Built-in Apps

| App | Identifier | MPI | Input Required | Default Input | Success Check |
|---|---|---|---|---|---|
| **LAMMPS (Kokkos)** | `lammps-kk` | Yes | Yes | `in.lammps` | "Total wall time:" in `log.lammps` |
| **VASP** | `vasp` | Yes | No | — | Auto-selects `vasp_gpu` or `vasp_std` based on system |
| **ORCA** | `orca` | Internal (bundled OpenMPI) | Yes | `input.inp` | `****ORCA TERMINATED NORMALLY****` in any `.out` file |
| **Python** | `python` | Yes (1 rank/node default) | Yes | — | Reads `PBX_JOB_STATUS_REPORT` via `report_status()` |
| **Julia** | `julia` | Yes (1 rank/node default) | Yes | — | Reads `PBX_JOB_STATUS_REPORT` |
| **LAMMPS (plain)** | `lammps` | Yes | Yes | `in.lammps` | "Total wall time:" in `log.lammps` |
| **GROMACS** | `gromacs` | Yes | Yes | `topol.tpr` | Last `mdrun` segment of `<name>.log` has "Finished mdrun" and no time-limit or signal stop |
| **Quantum ESPRESSO** | `qe` | Yes | Yes | — | "JOB DONE." in `<input>.out` and no stopped-early line |

### App-specific notes

**ORCA** manages its own MPI parallelism via a bundled OpenMPI — pbx does **not** wrap it with `mpirun`. Instead, pbx generates a `.nodes` file and passes `--host` to ORCA's internal launcher. Parallelism is controlled by `%pal nprocs N end` in the ORCA input file. Ensure ORCA's directory is first on `PATH` so its bundled `mpirun` takes priority over the system MPI.

**Python / Julia** scripts are launched via `mpiexec`/`srun` with 1 rank per node by default — they run constrained to the assigned nodes whether or not they use `mpi4py`/`MPI.jl`. Scripts that want to launch nested MPI subprocesses can read the prepared MPI command from the `PBX_MPI_PREFIX` environment variable that pbx exports into their environment.

**VASP** auto-detects whether to run `vasp_gpu` or `vasp_std` based on the system's GPU configuration. No `--input` flag needed.

**LAMMPS (Kokkos)** is `lammps-kk`; plain LAMMPS is `lammps` (below). The default rank-per-node policy is 1 rank per GPU. Override with `--ranks-per-node/-rpn` if your input benefits from a different ratio — in either direction. More ranks than GPUs makes them share, round-robin, with the ranks on a GPU sharing its CPU cores. Fewer ranks than GPUs gives each rank a contiguous block of GPUs, but note that Kokkos binds **one GPU per MPI rank** (`-k on g N` is GPUs per *node*, and it selects `local_rank % N`), so the surplus GPUs in a block go unused — 1 rank per GPU is the only way to drive them all. pbx warns when ranks outnumber GPUs, and when the GPU count does not divide evenly into the rank count, since unequal blocks cannot be described by a single `-k on g N`. `ranks_per_node` is capped at `pbx add` time by the node's usable core count; `pbx update` does not re-check it.

**LAMMPS (plain)** is LAMMPS without Kokkos. CPU jobs run `lmp -in <input>`; a job that holds GPUs runs the GPU package, `lmp -sf gpu -pk gpu N -in <input>`, with `N` the GPUs each rank can see (1 under srun `map_gpu` or a per-rank wrapper script). That needs a LAMMPS built with the GPU package. On a GPU system a job gets all GPUs unless `--nocc` is given (`--ngpus 0` alone still auto-assigns them), so add `--nocc 1.0` to run on CPUs; use `lammps-kk` for a Kokkos build. `OMP_NUM_THREADS` is set to 1 after `environment_setup` runs, so it cannot be changed there. Restart is not supported. Rank defaults and sharing work as for `lammps-kk`. Success needs `log.lammps`, so a `log` command in the input that moves the log makes the job Failed.

**GROMACS** runs `gmx_mpi mdrun -deffnm <name> -s <name>.tpr -cpi <name>.cpt -ntomp T [-nb gpu]` on a prebuilt `.tpr`; pbx does not run `gmx grompp`. Pass extra `mdrun` flags with `--app-args`, but not `-deffnm`, `-s` or `-cpi`, which pbx sets, nor `-noappend` or `-g`, which rename the log the success check reads. `-ntomp` is the cores the job holds per node divided by ranks per node. That is exact for GPU jobs and for CPU jobs on whole nodes; a CPU job at `--nocc` below 1 is assumed to hold the whole node, so it gets too many threads (e.g. `--nocc 0.5` with default ranks gives 2× oversubscription) and should pass `-ntomp` itself. `-nb gpu` is added for jobs with GPUs; a `-ntomp` or `-nb` in `--app-args` replaces them. `-cpi` starts from step 0 when no checkpoint exists, so a restart (`pbx update --status Restart`, or `--respawn`) re-runs the same command and resumes. A run stopped by `-maxh` or a signal prints "Finished mdrun" too, so it is marked **Failed**, not Done, and so is any non-zero exit. A thread-MPI `gmx` executable with more than one rank is refused at launch (the job fails with the reason in `pbx_job_<id>.err`), since it would start several independent simulations; point `executable_path` at `gmx_mpi`.

**Quantum ESPRESSO** (only `pw.x`; `ph.x` and the other QE programs are not covered) runs `pw.x -in <input> > <input>.out` (the input's extension replaced by `.out`); the input file is required and `-nk`-style flags go in `--app-args`. `pw.x` prints "JOB DONE." even when SCF did not converge or a time limit stopped it, and exits 0 by default, so the success check also rejects "convergence NOT achieved", "Maximum CPU time exceeded" and "Program stopped by user request". Ion-relaxation non-convergence is not detected. `OMP_NUM_THREADS` is set to 1 after `environment_setup` runs, so hybrid MPI+OpenMP runs cannot be set up here. A non-zero exit is Failed. Restart is not supported.

These three apps were written from the programs' documentation and source and have not yet been run on an HPC system; flags that pbx does not fix can be adjusted through `--app-args`.

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
| `restart(job_dict) -> dict \| None` | Resume a `Restart`-status job. Base raises `NotImplementedError`; override to patch `in_file`/`env_file`/`tag` and re-run, return `None`/`{}` to re-run as-is, or leave unimplemented to mark the job `Failed`. See [`pbx-run-details.md`](pbx-run-details.md). |
| `min_remaining_walltime(job_dict) -> int` | Minimum remaining batch walltime (seconds) required to dispatch a fresh job; default `0` (no gate). |
| `RUN_HOOKS_ON_COMPUTE: bool` | When `True`, `preprocess`/`postprocess` run on the assigned compute node via a subprocess wrapped with the resource launcher, instead of in-process in `pbx run` on the head node. Default `False`. Does not affect `restart()`. |

Set `RUN_HOOKS_ON_COMPUTE = True` when your hooks do non-trivial work (file staging on node-local scratch, NumPy/HDF5 post-analysis, anything that imports the heavy modules the job uses). The app class is re-instantiated on the compute node, so its `__init__` must be side-effect-free. The hook's return value (the status string from `postprocess`) is communicated back via a `PBX_HOOK_RETURN` file in the job directory, mirroring the `PBX_JOB_STATUS_REPORT` pattern.

See `parslbox/apps/EXAMPLE_NEW_APP.py` (from the repo root) for full templates.

### Where custom apps live

You can either:
- Point `module:` at any absolute path on disk (handy for one-off scripts).
- Drop your class file into `parslbox/apps/custom_apps/` and reference it by relative module name. This directory is preserved across pbx upgrades.
