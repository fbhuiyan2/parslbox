```
 ███████████                              ████  ███████████
▒▒███▒▒▒▒▒███                            ▒▒███ ▒▒███▒▒▒▒▒███
 ▒███    ▒███  ██████   ████████   █████  ▒███  ▒███    ▒███  ██████  █████ █████
 ▒██████████  ▒▒▒▒▒███ ▒▒███▒▒███ ███▒▒   ▒███  ▒██████████  ███▒▒███▒▒███ ▒▒███
 ▒███▒▒▒▒▒▒    ███████  ▒███ ▒▒▒ ▒▒█████  ▒███  ▒███▒▒▒▒▒███▒███ ▒███ ▒▒▒█████▒
 ▒███         ███▒▒███  ▒███      ▒▒▒▒███ ▒███  ▒███    ▒███▒███ ▒███  ███▒▒▒███
 █████       ▒▒████████ █████     ██████  █████ ███████████ ▒▒██████  █████ █████
▒▒▒▒▒         ▒▒▒▒▒▒▒▒ ▒▒▒▒▒     ▒▒▒▒▒▒  ▒▒▒▒▒ ▒▒▒▒▒▒▒▒▒▒▒   ▒▒▒▒▒▒  ▒▒▒▒▒ ▒▒▒▒▒
```

Your autopilot for HPC job orchestration. ParslBox manages multi-application workflows (LAMMPS, VASP, ORCA, Python, Julia, custom apps) across PBS and SLURM clusters with resource-aware scheduling, dependency tracking, and fault tolerance.

ParslBox provides a CLI (`pbx`), a Python API, and an MCP server for AI-agent integration.

## Highlights

- **Job database:** Persistent SQLite tracking for job paths, resource requirements, dependencies, and status across sessions. Multiple isolated databases via `PBX_DB_PATH`
- **Job lifecycle:** add, update, filter, run with dependency support (parent IDs, parent tags)
- **Built-in apps:** LAMMPS, VASP, ORCA, Python, Julia — plus pluggable custom apps
- **Resource-aware execution:**
  - Pack multiple sub-node jobs onto shared nodes (GPU or CPU)
  - Run multi-node MPI jobs with exclusive node allocation
  - Resource constraining for non-MPI apps (Python, Julia) via resource launcher
  - CPU-GPU affinity-aware placement
  - MPI backends: MPICH, OpenMPI, srun
- **Scheduler support:** PBS (`pbx qsub`) and SLURM (`pbx sbatch`) with configurable `--sched-opts`
- **Pre-configured HPC systems:** Polaris, Aurora (GPU & Tile modes), Sophia, Crux, LCRC Swing, LCRC Improv
- **Dynamic job discovery:** `--dynamic` (default) polls for newly added jobs during a run session. New jobs matching the same `--apps`/`--tags` filters are picked up every 60s. Failed jobs reset to Ready by the user (via `pbx update --status Ready` from another terminal) are also re-discovered and re-run. Use `--static` for collect-once behavior
- **Fault tolerance:** Node health tracking, quarantine, and auto-recovery
- **Script-driven status reporting:** Python/Julia scripts report success/failure via `report_status()` utility
- **Python API and MCP server** for programmatic and AI-agent integration
- **Rich CLI output** with tables and colors

## Installation

Requirements:
- Python >= 3.11, < 3.14
- Parsl >= 2025.9.8

### Using Poetry
```bash
conda create --name parslbox python=3.11.9
conda activate parslbox
pip install poetry

git clone https://github.com/fbhuiyan2/parslbox.git
cd parslbox
poetry install                          # core dependencies only
# poetry install --extras "simulation"  # + ase, pymatgen
# poetry install --extras "agentic"    # + uvicorn, mcp, pydantic
# poetry install --extras "simulation agentic"  # both extras
# poetry install --all-extras          # all optional packages
```

### Using pip
```bash
conda create --name parslbox python=3.11.9
conda activate parslbox

git clone https://github.com/fbhuiyan2/parslbox.git
cd parslbox
pip install .                           # core dependencies only
# pip install ".[simulation]"           # + ase, pymatgen
# pip install ".[agentic]"             # + uvicorn, mcp, pydantic
# pip install ".[simulation,agentic]"  # both extras
# pip install ".[all]"                 # all optional packages
```

### Setup
Run `pbx config` to interactively create a configuration file. Edit the generated config to set correct executable paths and environment setup before running jobs.

## Quick Start

Add jobs:
```bash
# Add a single LAMMPS job (requires system config name)
pbx add /path/to/sim --app lammps --config polaris --ngpus 2 --tag run1

# Add all subdirectories in current folder as VASP jobs
pbx add all --app vasp --config polaris --tag ManyVaspCalc

# Add a Python script job with arguments and environment file
pbx add /path/to/analysis --app python --config polaris --input plot.py \
  --args "--mode gpu --output-prefix test" --envfile env_setup.sh

# Add with explicit resources and dependencies
pbx add /path/to/calc --app vasp --config polaris --nnodes 1 --nocc 0.5 \
  --parents "10 11" --tag stage2
# or wait for all jobs with a tag to finish
pbx add /path/to/calc2 --app vasp --config polaris --parent-tag stage1
```

Submit via PBS or SLURM:
```bash
# Short flags
pbx qsub -c sophia -N myrun -q gpu --select 2 -T 90 -A myproject -a lammps -t production
pbx sbatch -c polaris -N myrun -p gpu --nodes 2 -T 90 -A myproject -a lammps -t production

# Long flags
pbx qsub --config sophia --job-name myrun --queue gpu --select 2 --walltime 90 --project myproject --apps lammps --tags production
pbx sbatch --config polaris --job-name myrun --partition gpu --nodes 2 --walltime 90 --account myproject --apps lammps --tags production
```

Inspect, filter, update, remove:
```bash
# List jobs (rich table)
pbx ls
pbx ls --status Running --app lammps --tag production
pbx ls --all
pbx ls -n 15
pbx ls -n -20

# Show fields per job (supports ranges: 1-5 8 14-20)
pbx info 1-5 8 --path --ngpus --envfile --parents

# Calculate resource requirements for a target system
pbx info 1-10 --req polaris

# Get IDs via filters (use in command composition)
pbx filter --status done --app vasp
pbx filter -s failed -a lammps -t test -p /path/part -i input.lammps

# Update fields and dependencies (supports ranges)
pbx update 1-5 --status Restart --tag high-priority
pbx update 10 --nnodes 2          # multi-node
pbx update 11 --nocc 0.25         # CPU-only fractional occupancy
pbx update 12 --add_deps "8 9" --rm_deps "7"
pbx update 13 --envfile ./env.sh
pbx update 14 --input new_input.dat
pbx update 15 --args "--new-flag value"

# Remove jobs (supports ranges)
pbx rm 1-5 8 10-15
pbx rm all
pbx rm $(pbx filter --status done)
```

## Supported Applications

### Built-in Apps

| App | MPI | Input Required | Default Input | Success Check |
|-----|-----|----------------|---------------|---------------|
| **lammps** | Yes | Yes | `in.lammps` | Checks for "Total wall time:" in `log.lammps` |
| **vasp** | Yes | No | — | Auto-selects `vasp_gpu` or `vasp_std` |
| **orca** | Internal | Yes | `input.inp` | Checks for "ORCA TERMINATED NORMALLY" in `.out` files |
| **python** | No | Yes | — | Reads `PBX_JOB_STATUS_REPORT` file via `report_status()` |
| **julia** | No | Yes | — | Reads `PBX_JOB_STATUS_REPORT` file |

ORCA manages its own MPI parallelism via a bundled OpenMPI — pbx does not wrap it with `mpirun`. Instead, pbx generates a `.nodes` file and passes `--host` to ORCA's internal launcher. Parallelism is controlled by `%pal nprocs N end` in the ORCA input file. Ensure ORCA's directory is first on `PATH` so its bundled `mpirun` takes priority over the system MPI.

Non-MPI apps (Python, Julia) are automatically constrained to their assigned node/resources via a resource launcher. Scripts can access the MPI command via the `PBX_MPI_PREFIX` environment variable if needed.

### Script-Driven Status Reporting

Python and Julia scripts must report their outcome so PBX can determine job success:

```python
# In your Python script
from parslbox.apps.utils import report_status

# ... do work ...

if success:
    report_status("done")
else:
    report_status("failed")
```

```julia
# In your Julia script
open("PBX_JOB_STATUS_REPORT", "w") do f
    write(f, "Done")   # or "Failed"
end
```

If the status file is not found after execution, the job is marked as Failed.

### Custom Apps

Custom apps can be registered via `config.yaml` without modifying ParslBox source code:

```yaml
custom_apps:
  my_app:
    module: "/path/to/my_app.py"
    class: "MyAppClass"

my_app:
  polaris:
    executable_path: "/path/to/exe"
    environment_setup: |
      module load my_module
```

Implement by inheriting from `AppBase` with `get_command_template()`. See [`parslbox/apps/EXAMPLE_NEW_APP.py`](parslbox/apps/EXAMPLE_NEW_APP.py) for templates including MPI and non-MPI examples.

## Supported HPC Systems

| System | Scheduler | GPUs/Node | GPU Type | Cores/Node | MPI Backend |
|--------|-----------|-----------|----------|------------|-------------|
| **polaris** | PBS | 4 | NVIDIA A100 | 64 | MPICH |
| **aurora-gpu** | PBS | 6 | Intel Max 1550 | 208 | MPICH |
| **aurora-tile** | PBS | 12 (6x2 tiles) | Intel Max 1550 | 208 | MPICH |
| **sophia** | PBS | 8 | NVIDIA A100 (DGX) | 128 | OpenMPI |
| **crux** | PBS | 0 (CPU-only) | — | 256 | MPICH |
| **lcrc-swing** | PBS | 8 | NVIDIA A100 | 64 | OpenMPI |
| **lcrc-improv** | PBS | 0 (CPU-only) | — | 128 | OpenMPI |
| **pinnacles-cenvalarc** | SLURM | 0 or 2 (auto-detected) | NVIDIA L40S / H200 NVL | 64 | srun |

Each system defines its own MPI defaults, scheduler templates, and resource detection methods. New systems can be added by creating a config class inheriting from `BaseSystemConfig`.

## Programmatic API (Python)

ParslBox provides a Python API for scripts and AI agents. The API delegates to core logic inside the command modules, ensuring CLI and API behavior stay in sync.

Main methods:
- `add_jobs(paths, app, config, ...)`
- `update_jobs(job_ids, ...)`
- `list_jobs(...)` — returns all jobs when called with no filters
- `get_job(job_id)`, `get_jobs_by_ids(ids)`
- `remove_job`, `remove_jobs`, `remove_all_jobs`
- `qsub(...)` — submit via PBS
- `sbatch(...)` — submit via SLURM
- `run(...)` — execute jobs directly

Exceptions: `ParslBoxError`, `ValidationError`, `JobNotFoundError`

```python
from parslbox.api import ParslBox
pbx = ParslBox()
job_ids, failures, log = pbx.add_jobs(
    paths=["/path/to/sim"], app="lammps", config="polaris", ngpus=2
)
jobs = pbx.list_jobs()
pbx.update_jobs(job_ids, status="Restart")
```

## MCP Server

ParslBox includes an MCP server for AI-agent integration. Install with `pip install ".[agentic]"` and start:

```bash
# HTTP mode (standalone server on port 9795)
python -m parslbox.mcp.mcp_server

# stdio mode (for Claude Code integration)
python -m parslbox.mcp.mcp_server --stdio
```

Exposed tools: `add_jobs`, `submit_pbs_job`, `submit_slurm_job`, `remove_job`, `update_job`, `filter_jobs`, `list_jobs`, `get_job`, `get_jobs`.

### Claude Code Integration

The project includes a `.mcp.json` for automatic MCP server discovery. When running Claude Code from the project directory, it will offer to connect to the ParslBox MCP server.

For global access (any directory), add to `~/.claude/.mcp.json`:
```json
{
  "mcpServers": {
    "parslbox": {
      "command": "/path/to/conda/envs/parslbox/bin/python",
      "args": ["-m", "parslbox.mcp.mcp_server", "--stdio"],
      "cwd": "/path/to/parslbox"
    }
  }
}
```

See [`examples/chemgraph_parslbox_example/`](examples/chemgraph_parslbox_example/) for an HTTP client example.

## Commands Overview

Note: Users submit via `pbx qsub` or `pbx sbatch`. These generate a `submit.sh` and submit it to the scheduler; `submit.sh` invokes `pbx run` under the hood.

- **pbx config** — Interactive wizard to create/reconfigure configuration
  - Prompts for config path, system selection, app selection, and database creation
  - Arguments: optional path (`.` for current dir, `~` for home, or custom path)

- **pbx add** — Add jobs to the database
  - Arguments: paths (one or more directories, or `all`)
  - Required: `--app/-a`, `--config/-c`
  - Options: `--tag/-t`, `--input/-i`, `--args`, `--ngpus/-g`, `--nnodes/-n`, `--nocc/-o`, `--ranks-per-node/-rpn`, `--mpiopts`, `--envfile/-e`
  - Dependencies: `--parents/-P "1 2 3"`, `--parent-tag`
  - Initial status: `--status/-s` (default Ready)
  - `--args` appends arguments to input file (e.g., `python script.py --flag value`)
  - For `--nnodes > 1`, pbx ignores `--ngpus` and `--nocc`
  - Single-node GPU jobs auto-assign all GPUs on GPU systems when `-g` is not specified

- **pbx qsub** — Submit PBS job
  - Required: `--config/-c`, `--job-name/-N`, `--queue/-q`, `--select`, `--walltime/-T`, `--project/-A`
  - `--walltime` defaults to minutes; supports `h` (hours) and `d` (days) suffixes (e.g., `90`, `4.25h`, `3.5d`)
  - Optional: `--run-dir`, `--apps/-a`, `--tags/-t`, `--retries`, `--sched-opts`, `--dynamic/--static`
  - `--sched-opts` adds extra `#PBS` directives (repeatable)
  - `--dynamic` (default) enables live discovery of new jobs during the run; `--static` for collect-once behavior

- **pbx sbatch** — Submit SLURM job
  - Required: `--config/-c`, `--job-name/-N`, `--partition/-p`, `--nodes`, `--walltime/-T`, `--account/-A`
  - Optional: `--run-dir`, `--apps/-a`, `--tags/-t`, `--retries`, `--sched-opts`, `--dynamic/--static`

- **pbx ls** — List jobs with filtering and pagination
  - Filters: `--status/-s`, `--app/-a`, `--tag/-t`
  - `--all` shows all; `-n N` first N; `-n -N` last N; `-n 0` all
  - Auto-paginates (first 10 + last 10) when > 25 jobs

- **pbx info** — Show detailed job information. Supports ID ranges (e.g., `pbx info 1-5 8`)
  - Field selectors: `--path/-p`, `--ngpus/-n`, `--app/-a`, `--status/-s`, `--tag/-t`, `--input/-i`, `--sched-job-id/-j`, `--timestamp/-ts`, `--envfile/-e`, `--parents/-P`
  - `--req/-r SYSTEM` — Calculate resource requirements for target system (simultaneous vs optimal packing)

- **pbx filter** — Output space-separated job IDs for command composition
  - Filters: `--status/-s`, `--app/-a`, `--tag/-t`, `--path/-p`, `--in-file/-i`

- **pbx update** — Update job fields and dependencies. Supports ID ranges (e.g., `pbx update 1-5 8`)
  - Fields: `--status`, `--tag`, `--input/-i`, `--args`, `--ngpus/-g`, `--envfile/-e`, `--nnodes/-n`, `--nocc/-o`, `--ranks-per-node/-rpn`
  - Dependencies: `--add_deps/--padd`, `--rm_deps/--parm`
  - Validates dependencies and prevents circular references

- **pbx rm** — Remove jobs by IDs or ranges (e.g., `pbx rm 1-5 8`), or `pbx rm all` (with confirmation)

- **pbx run** (internal) — Engine used by qsub/sbatch; not for direct use
  - `--dynamic/--static` controls live job discovery (default: `--dynamic`)

## Configuration

Environment variables:
- `PBX_DB_PATH` — database file or directory path
- `PBX_CONFIG_PATH` — config file or directory path

Defaults (when env vars are not set):
- Database: `~/.parslbox/job_database_pbx.db`
- Config: `~/.parslbox/config.yaml`
- Runs: `~/.parslbox/runs/<timestamp>/`

```bash
export PBX_DB_PATH=/scratch/mydbs/pbx.db
export PBX_CONFIG_PATH=/scratch/mycfgs/config.yaml
```

Using separate `PBX_DB_PATH` and/or `PBX_CONFIG_PATH` allows multiple isolated databases and configurations.

## Resource Manager

- **Single-node GPU jobs:** Assign specific GPU IDs (e.g., 0,1) so multiple jobs can share a node
- **CPU-only jobs:** Use `--nocc` for fractional node occupancy (e.g., 0.25); multiple jobs co-reside up to 1.0
- **Multi-node jobs:** Exclusive free nodes with MPI hostlist generation
- **Non-MPI apps:** Resource launcher constrains execution to assigned node/resources
- **Backlog scheduling:** Dependency-aware rescheduling as resources free up
- **Node health tracking:** Quarantine nodes after repeated failures, auto-recover when healthy

### MPI CPU Binding

ParslBox generates MPI launch commands with CPU binding flags appropriate for each job type (subnode, fullnode, multinode) and scheduler:

- **PBS systems (mpiexec/MPICH, mpirun/OpenMPI):** Configurable via `cpu_bind_method` in the `mpi:` config section. Options: `none`, `rankfile`, `list`, `depth`. The `rankfile` and `list` methods provide GPU-affinity-aware core assignments.
- **SLURM systems (srun):** Native SLURM resource binding. CPU binding via `--cpus-per-task=D --cpu-bind=cores|threads` (configurable with `cpu_bind_method: cores` or `threads`). GPU jobs use native SLURM flags: `--gpus-per-node` + `--gpu-bind=map_gpu` for full/multi-node, `--gpus-per-task` + `--mem-per-gpu` for sub-node. Sub-node jobs include `--exact` and `-u` for proper resource isolation.

Details: [`parslbox/resource_manager/README.md`](parslbox/resource_manager/README.md)

## Job Lifecycle

Statuses:
- Ready → Submitted → Running → Done | Failed | Killed
- Restart — for recoverable errors / re-runs
- Warning — if an app returns an invalid/unknown status
- Killed — when walltime is exceeded (SIGTERM handler marks active jobs)

## Examples

- [`examples/strong_scaling/`](examples/strong_scaling/) — LAMMPS strong scaling orchestration with automated analysis and publication-ready plots
- [`examples/weak_scaling/`](examples/weak_scaling/) — LAMMPS weak scaling orchestration with intelligent system replication
- [`examples/chemgraph_parslbox_example/`](examples/chemgraph_parslbox_example/) — MCP server integration with ChemGraph AI agent

## Contributing

Issues and PRs welcome at:
- https://github.com/fbhuiyan2/parslbox/issues

## License

Author and developer: Fakhrul Hasan Bhuiyan

Copyright Argonne UChicago LLC, 2026. All rights reserved.
