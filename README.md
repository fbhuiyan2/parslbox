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

Your autopilot for HPC job orchestration. ParslBox manages multi-application workflows (LAMMPS, VASP, Python, custom apps) across PBS and SLURM clusters with resource-aware scheduling, dependency tracking, and fault tolerance.

ParslBox provides a CLI (pbx), a Python API, and an MCP server for AI-agent integration.

## Highlights:
- Job database: Persistent job tracking using SQLite database for job paths, resource requirements, dependencies, and status across sessions
  - Multiple databases supported through environment variable `PBX_DB_PATH`
- Job lifecycle: add, update, filter, run with dependency support (parents, tags)
- Built-in apps: LAMMPS, VASP, Python — plus pluggable custom apps
- Resource-aware execution:
  - Pack multiple sub-node jobs onto shared nodes (GPU or CPU)
  - Run multi-node MPI jobs with exclusive node allocation
  - CPU-GPU affinity-aware placement
  - MPI launchers: openmpi, mpich, srun
- Scheduler support: PBS (pbx qsub) and SLURM (pbx sbatch)
- Pre-configured HPC systems: Polaris, Aurora, Sophia, Crux, LCRC Swing
- Fault tolerance with node health tracking and quarantine
- Python API and MCP server for programmatic and AI-agent integration
- Rich CLI output (tables, colors)

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

### Verify
```bash
# First call initializes ~/.parslbox/config.yaml and the job database
# (or custom locations via PBX_CONFIG_PATH / PBX_DB_PATH)
pbx ls
```

On first run, a default config is created at ~/.parslbox/config.yaml (unless PBX_CONFIG_PATH overrides it). Edit this file to set correct executable paths, environment setup, and system settings before running jobs.


## Quick Start

Add jobs:
```bash
# Add a single LAMMPS job (requires system config name)
pbx add /path/to/sim --app lammps --config polaris --ngpus 2 --tag run1

# Add all subdirectories in current folder as VASP jobs
pbx add all --app vasp --config polaris --tag ManyVaspCalc

# Add with explicit resources and dependencies
pbx add /path/to/calc --app vasp --config polaris --nnodes 1 --nocc 0.5 \
  --parents "10 11" --tag stage2
# or wait for all jobs with a tag to finish
pbx add /path/to/calc2 --app vasp --config polaris --parent-tag stage1
```

Submit via PBS:
```bash
pbx qsub \
  --config sophia \
  --job-name myrun \
  --queue gpu \
  --select 2 \
  --walltime 90 \
  --project myproject \
  --apps lammps --tags production
# Creates a timestamped run dir with submit.sh and calls qsub.
# submit.sh invokes: pbx run --config sophia [--apps ... --tags ... --retries ...]
```

Inspect, filter, update, remove:
```bash
# List jobs (rich table)
pbx ls
pbx ls --status Running --app lammps --tag production
pbx ls --all
pbx ls -n 15
pbx ls -n -20

# Show fields per job
pbx info 1 2 --path --ngpus --envfile --parents

# Get IDs via filters (use in command composition)
pbx filter --status done --app vasp
pbx filter -s failed -a lammps -t test -p /path/part -i input.lammps

# Update fields and dependencies
pbx update 4 5 --status Restart --tag high-priority
pbx update 10 --nnodes 2          # multi-node
pbx update 11 --nocc 0.25    # CPU-only fractional occupancy
pbx update 12 --add_deps "8 9" --rm_deps "7"
pbx update 13 --envfile ./env.sh
pbx update 14 --input new_input.dat  # update input file

# Remove jobs
pbx rm 1 2 3
pbx rm all
pbx rm $(pbx filter --status done)
```

## Programmatic API (Python)

ParslBox provides a compact Python API for scripts and AI agents. The API delegates to core logic inside the command modules (add/update/qsub), ensuring CLI and API behavior stay in sync without duplication.

Main methods:
- add_jobs(paths, app, config, ...)
- update_jobs(job_ids, ...)
- list_jobs(...): when called with no filters, returns all jobs
- get_job(job_id), get_jobs_by_ids(ids)
- remove_job, remove_jobs, remove_all_jobs
- qsub(...): submit to the scheduler using the same template as the CLI

Exceptions: ParslBoxError, ValidationError, JobNotFoundError

Example:
```python
from parslbox.api import ParslBox
pbx = ParslBox()
job_id = pbx.add_job("/path/to/sim", app="lammps", config="polaris", ngpus=2)
jobs = pbx.list_jobs()  # returns all jobs if no filters
pbx.update_job(job_id, status="Submitted")
```

## MCP Server

ParslBox includes an MCP server for AI-agent integration. Install with `pip install ".[agentic]"` and start the server:

```bash
python -m parslbox.mcp.mcp_server
```

The server exposes tools for job management (add, remove, update, filter, submit) over HTTP on port 9005. See [`examples/chemgraph_parslbox_example/`](examples/chemgraph_parslbox_example/) for a full setup and client example.

## Commands Overview

Note on usage:
- Users should submit via pbx qsub. The qsub command generates a submit.sh and submits it to the scheduler; submit.sh invokes pbx run under the hood.
- pbx run is the engine used by qsub and is not intended to be called directly by users.

- pbx config
  - Interactive wizard to create or reconfigure ParslBox configuration
  - Prompts for config path, system selection, app selection, and database creation
  - Arguments: optional path (`.` for current dir, `~` for home, or custom path)

- pbx add
  - Arguments: paths (one or more directories, or 'all')
  - Required: --app/-a, --config/-c
  - Common options: --tag/-t, --input/-i, --ngpus/-g, --nnodes/-n, --nocc/-o, --ranks-per-node/-rpn, --mpiopts, --envfile/-e
  - Dependencies: --parents/-P "1 2 3", --parent-tag
  - Initial status: --status/-s (default Ready)
  - Input handling:
    - If the app requires input and no file is provided, pbx prompts for a filename.
    - If the app defines a default input, that default is used unless overridden via --input.
  - Single-node vs multi-node resources:
    - For --nnodes > 1, pbx ignores --ngpus and --nocc and displays actual GPU count, nocc:NA.
  - Ranks per node:
    - GPU jobs: --ranks-per-node is ignored; pbx uses 1 rank per GPU.
    - CPU jobs: If not specified, pbx uses ranks_per_node = cores_per_node * node_occupancy (minimum 1).
  - Python environment file:
    - For --app python without --envfile/-e, pbx warns and asks for confirmation to proceed without an env file.

- pbx qsub
  - Required: --config/-c, --job-name/-N, --queue/-q, --select, --walltime/-T, --project/-A
  - Optional: --filesystems, --run-dir, --apps/-a, --tags/-t, --retries
  - Behavior: creates run dir, generates submit.sh from PBS template, runs qsub submit.sh
  - Details:
    - If --filesystems is omitted, the filesystems line is removed from the generated script.
    - --apps and --tags are passed to pbx run inside submit.sh to select which jobs to execute.

- pbx sbatch
  - Required: --config/-c, --job-name/-N, --partition/-p, --nodes, --walltime/-T, --account/-A
  - Optional: --run-dir, --apps/-a, --tags/-t, --retries
  - Behavior: creates run dir, generates submit.sh from SLURM template, runs sbatch submit.sh
  - Same filtering behavior as pbx qsub (--apps, --tags passed to pbx run)

- pbx ls
  - Filters: --status/-s, --app/-a, --tag/-t
  - Pagination:
    - --all shows all jobs.
    - -n N shows first N jobs; -n -N shows last N jobs; -n 0 shows all.
    - If > 25 jobs and no flags are used, ls shows first 10 and last 10 with "..." separator.
  - Display formatting:
    - ID column shows parent dependencies as "ID (p1,p2,...,pn)" (truncated beyond 4).
    - Scheduler job IDs are truncated for readability.
    - Paths are truncated to show leading and trailing segments.
  - Resource string:
    - Displays as n:{nodes}-r:{total_ranks}-g:{gpu_count}-nocc:{fraction|NA}
    - For multi-node GPU jobs: total_ranks = total_gpus (1 rank per GPU)
    - For multi-node CPU jobs: total_ranks = num_nodes * ranks_per_node
    - For single-node jobs: total_ranks = ngpus (GPU) or ranks_per_node (CPU)

- pbx info
  - Field selectors: --path/-p, --ngpus/-n, --app/-a, --status/-s, --tag/-t,
    --input/-i, --sched-job-id/-j, --timestamp/-ts, --envfile/-e, --parents/-P
  - When any field selector is used, the ID column is always included.
  - --parents/-P shows a dedicated Parents column with full parent lists; without it, the ID column includes a truncated parent view.
  - If some job IDs are not found, info reports them and continues for found jobs.

- pbx filter
  - Filters: --status/-s, --app/-a, --tag/-t, --path/-p, --in-file/-i
  - Outputs space‑separated job IDs for command composition
  - If no jobs match, outputs nothing (silent)

- pbx update
  - Fields: --status, --tag, --input/-i, --ngpus/-g, --envfile/-e,
    --nnodes/-n, --nocc/-o, --ranks-per-node/-rpn
  - Dependencies: --add_deps/--padd, --rm_deps/--parm
  - Validation:
    - --nnodes must be ≥ 1; --nocc in (0.0, 1.0]; --ranks-per-node ≥ 1.
    - --input file validation: input files are validated against the job's existing app; warnings cause update failures.
  - CPU vs GPU updates:
    - Setting --nocc for jobs with ngpus > 0 prompts confirmation to convert to CPU-only (ngpus=0).
  - Dependency updates:
    - Validates parent IDs and prevents circular dependencies (a job cannot be its own parent).
    - Shows warnings when removing non-existent parents and reports a per-job summary.
  - Examples:
    - pbx update 14 --ranks-per-node 8
    - pbx update 12 --add_deps "8 9" --rm_deps "7"
    - pbx update 11 --nocc 0.5
    - pbx update 15 --input new_file.dat

- pbx rm
  - Remove by explicit IDs, or pbx rm all (with confirmation)

- pbx run (internal)
  - Engine used by qsub; not intended for direct user invocation.
  - Options (for completeness): --config/-c, --apps/-a, --tags/-t, --retries, --run-dir
  - Behavior:
    - Discovers jobs with status Ready or Restart, applies --apps/--tags filters,
      enforces parent dependencies (parents must be Done), assigns resources,
      submits apps, and dynamically schedules backlog as resources free up.
    - Writes logs to the run directory (log.pbx).

## Configuration

ParslBox uses environment-configurable paths for the job database and the user config via `parslbox/helpers/path_utils.py`.

Environment variables:
- PBX_DB_PATH
  - Can be a directory or a full `.db` file path
  - If a directory is provided, ParslBox uses `<dir>/job_database_pbx.db`
- PBX_CONFIG_PATH
  - Can be a directory or a full `.yaml`/`.yml` file path
  - If a directory is provided, ParslBox uses `<dir>/config.yaml`

Defaults (when env vars are not set):
- Database: `~/.parslbox/job_database_pbx.db`
- Config file: `~/.parslbox/config.yaml`
- Runs: `~/.parslbox/runs/<timestamp>/`

Both CLI and API honor these paths. Set them per session, for example:
```bash
export PBX_DB_PATH=/scratch/mydbs/pbx.db          # full file path
export PBX_CONFIG_PATH=/scratch/mycfgs             # directory, becomes /scratch/mycfgs/config.yaml
```

Using PBX_DB_PATH and/or PBX_CONFIG_PATH to set the paths allow users to use multiple (and isolated) job databases and/or config files with PBX.

## Resource Manager (summary)

- Single‑node GPU jobs: assign specific GPU IDs (e.g., 0,1) so multiple GPU jobs can share a node when capacity allows.
- CPU‑only jobs: use --nocc to share a node fractionally (e.g., 0.25); multiple jobs can co‑reside up to occupancy 1.0.
- Multi‑node jobs: require exclusive free nodes; MPI hostlist is generated.
- Backlog and scheduling when resources are temporarily unavailable; dependency‑aware rescheduling after resources free up.
- Node health tracking: quarantine nodes after repeated failures, auto‑recover when healthy.

### MPI CPU Binding

ParslBox generates MPI launch commands with CPU binding flags appropriate for each job type (subnode, fullnode, multinode) and scheduler:

- **PBS systems (mpiexec/MPICH, mpirun/OpenMPI):** Configurable via `cpu_bind_method` in the `mpi:` config section. Options include `none`, `rankfile`, `list`, and `depth`. The `rankfile` and `list` methods provide GPU-affinity-aware core assignments using the resource manager's per-rank CPU maps.
- **SLURM systems (srun):** CPU binding is handled automatically. ParslBox generates `--cpus-per-task=D --cpu-bind=cores` for all job types, plus `--exact` for subnode jobs to prevent over-allocation. GPU jobs use the same wrapper scripts as PBS backends (the wrappers detect SLURM rank via `SLURM_PROCID`/`SLURM_LOCALID` environment variables).

Details and examples: parslbox/resource_manager/README.md

## Job Lifecycle

Statuses used across the system:
- Ready → Restart → Submitted → Running → Done | Failed
- Warning is used if an app returns an invalid/unknown status

## Supported Applications

Built-in apps: lammps, vasp, python

Custom apps can be added via config.yaml using the plugin system. See [`parslbox/apps/EXAMPLE_NEW_APP.py`](parslbox/apps/EXAMPLE_NEW_APP.py) for a template.

## Contributing

Issues and PRs welcome at:
- https://github.com/fbhuiyan2/parslbox/issues

## License

Author and developer: Fakhrul Hasan Bhuiyan

Copyright Argonne UChicago LLC, 2026. All rights reserved.
