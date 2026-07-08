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
  - CPU-GPU affinity-aware placement
  - MPI backends: MPICH, OpenMPI, srun
- **Scheduler support:** PBS (`pbx qsub`) and SLURM (`pbx sbatch`) with configurable `--sched-opts`
- **Pre-configured HPC systems:** Polaris, Aurora (GPU & Tile modes), Sophia, Crux, LCRC Swing, LCRC Improv, Pinnacles-CenvalArc, Perlmutter (GPU & CPU), plus `*-mpi`/`*-srun` launcher variants for large-scale (>10k worker) runs
- **Dynamic job discovery:** `--dynamic` (default) re-queries the DB for runnable jobs on every dispatch pass. New jobs matching the same `--apps`/`--tags` filters, and jobs the user flips back to `Ready`/`Restart` from another terminal (via `pbx update --status … <ids>`), are picked up automatically — jobs re-dispatched via `Restart` go through the app's `restart()` hook just like any other `Restart` job. Because each run atomically claims only what it dispatches, multiple `pbx run` allocations can safely share one DB in dynamic mode. Use `--static` for collect-once behavior (claim the runnable set once up front; no re-query)
- **Self-respawn chain:** `pbx qsub --respawn N` produces a self-perpetuating submission chain that auto-resubmits at every walltime boundary. Running jobs preempted at walltime are marked `Restart`; the next link calls each app's `restart()` hook lazily as each `Restart` job is dispatched, so apps can decide how to resume. See [`docs/pbx-run-details.md`](docs/pbx-run-details.md)
- **Fault tolerance:** Node health tracking, quarantine, and auto-recovery
- **Script-driven status reporting:** Python/Julia scripts report success/failure via `report_status()` utility
- **Python API and MCP server** for programmatic and AI-agent integration
- **Rich CLI output** with tables and colors

## Installation

Requirements:
- Python >= 3.11, < 3.14
- Parsl >= 2025.9.8

Clone the repo first:

```bash
git clone https://github.com/fbhuiyan2/parslbox.git
cd parslbox
```

Then pick a Python environment manager below. Poetry (recommended) installs into the active environment and uses the committed `poetry.lock` for reproducible dependency resolution. pip is offered as an alternative.

### conda

```bash
conda create -n parslbox python=3.11.9
conda activate parslbox
pip install poetry
poetry install                          # core dependencies
# poetry install --extras "simulation"  # + ase, pymatgen
# poetry install --extras "agentic"     # + uvicorn, mcp, pydantic
# poetry install --extras "simulation agentic"  # both
# poetry install --all-extras
```

<details><summary>Or with pip</summary>

```bash
pip install .                         # core
# pip install ".[simulation]"
# pip install ".[agentic]"
# pip install ".[simulation,agentic]"
# pip install ".[all]"
```
</details>

### Python venv

```bash
# First confirm a suitable Python is on PATH:
#   which python           # or `which python3.11`
#   python --version       # should be >= 3.11, < 3.14
python3.11 -m venv .venv
source .venv/bin/activate
pip install poetry
poetry install                          # (same extras options as above)
```

<details><summary>Or with pip</summary>

```bash
pip install .                         # (same extras options as above)
```
</details>

### uv

```bash
uv venv -p 3.11.9 .uvenv
source .uvenv/bin/activate
uv pip install poetry
poetry install                          # (same extras options as above)
```

<details><summary>Or with uv pip</summary>

```bash
uv pip install .                      # (same extras options as above)
```
</details>

### Setup

Run `pbx config` to interactively create a configuration file. Edit the generated config to set correct executable paths and environment setup before running jobs.

## Quick Start

Add jobs:
```bash
# Add a single LAMMPS job (requires system config name)
pbx add /path/to/sim --app lammps-kk --config polaris --ngpus 2 --tag run1

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
pbx qsub -c sophia -N myrun -q gpu --select 2 -T 90 -A myproject -a lammps-kk -t production
pbx sbatch -c polaris -N myrun -p gpu --nodes 2 -T 90 -A myproject -a lammps-kk -t production

# Long flags
pbx qsub --config sophia --job-name myrun --queue gpu --select 2 --walltime 90 --project myproject --apps lammps-kk --tags production
pbx sbatch --config polaris --job-name myrun --partition gpu --nodes 2 --walltime 90 --account myproject --apps lammps-kk --tags production

# Glob tags: use `*` to match a substring. Quote to prevent shell expansion.
pbx qsub -c sophia -N myrun -q gpu --select 2 -T 90 -A myproject -a lammps-kk -t '*nomix,prod-run'

# Self-respawn chain: auto-resubmit at every walltime boundary, up to 3 times
pbx qsub -c sophia -N sweep -q gpu --select 4 -T 4h -A myproject -a lammps-kk -t sweep \
  --respawn 3
```

## Supported Applications

Built-in: **lammps-kk**, **vasp**, **orca**, **python**, **julia**. Custom apps can be registered via `config.yaml`. Full details, per-app notes, status reporting protocol, and custom-app templates: [`docs/apps.md`](docs/apps.md).

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
| **perlmutter-gpu** | SLURM | 4 | NVIDIA A100 | 128 | srun |
| **perlmutter-cpu** | SLURM | 0 (CPU-only) | — | 128 | srun |
| **aurora-tile-mpi** | PBS | 12 (6x2 tiles) | Intel Max 1550 | 208 | MPICH |
| **perlmutter-gpu-srun** | SLURM | 4 | NVIDIA A100 | 128 | srun |

Each system defines its own MPI defaults, scheduler templates, and resource detection methods. New systems can be added by creating a config class inheriting from `BaseSystemConfig`.

**Launcher variants** (`*-mpi` / `*-srun`): hardware-identical to their base configs (`aurora-tile`, `perlmutter-gpu`) but use `MpiExecLauncher` / `SrunLauncher` instead of `SimpleLauncher`. This places one Parsl manager per compute node (workers distributed across nodes) rather than concentrating all workers on the head node. Use for runs above ~10k workers, where head-node RAM would otherwise be the scaling ceiling. The base configs remain the default and are recommended for smaller runs.

## Programmatic API (Python)

ParslBox provides a Python API for scripts and AI agents. The API delegates to core logic inside the command modules, ensuring CLI and API behavior stay in sync.

Main methods:
- `add_jobs(paths, app, config, ...)`
- `update_jobs(job_ids, ...)`
- `list_jobs(...)` — returns all jobs when called with no filters
- `filter_jobs(...)` — returns job IDs matching filters (status, app, tag, path, input)
- `get_job(job_id)`, `get_jobs_by_ids(ids)`
- `remove_job`, `remove_jobs`, `remove_all_jobs`
- `qsub(...)` — submit via PBS
- `sbatch(...)` — submit via SLURM
- `run(...)` — execute jobs directly
- `qdel(jobid, grace=30)` / `scancel(jobid, grace=30)` — graceful cancel of a running PBX batch job (SIGTERM → wait `grace` seconds → hard kill)

Exceptions: `ParslBoxError`, `ValidationError`, `JobNotFoundError`

Full per-method reference with examples: [`docs/api-details.md`](docs/api-details.md).

## MCP Server

ParslBox includes an MCP server for AI-agent integration. Install with `pip install ".[agentic]"` and start:

```bash
# HTTP mode (standalone server on port 9795)
python -m parslbox.mcp.mcp_server

# stdio mode (for Claude Code integration)
python -m parslbox.mcp.mcp_server --stdio
```

Exposed tools: `add_jobs`, `submit_pbs_job`, `submit_slurm_job`, `cancel_pbs_job`, `cancel_slurm_job`, `remove_job`, `update_job`, `filter_jobs`, `list_jobs`, `get_job`, `get_jobs`.

### Claude Code Integration

The project ships a [`.mcp.json`](.mcp.json) for automatic discovery — Claude Code launched from the repo directory will offer to connect. For global access, copy that file to `~/.claude/.mcp.json` and edit the two `/path/to/...` placeholders.

See [`examples/chemgraph_parslbox_example/`](examples/chemgraph_parslbox_example/) for an HTTP client example.

## Commands Overview

Full per-command reference with examples lives in [`docs/commands.md`](docs/commands.md).

## Configuration

Environment variables:
- `PBX_DB_PATH` — database file or directory path
- `PBX_CONFIG_PATH` — config file or directory path
- `PBX_RUN_DELAY` — seconds to sleep between consecutive job submissions (default: `0.2`). Bump up on systems like Perlmutter where rapid `srun` invocations can overload `slurmctld`.

Defaults (when env vars are not set):
- Database: `~/.parslbox/job_database_pbx.db`
- Config: `~/.parslbox/config.yaml`
- Runs: `~/.parslbox/runs/<timestamp>/`

```bash
export PBX_DB_PATH=/scratch/mydbs/pbx.db
export PBX_CONFIG_PATH=/scratch/mycfgs/config.yaml
export PBX_RUN_DELAY=0.5
```

Using separate `PBX_DB_PATH` and/or `PBX_CONFIG_PATH` allows multiple isolated databases and configurations. All three env vars are automatically propagated into the qsub/sbatch submission script.

## Resource Manager

- **Single-node GPU jobs:** Assign specific GPU IDs (e.g., 0,1) so multiple jobs can share a node
- **CPU-only jobs:** Use `--nocc` for fractional node occupancy (e.g., 0.25); multiple jobs co-reside up to 1.0
- **Multi-node jobs:** Exclusive free nodes with MPI hostlist generation
- **Non-MPI apps:** Resource launcher constrains execution to assigned node/resources
- **Dependency-aware scheduling:** Jobs dispatch once their parents finish and resources free up (parents satisfy on `Done`/`Warning`)
- **Node health tracking:** Quarantine a node only after multiple *distinct* jobs fail on it (a single job's crash — even a multi-node one — never quarantines a node), auto-recover when healthy

### MPI CPU Binding

ParslBox generates MPI launch commands with CPU binding flags appropriate for each job type (subnode, fullnode, multinode) and scheduler:

- **PBS systems (mpiexec/MPICH, mpirun/OpenMPI):** Configurable via `cpu_bind_method` in the `mpi:` config section. Options: `none`, `rankfile`, `list`, `depth`. The `rankfile` and `list` methods provide GPU-affinity-aware core assignments.
- **SLURM systems (srun):** Native SLURM resource binding. Full/multi-node GPU: `--gpus-per-node` + `--gpu-bind=map_gpu|mask_gpu` with `--cpu-bind=cores|threads`. Sub-node GPU: `--gpu-bind=map_gpu:{pbx_gpu_ids}` + `--cpu-bind=mask_cpu:{hex}`. All srun jobs include `--overlap` (required when running under `SrunLauncher`-based configs like `perlmutter-gpu-srun`; also enables sub-node packing) and `-N {nodes}` for explicit node control.

Details: [`parslbox/resource_manager/README.md`](parslbox/resource_manager/README.md)

## Job Lifecycle

Statuses:
- Ready → Submitted → Running → Done | Failed | Killed
- Restart → Resubmitted → Running → … — the restart path. `Submitted`/`Resubmitted` are the *claimed* states: a `pbx run` atomically flips a job to `Submitted` (from `Ready`) or `Resubmitted` (from `Restart`), stamping its batch id (`sched_job_id`) so multiple concurrent runs sharing one DB never double-claim.
- Warning — if an app returns an invalid/unknown status (satisfies dependencies like `Done`)
- Killed — when walltime is exceeded **in a no-respawn run**, or whenever the user runs `pbx qdel`/`pbx scancel`. At shutdown the orchestrator reconciles its own jobs per state: `Running → Killed` (or `Restart`/`Failed` in a `--respawn` run), and claimed-but-not-yet-running jobs go back to the pool (`Submitted → Ready`, `Resubmitted → Restart`). If the signal handler can't finish its DB writes in time, a post-kill reconciliation step in `pbx qdel`/`scancel` applies the same per-state rules, scoped to the killed batch's `sched_job_id`.
- Restart — set by either the user (`pbx update --status Restart`) or by the orchestrator in a `--respawn` chain at walltime. Either way, the next `pbx run` calls the app's `restart()` hook lazily per-job as each `Restart` row is dispatched (patch fields and re-run / re-run as-is / mark Failed), right after resources are assigned.

Full state-transition table and end-to-end chain walkthrough: [`docs/pbx-run-details.md`](docs/pbx-run-details.md).

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
