# ParslBox

Your autopilot for running HPC simulations. CLI orchestration built based on Parsl. Manage jobs for LAMMPS, VASP, and Python apps with resource‑aware scheduling, dependency tracking, and PBS submission. HPC configurations come out-of-the-box. Adding new apps and new HPC configurations is super simple.


## Highlights

- Job lifecycle management with dependency support (parents, tags)
- Multi-application plugins: lammps, vasp, python
- Resource‑aware execution:
  - Single‑node GPU jobs with explicit GPU assignment
  - CPU‑only jobs via fractional node occupancy
  - Multi‑node MPI jobs with exclusive node allocation
- PBS integration via pbx qsub
- Auto‑generated config template and run directories
- Rich CLI output (tables, colors)

## Installation

Requirements:
- Python >= 3.11, < 3.14
- Parsl >= 2025.9.8

Using Conda + Poetry:
```bash
conda create --name parslbox python=3.11.9
conda activate parslbox
pip install poetry

# Clone and install
git clone https://github.com/fbhuiyan2/parslbox.git
cd parslbox
poetry install

# First call initializes ~/.parslbox/config.yaml and the job database
pbx ls
```

On first run, a default config is created at ~/.parslbox/config.yaml. Edit this file to set correct executable paths, environment setup, and system settings before running jobs.

## Quick Start

Add jobs:
```bash
# Add a single LAMMPS job (requires system config name)
pbx add /path/to/sim --app lammps --config polaris --ngpus 2 --tag run1

# Add all subdirectories in current folder as VASP jobs
pbx add all --app vasp --config polaris --tag ManyVaspCalc

# Add with explicit resources and dependencies
pbx add /path/to/calc --app vasp --config polaris --nnodes 1 --nodealloc 0.5 \
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

# Show fields per job
pbx info 1 2 --path --ngpus --envfile --parents

# Get IDs via filters (use in command composition)
pbx filter --status done --app vasp
pbx filter -s failed -a lammps -t test -p /path/part -i input.lammps

# Update fields and dependencies
pbx update 4 5 --status Restart --tag high-priority
pbx update 10 --nnodes 2          # multi-node
pbx update 11 --nodealloc 0.25    # CPU-only fractional occupancy
pbx update 12 --add_deps "8 9" --rm_deps "7"
pbx update 13 --envfile ./env.sh

# Remove jobs
pbx rm 1 2 3
pbx rm all
pbx rm $(pbx filter --status done)
```

## Commands Overview

Note on usage:
- Users should submit via pbx qsub. The qsub command generates a submit.sh and submits it to the scheduler; submit.sh invokes pbx run under the hood.
- pbx run is the engine used by qsub and is not intended to be called directly by users.

- pbx add
  - Arguments: paths (one or more directories, or 'all')
  - Required: --app/-a, --config/-c
  - Common options: --tag/-t, --input/-i, --ngpus/-g, --nnodes/-n, --nodealloc/-na, --mpiopts, --envfile/-e
  - Dependencies: --parents/-P "1 2 3", --parent-tag
  - Initial status: --status/-s (default Ready)

- pbx qsub
  - Required: --config/-c, --job-name/-N, --queue/-q, --select, --walltime/-T, --project/-A
  - Optional: --filesystems, --run-dir, --apps/-a, --tags/-t, --retries
  - Behavior: creates run dir, generates submit.sh from config template, runs qsub submit.sh

- pbx ls
  - Filters: --status/-s, --app/-a, --tag/-t
  - Displays resources as n:{nodes}-g:{gpus|auto}-nocc:{fraction|NA}

- pbx info
  - Field selectors: --path/-p, --ngpus/-n, --app/-a, --status/-s, --tag/-t,
    --input/-i, --sched-job-id/-j, --timestamp/-ts, --envfile/-e, --parents/-P

- pbx filter
  - Filters: --status/-s, --app/-a, --tag/-t, --path/-p, --in-file/-i
  - Outputs space‑separated job IDs for command composition

- pbx update
  - Fields: --status, --app, --tag, --input/-i, --ngpus/-g, --envfile/-e,
    --nnodes/-n, --nodealloc/-na
  - Dependencies: --add_deps/--padd, --rm_deps/--parm

- pbx rm
  - Remove by explicit IDs, or pbx rm all (with confirmation)

- pbx run (internal)
  - Engine used by qsub; not intended for direct user invocation.
  - Options (for completeness): --config/-c, --apps/-a, --tags/-t, --retries, --run-dir

## Configuration

- A template is created at ~/.parslbox/config.yaml on first run.
- Edit system entries (e.g., polaris, sophia) and per‑app settings (environment setup, executable paths, MPI options).
- See parslbox/configs/*.py for programmatic configs and examples used by the engine.

Data locations:
- Database: ~/.parslbox/job_database.db
- Runs: ~/.parslbox/runs/<timestamp>/
- Config file: ~/.parslbox/config.yaml

## Resource Manager (summary)

- Single‑node GPU jobs: assign specific GPU IDs (e.g., 0,1) so multiple GPU jobs can share a node when capacity allows.
- CPU‑only jobs: use --nodealloc to share a node fractionally (e.g., 0.25); multiple jobs can co‑reside up to occupancy 1.0.
- Multi‑node jobs: require exclusive free nodes; MPI hostlist is generated.
- Backlog and scheduling when resources are temporarily unavailable; dependency‑aware rescheduling after resources free up.

Details and examples: parslbox/resource_manager/README.md

## Job Lifecycle

Statuses used across the system:
- Ready → Restart → Submitted → Running → Done | Failed
- Warning is used if an app returns an invalid/unknown status

## Supported Applications

- lammps
- vasp
- python

## Contributing

Issues and PRs welcome at:
- https://github.com/fbhuiyan2/parslbox/issues

## License

Author and developer: Fakhrul Hasan Bhuiyan

Copyright Argonne UChicago LLC, 2025. All rights reserved.
