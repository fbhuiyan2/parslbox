---
name: parslbox
description: Help with ParslBox (pbx) CLI and Python API usage for HPC job orchestration (add/list/filter/update/remove jobs, qsub submission, config/db paths, resource semantics). Use when users ask how to run ParslBox, troubleshoot pbx commands/config, or script workflows via parslbox.api.
---

# ParslBox (pbx) CLI + API

## Ground rules
- Prefer `pbx qsub` for scheduler submission; treat `pbx run` as internal.
- When unsure, ask for: system config name, app (lammps/vasp/python), intended resources (ngpus/nnodes/nocc), and dependency pattern (parents/parent-tag).

## Quick start (CLI)
- Initialize config + DB (first run):
  - `pbx ls`
- Add jobs:
  - Single directory: `pbx add /path/to/sim --app lammps --config polaris --ngpus 2 --tag run1`
  - All subdirs: `pbx add all --app vasp --config polaris --tag ManyVaspCalc`
  - With deps: `pbx add /p/calc --app vasp --config polaris --nnodes 1 --nocc 0.5 --parents "10 11" --tag stage2`
  - Wait on a tag: `pbx add /p/calc2 --app vasp --config polaris --parent-tag stage1`
- Inspect/filter:
  - `pbx ls --status Running --app lammps --tag production`
  - `pbx info 1 2 --path --ngpus --envfile --parents`
  - `pbx filter --status done --app vasp`
- Update:
  - `pbx update 4 5 --status Restart --tag high-priority`
  - `pbx update 10 --nnodes 2`
  - `pbx update 11 --nocc 0.25`
  - `pbx update 12 --add_deps "8 9" --rm_deps "7"`
- Remove:
  - `pbx rm 1 2 3`
  - `pbx rm all`

## Quick start (PBS submission)
- Submit via qsub (generates run dir + submit.sh and calls qsub):
  - `pbx qsub --config sophia --job-name myrun --queue gpu --select 2 --walltime 90 --project myproject --apps lammps --tags production`

## Programmatic API
- Prefer using `parslbox.api.ParslBox` to keep parity with CLI behavior.

```python
from parslbox.api import ParslBox
pbx = ParslBox()
job_id = pbx.add_job("/path/to/sim", app="lammps", config="polaris", ngpus=2)
jobs = pbx.list_jobs()
pbx.update_job(job_id, status="Submitted")
```

- Main methods to look for:
  - add_job(s), update_job(s), list_jobs, get_job, get_jobs_by_ids, remove_job(s), remove_all_jobs, qsub
- Common exceptions:
  - ParslBoxError, ValidationError, JobNotFoundError

## Configuration and paths
- Env vars:
  - `PBX_DB_PATH`: dir or full `.db` path (dir => `<dir>/job_database_pbx.db`)
  - `PBX_CONFIG_PATH`: dir or full `.yml/.yaml` path (dir => `<dir>/config.yaml`)
- Defaults:
  - `~/.parslbox/job_database_pbx.db`
  - `~/.parslbox/config.yaml`
  - `~/.parslbox/runs/<timestamp>/`

## Resource semantics (important)
- Single-node GPU jobs: explicit GPU IDs; multiple jobs can share a node if capacity allows.
- CPU-only jobs: `--nocc` fractional occupancy (0.0, 1.0].
- Multi-node jobs (`--nnodes > 1`): exclusive nodes; GPU count/ranks derived; `--ngpus/--nocc` ignored.
- Ranks-per-node:
  - GPU jobs: 1 rank per GPU (ignore `--ranks-per-node`)
  - CPU jobs: if not set, computed from cores_per_node * node_occupancy (min 1)

## Troubleshooting checklist
- “Nothing runs”: confirm jobs are `Ready`/`Restart`, parents are `Done`, and filters (`--apps/--tags`) match.
- “Wrong resources”: check `--nnodes` vs `--ngpus/--nocc` rules; verify `--ranks-per-node` expectations.
- “Config/db confusion”: print `PBX_CONFIG_PATH`/`PBX_DB_PATH`, confirm the created defaults under `~/.parslbox/`.
