# ParslBox Commands

Reference for every `pbx` subcommand. Where a flag has both forms, the short alias appears in parentheses.

Submission flow: `pbx qsub` / `pbx sbatch` generate a `submit.sh` and submit it to the scheduler. The script invokes `pbx run` under the hood — `pbx run` itself is internal, not for direct use.

## Quick reference

| Command | What it does |
|---|---|
| [`pbx config`](#pbx-config) | Interactive wizard to create or reconfigure a ParslBox configuration file. |
| [`pbx add`](#pbx-add) | Add one or more job directories to the database with app, resources, tags, and dependencies. |
| [`pbx ls`](#pbx-ls) | List jobs in a Rich table with optional filters and pagination. |
| [`pbx info`](#pbx-info) | Show per-field details for specific jobs; also previews MPI commands and resource requirements. |
| [`pbx filter`](#pbx-filter) | Output space-separated job IDs matching include/exclude filters, for shell composition. |
| [`pbx update`](#pbx-update) | Modify fields and dependencies of existing jobs. |
| [`pbx rm`](#pbx-rm) | Delete jobs from the database by ID range or `all`. |
| [`pbx qsub` / `pbx sbatch`](#pbx-qsub--pbx-sbatch) | Generate a `submit.sh` and submit it to PBS / SLURM. |
| [`pbx qdel` / `pbx scancel`](#pbx-qdel-jobid--pbx-scancel-jobid) | Gracefully cancel a running ParslBox batch job (SIGTERM → grace → hard kill). |

| Internal | What it does |
|---|---|
| [`pbx run`](#pbx-run-internal) | Engine invoked by qsub/sbatch's generated submit script. Not for direct use. |

---

## `pbx config`

Interactive wizard to create or reconfigure a ParslBox configuration.

Prompts for config path, system selection (Polaris, Aurora, Sophia, etc.), app selection, and whether to initialize a database.

```bash
pbx config                  # writes to ~/.parslbox/config.yaml
pbx config .                # writes to current directory
pbx config ~                # writes to home directory
pbx config /custom/path     # writes to custom path
```

---

## `pbx add`

Add jobs to the database. New IDs print as a compressed range (`✅ Added 5000 job(s), IDs: 1-5000`); failures are grouped by error message with the affected paths listed under each.

**Required**: `--app/-a`, `--config/-c`
**Resource**: `--ngpus/-g`, `--nnodes/-n`, `--nocc/-o`, `--ranks-per-node/-rpn`, `--mpiopts`
**Metadata**: `--tag/-t`, `--input/-i`, `--args`, `--envfile/-e`, `--status/-s`
**Dependencies**: `--parents/-P "1 2 3"` (supports ranges, e.g. `'1-5 8 14-20'`), `--parent-tag <tag>`

Notes:
- For `--nnodes > 1`, `--ngpus` and `--nocc` are ignored.
- Single-node GPU jobs auto-assign all GPUs on GPU systems when `-g` is not specified.
- `--args` appends to the input file invocation (e.g., `python script.py --flag value`).

```bash
# Single LAMMPS job (long + short flags)
pbx add /path/to/sim --app lammps-kk --config polaris --ngpus 2 --tag run1
pbx add /path/to/sim -a lammps-kk -c polaris -g 2 -t run1

# All subdirectories as VASP jobs — cwd, or an explicit base dir
pbx add all -a vasp -c polaris -t ManyVaspCalc
pbx add all:/abs/path/to/runs -a vasp -c polaris -t ManyVaspCalc

# Python job with args and env file
pbx add /path/to/analysis -a python -c polaris -i plot.py \
  --args "--mode gpu --output-prefix test" -e env_setup.sh

# CPU-only fractional occupancy + explicit parent IDs (range syntax supported)
pbx add /path/to/calc -a vasp -c polaris -n 1 -o 0.5 -P "1-5 10 14-20" -t stage2

# Wait for all jobs with a tag to finish
pbx add /path/to/calc2 -a vasp -c polaris --parent-tag stage1
```

---

## `pbx ls`

List jobs as a Rich table.

Takes an optional positional `COUNT`: `N` for the first N rows, `-N` for the last N, or
`all` for everything. Omit it and the table auto-paginates (first 10 + last 10) when more
than 25 jobs match.

| Argument | Purpose |
|---|---|
| `COUNT` | `N` first N, `-N` last N, `all` everything; omit for the paginated view |

| Flag | Short | Purpose |
|---|---|---|
| `--status` | `-s` | filter by status |
| `--app` | `-a` | filter by app |
| `--tag` | `-t` | filter by tag (supports `*` glob) |
| `--nnodes` | `-n` | filter by number of nodes (exact match) |

```bash
pbx ls
pbx ls all
pbx ls 15           # first 15
pbx ls -20          # last 20
pbx ls --status Running --app lammps-kk --tag production
pbx ls -s Running -a lammps-kk -t production
pbx ls -n 2         # only 2-node jobs
pbx ls -20 -n 2     # last 20 of the 2-node jobs
pbx ls -t '*test'   # tag glob
```

---

## `pbx info`

Show per-field details. Accepts ID ranges like `1-5 8 14-20`.

**Field selectors** (combine freely; `ID` is auto-included with any selector):

| Flag | Short | Field |
|---|---|---|
| `--path` | — | path, always full |
| — | `-p` | path, auto-truncated when more than 3 fields shown |
| `--ngpus` | `-g` | GPU count |
| `--nodes` | `-n` | node count |
| `--ranks` | — | ranks per node |
| `--nocc` | `-o` | node occupancy |
| `--resrc` | `-r` | combined resource string (`n:5-r:60-g:60-nocc:NA`) |
| `--app` | `-a` | app |
| `--status` | `-s` | status |
| `--tag` | `-t` | tag |
| `--input` | `-i` | input file |
| `--sched-job-id` | `-j` | scheduler job ID |
| `--timestamp` | `-ts` | created at |
| `--envfile` | `-e` | env file |
| `--parents` | `-P` | parent IDs (no truncation) |

**Analysis tools** (long-form only — `-r` and `-c` no longer alias these):
- `--req SYSTEM` — resource requirements for a target system (simultaneous vs optimal packing)
- `--cmdline SYSTEM` — preview the MPI/srun command line that would run

```bash
pbx info 1                                      # default 10-field view
pbx info 1-5 8                                  # ID ranges
pbx info 1-5 8 -p -g -e -P                     # path auto-truncates (5 fields > 3)
pbx info 1-5 8 --path -g -e -P                 # --path forces full path
pbx info 1 -r                                   # show only the resource string
pbx info 1-10 --req polaris                     # how many nodes would these need on polaris?
pbx info 5 --cmdline aurora-tile                # preview the MPI command
```

---

## `pbx filter`

Output space-separated job IDs for shell composition. Silent on no match.

**Include**:

| Flag | Short | Match |
|---|---|---|
| `--status` | `-s` | exact (case-insensitive on first letter) |
| `--app` | `-a` | exact |
| `--tag` | `-t` | exact, or `*` glob (e.g., `'*prod'`) |
| `--path` | `-p` | substring |
| `--in-file` | `-i` | substring |
| `--nnodes` | `-n` | exact number of nodes |

**Exclude**:

| Flag | Short alias | Drops |
|---|---|---|
| `--exclude-status` | `--xstatus` | jobs with this status |
| `--exclude-app` | `--xapp` | jobs with this app |
| `--exclude-tag` | `--xtag` | jobs with this tag (supports `*` glob) |

Untagged jobs are never excluded by a `*` glob on `--xtag`.

```bash
pbx filter --status done --app vasp
pbx filter -s failed -a lammps -t test -p /path/part -i input.lammps

# Exclude composition
pbx filter -s Ready --xtag '*test'
pbx filter -a lammps-kk --xapp vasp --xstatus Failed

# Compose with other commands
pbx rm   $(pbx filter --status done)
pbx info $(pbx filter -s Ready --xtag '*test')
```

---

## `pbx update`

Update one or more job fields. Accepts ID ranges as separate shell words (`pbx update 1-5 8 14-20`) — do not quote the whole list.

Successful IDs print as compressed ranges (`1-5 8 14-20`). Failures are grouped by error message, one line per distinct error.

**Fields**: `--status/-s`, `--tag/-t`, `--input/-i`, `--args`, `--envfile/-e`, `--ngpus/-g`, `--nnodes/-n`, `--nocc/-o`, `--ranks-per-node/-rpn`
**Dependencies**: `--add_deps/--padd "8 9"`, `--rm_deps/--parm "7"`

Validates dependencies and prevents circular references.

```bash
pbx update 1-5 --status Restart --tag high-priority
pbx update 10 --nnodes 2          # promote to multi-node
pbx update 11 --nocc 0.25         # fractional CPU occupancy
pbx update 12 --add_deps "8 9" --rm_deps "7"
pbx update 13 --envfile ./env.sh
pbx update 14 --input new_input.dat
pbx update 15 --args "--new-flag value"
```

---

## `pbx rm`

Remove jobs from the database. Supports ID ranges and `all` (with confirmation).

```bash
pbx rm 1-5 8 10-15
pbx rm all
pbx rm $(pbx filter --status done)
```

---

## `pbx qsub` / `pbx sbatch`

Generate a `submit.sh` and submit it to PBS / SLURM. Both commands share most flags.

**Required**: `--config/-c`, `--job-name/-N`, `--queue/-q`, `--select`, `--walltime/-T`, `--project/-A`

Both commands share identical flag names — there is no `--partition`, `--nodes`, or `--account`. The meaning of two flags differs by scheduler:
- `--queue/-q` — PBS queue name (qsub) / SLURM partition name (sbatch).
- `--select` — PBS select spec (qsub), e.g. `4` or `2:ncpus=32:ngpus=4` / number of nodes (sbatch), e.g. `2`.

**Optional**: `--run-dir`, `--apps/-a`, `--tags/-t`, `--retries`, `--sched-opts`, `--dynamic`/`--static`, `--respawn N`, `--loglevel`

Notes:
- `--walltime` defaults to **minutes**; supports `h` and `d` suffixes (`90`, `4.25h`, `3.5d`).
- `--retries N` is passed through to Parsl: each individual ParslBox job that fails (non-zero exit, app exception) is retried up to `N` times before being marked `Failed`. Default `0` (no retry).
- `--sched-opts` adds extra scheduler directives. **Each value is a complete directive line including the `#PBS` / `#SBATCH` prefix** (e.g., `'#PBS -l filesystems=home:eagle'`, `'#SBATCH --qos=regular'`). Repeatable; directives matching a template key override it.
- `--dynamic` (default) re-queries the DB for runnable (`Ready`/`Restart`) jobs matching the same `--apps`/`--tags` filters on every dispatch pass, so jobs added or flipped mid-run are picked up; it also lets multiple runs share one DB. `--static` claims the runnable set once up front and does not re-query.
- `--respawn N` enables the self-respawn chain. A `respawn_template.sh` is generated alongside `submit.sh`. At each walltime expiry, `Running` jobs are marked `Restart` (instead of `Killed`) and the next link is auto-submitted with `--respawn (N-1)`. When `--respawn 0`, `Running` jobs go to `Failed` and the chain ends. Full lifecycle + per-link behavior: [`pbx-run-details.md`](pbx-run-details.md).
- **Tag globs**: each `--tags` token may be a literal or a `*` glob (`*prod`, `run*`, `*3c*`). Globs are resolved against the DB before submission. **If any token matches no existing tag, submission aborts.** Quote globs to stop the shell from expanding `*`.
- **Job-count guard**: before submitting, qsub/sbatch query the DB for matching Ready/Restart jobs. **If zero match, submission aborts** to avoid wasting the allocation.
- After generation, the submit script is printed in a cyan box with the `pbx run` line highlighted in red. When `--respawn` is set, `respawn_template.sh` is also printed (yellow note explains the placeholder behavior).

```bash
# Short flags
pbx qsub   -c sophia        -N myrun -q gpu     --select 2 -T 90 -A myproject -a lammps-kk -t production
pbx sbatch -c perlmutter-gpu -N myrun -q regular --select 2 -T 90 -A myproject -a lammps-kk -t production

# Long flags
pbx qsub --config sophia --job-name myrun --queue gpu --select 2 \
  --walltime 90 --project myproject --apps lammps-kk --tags production
pbx sbatch --config perlmutter-gpu --job-name myrun --queue regular --select 2 \
  --walltime 90 --project myproject --apps lammps-kk --tags production

# Tag glob (quote to dodge shell expansion)
pbx qsub -c sophia -N myrun -q gpu --select 2 -T 90 -A myproject -a lammps-kk -t '*nomix,prod-run'

# Extra scheduler directives (full directive lines, repeatable)
pbx qsub -c polaris -N myrun -q prod --select 4 -T 4h -A myproject \
  --sched-opts '#PBS -l filesystems=home:eagle' \
  --sched-opts '#PBS -l place=scatter'
pbx sbatch -c perlmutter-gpu -N myrun -q regular --select 2 -T 2h -A m1234 \
  --sched-opts '#SBATCH --qos=regular' \
  --sched-opts '#SBATCH --constraint=gpu'

# Walltime suffixes
pbx qsub ... -T 90     # 90 minutes
pbx qsub ... -T 4.25h  # 4h 15m
pbx qsub ... -T 3.5d   # 3d 12h

# Self-respawn chain (auto-resubmit at every walltime, up to 3 more times)
pbx qsub -c sophia -N sweep -q gpu --select 4 -T 4h -A myproject -a lammps-kk -t sweep \
  --respawn 3
```

---

## `pbx qdel <jobid>` / `pbx scancel <jobid>`

Gracefully cancel a running ParslBox batch job. **Always prefer these over raw `qdel`/`scancel`.**

- `--grace/-g N` — seconds between SIGTERM and the hard kill (default `30`).
- Sends SIGTERM first so the orchestrator can reconcile its in-flight jobs in the database, then runs the scheduler's kill command.
- Raw `qdel`/`scancel` give only the site's default kill grace (often ~2s), which can leave jobs stuck in `Running`/claimed states.
- **A `Running` job is always marked `Killed`, even under `--respawn`** (claimed-but-not-yet-running jobs revert to the pool: `Submitted` → `Ready`, `Resubmitted` → `Restart`). If you want to pause-and-resume a respawn chain rather than terminate it, flip the killed jobs back to `Restart` manually after the cancel (`pbx update --status Restart <ids>`) and run `pbx qsub --respawn N` again.
- **DB reconciliation.** After the scheduler kill (whether it succeeded or not), pbx queries the DB for any non-terminal jobs under this batch (`sched_job_id` match) and reconciles them per state: `Running` → `Killed`, `Submitted` → `Ready`, `Resubmitted` → `Restart`. This catches cases where the orchestrator's signal handler couldn't complete its cleanup before the process exited (DB contention, alarm timeout, etc.). Scoped by `sched_job_id` so concurrent batch jobs are unaffected. If the hard-kill step returned an error but reconciliation cleaned up stuck jobs, the command exits `0` with a warning — the batch is dead and the DB is consistent, which was the user intent.

```bash
pbx qdel    1234567        # 30s default grace
pbx scancel 7654321 -g 60  # custom grace
```

---

## `pbx run` (internal)

Engine used by qsub/sbatch — not for direct use. Full runtime reference: [`pbx-run-details.md`](pbx-run-details.md).

- `--dynamic` (default) re-queries the DB for runnable jobs each dispatch pass (and lets multiple runs share one DB); `--static` claims the runnable set once up front. Neither idles — a run exits when nothing runnable remains.
- Triggers a graceful shutdown automatically before walltime (30s grace, 90s under `--respawn`) so in-flight jobs are reconciled cleanly: `Running` → `Killed` by default (or `Restart`/`Failed` under `--respawn`), and claimed-but-not-yet-running jobs revert to `Ready`/`Restart`.
- `--walltime-seconds` (required) is injected by the generated batch script; it is what the shutdown timing above is measured against.
- `--respawn N` is set internally by `pbx qsub --respawn N` / `pbx sbatch --respawn N`. It turns on the walltime-time auto-resubmission step. Do not invoke `pbx run` with it directly — use `pbx qsub --respawn N`. The `restart()` hook runs lazily per-job as each `Restart`-status job is dispatched, at every `pbx run` invocation, regardless of `--respawn`.
