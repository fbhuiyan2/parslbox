---
name: parslbox-cli
description: Help with ParslBox `pbx` shell commands — adding/listing/filtering/updating/removing jobs, scheduler submission (qsub/sbatch), job cancellation (qdel/scancel), `pbx config` setup, and `pbx local` projects that run on a remote machine over Globus. Use when the user runs `pbx` in a terminal, asks how to submit/manage jobs from the CLI, or troubleshoots command/flag behavior. Do NOT use for Python `from parslbox.api import ParslBox` questions — those go to parslbox-api.
---

# ParslBox CLI (`pbx`)

Authoritative reference: [docs/commands.md](docs/commands.md). Read it for any non-trivial flag, edge case, or output format question. This skill is a fast index.

## Commands at a glance

| Command | Purpose |
|---|---|
| `pbx config` | Interactive config + DB setup; first-run or per-project (`pbx config .` for cwd, `pbx config /path` for any dir) |
| `pbx add` | Register a job (or a directory tree of jobs) in the DB |
| `pbx ls` | List jobs (one-line per job, filterable); optional positional count — `pbx ls 10`, `pbx ls -20`, `pbx ls all` |
| `pbx info` | Detailed view of specific job IDs |
| `pbx filter` | Filter jobs by status/app/tag (with `--exclude-status` / `--exclude-app` / `--exclude-tag` for exclusion) |
| `pbx update` | Edit job fields (status, resources, deps, tags, in_file, args) |
| `pbx rm` | Delete jobs from the DB |
| `pbx qsub` | Submit a PBS batch that runs `pbx run` inside the allocation |
| `pbx sbatch` | Same as `qsub` but for SLURM |
| `pbx qdel` | Cancel a `pbx qsub`-submitted PBS job |
| `pbx scancel` | Cancel a `pbx sbatch`-submitted SLURM job |
| `pbx local` | Local-to-remote projects: `init`, `status`, `push`, `pull` (see below) |
| `pbx run` | Internal — the orchestrator invoked by qsub/sbatch (do not run by hand unless on a login node test) |

Use `--help` on any command to see flags.

## Conventions worth knowing

- **Job IDs accept ranges**, passed as separate shell words — do **not** quote the whole list: `pbx update 1-5 8 14-20 --status Restart`. Same for `rm` and `info`. Quoting a single token is harmless (`pbx rm "1-5"`), but quoting a multi-token list (`"1-5 8"`) fails to parse. `add --parents` is the opposite — it is one option value, so it **must** be quoted: `--parents "1-5 8 14-20"`.
- **Bulk add**: `pbx add all` registers every subdirectory of the cwd as a job; `pbx add all:<dir>` uses subdirectories of `<dir>`. All options in the call (app, config, resources, tag, env file) apply to every job.
- **Tags**: `--tag` (singular) for filtering on `add`/`update`/`filter`/`ls`; `--tags` (plural, comma-separated) on `qsub`/`sbatch`/`run`. `pbx info --tag` is a *display* flag (show only the tag column), not a filter. `*` glob for partial matching (e.g., `--tag 'stage*'` matches `stage1`, `stage-prod`). Exclusion via `--exclude-tag` / `--xtag` is `pbx filter`-only.
- **Bulk output is summarized.** `add`/`update` print IDs as compressed ranges (`Added 5000 job(s), IDs: 1-5000`), and group failures by error message rather than repeating it per job. Do not expect a comma-separated ID list to parse out of the output.
- **`-n` always means nodes**, never a row count — `--nnodes` on `add`/`update`/`ls`/`filter`, `--nodes` on `info`. On `ls`/`filter` it filters by exact node count (`pbx ls -n 2` = only 2-node jobs).
- **How many rows `pbx ls` prints** is a positional argument, not a flag: `pbx ls 10` (first 10), `pbx ls -20` (last 20), `pbx ls all` (everything). Bare `pbx ls` truncates to first 10 + last 10 once more than 25 jobs match. Composes with filters in any order: `pbx ls -20 -s Ready -n 2`.
- **Status values**: `Ready`, `Submitted`, `Running`, `Done`, `Failed`, `Killed`, `Restart`, `Resubmitted`, `Warning` (case-insensitive). `Submitted`/`Resubmitted` are the *claimed* states (a run atomically flipped `Ready`→`Submitted` / `Restart`→`Resubmitted` and stamped its batch id) — normally transient; you set jobs to `Ready`/`Restart`, not to these.
- **Self-respawn chains**: `pbx qsub --respawn N` / `pbx sbatch --respawn N` enables walltime-driven auto-resubmission. `N` is the number of remaining auto-resubmissions (decremented per link; `0` = chain ends after this run). See [docs/pbx-run-details.md](docs/pbx-run-details.md).
- **Dynamic discovery**: `--dynamic` (default) re-queries the DB for runnable jobs on every dispatch pass, so jobs added or flipped to `Ready`/`Restart` mid-run are picked up automatically; it also lets multiple `pbx run` allocations share one DB (each atomically claims what it dispatches). `--static` claims the runnable set once up front and does not re-query.

## Local projects (`pbx local`)

A directory holding `job_database_pbx-local.db` **and** `.pbxlocal.yaml` is a *local project*: its database stores the paths jobs will have on a remote HPC machine, from the first row. `pbx add ./run_a` there stores `/lus/flare/.../proj1/run_a`. Full walkthrough: [docs/remote-workflow.md](docs/remote-workflow.md).

- `pbx local init` — create or resume a project in the cwd. Cannot set `PBX_DB_PATH` itself; it prints the `export` line. Passing `--remote-root` makes it non-interactive.
- `pbx local status` — which side is ahead, endpoint health, the export line. Read-only. Follows `PBX_DB_PATH`, so it works from any directory. `--local-only` skips the network.
- `pbx local push` / `pull` — move the database. Each refuses if the side it would overwrite has changed since the last sync; there is no merge. `--force` overrides.
- `pbx local push --with-dirs` also sends job directories over Globus Transfer. Needs `transfer_local`/`transfer_remote` in `.pbxlocal.yaml` plus `PBX_GLOBUS_CLIENT_ID`. `transfer_remote_root` (which prefix of the remote path the collection publishes as its own `/`) is **detected on the first such push** and saved — hand-set it only to override. `--apps`/`--tags` scope which directories go; `--tags` globs, and a tag matching nothing is an error.
- `pbx qsub` / `pbx sbatch` in a local project push, submit on the remote's Globus Compute endpoint, and pull back. `--no-local` submits from here instead; `--push-dirs` also sends job directories over Globus Transfer.
- **Paths on `pbx add`**: under the project root, must exist here and are stored with the remote prefix; outside it, must be absolute and are stored unchanged (already remote paths, not checked locally).
- The export line names the **file**. Exporting the directory resolves to `job_database_pbx.db`, a separate empty database. Every `pbx` command refuses before creating it and prints the corrected `export` line. Only checked where a `.pbxlocal.yaml` is present; elsewhere a directory is a fine `PBX_DB_PATH`.

## App names

`lammps-kk`, `vasp`, `orca`, `python`, `julia`. Custom apps register in `config.yaml`. See [docs/apps.md](docs/apps.md).

## Config & DB paths

- `PBX_CONFIG_PATH`: dir or `.yml`/`.yaml` file (dir → `<dir>/config.yaml`)
- `PBX_DB_PATH`: dir or `.db` file (dir → `<dir>/job_database_pbx.db`)
- Both must be **absolute** paths (`~` is expanded); a relative value is a hard error
- Defaults: `~/.parslbox/config.yaml`, `~/.parslbox/job_database_pbx.db`, runs in `~/.parslbox/runs/<timestamp>/`
- `PBX_GLOBUS_CLIENT_ID`: Native App client id for Globus Transfer logins. Only needed for `pbx local push --with-dirs`; the database itself syncs over the Compute endpoint

## When asked something specific

Read [docs/commands.md](docs/commands.md) for the exact command. For app-specific behavior (e.g., LAMMPS rank policy, Python status reporting) read [docs/apps.md](docs/apps.md). For `pbx run` runtime details / restart chain semantics read [docs/pbx-run-details.md](docs/pbx-run-details.md). For local-to-remote projects read [docs/remote-workflow.md](docs/remote-workflow.md).
