---
name: parslbox-api
description: Help with the ParslBox Python API (`from parslbox.api import ParslBox`) — programmatic job add/list/filter/update/remove, scheduler submission (qsub/sbatch), cancellation (qdel/scancel), local-to-remote projects (local_init/status/push/pull), and orchestrator runs from Python scripts. Use when the user writes or debugs Python code that imports `parslbox`, instantiates `ParslBox()`, or wraps pbx in their own orchestrator. Do NOT use for `pbx` shell command questions — those go to parslbox-cli.
---

# ParslBox Python API

Authoritative reference: [docs/api-details.md](docs/api-details.md). Read it for method signatures, parameter details, return shapes, and exception semantics. This skill is a fast index.

## Entry point

```python
from parslbox.api import ParslBox

pbx = ParslBox()                  # uses default config & DB paths
# or:
pbx = ParslBox(config_path="...", db_path="...")
```

`ParslBox.__init__` checks (does not create) the config; if missing, raises. Run `pbx config` from the shell first, or call `parslbox.commands.config.config_setup(...)` programmatically.

`db_path` / `config_path` apply to every method including `qsub` and `sbatch` — they pick the DB that tag globs and the runnable-jobs guard resolve against, and both are baked into the generated `submit.sh`. Both must be absolute paths — a relative value is rejected, since it would resolve against the allocation's working directory inside the batch job. `PBX_DB_PATH` / `PBX_CONFIG_PATH` do the same job as defaults, but are read once at import, so pass the arguments if one process must target more than one DB.

## Methods at a glance

| Method | Returns | Purpose |
|---|---|---|
| `add_jobs(paths, app, config, ...)` | `(ids, failures, msg_log)` | Register one or many jobs. `paths` is always `List[str]`. |
| `list_jobs(status=, app=, tag=, path=, in_file=, num_nodes=)` | `List[dict]` | Job rows with optional filters (`num_nodes` is an exact match) |
| `filter_jobs(status=, app=, tag=, path=, in_file=, num_nodes=, exclude_status=, exclude_app=, exclude_tag=)` | `List[int]` | Just the job IDs matching the filters (note: IDs only, not full rows) |
| `get_job(job_id)` | `dict` | Full row for one job |
| `get_jobs_by_ids(ids)` | `List[dict]` | Full rows for a list of IDs |
| `update_jobs(job_ids, status=, tag=, input_file=, ngpus=, env_file=, nnodes=, node_occupancy=, ranks_per_node=, add_deps=, rm_deps=, app_args=)` | `(ids, failures, msg_log)` | Edit any of the listed fields. `app_args` mirrors CLI `--args`: it rebuilds each job's `in_file` as `<base script> <app_args>`, replacing any args already there. |
| `remove_job(id)` / `remove_jobs(ids)` / `remove_all_jobs()` | `bool` / `int` / `int` | Delete |
| `qsub(config, job_name, queue, select, walltime, project=, apps=, tags=, sched_opts=, respawn=, ...)` | `dict` | Build + submit a PBS batch. With `respawn=N` set, returns includes `respawn_template_file`. |
| `sbatch(...)` | `dict` | Same as `qsub` for SLURM. Same `respawn` semantics. |
| `qdel(jobid, grace=30)` | `dict` | Graceful PBS cancel (SIGTERM → grace → `qdel`). Returns dict with `success`, `reconciled_count`, etc. |
| `scancel(jobid, grace=30)` | `dict` | Same for SLURM. |
| `local_init(directory, remote_root, compute_endpoint=, transfer_local=, transfer_remote=, transfer_remote_root=, remote_config=, nested_ok=False)` | `dict` | Create or adopt a local project. `nested_ok=False` raises if a project exists in a parent directory. `transfer_remote_root` overrides the collection root that the first directory push otherwise detects. |
| `local_status(check_remote=True)` | `dict` | Which side is ahead, endpoint health. Always has `is_local_project`; the rest only when that is `True`. |
| `local_push(force=False, with_dirs=False, apps=, tags=, wait_for_dirs=True, sync_level='checksum')` | `dict` | Send the database up. `with_dirs=True` also sends job directories over Globus Transfer — that needs `transfer_local`/`transfer_remote` in `.pbxlocal.yaml` **and** `PBX_GLOBUS_CLIENT_ID` set to a registered Globus Native App id. `tags` accepts `*` globs; a tag matching nothing is an error. Raises `ValidationError` if the remote moved and `force` is unset, if the Compute endpoint is unreachable, or if Transfer refuses. |
| `local_pull(force=False)` | `dict` | Bring the database down. Raises `ValidationError` if the local moved and `force` is unset. |
| `run(...)` | — | Reserved / not implemented (raises `NotImplementedError`). Use the `pbx run` CLI for in-allocation execution. |

## Exceptions

`ParslBoxError` (base), `ValidationError`, `JobNotFoundError`. The `local_*` methods raise `ValidationError` for every expected failure, including an unreachable Compute endpoint and a refused Globus transfer. Plus `FileNotFoundError` from `qsub`/`sbatch` when the scheduler command itself isn't on PATH. See [docs/api-details.md](docs/api-details.md) for the per-method raise contract.

## App identifiers

`lammps-kk`, `vasp`, `orca`, `python`, `julia` — pass these as the `app` field on `add_jobs`. Custom apps register in `config.yaml`. See [docs/apps.md](docs/apps.md).

## Self-respawn chains

`qsub(..., respawn=N)` (or `sbatch(..., respawn=N)`) enables the walltime-driven resubmit chain. `N` is the number of remaining auto-resubmissions in the chain (decremented per link; `0` = chain ends after this run; `None` = chain disabled entirely, the default). The response dict includes `respawn_template_file` (a path you can inspect/modify before the next link fires). See [docs/pbx-run-details.md](docs/pbx-run-details.md).

## Local projects

When `db_path` has a `.pbxlocal.yaml` beside it, the instance is on a *local project*: its database stores remote paths, and `qsub`/`sbatch` push the database to a Globus Compute endpoint, submit there, and pull back. `no_local=True` submits from this machine instead; `push_dirs=True` also sends job directories over Globus Transfer.

`add_jobs` follows the project's path rule automatically: paths under the project root must exist locally and are stored with the remote prefix, paths outside it must be absolute and are stored unchanged. Nothing else in the API changes. See [docs/remote-workflow.md](docs/remote-workflow.md).

## Note on MCP

ParslBox also exposes its API over MCP (Model Context Protocol) — Claude Code and other MCP-aware agents discover the tools automatically once `.mcp.json` is wired up. No skill needed; this skill is for direct Python use.

## When asked something specific

Read [docs/api-details.md](docs/api-details.md) for method signatures and parameter semantics. For app-specific behavior read [docs/apps.md](docs/apps.md). For orchestrator/restart-chain details read [docs/pbx-run-details.md](docs/pbx-run-details.md). For local-to-remote projects read [docs/remote-workflow.md](docs/remote-workflow.md).
