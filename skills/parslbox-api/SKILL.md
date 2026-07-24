---
name: parslbox-api
description: Help with the ParslBox Python API (`from parslbox.api import ParslBox`) — programmatic job add/list/filter/update/remove, scheduler submission (qsub/sbatch), cancellation (qdel/scancel), and orchestrator runs from Python scripts. Use when the user writes or debugs Python code that imports `parslbox`, instantiates `ParslBox()`, or wraps pbx in their own orchestrator. Do NOT use for `pbx` shell command questions — those go to parslbox-cli.
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

## Methods at a glance

| Method | Returns | Purpose |
|---|---|---|
| `add_jobs(paths, app, config, ...)` | `(ids, failures, msg_log)` | Register one or many jobs. `paths` is always `List[str]`. |
| `list_jobs(status=, app=, tag=, path=, in_file=)` | `List[dict]` | Job rows with optional filters |
| `filter_jobs(status=, app=, tag=, path=, in_file=, exclude_status=, exclude_app=, exclude_tag=)` | `List[int]` | Just the job IDs matching the filters (note: IDs only, not full rows) |
| `get_job(job_id)` | `dict` | Full row for one job |
| `get_jobs_by_ids(ids)` | `List[dict]` | Full rows for a list of IDs |
| `update_jobs(job_ids, status=, tag=, input_file=, ngpus=, env_file=, nnodes=, node_occupancy=, ranks_per_node=, add_deps=, rm_deps=)` | `(ids, failures, msg_log)` | Edit any of the listed fields. CLI `--args` is *not* a separate API param — append args to `input_file` instead. |
| `remove_job(id)` / `remove_jobs(ids)` / `remove_all_jobs()` | `bool` / `int` / `int` | Delete |
| `qsub(config, job_name, queue, select, walltime, project=, apps=, tags=, sched_opts=, respawn=, ...)` | `dict` | Build + submit a PBS batch. With `respawn=N` set, returns includes `respawn_template_file`. |
| `sbatch(...)` | `dict` | Same as `qsub` for SLURM. Same `respawn` semantics. |
| `qdel(jobid, grace=30)` | `dict` | Graceful PBS cancel (SIGTERM → grace → `qdel`). Returns dict with `success`, `reconciled_count`, etc. |
| `scancel(jobid, grace=30)` | `dict` | Same for SLURM. |
| `run(...)` | — | Run the orchestrator in-process (rarely needed — `qsub`/`sbatch` wrap this for you). |

## Exceptions

`ParslBoxError` (base), `ValidationError`, `JobNotFoundError`. Plus `FileNotFoundError` from `qsub`/`sbatch` when the scheduler command itself isn't on PATH. See [docs/api-details.md](docs/api-details.md) for the per-method raise contract.

## App identifiers

`lammps-kk`, `vasp`, `orca`, `python`, `julia` — pass these as the `app` field on `add_jobs`. Custom apps register in `config.yaml`. See [docs/apps.md](docs/apps.md).

## Self-respawn chains

`qsub(..., respawn=N)` (or `sbatch(..., respawn=N)`) enables the walltime-driven resubmit chain. `N` is the number of remaining auto-resubmissions in the chain (decremented per link; `0` = chain ends after this run; `None` = chain disabled entirely, the default). The response dict includes `respawn_template_file` (a path you can inspect/modify before the next link fires). See [docs/pbx-run-details.md](docs/pbx-run-details.md).

## Note on MCP

ParslBox also exposes its API over MCP (Model Context Protocol) — Claude Code and other MCP-aware agents discover the tools automatically once `.mcp.json` is wired up. No skill needed; this skill is for direct Python use.

## When asked something specific

Read [docs/api-details.md](docs/api-details.md) for method signatures and parameter semantics. For app-specific behavior read [docs/apps.md](docs/apps.md). For orchestrator/restart-chain details read [docs/pbx-run-details.md](docs/pbx-run-details.md).
