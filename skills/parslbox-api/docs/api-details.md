# ParslBox Python API

The `parslbox.api.ParslBox` class is the programmatic entry point. It mirrors the CLI (`pbx ...`) — same validations, same behavior — and delegates to the shared core helpers, so CLI / API / MCP stay in sync.

```python
from parslbox.api import ParslBox

pbx = ParslBox()                               # uses default DB + config
# pbx = ParslBox(db_path=Path("/tmp/x.db"),    # or pin both explicitly
#                config_path=Path("/tmp/cfg.yaml"))
```

`ParslBox.__init__` creates the DB if missing but **requires the config file to already exist** — run `pbx config` first (or pass `config_path=` to an existing one).

`db_path` and `config_path` govern every method on the instance, `qsub`/`sbatch` included: they select the database that tag globs and the runnable-jobs guard are checked against, and both paths are written into the generated `submit.sh` so `pbx run` opens the same ones inside the allocation. Setting `PBX_DB_PATH` / `PBX_CONFIG_PATH` in the environment stays equivalent — it just supplies the defaults. Pass the arguments when one process needs to target several databases, since the environment is only read once, at import.

## Exceptions

```python
from parslbox.api import ParslBoxError, ValidationError, JobNotFoundError
```

- `ParslBoxError` — base class for all API errors.
- `ValidationError` — bad arguments, unknown app/config, unmatched tag globs, no runnable jobs at submit time, etc.
- `JobNotFoundError` — `get_job(id)` when the id doesn't exist.

## Methods

| Method | Purpose |
| --- | --- |
| `add_jobs(paths, app, config, ...)` | Register one or more jobs in the DB |
| `update_jobs(job_ids, ...)` | Mutate status / resources / dependencies of existing jobs |
| `list_jobs(...)` | Return full job rows (dicts), optionally filtered |
| `filter_jobs(...)` | Return matching job IDs only; supports include + exclude filters |
| `get_job(job_id)` | Fetch one job by ID (raises if missing) |
| `get_jobs_by_ids(job_ids)` | Fetch many jobs by ID (silently drops missing) |
| `remove_job(job_id)` | Delete one job |
| `remove_jobs(job_ids)` | Delete many jobs |
| `remove_all_jobs()` | Wipe the DB |
| `qsub(...)` | Generate and submit a PBS batch script |
| `sbatch(...)` | Generate and submit a SLURM batch script |
| `qdel(jobid, grace=30)` | Graceful PBS cancel (SIGTERM → wait → hard kill) |
| `scancel(jobid, grace=30)` | Graceful SLURM cancel |
| `run(...)` | (Reserved — use `pbx run` CLI for full execution control) |

---

## `add_jobs`

Register one or more job directories in the DB. Returns `(successful_ids, failures, msg_log)` — failures are per-path so a bad entry doesn't poison the batch.

```python
job_ids, failures, log = pbx.add_jobs(
    paths=["/scratch/sim1", "/scratch/sim2"],
    app="lammps-kk",
    config="polaris",
    ngpus=2,
    tag="film-bulk",
    input_file="in.lammps",
)
print(f"added {len(job_ids)} jobs: {job_ids}")
for path, err in failures:
    print(f"skip {path}: {err}")
for w in log["warnings"]: print("warn:", w)
```

Sub-node CPU job with fractional occupancy:

```python
ids, _, _ = pbx.add_jobs(
    paths=["/scratch/analysis"],
    app="python", config="polaris",
    input_file="analyze.py",
    node_occupancy=0.25,           # 4 of these pack onto one node
    env_file="env_setup.sh",
)
```

Job that waits on parents (explicit IDs and/or by tag):

```python
ids, _, _ = pbx.add_jobs(
    paths=["/scratch/stage2"],
    app="vasp", config="polaris",
    nnodes=1, ngpus=4,
    parents=[10, 11, 12],          # ints; ranges only via CLI
    parent_tag="stage1",           # also waits for every Done-tagged job
    tag="stage2",
)
```

---

## `update_jobs`

Patch one or more existing jobs. Only the kwargs you pass get changed. Returns `(updated_ids, failures, msg_log)`.

```python
# Mark a batch as Restart so the next run picks them back up
updated, failures, log = pbx.update_jobs(
    job_ids=[101, 102, 103],
    status="Restart",
)

# Bump resources on a single job and swap its input file
pbx.update_jobs(
    job_ids=[55],
    ngpus=4, nnodes=2, ranks_per_node=12,
    input_file="in.lammps.large",
)

# Add / remove dependency edges without rewriting the parent list
pbx.update_jobs(job_ids=[200], add_deps=[10, 11], rm_deps=[7])

# Change app arguments (mirrors CLI --args). Rebuilds in_file as
# "<base script> <app_args>", replacing any args already set.
pbx.update_jobs(job_ids=[101, 102], app_args="-var T 300")
```

---

## `list_jobs`

Returns full row dicts. No args → every job in the DB.

```python
all_jobs = pbx.list_jobs()

ready_lammps = pbx.list_jobs(status="Ready", app="lammps-kk")
for j in ready_lammps:
    print(j["job_id"], j["path"], j["tag"], j["num_nodes"], j["ngpus"])

# Substring match on path / input file
in_scratch = pbx.list_jobs(path="/scratch")
plot_jobs  = pbx.list_jobs(in_file="plot.py")

# Exact node count
two_node = pbx.list_jobs(num_nodes=2)
```

> `list_jobs` filters use exact match for `status`/`app`/`tag`/`num_nodes` and substring match for `path`/`in_file`. For **tag glob support** (`*test`, `prod*`, `*3c*`) and exclude filters, use `filter_jobs`.

---

## `filter_jobs`

Same include filters as `list_jobs`, plus tag-glob support and three exclude flags (`exclude_status`, `exclude_app`, `exclude_tag`). Returns just IDs — perfect for piping into `update_jobs` / `remove_jobs` / `get_jobs_by_ids`.

```python
# All Ready jobs tagged like prod-* but NOT prod-test
ids = pbx.filter_jobs(status="Ready", tag="prod*", exclude_tag="prod-test")
print(ids)                                        # e.g. [12, 14, 18]

# Everything except python jobs
ids = pbx.filter_jobs(exclude_app="python")

# Compose: re-queue every Failed job that isn't tagged *experimental
to_retry = pbx.filter_jobs(status="Failed", exclude_tag="*experimental")
pbx.update_jobs(to_retry, status="Restart")
```

Untagged jobs are never excluded by a `*` tag glob (a wildcard for tags shouldn't sweep up untagged jobs).

---

## `get_job` / `get_jobs_by_ids`

```python
job = pbx.get_job(42)
print(job["app"], job["status"], job["path"])

# Bulk fetch — missing IDs are silently dropped (no exception)
rows = pbx.get_jobs_by_ids([1, 2, 3, 9999])
for j in rows:
    print(j["job_id"], "->", j["status"])

# get_job raises if missing
from parslbox.api import JobNotFoundError
try:
    pbx.get_job(999_999)
except JobNotFoundError as e:
    print("not in DB:", e)
```

---

## `remove_job` / `remove_jobs` / `remove_all_jobs`

```python
ok = pbx.remove_job(42)                # True if it existed
n  = pbx.remove_jobs([10, 11, 12])     # returns count actually deleted
n  = pbx.remove_all_jobs()             # wipe DB; returns count

# Idiom: remove every Done job older than some criterion
done_ids = pbx.filter_jobs(status="Done")
pbx.remove_jobs(done_ids)
```

---

## `qsub` — submit a PBS batch

Generates the batch script, validates `tags` against the DB (literal **and** glob entries must each match at least one Ready job), aborts cleanly if nothing is runnable, and submits. Returns a dict with `success`, `job_id`, `run_dir`, and — when validation ran — `matched_jobs` + `resolved_tags`. When `respawn` is set, the result dict also includes `respawn_template_file` (path to the generated `respawn_template.sh`).

```python
result = pbx.qsub(
    config="polaris",
    job_name="prod-run",
    queue="prod",
    select="4",                    # 4 nodes; can also be "2:ncpus=32:ngpus=4"
    walltime="90",                 # minutes; or "4.25h", "3.5d"
    project="MYPROJ",
    apps=["lammps-kk"],
    tags=["prod-*", "film-bulk"],  # globs allowed; resolved against DB
    retries=1,
)
print(result["job_id"], result["run_dir"])
print("matched", result.get("matched_jobs"), "tags →", result.get("resolved_tags"))
```

Self-respawn chain — pass `respawn=N` (non-negative int). pbx generates `respawn_template.sh` alongside `submit.sh`; at every walltime expiry the orchestrator marks `Running` jobs `Restart` and auto-submits the next link with `respawn` decremented by 1. When the counter reaches 0, `Running` jobs go to `Failed` and the chain ends. Full lifecycle: [`pbx-run-details.md`](pbx-run-details.md).

```python
result = pbx.qsub(
    config="sophia", job_name="sweep", queue="gpu",
    select="4", walltime="4h", project="MYPROJ",
    apps=["lammps-kk"], tags=["sweep"],
    respawn=3,
)
print("respawn template:", result.get("respawn_template_file"))
```

Pass extra PBS directives verbatim:

```python
pbx.qsub(
    config="polaris", job_name="x", queue="debug",
    select="1", walltime=30, project="MYPROJ",
    sched_opts=["#PBS -l filesystems=home:eagle", "#PBS -M me@example.com", "#PBS -m bea"],
)
```

Handle a validation failure (unknown app, unmatched tag glob, no Ready jobs):

```python
from parslbox.api import ValidationError
try:
    pbx.qsub(config="polaris", job_name="x", queue="prod",
             select="1", walltime=30, tags=["does-not-exist"])
except ValidationError as e:
    print("won't submit:", e)
```

---

## `sbatch` — submit a SLURM batch

Same shape as `qsub`, including `respawn`. `select` is the node count; `queue` is the partition.

```python
result = pbx.sbatch(
    config="perlmutter-gpu",
    job_name="prod-run",
    queue="regular",               # partition
    select="2",                    # 2 nodes
    walltime="4h",
    project="m1234",
    apps=["lammps-kk", "vasp"],
    tags=["prod-*"],
    sched_opts=["#SBATCH --qos=regular", "#SBATCH --constraint=gpu"],
    respawn=2,  # optional self-respawn chain
)
print(result["job_id"], result["run_dir"])
```

---

## `qdel` / `scancel` — graceful cancel

Sends SIGTERM, gives the orchestrator `grace` seconds to reconcile active jobs in the DB, then hard-kills the batch. After the scheduler-level kill, the DB is reconciled per state for any non-terminal jobs under this batch (matched by `sched_job_id`): `Running` → `Killed`, `Submitted` → `Ready`, `Resubmitted` → `Restart`. This catches cases where the orchestrator's signal handler couldn't complete its DB writes before the process exited. Scoped by `sched_job_id`, so concurrent batch jobs are unaffected.

```python
pbx.qdel("123456.polaris-pbs-01", grace=30)
pbx.scancel("789012", grace=60)
```

Both return a dict — `{"success": True, "jobid": ..., "grace": ..., "reconciled_count": N}` on success, `{"success": False, "jobid": ..., "stage": ..., "error": ..., "reconciled_count": N}` on failure. `reconciled_count` is the number of stuck jobs cleaned up (typically `0` when the signal handler finished cleanly).

---

## End-to-end example

```python
from parslbox.api import ParslBox, ValidationError

pbx = ParslBox()

# 1. Register a batch of LAMMPS jobs
job_ids, failures, _ = pbx.add_jobs(
    paths=[f"/scratch/run{i}" for i in range(10)],
    app="lammps-kk", config="polaris",
    ngpus=4, nnodes=1, tag="sweep-v2", input_file="in.lammps",
)
if failures:
    print("dropped:", failures)

# 2. Submit them under a single PBS allocation
try:
    res = pbx.qsub(
        config="polaris", job_name="sweep", queue="prod",
        select="5", walltime="3h", project="MYPROJ",
        apps=["lammps-kk"], tags=["sweep-v2"], retries=1,
    )
    print(f"submitted PBS job {res['job_id']} ({res['matched_jobs']} runnable)")
except ValidationError as e:
    print("aborted:", e); raise

# 3. Later: re-queue anything that failed, drop the experimental ones
failed = pbx.filter_jobs(status="Failed", tag="sweep-v2",
                         exclude_tag="*experimental")
if failed:
    pbx.update_jobs(failed, status="Restart")
    print(f"re-queued {len(failed)} jobs for the next submission")
```
