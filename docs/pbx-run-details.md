# `pbx run` Runtime Behavior

Reference for every scenario the in-allocation engine handles: normal runs, walltime expiry, external cancellation, and the self-restart chain that `--restart` enables.

`pbx run` itself is internal — it is invoked by the `submit.sh` that `pbx qsub` / `pbx sbatch` generate. You configure its behavior through qsub/sbatch flags.

---

## Job statuses at a glance

| Status | Meaning |
| --- | --- |
| `Ready` | Eligible to run when resources are free and parents are `Done`. |
| `Submitted` | Handed to Parsl; about to start. |
| `Running` | Executing on a compute node. |
| `Done` | Finished successfully (app reported success). |
| `Failed` | App reported failure, exited non-zero, or — in restart-mode — chain exhausted. |
| `Killed` | Walltime ran out (non-restart-mode run) **or** the user ran `pbx qdel` / `pbx scancel`. |
| `Restart` | Two distinct meanings — see below. **Important.** |
| `Warning` | App returned an invalid/unknown status string. |

### `Restart` has two meanings — read carefully

`Restart` is the **only** status whose semantics depend on whether the current `pbx run` was invoked with `--restart-mode` (which `pbx qsub --restart` / `pbx sbatch --restart` set internally).

| Set by | Semantics | What happens at the next `pbx run` startup |
| --- | --- | --- |
| **User** (`pbx update <id> --status Restart`) | "I want this job to run again — just re-execute it." | Treated identically to `Ready`. The app's `restart()` hook is **not** called (the orchestrator is not in restart-mode). |
| **Orchestrator** (in-flight at walltime under `--restart-mode`, when `--max-restarts > 0`) | "This job was preempted mid-execution by walltime expiry. The app must decide how to resume it." | The next link's startup invokes `app.restart(job_dict)` for every `Restart` job before the run loop. See [Three-scene contract](#the-three-scene-restart-contract) below. |

The status field is the same in both cases. The disambiguator is `--restart-mode` on the *consuming* `pbx run`, not on the row itself. If you set `Restart` manually and then submit a chain with `pbx qsub --restart`, those manually-flipped jobs **will** be put through `app.restart()` at startup — which is usually what you want, but worth knowing.

---

## Non-restart-mode runs (the default)

What `pbx qsub` / `pbx sbatch` produce when `--restart` is **not** passed. The submit script's `pbx run` line has no `--restart-mode` flag.

### Startup

1. Parsl loads the system config.
2. The orchestrator queries the DB for jobs matching `--apps` / `--tags` filters and status `Ready` or `Restart`.
3. Those jobs go into the dispatch loop — `Restart` is treated identically to `Ready` (no hook is called).
4. The walltime timer arms itself ~30 s before the scheduler's stated walltime.

### Walltime expiry (internal timer fires)

1. The orchestrator stops dispatching new jobs.
2. All in-flight (`Running` / `Submitted`) jobs are marked **`Killed`**.
3. Parsl is shut down cleanly.
4. The batch job exits. **No resubmission.**

The user must manually re-queue (`pbx update <ids> --status Ready` or `--status Restart`) and resubmit (`pbx qsub` / `pbx sbatch`) if they want the work to continue.

### External cancellation (`pbx qdel <jobid>` / `pbx scancel <jobid>`)

1. The CLI sends SIGTERM to the orchestrator (not directly to the scheduler).
2. The SIGTERM handler marks all in-flight jobs **`Killed`** in the DB and triggers a graceful Parsl shutdown.
3. After `--grace` seconds (default 30 s), the hard `qdel`/`scancel` runs.

The grace window is the reason you should always prefer `pbx qdel` over raw `qdel` — without it, in-flight rows can remain stuck in `Running`/`Submitted`.

### Normal completion

When the dispatch queue is exhausted and all in-flight work has finished, the orchestrator exits cleanly. Each job ends in `Done`, `Failed`, or `Warning` based on what its app reported.

---

## Restart-mode runs (`pbx qsub --restart --max-restarts N`)

When you pass `--restart --max-restarts N`, `pbx qsub` / `pbx sbatch` generate two scripts in the run directory:

```
run_dir/
├── submit.sh                  ← submitted to the scheduler now; pbx never touches after creation
├── restart_template.sh        ← created once; pbx never overwrites; user-editable resource placeholders
└── restart_link_<idx>.sh      ← per-chain-link ephemeral; pbx writes one at each walltime resubmit
```

Both `submit.sh` and `restart_template.sh` embed `pbx run --restart-mode --max-restarts N ...` so chain semantics travel through every link.

### Worked example: `--max-restarts 3`

We will walk through the full chain: 4 batch jobs (`Link 0` through `Link 3`), each generating the next from `restart_template.sh`.

```
Link 0:  pbx qsub --restart --max-restarts 3   →   submit.sh has --max-restarts 3
   │  walltime
   ▼
Link 1:  restart_link_1.sh                      →   pbx run --max-restarts 2
   │  walltime
   ▼
Link 2:  restart_link_2.sh                      →   pbx run --max-restarts 1
   │  walltime
   ▼
Link 3:  restart_link_3.sh                      →   pbx run --max-restarts 0   (chain end)
```

#### Link 0 (initial submission)

**Startup.** Same as a non-restart-mode run. No jobs have status `Restart` yet (unless the user pre-set some), so `app.restart()` is **not** called for anything. Ready/Restart jobs flow into the dispatch loop.

> If the user *did* manually flip some jobs to `Restart` before submitting, the orchestrator's startup hook will treat them as Scene B / C (whichever the app returns) and re-queue them as `Ready`. The hook does not care whether a row was flipped by the user or by a prior link.

**Walltime expiry.** The internal timer fires ~30 s before the scheduler's stated walltime:

1. Stop dispatching new jobs.
2. All in-flight (`Running` / `Submitted`) jobs are marked **`Restart`** (not `Killed` — this is the key difference from non-restart-mode).
3. The orchestrator inspects the DB: how many `Ready` + `Restart` jobs remain?
4. It computes the optimal node count for those jobs (see [Resource recalc](#resource-recalc--node-cap)), capped at the original allocation size.
5. It reads `restart_template.sh`, substitutes `<<PBX_AUTO_SELECT>>` / `<<PBX_AUTO_NODES>>` with the computed value, rewrites the run line's `--max-restarts 3` to `--max-restarts 2`, and writes `restart_link_1.sh` to the run directory.
6. It runs `qsub restart_link_1.sh` (or `sbatch ...`) from the compute node, captures the output, logs the new job ID.
7. Parsl shuts down; the batch job exits.

The next scheduler allocation is now queued. If qsub/sbatch is not reachable from compute nodes, the resubmit step fails and the chain stops — see [Chain termination](#chain-termination).

**External SIGTERM during Link 0.** Same as the non-restart-mode case: in-flight → **`Killed`**, no resubmission. External cancellation always stops the chain.

#### Link 1 (first restart)

The scheduler eventually starts `restart_link_1.sh`. Its `pbx run` line includes `--restart-mode --max-restarts 2`.

**Startup hook runs.** This is the load-bearing difference from Link 0:

1. The orchestrator queries the DB for jobs matching `--apps`/`--tags` filters with status `Restart`.
2. For each such job, it calls `app.restart(job_dict)` — where `app` is the application instance that owns this job's `app` field.
3. The hook partitions outcomes into three buckets (see [Three-scene contract](#the-three-scene-restart-contract)):
   - **Patched** (dict returned) → DB row updated with the patch fields, status flipped to `Ready`.
   - **Re-run** (`None` or `{}` returned) → status flipped to `Ready`, no field changes.
   - **Failed** (raised `NotImplementedError`, or app not loaded) → status flipped to `Failed`.
4. After the hook completes, the orchestrator enters its normal dispatch loop. Only Patched + Re-run jobs are now `Ready` and pickable.

**Normal dispatch + walltime.** From here, Link 1 behaves exactly like Link 0: jobs run; walltime expires; in-flight jobs are marked `Restart`; `restart_link_2.sh` is generated with `--max-restarts 1`; the orchestrator resubmits.

#### Link 2 (second restart)

Identical to Link 1. Startup hook fires on jobs still `Restart`-marked (including any that didn't finish during Link 1). At walltime, in-flight → `Restart`, `restart_link_3.sh` generated with `--max-restarts 0`, resubmitted.

#### Link 3 (last link, `--max-restarts 0`)

The final link has `--max-restarts 0` baked into its `pbx run` line. Two things change:

**Startup hook still runs.** The hook isn't gated on the counter — it always runs in restart-mode, so any leftover `Restart` jobs get their last chance through `app.restart()` and re-enter `Ready`.

**Walltime expiry: in-flight → `Failed`, no resubmission.** Because `--max-restarts == 0`, the orchestrator marks in-flight jobs **`Failed`** (with the reason "chain exhausted: max restarts reached") instead of `Restart`, skips the resubmit step entirely, and exits. The chain ends.

If you want to continue the chain after exhaustion, you can manually re-queue the failed jobs (`pbx update <ids> --status Restart`) and resubmit (`pbx qsub --restart --max-restarts M ...`).

### External cancellation during any link

External SIGTERM (`pbx qdel <jobid>` / `pbx scancel <jobid>`) always:

- Marks in-flight jobs **`Killed`** (not `Restart`).
- Skips the resubmit step.
- Ends the chain.

There is no "stop after this link, but don't kill the running work" option. External cancellation is the supported way to stop a chain mid-flight.

---

## The three-scene `restart()` contract

Each app subclass under `parslbox/apps/` can override the `restart(job_dict) -> dict | None` hook on its `BaseApp` subclass. The orchestrator's startup hook in restart-mode passes the job's full DB row in and partitions the return value:

### Scene A — patch and re-run with new fields

The app inspects the job's state on disk (checkpoint files, last-completed step, etc.) and returns a dict of fields to change. The orchestrator applies the patch, sets `status='Ready'`, and `preprocess()` is **skipped** when the job is re-dispatched (since the app already prepared resume state).

```python
# parslbox/apps/lammps.py  (sketch)
class Lammps(BaseApp):
    def restart(self, job_dict):
        ckpt = find_latest_checkpoint(job_dict["path"])
        if not ckpt:
            return None                       # → Scene B fallback
        return {
            "in_file": rewrite_input_for_restart(job_dict["in_file"], ckpt),
            # any of: in_file, env_file, ngpus, num_nodes, node_occupancy,
            # ranks_per_node, mpi_opts, tag
        }
```

**Patchable fields:** `in_file`, `env_file`, `ngpus`, `num_nodes`, `node_occupancy`, `ranks_per_node`, `mpi_opts`, `tag`. Any other key in the returned dict is logged as a warning and ignored.

### Scene B — re-run as-is

The app decides this job can simply be re-executed from scratch (idempotent, no resume state needed). Return `None` or `{}`. The orchestrator flips status to `Ready`; no field changes; `preprocess()` is skipped on re-dispatch.

```python
class Python(BaseApp):
    def restart(self, job_dict):
        return None      # just re-run
```

### Scene C — no restart capability

The app cannot resume — maybe its tool doesn't checkpoint, maybe the resume logic isn't implemented. **This is the default for the shipped apps in v1** (`appbase.py` raises `NotImplementedError`). The orchestrator marks the job `Failed` and moves on.

```python
class AppBase:
    def restart(self, job_dict):
        raise NotImplementedError(
            f"{type(self).__name__} does not support restart. "
            "Override restart() to enable --restart for jobs of this app."
        )
```

To add restart support to a custom app, override `restart()`. The default raise is intentional — silently re-running may corrupt state, so apps must opt in.

---

## Resource placeholders and template lifecycle

`restart_template.sh` is generated once at initial `pbx qsub --restart` / `pbx sbatch --restart` and is **never overwritten by pbx**. Its resource line uses placeholders:

| Placeholder | Replaced by | Used in |
| --- | --- | --- |
| `<<PBX_AUTO_SELECT>>` | computed integer node count | PBS `#PBS -l select=...` |
| `<<PBX_AUTO_NODES>>` | computed integer node count | SLURM `#SBATCH --nodes=...` |

You can edit `restart_template.sh` freely between links — your edits persist for the rest of the chain. The one line pbx **does** rewrite at every link is the `pbx run` invocation:

- `--max-restarts <N>` is regex-replaced with the decremented value. Any manual edit to that number will be overwritten next cycle. **Do not edit it.**
- The header comment at the top of every generated template restates this.

If a required arg is missing from the `pbx run` line at resubmit time (`--restart-mode`, `--max-restarts`, `--config`, `--run-dir`), the orchestrator logs a clear error and stops the chain — it will not submit a malformed script.

### Resource recalc + node cap

At each walltime resubmit, the orchestrator counts schedulable jobs (`Ready` + `Restart`) and computes the optimal (tightest-pack) node count: GPU jobs packed by GPUs-per-node, CPU jobs packed by `node_occupancy`. The result is **capped at the original allocation's node count** (parsed from `submit.sh`) — pbx will never silently request more than the user originally asked for.

If the original allocation used a complex PBS select string (e.g. `select=2:ncpus=32:ngpus=4`) that can't be parsed as a plain integer, the cap is skipped — your edits to `restart_template.sh` are then the only safeguard.

---

## Chain termination

A chain ends — no further auto-resubmission — when **any** of:

1. **`--max-restarts == 0`** at walltime. In-flight jobs go to `Failed`. (See [Link 3](#link-3-last-link---max-restarts-0).)
2. **Zero in-flight jobs at walltime.** Everything finished as `Done`/`Failed` already — there is nothing to resubmit for.
3. **`qsub` / `sbatch` returns non-zero** at resubmit. The orchestrator logs full stderr + a "chain stopped" note. Most common cause: the site does not allow scheduler commands from inside compute jobs.
4. **`qsub` / `sbatch` not found on the compute node.** Same outcome as 3.
5. **Template validation fails.** A required arg was edited out of the `pbx run` line.
6. **External SIGTERM.** `pbx qdel <jobid>` / `pbx scancel <jobid>` always stops the chain.

In every case the previously-running link still marks its in-flight jobs as `Restart` (or `Killed` for SIGTERM and case 1) and exits cleanly. You can pick up where it left off with `pbx qsub --restart --max-restarts M ...` and the next link's startup hook will process those `Restart` rows.

---

## Quick reference: which status, when?

| Trigger | Non-restart-mode | Restart-mode, `--max-restarts > 0` | Restart-mode, `--max-restarts == 0` |
| --- | --- | --- | --- |
| Walltime expiry | in-flight → `Killed` | in-flight → `Restart` + resubmit | in-flight → `Failed` |
| `pbx qdel` / `pbx scancel` | in-flight → `Killed` | in-flight → `Killed`, chain stops | in-flight → `Killed`, chain stops |
| App reports success | `Done` | `Done` | `Done` |
| App reports failure / exits non-zero | `Failed` | `Failed` | `Failed` |
| App returns unknown status | `Warning` | `Warning` | `Warning` |
| Parents not `Done` yet | stays `Ready` | stays `Ready` | stays `Ready` |
| Startup, job is `Restart` | dispatched as if `Ready`; no hook | `app.restart()` called → Patched/Re-run/Failed | `app.restart()` called → Patched/Re-run/Failed |
