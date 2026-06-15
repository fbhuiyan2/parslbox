# `pbx run` Runtime Behavior

Reference for every scenario the in-allocation engine handles: normal runs, walltime expiry, external cancellation, and the self-respawn chain that `--respawn N` enables.

`pbx run` itself is internal — it is invoked by the `submit.sh` that `pbx qsub` / `pbx sbatch` generate. You configure its behavior through qsub/sbatch flags.

---

## Job statuses at a glance

| Status | Meaning |
| --- | --- |
| `Ready` | Eligible to run when resources are free and parents are `Done`. |
| `Submitted` | Handed to Parsl; about to start. |
| `Running` | Executing on a compute node. |
| `Done` | Finished successfully (app reported success). |
| `Failed` | App reported failure, exited non-zero, or — in a respawn chain — chain exhausted. |
| `Killed` | Walltime ran out (no `--respawn`) **or** the user ran `pbx qdel` / `pbx scancel`. |
| `Restart` | Two distinct meanings — see below. **Important.** |
| `Warning` | App returned an invalid/unknown status string. |

### `Restart` semantics

`Restart` jobs get their app's `restart()` hook called **lazily per-job at dispatch time** — inside `create_parsl_future`, right after resources are assigned and right before `preprocess()` runs. Symmetric with `preprocess()`: on-demand, only for jobs the orchestrator is actually about to run. Jobs that get backlogged (no resources), gated (walltime floor), or filtered out for any reason never trigger `restart()`.

| Set by | Why it ended up as Restart |
| --- | --- |
| **User** (`pbx update <id> --status Restart`) | "I want this job to resume — use the app's checkpoint logic if it has one, or just re-execute if it doesn't." |
| **Orchestrator** (in-flight at walltime under `--respawn N` with `N > 0`) | "This job was preempted mid-execution by walltime expiry. The next chain link picks it up." |

In both cases the next `pbx run` invokes `app.restart(job_dict)` once per Restart-status job as that job reaches dispatch. See [Three-scene contract](#the-three-scene-restart-contract) below. Apps that don't override `restart()` raise `NotImplementedError` and the job is marked `Failed` with "app does not support restart" — so manually marking a job Restart only makes sense for apps that actually implement the hook.

`--respawn` does **not** gate `restart()`. It controls only the walltime auto-resubmit chain (next-link script generation + qsub from compute) and the wider 90 s shutdown grace.

**Strict contract — no resource patches.** `restart()` may return a dict patching `in_file`, `env_file`, or `tag` only. Returning a resource field (`ngpus`, `num_nodes`, `node_occupancy`, `ranks_per_node`, `mpi_opts`) is a contract violation and marks the job `Failed` — because resources are allocated **before** `restart()` runs in the lazy design, so patches there would be silently ignored. If a job needs different resources for its restart-continuation, the user must `pbx update --status Restart --ngpus N ...` before submitting the next run.

---

## No-respawn runs (the default)

What `pbx qsub` / `pbx sbatch` produce when `--respawn` is **not** passed. The submit script's `pbx run` line has no `--respawn` flag.

### Startup

1. Parsl loads the system config.
2. The orchestrator queries the DB for jobs matching `--apps` / `--tags` filters and status `Ready` or `Restart`. Both statuses enter the dispatch loop together (Restart first as priority).
3. `app.restart()` is **not** called at startup — it runs lazily per-job inside `create_parsl_future` once resources are assigned (see [Restart semantics](#restart-semantics)). This is only the chain auto-resubmit that gets gated on `--respawn`; restart() itself fires regardless.
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
4. After the hard kill (success or failure), pbx reconciles the DB: any jobs left in `Running`/`Submitted` whose `sched_job_id` matches the killed batch are force-flipped to `Killed`. Scoped by `sched_job_id`, so concurrent batch jobs are unaffected.

The grace window is the reason you should always prefer `pbx qdel` over raw `qdel` — without it, in-flight rows can remain stuck in `Running`/`Submitted`. The reconciliation in step 4 is a safety net for the (rare) cases where the signal handler couldn't complete its DB cleanup before the process exited (DB contention, alarm timeout, exception). Without it, those jobs would stay stuck in `Running` indefinitely until a manual `pbx update`.

`pbx qdel`/`pbx scancel` **always** mark `Killed`, regardless of `--respawn` mode. If you want to pause-and-resume a respawn chain rather than terminate it, flip the jobs back to `Restart` manually after the cancel and run `pbx qsub --respawn N` again.

### Normal completion

When the dispatch queue is exhausted and all in-flight work has finished, the orchestrator exits cleanly. Each job ends in `Done`, `Failed`, or `Warning` based on what its app reported.

### Per-app minimum-remaining-walltime floor

Apps may override `min_remaining_walltime(job_dict) -> int` on their `AppBase` subclass to declare a minimum remaining batch walltime (seconds) required for a fresh dispatch. The orchestrator checks this floor at every dispatch site — initial dispatch, mid-run dynamic-discovery, and backlog reschedule. Jobs failing the floor are skipped, stay in DB at their current `Ready`/`Restart` status, and get picked up by the next `pbx run`. The default is `0` (no gate). Restart-continuations (jobs the current run already loaded via `app.restart()`) are exempt — they have checkpoint state and brief runtime still advances the simulation.

Use this for apps where a short runtime is wasted compute — e.g. long MD simulations that need at least an hour to write a meaningful checkpoint. `LammpsKkRestart` ships with a 1-hour floor.

---

## Respawn-chain runs (`pbx qsub --respawn N`)

When you pass `--respawn N`, `pbx qsub` / `pbx sbatch` generate two scripts in the run directory:

```
run_dir/
├── submit.sh                  ← submitted to the scheduler now; pbx never touches after creation
├── respawn_template.sh        ← created once; pbx never overwrites; user-editable resource placeholders
└── respawn_link_<idx>.sh      ← per-chain-link ephemeral; pbx writes one at each walltime resubmit
```

Both `submit.sh` and `respawn_template.sh` embed `pbx run --respawn N ...` so chain semantics travel through every link.

### Worked example: `--respawn 3`

We will walk through the full chain: 4 batch jobs (`Link 0` through `Link 3`), each generating the next from `respawn_template.sh`.

```
Link 0:  pbx qsub --respawn 3        →   submit.sh has --respawn 3
   │  walltime
   ▼
Link 1:  respawn_link_1.sh           →   pbx run --respawn 2
   │  walltime
   ▼
Link 2:  respawn_link_2.sh           →   pbx run --respawn 1
   │  walltime
   ▼
Link 3:  respawn_link_3.sh           →   pbx run --respawn 0   (chain end)
```

#### Link 0 (initial submission)

**Startup.** Same as a no-respawn run. Ready/Restart jobs flow into the dispatch loop together (Restart first as priority). `restart()` is called lazily per-job inside `create_parsl_future` once resources are assigned — see [Restart semantics](#restart-semantics).

> If the user *did* manually flip some jobs to `Restart` before submitting, the orchestrator treats them like any other Restart job — runs them through `app.restart()` (Scene A/B/C depending on the app's return value) when each one reaches dispatch.

**Walltime expiry.** The internal timer fires ~90 s before the scheduler's stated walltime (respawn uses a wider grace than the no-respawn 30 s — the resubmit path needs the extra runway):

1. Stop dispatching new jobs.
2. All in-flight (`Running` / `Submitted`) jobs are marked **`Restart`** (not `Killed` — this is the key difference from no-respawn runs).
3. The orchestrator inspects the DB: how many `Ready` + `Restart` jobs remain?
4. It computes the optimal node count for those jobs (see [Resource recalc](#resource-recalc--node-cap)), capped at the original allocation size.
5. It reads `respawn_template.sh`, substitutes `<<PBX_AUTO_SELECT>>` / `<<PBX_AUTO_NODES>>` with the computed value, rewrites the run line's `--respawn 3` to `--respawn 2`, and writes `respawn_link_1.sh` to the run directory.
6. It runs `qsub respawn_link_1.sh` (or `sbatch ...`) from the compute node, captures the output, logs the new job ID.
7. Parsl shuts down; the batch job exits.

The next scheduler allocation is now queued. If qsub/sbatch is not reachable from compute nodes, the resubmit step fails and the chain stops — see [Chain termination](#chain-termination).

**External SIGTERM during Link 0.** Same as the no-respawn case: in-flight → **`Killed`**, no resubmission. External cancellation always stops the chain.

#### Link 1 (first respawn)

The scheduler eventually starts `respawn_link_1.sh`. Its `pbx run` line includes `--respawn 2`.

**Per-job restart fires lazily during dispatch.** Unlike Link 0 (where most jobs are fresh Ready), Link 1 typically has a large `Restart`-status backlog from Link 0's walltime kill. The orchestrator collects both Ready and Restart jobs into the dispatch loop. For each Restart-status job that survives the dispatch gate (deps, resources, walltime floor), `app.restart(job_dict)` is called inside `create_parsl_future` once resources are assigned. The return value partitions into three buckets (see [Three-scene contract](#the-three-scene-restart-contract)):

- **Patched** (dict returned) → patch fields applied to the in-memory job dict and ride along on the buffered Running write (single batched DB call covers status + patch).
- **Re-run** (`None` or `{}` returned) → job dispatches unchanged, just with file-mode 'a'+banner so the new run's stdout/stderr appends instead of clobbering prior content.
- **Failed** (raised `NotImplementedError`, returned a forbidden resource field, or app not loaded) → job marked `Failed`, dispatch aborts for that job.

**Normal dispatch + walltime.** Link 1 behaves like Link 0: jobs run; walltime expires; in-flight jobs marked `Restart`; `respawn_link_2.sh` generated with `--respawn 1`; resubmit.

#### Link 2 (second respawn)

Identical to Link 1. Per-job `restart()` fires on jobs still `Restart`-marked (including any that didn't finish during Link 1). At walltime, in-flight → `Restart`, `respawn_link_3.sh` generated with `--respawn 0`, resubmitted.

#### Link 3 (last link, `--respawn 0`)

The final link has `--respawn 0` baked into its `pbx run` line. Two things change:

**Per-job `restart()` still fires.** It isn't gated on the counter (or on `--respawn` itself) — it always runs for Restart-status jobs that reach dispatch, so any leftover `Restart` jobs get their last chance through `app.restart()`.

**Walltime expiry: in-flight → `Failed`, no resubmission.** Because `--respawn == 0`, the orchestrator marks in-flight jobs **`Failed`** (with the reason "chain exhausted: respawn reached 0") instead of `Restart`, skips the resubmit step entirely, and exits. The chain ends.

If you want to continue the chain after exhaustion, you can manually re-queue the failed jobs (`pbx update <ids> --status Restart`) and resubmit (`pbx qsub --respawn M ...`).

### External cancellation during any link

External SIGTERM (`pbx qdel <jobid>` / `pbx scancel <jobid>`) always:

- Marks in-flight jobs **`Killed`** (not `Restart`).
- Skips the resubmit step.
- Ends the chain.

There is no "stop after this link, but don't kill the running work" option. External cancellation is the supported way to stop a chain mid-flight.

---

## The three-scene `restart()` contract

Each app subclass under `parslbox/apps/` can override the `restart(job_dict) -> dict | None` hook on its `AppBase` subclass. The orchestrator calls it lazily per-job inside `create_parsl_future` for every Restart-status job that reaches dispatch (regardless of `--respawn`). The return value partitions into three scenes:

### Scene A — patch and re-run with new fields

The app inspects the job's state on disk (checkpoint files, last-completed step, etc.) and returns a dict of fields to change. The patch is applied to the in-memory job dict and folded into the buffered Running write — one batched DB call covers both the status flip and the patch. `preprocess()` runs after restart() in the same dispatch sequence.

```python
# parslbox/apps/lammps.py  (sketch)
class Lammps(AppBase):
    def restart(self, job_dict):
        ckpt = find_latest_checkpoint(job_dict["path"])
        if not ckpt:
            return None                       # → Scene B fallback
        return {
            "in_file": rewrite_input_for_restart(job_dict["in_file"], ckpt),
            # allowed keys: in_file, env_file, tag
        }
```

**Patchable fields:** `in_file`, `env_file`, `tag`. Any other non-resource key is logged as a warning and dropped (job still dispatches with the valid fields).

**Forbidden (contract violation → Failed):** `ngpus`, `num_nodes`, `node_occupancy`, `ranks_per_node`, `mpi_opts`. Resources are allocated **before** `restart()` runs in the lazy design, so patching them here would be silently ignored. The orchestrator marks the job `Failed` with a clear error if any of these keys appear in the returned dict.

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
            "Override restart() to enable Restart-status job handling for this app."
        )
```

To add restart support to a custom app, override `restart()`. The default raise is intentional — silently re-running may corrupt state, so apps must opt in.

---

## Resource placeholders and template lifecycle

`respawn_template.sh` is generated once at initial `pbx qsub --respawn N` / `pbx sbatch --respawn N` and is **never overwritten by pbx**. Its resource line uses placeholders:

| Placeholder | Replaced by | Used in |
| --- | --- | --- |
| `<<PBX_AUTO_SELECT>>` | computed integer node count | PBS `#PBS -l select=...` |
| `<<PBX_AUTO_NODES>>` | computed integer node count | SLURM `#SBATCH --nodes=...` |

You can edit `respawn_template.sh` freely between links — your edits persist for the rest of the chain. The one line pbx **does** rewrite at every link is the `pbx run` invocation:

- `--respawn <N>` is regex-replaced with the decremented value. Any manual edit to that number will be overwritten next cycle. **Do not edit it.**
- The header comment at the top of every generated template restates this.

If a required arg is missing from the `pbx run` line at resubmit time (`--respawn`, `--config`, `--run-dir`), the orchestrator logs a clear error and stops the chain — it will not submit a malformed script.

### Resource recalc + node cap

At each walltime resubmit, the orchestrator counts schedulable jobs (`Ready` + `Restart`) and computes the optimal (tightest-pack) node count: GPU jobs packed by GPUs-per-node, CPU jobs packed by `node_occupancy`. The result is **capped at the original allocation's node count** (parsed from `submit.sh`) — pbx will never silently request more than the user originally asked for.

If the original allocation used a complex PBS select string (e.g. `select=2:ncpus=32:ngpus=4`) that can't be parsed as a plain integer, the cap is skipped — your edits to `respawn_template.sh` are then the only safeguard.

---

## Chain termination

A chain ends — no further auto-resubmission — when **any** of:

1. **`--respawn 0`** at walltime. In-flight jobs go to `Failed`. (See [Link 3](#link-3-last-link---respawn-0).)
2. **Zero in-flight jobs at walltime.** Everything finished as `Done`/`Failed` already — there is nothing to resubmit for.
3. **`qsub` / `sbatch` returns non-zero** at resubmit. The orchestrator logs full stderr + a "chain stopped" note. Most common cause: the site does not allow scheduler commands from inside compute jobs.
4. **`qsub` / `sbatch` not found on the compute node.** Same outcome as 3.
5. **Template validation fails.** A required arg was edited out of the `pbx run` line.
6. **External SIGTERM.** `pbx qdel <jobid>` / `pbx scancel <jobid>` always stops the chain.

In every case the previously-running link still marks its in-flight jobs as `Restart` (or `Killed` for SIGTERM and case 1) and exits cleanly. You can pick up where it left off with `pbx qsub --respawn M ...` and the next link's startup hook will process those `Restart` rows.

---

## Quick reference: which status, when?

| Trigger | No `--respawn` | `--respawn N > 0` | `--respawn 0` |
| --- | --- | --- | --- |
| Walltime expiry | in-flight → `Killed` | in-flight → `Restart` + resubmit | in-flight → `Failed` |
| `pbx qdel` / `pbx scancel` | in-flight → `Killed` | in-flight → `Killed`, chain stops | in-flight → `Killed`, chain stops |
| App reports success | `Done` | `Done` | `Done` |
| App reports failure / exits non-zero | `Failed` | `Failed` | `Failed` |
| App returns unknown status | `Warning` | `Warning` | `Warning` |
| Parents not `Done` yet | stays `Ready` | stays `Ready` | stays `Ready` |
| Startup, job is `Restart` | `app.restart()` called → Patched/Re-run/Failed | `app.restart()` called → Patched/Re-run/Failed | `app.restart()` called → Patched/Re-run/Failed |
