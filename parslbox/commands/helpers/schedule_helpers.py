"""
Scheduling / dispatch engine for `pbx run`.

Separates the two concerns that used to be tangled in run.py:

  - **claim** (a DB status flip, concurrency-safe) vs
  - **schedule** (in-memory: assign resources, build the Parsl future).

Dynamic mode (default, shared-DB, concurrent runs): each pass re-queries the DB
for runnable jobs that fit current free capacity, atomically claims a coarse
capacity-sized pick (stamped with this run's owner id), assigns + dispatches only
what it won, and reverts the rare post-claim assign miss. No backlog.

Static mode (single owner): all runnable jobs are claimed up front into the
resource manager's in-memory backlog, which is then drained (dep-ready + fits)
as jobs finish.

`assign_resources` is the single authority on "does it fit"; the coarse pick just
bounds how much we claim.
"""

import logging
import os
import time
from concurrent.futures import as_completed, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from parslbox.database import database
from parslbox.resource_manager.exceptions import InsufficientResources
from parslbox.resource_manager.models import create_job_resource_spec
from parslbox.resource_manager.mpi_command_builder import (
    build_mpi_command,
    build_single_rank_launcher,
)
from parslbox.commands.helpers.run_cmd_helpers import (
    choose_output_mode,
    should_gate_dispatch,
    validate_and_normalize_status,
)
from parslbox.commands.helpers.hook_dispatch import dispatch_hook_on_compute

logger = logging.getLogger(__name__)


@dataclass
class SchedulerContext:
    """Bundle of shared handles threaded through the dispatch functions."""
    db_path: Path
    scheduler: str
    owner: str                       # this run's sched_job_id (claim owner token)
    config_name: str
    resource_manager: object
    job_tracker: object
    status_buffer: object
    system_config: object
    app_instances: dict
    app_configs: dict
    mpi_configs: dict
    restarting_job_ids: set
    fut_to_item: dict
    shutdown_at: float
    full_yaml_config: dict = field(default_factory=dict)

    def remaining_walltime(self) -> float:
        return self.shutdown_at - time.time()


def coarse_pick(candidates: List[dict], budget: int) -> List[dict]:
    """Greedily take candidates whose cumulative `num_nodes` fits `budget`.

    Node-level and lax by design (sub-node packing is discovered later by
    `assign_resources` + the refill loop). Iteration order is preserved
    (Restart-first, smallest-first from `get_runnable_jobs`).
    """
    pick: List[dict] = []
    used = 0
    for job in candidates:
        if used >= budget:
            break
        n = max(1, int(job.get('num_nodes', 1) or 1))
        if used + n <= budget:
            pick.append(job)
            used += n
    return pick


def _ensure_app_loaded(app_name: str, ctx: SchedulerContext) -> bool:
    """Load app instance / config / MPI config on demand (dynamic discovery of
    an app type not present at startup). Returns True if usable."""
    if app_name in ctx.app_instances:
        return True
    try:
        from parslbox.apps.app_registry import get_app_instance
        from parslbox.utils.pbx_config_utils import load_app_config
        from parslbox.resource_manager.mpi_config import load_mpi_config

        ctx.app_instances[app_name] = get_app_instance(app_name)
        ctx.app_configs[app_name] = load_app_config(
            app_name=app_name, system_name=ctx.config_name
        )
        ctx.mpi_configs[app_name] = load_mpi_config(
            system_name=ctx.config_name,
            app_name=app_name,
            system_config_class=ctx.system_config,
            yaml_config=ctx.full_yaml_config,
        )
        logger.info(f"Loaded app context for dynamically discovered app '{app_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to load app '{app_name}': {e}")
        return False


def create_parsl_future(job: dict, ctx: SchedulerContext, is_restart: bool) -> bool:
    """Build and submit the Parsl future for an already-claimed, already-assigned
    job. Buffers the Running status (plus any restart() patch) on success, or
    frees resources + buffers Failed on error. Returns True on success.

    The job must already be claimed (DB status Submitted/Resubmitted, owned by
    this run) and have a resource assignment. `is_restart` selects restart()
    handling and append-vs-write output mode.
    """
    job_id = job['job_id']
    app_name = job['app']
    job_path = Path(job['path'])

    app_instance = ctx.app_instances[app_name]
    app_config = ctx.app_configs[app_name]
    mpi_config = ctx.mpi_configs[app_name]
    resource_manager = ctx.resource_manager
    status_buffer = ctx.status_buffer

    # Carries restart() patch fields (if any) into the buffered Running write.
    pending_patch: dict = {}

    try:
        assignment = resource_manager.get_job_assignment(job_id)
        if not assignment:
            raise ValueError(f"No resource assignment found for job {job_id}")

        job_spec = create_job_resource_spec(job)

        mpi_commands = build_mpi_command(
            mpi_config=mpi_config,
            system_config=ctx.system_config,
            assignment=assignment,
            job_spec=job_spec,
            job_path=str(job_path),
        )
        if mpi_config.env_setup:
            mpi_commands['PBX_MPI_ENV_SETUP'] = mpi_config.env_setup
        mpi_commands['PBX_MPI_BACKEND'] = mpi_config.backend.value
        mpi_commands['PBX_GPU_TILE_MODE'] = 'tile' in type(ctx.system_config).__name__.lower()

        hostnames = list(assignment.hostnames)
        if mpi_config.use_short_hostnames:
            hostnames = [h.split('.')[0] for h in hostnames]
        mpi_commands['PBX_HOSTNAMES'] = ','.join(hostnames)
        mpi_commands['PBX_CORES_PER_NODE'] = str(ctx.system_config.CORES_PER_NODE)
        mpi_commands['PBX_RANKS_PER_NODE'] = str(job_spec.ranks_per_node)

        logger.info(f"Job {job_id}: Generated MPI command - {mpi_commands.get('PBX_MPI_PREFIX', 'None')}")

        hook_gpu_env_vars = None
        if app_instance.RUN_HOOKS_ON_COMPUTE:
            single_rank_launcher = build_single_rank_launcher(
                mpi_config=mpi_config,
                assignment=assignment,
            )
            mpi_commands['PBX_SINGLE_RANK_LAUNCHER'] = single_rank_launcher
            logger.info(f"Job {job_id}: Single-rank launcher - {single_rank_launcher}")
            hook_gpu_env_vars = assignment.get_env_vars(
                mpi_backend=mpi_commands.get('PBX_MPI_BACKEND'),
                tile_mode=mpi_commands.get('PBX_GPU_TILE_MODE', False),
            )

        # Lazy per-job restart() — after resources are assigned, before dispatch.
        if is_restart:
            from parslbox.commands.helpers.restart_helpers import apply_restart_for_job
            bucket, patch, err = apply_restart_for_job(job, app_instance, logger)
            if bucket == 'failed':
                raise RuntimeError(f"restart() failed: {err}")
            ctx.restarting_job_ids.add(job_id)
            if bucket == 'patched':
                job.update(patch)
                pending_patch = patch
                logger.info(f"Job {job_id}: restart() patched fields {sorted(patch)}")
            else:
                logger.info(f"Job {job_id}: restart() returned no patch (rerun as-is)")

        logger.info(f"Running preprocessing for Job ID {job_id}...")
        if app_instance.RUN_HOOKS_ON_COMPUTE:
            dispatch_hook_on_compute(
                app_name=job['app'],
                method_name='preprocess',
                single_rank_launcher=mpi_commands['PBX_SINGLE_RANK_LAUNCHER'],
                env_file=job.get('env_file'),
                gpu_env_vars=hook_gpu_env_vars,
                job_id=job_id,
                job_path=job_path,
                db_path=ctx.db_path,
                app_config=app_config,
                config_name=ctx.config_name,
            )
        else:
            app_instance.preprocess(
                job_id=job_id,
                job_path=job_path,
                db_path=ctx.db_path,
                app_config=app_config,
                config_name=ctx.config_name,
            )

        stdout_path = job_path / f"pbx_job_{job_id}.out"
        stderr_path = job_path / f"pbx_job_{job_id}.err"
        file_mode = choose_output_mode(
            job_id,
            'Restart' if is_restart else 'Ready',
            ctx.restarting_job_ids,
            ctx.remaining_walltime(),
            stdout_path, stderr_path,
        )

        fut = app_instance.parsl_app(
            job_id=job_id,
            job_path=job_path,
            db_path=ctx.db_path,
            assignment=assignment,
            mpi_commands=mpi_commands,
            app_config=app_config,
            config_name=ctx.config_name,
            in_file=job['in_file'],
            mpi_opts=job['mpi_opts'],
            env_file=job.get('env_file'),
            stdout=(str(stdout_path), file_mode),
            stderr=(str(stderr_path), file_mode),
        )

        ctx.fut_to_item[fut] = {
            'future': fut,
            'job': job,
            'app_instance': app_instance,
            'assignment': assignment,
            'resource_manager': resource_manager,
            'single_rank_launcher': mpi_commands.get('PBX_SINGLE_RANK_LAUNCHER'),
            'env_file': job.get('env_file'),
            'gpu_env_vars': hook_gpu_env_vars,
        }

        logger.info(f"Job {job_id}: Successfully created Parsl future")

        ctx.job_tracker.update_job_status(job_id, 'Running')
        status_buffer.add_status_update(job_id, status='Running', **pending_patch)
        return True

    except Exception as e:
        logger.error(f"Job {job_id}: Failed to submit Parsl app: {e}")
        resource_manager.free_resources_with_health_check(
            job_id, job_succeeded=False, error_message=str(e)
        )
        ctx.job_tracker.update_job_status(job_id, "Failed")
        status_buffer.add_status_update(job_id, status="Failed")
        return False


def _try_dispatch(job: dict, ctx: SchedulerContext, is_restart: bool) -> str:
    """Assign resources to an owned job and build its future.

    Returns 'dispatched' | 'nofit' (InsufficientResources — caller decides
    revert vs keep-in-backlog) | 'failed' (buffered Failed already).
    """
    job_id = job['job_id']
    try:
        ctx.resource_manager.assign_resources(job)
    except InsufficientResources as e:
        logger.debug(f"Job {job_id}: does not fit current free resources: {e}")
        return 'nofit'
    except Exception as e:
        logger.error(f"Job {job_id}: Failed to allocate resources: {e}")
        ctx.job_tracker.update_job_status(job_id, "Failed")
        ctx.status_buffer.add_status_update(job_id, status="Failed")
        return 'failed'

    ok = create_parsl_future(job, ctx, is_restart)
    if ok:
        time.sleep(float(os.getenv("PBX_RUN_DELAY", "0.2")))
        return 'dispatched'
    return 'failed'


def dispatch_dynamic(
    ctx: SchedulerContext,
    apps: Optional[List[str]],
    tags: Optional[List[str]],
) -> int:
    """One dynamic dispatch pass: fill free capacity from the DB, claiming a
    coarse pick each inner round and refilling until saturated, dry, or a round
    wins nothing. Returns the number of jobs dispatched.
    """
    total_dispatched = 0

    while True:
        budget = ctx.resource_manager.free_node_capacity()
        if budget <= 0:
            break

        candidates = database.get_runnable_jobs(
            ctx.db_path, apps=apps, tags=tags, max_nodes=budget
        )
        # Skip jobs this run is already running, deps not satisfied, or gated
        # by the app's walltime floor.
        remaining = ctx.remaining_walltime()
        eligible = []
        for job in candidates:
            if job['job_id'] in ctx.fut_to_item:
                continue
            if not _ensure_app_loaded(job['app'], ctx):
                _fail_job(job['job_id'], ctx)
                continue
            if not ctx.job_tracker.parents_satisfied(job):
                continue
            if should_gate_dispatch(ctx.app_instances[job['app']], job, remaining):
                continue
            eligible.append(job)

        if not eligible:
            break

        pick = coarse_pick(eligible, budget)
        ready_ids = [j['job_id'] for j in pick if j['status'] == 'Ready']
        restart_ids = [j['job_id'] for j in pick if j['status'] == 'Restart']

        owned_ids = set(database.claim_jobs(ctx.db_path, ready_ids, restart_ids, ctx.owner))
        if not owned_ids:
            break

        round_dispatched = 0
        for job in pick:
            job_id = job['job_id']
            if job_id not in owned_ids:
                continue  # lost the race to another run
            is_restart = job['status'] == 'Restart'
            ctx.job_tracker.register_jobs([job])
            ctx.job_tracker.update_job_status(
                job_id, 'Resubmitted' if is_restart else 'Submitted'
            )
            outcome = _try_dispatch(job, ctx, is_restart)
            if outcome == 'dispatched':
                round_dispatched += 1
            elif outcome == 'nofit':
                # Coarse pick over-claimed (sub-node packing tighter than
                # node-count implied) — hand it back so another/next pass can
                # take it. Rare.
                database.revert_claims(ctx.db_path, [job_id], owner=ctx.owner)
                ctx.job_tracker.update_job_status(job_id, job['status'])
            # 'failed' already buffered Failed

        total_dispatched += round_dispatched
        if round_dispatched == 0:
            break  # won nothing usable this round — stop, let the caller wait

    return total_dispatched


def dispatch_static(ctx: SchedulerContext) -> int:
    """One static dispatch pass: drain dep-ready, fitting jobs from the resource
    manager's in-memory backlog (all jobs were claimed up front). Jobs that
    don't fit stay in the backlog for a later pass. Returns count dispatched.
    """
    rm = ctx.resource_manager
    remaining = ctx.remaining_walltime()

    ready_jobs = rm.get_dependency_ready_jobs_from_backlog()
    if not ready_jobs:
        return 0

    eligible = [
        j for j in ready_jobs
        if not should_gate_dispatch(ctx.app_instances[j['app']], j, remaining)
    ]
    if not eligible:
        return 0

    budget = rm.free_node_capacity()
    pick = coarse_pick(eligible, budget) if budget > 0 else []

    dispatched = 0
    for job in pick:
        job_id = job['job_id']
        is_restart = job['status'] == 'Restart'
        outcome = _try_dispatch(job, ctx, is_restart)
        if outcome == 'dispatched':
            rm._backlogged_jobs_set.discard(job_id)
            dispatched += 1
        elif outcome == 'failed':
            rm._backlogged_jobs_set.discard(job_id)
        # 'nofit' → leave in backlog for a later pass
    return dispatched


def handle_completion(fut, ctx: SchedulerContext) -> None:
    """Process one finished future: determine final status (check_success →
    postprocess), buffer it, and free resources with health tracking."""
    item = ctx.fut_to_item.pop(fut)
    job = item['job']
    app_instance = item['app_instance']
    resource_manager = item['resource_manager']
    job_id = job['job_id']
    job_path = Path(job['path'])
    db_path = ctx.db_path

    error_message = None
    try:
        fut.result()
        logger.info(f"Job {job_id}: Execution completed without errors.")
    except Exception as e:
        error_message = str(e)
        logger.error(f"Job {job_id}: Execution error occurred: {error_message}")

    job_status = app_instance.check_success(
        job_id=job_id, job_path=job_path, db_path=db_path, error_message=error_message
    )
    job_status = validate_and_normalize_status(job_status, job_id)

    if job_status == "Done":
        logger.info(f"Job {job_id}: Success check passed. Running post-processing...")
        try:
            if app_instance.RUN_HOOKS_ON_COMPUTE:
                final_status = dispatch_hook_on_compute(
                    app_name=job['app'],
                    method_name='postprocess',
                    single_rank_launcher=item['single_rank_launcher'],
                    env_file=item['env_file'],
                    gpu_env_vars=item['gpu_env_vars'],
                    job_id=job_id,
                    job_path=job_path,
                    db_path=db_path,
                )
            else:
                final_status = app_instance.postprocess(
                    job_id=job_id, job_path=job_path, db_path=db_path,
                )
            if final_status:
                job_status = validate_and_normalize_status(final_status, job_id)
        except Exception as e:
            logger.error(f"Job {job_id}: Post-processing failed: {e}")
            job_status = "Failed"

    ctx.job_tracker.update_job_status(job_id, job_status)
    ctx.status_buffer.add_status_update(job_id, status=job_status)
    logger.info(f"Job {job_id}: Final status = {job_status} (buffered)")

    if job_status in ('Done', 'Failed', 'Warning'):
        ctx.restarting_job_ids.discard(job_id)

    try:
        job_succeeded = (job_status == "Done")
        stderr_error = None
        if not job_succeeded:
            stderr_file = job_path / f"pbx_job_{job_id}.err"
            try:
                if stderr_file.is_file():
                    stderr_content = stderr_file.read_text(errors='replace')
                    stderr_tail = '\n'.join(stderr_content.splitlines()[-50:])
                    if stderr_tail.strip():
                        stderr_error = stderr_tail
            except Exception as e:
                logger.debug(f"Job {job_id}: Could not read stderr file: {e}")

        resource_manager.free_resources_with_health_check(
            job_id=job_id, job_succeeded=job_succeeded, error_message=stderr_error
        )
    except Exception as e:
        logger.error(f"Job {job_id}: Failed to free resources: {e}")


def _fail_job(job_id: int, ctx: SchedulerContext) -> None:
    """Mark a job Failed in tracker + buffer (used when its app can't load)."""
    ctx.job_tracker.register_jobs(
        [{'job_id': job_id, 'status': 'Failed', 'parents': None}]
    )
    ctx.job_tracker.update_job_status(job_id, "Failed")
    ctx.status_buffer.add_status_update(job_id, status="Failed")


def drain_completions(ctx: SchedulerContext, poll_interval: float) -> int:
    """Process every future that has *already* finished; block only if none has.

    Why drain instead of handling one at a time: the caller flushes the status
    buffer once per loop pass, so the number of DB writes equals the number of
    passes, not the number of jobs. Handling a single completion per pass made
    every flush carry exactly one row — `update_jobs` is one
    `UPDATE ... WHERE job_id IN (...)` (one connection, one commit) regardless
    of how many ids it carries, so a row is nearly free but a *write* is not.
    At 24k jobs that was 24k connection-opens + WAL commits on a shared
    filesystem instead of a handful.

    Batch size self-tunes and needs no tuning knob: slower flushes let more
    completions pile up, which makes the next batch bigger, which cuts the
    flush count back down.

    Trade-off — re-dispatch delay: the caller cannot refill freed capacity until
    this returns, so a large batch keeps newly-freed nodes idle for the length
    of the drain. Resources are released progressively inside
    `handle_completion` (see `free_resources_with_health_check`), so nothing is
    lost, only deferred. This is deliberately unbounded: capping the batch would
    trade a guaranteed write amplification for a speculative utilisation gain
    that only materialises when there is backlogged work waiting.

    Returns the number of completions handled.
    """
    # Snapshot: handle_completion pops from ctx.fut_to_item, so iterating the
    # live dict would raise "dictionary changed size during iteration".
    done_now = [fut for fut in list(ctx.fut_to_item) if fut.done()]

    if not done_now:
        # Nothing ready — fall back to the original blocking wait so the caller
        # re-checks walltime / re-queries for new work at a bounded cadence.
        try:
            for fut in as_completed(list(ctx.fut_to_item.keys()), timeout=poll_interval):
                if fut not in ctx.fut_to_item:
                    continue
                handle_completion(fut, ctx)
                return 1
        except FutureTimeoutError:
            pass
        return 0

    handled = 0
    for fut in done_now:
        if fut not in ctx.fut_to_item:
            continue
        handle_completion(fut, ctx)
        handled += 1
        # Walltime guard: the caller's shutdown check only runs between passes,
        # so a long drain could sail past shutdown_at and lose the grace period
        # needed to reconcile state. Bail out and let the caller's top-of-loop
        # check fire; anything handled so far is already in the buffer and gets
        # flushed by perform_shutdown's STEP 1.
        if ctx.remaining_walltime() <= 0:
            break
    return handled


def check_run_complete(ctx: SchedulerContext, dynamic: bool) -> Optional[str]:
    """Decide whether the run is finished, flushing pending state before giving up.

    Returns:
        None       - futures still in flight; keep going.
        'continue' - nothing running, but buffered updates were just written to
                     the DB. Take one more dispatch pass before concluding.
        'break'    - nothing running and nothing pending; the run is done.

    The 'continue' case is what makes deferred writes safe. Dependency checks
    (`parents_satisfied`) read parent status from the DB, not from the buffer.
    If the last running job is a parent, its completion is buffered and its
    child still looks un-ready — so without this guard the no-idle exit would
    fire and the child would be silently skipped (fan-in / aggregator jobs).
    Flushing here and taking one more pass turns that into a non-event.

    Terminates: the extra pass either dispatches something (futures non-empty,
    so we never reach this branch) or dispatches nothing, in which case the
    buffer is now empty, `flush_all()` short-circuits to 0, and we break.
    """
    if ctx.fut_to_item:
        return None

    if ctx.status_buffer.flush_all() > 0:
        return 'continue'

    # Static orphans — claimed up front but never became runnable (e.g. a
    # failed parent, or too big to ever fit). Return them to the pool rather
    # than stranding them in Submitted/Resubmitted. Reads the backlog set
    # directly; get_resource_status()['backlogged_jobs'] is the same value but
    # scans every node to build a dict we'd throw away.
    if not dynamic and ctx.resource_manager._backlogged_jobs_set:
        orphans = list(ctx.resource_manager._backlogged_jobs_set)
        reverted = database.revert_claims(ctx.db_path, orphans, owner=ctx.owner)
        logger.info(
            f"Static: reverted {reverted} un-runnable claimed job(s) to Ready/Restart."
        )

    return 'break'
