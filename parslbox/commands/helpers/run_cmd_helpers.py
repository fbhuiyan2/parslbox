import logging
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Set

import parsl
from parslbox.database import database

# Valid job status values (stored in lowercase for comparison)
VALID_JOB_STATUSES = ["ready", "done", "failed", "killed", "restart", "running", "submitted", "resubmitted", "warning"]


_RESTART_BANNER_TEMPLATE = (
    "\n{bar}\n=== Restart {ts}\n=== Remaining Walltime: {wt}\n{bar}\n\n"
)


def _fmt_remaining_walltime(s: float) -> str:
    """Format remaining walltime seconds as HH:MM:SS (clamped at 0)."""
    s = max(0, int(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def choose_output_mode(
    job_id: int,
    current_status: str,
    restarting_job_ids: Set[int],
    remaining_walltime_s: float,
    *paths: Path,
) -> str:
    """Decide 'a' vs 'w' for a job's stdout/stderr file mode.

    A run is a restart-continuation when EITHER:
      - the job is in `restarting_job_ids` (added by `apply_restart_for_job`
        inside create_parsl_future when the job's `restart()` hook returned
        a successful bucket), OR
      - the DB still shows status='Restart' at the moment we open the file
        (covers any future code path that puts a Restart-status job into
        create_parsl_future without routing through the per-job hook).

    For restart-continuations: returns 'a'. For each path that already exists
    and has content, appends a banner line so the boundary between runs is
    obvious when reading the file.

    For fresh runs (Ready / not in the set): returns 'w'. Any stale content
    from a prior life (e.g., the user reset a Failed job to Ready) is
    intentionally clobbered.
    """
    is_restart_continuation = (
        job_id in restarting_job_ids
        or current_status == 'Restart'
    )
    if not is_restart_continuation:
        return 'w'
    bar = "=" * 70
    banner = _RESTART_BANNER_TEMPLATE.format(
        bar=bar,
        ts=datetime.now().isoformat(),
        wt=_fmt_remaining_walltime(remaining_walltime_s),
    )
    for p in paths:
        if p.exists() and p.stat().st_size > 0:
            with open(p, 'a') as f:
                f.write(banner)
    return 'a'


def should_gate_dispatch(
    app,
    job_dict: dict,
    remaining_walltime_s: float,
) -> bool:
    """Decide whether a job should be SKIPPED at dispatch time because the
    batch's remaining walltime is below the app's declared floor.

    Returns True (skip dispatch — leave job in DB Ready/Restart) when BOTH:
      - app declares a positive floor via min_remaining_walltime(job_dict)
      - remaining_walltime_s < that floor

    Universal: applies whether --respawn is on or off, and regardless of
    whether the job is a fresh Ready job or a Restart-status job (the
    restart() hook is now called lazily AFTER this gate inside
    create_parsl_future, so there's nothing to "protect" pre-gate). Under
    --respawn, if all remaining work gets gated and futures drain, pbx run
    exits via the existing exit-check path; chain continuation depends on
    the walltime trigger, which this gate does not affect.
    """
    floor = app.min_remaining_walltime(job_dict)
    if floor <= 0:
        return False
    return remaining_walltime_s < floor


def get_default_run_dir() -> Path:
    """Generate default run directory with current time and date in hhmmss_ddmmyy format."""
    now = datetime.now()
    time_str = now.strftime("%H%M%S")  # hhmmss format (hours + minutes + seconds)
    date_str = now.strftime("%d%m%y")  # ddmmyy format
    dir_name = f"{time_str}_{date_str}"
    return Path.home() / ".parslbox" / "runs" / dir_name # / "log.pbx"

def validate_and_normalize_status(status: str, job_id: int = None) -> str:
    """
    Validate job status and return normalized (capitalized) version.
    If invalid, return 'Warning' and log the issue.
    
    Args:
        status: Status string to validate
        job_id: Optional job ID for logging context
        
    Returns:
        Normalized status string (capitalized)
    """
    logger = logging.getLogger(__name__)
    
    if not status or not isinstance(status, str):
        if job_id:
            logger.warning(f"Job {job_id}: Invalid status type '{type(status)}' with value '{status}'. Setting to 'Warning'.")
        else:
            logger.warning(f"Invalid status type '{type(status)}' with value '{status}'. Setting to 'Warning'.")
        return "Warning"
    
    status_lower = status.lower().strip()
    if status_lower in VALID_JOB_STATUSES:
        return status_lower.capitalize()
    else:
        if job_id:
            logger.warning(f"Job {job_id}: Invalid status '{status}' returned. Valid statuses are: {VALID_JOB_STATUSES}. Setting to 'Warning'.")
        else:
            logger.warning(f"Invalid status '{status}' provided. Valid statuses are: {VALID_JOB_STATUSES}. Setting to 'Warning'.")
        return "Warning"


def parse_parents(parents_str):
    """Parse parents string into list of job IDs."""
    if not parents_str:
        return []
    
    return [int(x) for x in json.loads(parents_str)]


# NOTE: are_parents_done() and get_dependency_ready_jobs() functions have been removed
# as they are now handled by JobTracker for better performance and to eliminate database reads


def create_shutdown_handler(status_buffer, logger, parsl_loaded_flag, job_tracker=None,
                            respawn_ctx=None, owner=None):
    """
    Create a signal handler that flushes the status buffer on termination.

    This handler is called when the batch job receives SIGTERM (typically when
    walltime is exceeded) or SIGINT (manual interruption). It ensures that all
    buffered status updates are written to the database before the process exits.

    Includes timeout protection via signal.alarm() to force exit if cleanup hangs.

    External-SIGTERM path always marks active jobs `Killed` and does NOT
    auto-resubmit, regardless of respawn mode. (Respawn auto-resubmit is only
    triggered by the walltime path in the main loop.)

    Args:
        status_buffer: StatusBuffer instance to flush
        logger: Logger instance for logging
        parsl_loaded_flag: Dict with 'loaded' key tracking Parsl state
        job_tracker: JobTracker instance (source of active job IDs)
        respawn_ctx: Ignored on the signal path — kept for symmetry with
            the walltime call; this path always marks Killed without resubmit.

    Returns:
        Signal handler function
    """
    def handler(signum, frame):
        signal_name = signal.Signals(signum).name
        # Signal path: minimal budget (PBS Pro default kill_delay ~2s on many
        # sites). Skip Parsl cleanup — the scheduler will reap the cgroup.
        # Users who want a clean shutdown should use `pbx qdel` / `pbx scancel`,
        # which control the grace period themselves.
        perform_shutdown(
            status_buffer=status_buffer,
            job_tracker=job_tracker,
            parsl_loaded_flag=parsl_loaded_flag,
            logger=logger,
            reason=f"signal {signal_name}",
            cleanup_parsl=False,
            respawn_ctx=None,  # signal path never auto-resubmits
            owner=owner,
        )
        sys.exit(130)

    return handler


def perform_shutdown(status_buffer, job_tracker, parsl_loaded_flag, logger,
                     reason: str, cleanup_parsl: bool, respawn_ctx=None,
                     owner: str = None):
    """
    Shared shutdown sequence used by both the signal handler and the
    walltime-triggered check in the main loop.

    Ordering:
      STEP 1: flush the status buffer (writes any pending Running/Done/Failed)
              — done FIRST so the DB snapshot in STEP 0 is accurate (a job
              dispatched this pass is buffered Running, not yet in the DB).
      STEP 0: snapshot this run's non-terminal jobs from the DB, scoped to
              `owner` (sched_job_id), bucketed by current status.
      STEP 2: reconcile per state (only ever touches this run's own jobs):
              - Submitted   → Ready   (claimed, never ran → back to the pool)
              - Resubmitted → Restart (claimed, never ran → back to the pool)
              - Running     → Restart (walltime + respawn > 0, chain continues)
                            → Failed  (walltime + respawn == 0, chain exhausted)
                            → Killed  (signal / qdel / scancel)
      STEP 3: Parsl cleanup — only when cleanup_parsl=True (walltime path).
      STEP 4: Respawn auto-resubmit — only under respawn > 0 at walltime, when
              at least one job was reconciled into a runnable state.

    Args:
        status_buffer: StatusBuffer instance.
        job_tracker: JobTracker instance (unused for the snapshot now; kept for
            signature compatibility / future use).
        parsl_loaded_flag: Dict with 'loaded' key.
        logger: Logger instance.
        reason: Human-readable trigger (e.g. "signal SIGTERM", "walltime").
        cleanup_parsl: If True, call parsl.dfk().cleanup().
        respawn_ctx: Optional dict (respawn, run_dir, system_config, db_path,
            app_filter, tag_filter). When present AND reason == "walltime",
            triggers the respawn branching for Step 2 / Step 4.
        owner: This run's sched_job_id — the claim owner token. Reconciliation
            is scoped to it so a run never touches another orchestrator's jobs.
    """
    import time

    start_time = time.time()
    logger.warning(f"=" * 80)
    logger.warning(f"Shutdown ({reason}) - initiating sequence at {start_time}")
    logger.warning(f"=" * 80)

    # Hard deadline so a single hung step (typically the status-buffer flush
    # over a slow shared filesystem) can't sit forever. Sized to give Lustre/
    # NFS-backed SQLite room to drain pending writes — 10s was tight enough
    # that real campaigns regularly tripped it and left jobs stuck in
    # Running/Submitted. 25s comfortably fits under both the `pbx qdel`
    # default `--grace 30` and typical scheduler kill_delay settings.
    #
    # The post-cancel DB reconciliation in cancel_helpers is still the
    # authoritative backstop — this alarm just reduces how often that
    # backstop needs to fire.
    signal.alarm(25)

    db_path = status_buffer.db_path

    # STEP 1: flush status buffer FIRST — writes any pending Running/Done/Failed
    # so the DB snapshot below reflects reality (a job dispatched this pass is
    # buffered Running, not yet persisted).
    try:
        flush_start = time.time()
        count = status_buffer.flush_all()
        flush_duration = time.time() - flush_start
        if count > 0:
            logger.info(f"Shutdown: Step 1 - Flushed {count} buffered update(s) "
                        f"in {flush_duration:.3f}s")
        else:
            logger.info(f"Shutdown: Step 1 - No pending updates to flush "
                        f"({flush_duration:.3f}s)")
    except Exception as e:
        logger.error(f"Shutdown: Step 1 - Status buffer flush FAILED: {e}")

    # STEP 0: snapshot this run's non-terminal jobs from the DB, owner-scoped,
    # bucketed by current status.
    running_ids, submitted_ids, resubmitted_ids = [], [], []
    if owner:
        try:
            rows = database.get_jobs_by_sched_id(
                db_path, owner, statuses=["Running", "Submitted", "Resubmitted"]
            )
            for r in rows:
                if r['status'] == 'Running':
                    running_ids.append(r['job_id'])
                elif r['status'] == 'Submitted':
                    submitted_ids.append(r['job_id'])
                elif r['status'] == 'Resubmitted':
                    resubmitted_ids.append(r['job_id'])
            logger.info(
                f"Shutdown: Step 0 - owner={owner}: {len(running_ids)} running, "
                f"{len(submitted_ids)} submitted, {len(resubmitted_ids)} resubmitted"
            )
        except Exception as e:
            logger.error(f"Shutdown: Step 0 - Failed to snapshot owned jobs: {e}")
    else:
        logger.warning("Shutdown: Step 0 - no owner id; skipping DB reconcile")

    # STEP 2: reconcile per state (owner-scoped, so never another run's jobs).
    restart_marked = 0
    try:
        # Claimed-but-never-ran jobs go back to the pool regardless of respawn.
        reverted = database.revert_claims(
            db_path, submitted_ids + resubmitted_ids, owner=owner
        )
        if reverted:
            logger.info(
                f"Shutdown: Step 2 - reverted {reverted} claimed job(s) "
                f"(Submitted→Ready / Resubmitted→Restart)"
            )

        # Running jobs: target depends on trigger + respawn.
        if respawn_ctx is not None and reason == "walltime":
            running_target = "Restart" if respawn_ctx["respawn"] > 0 else "Failed"
        else:
            running_target = "Killed"

        if running_ids:
            database.update_jobs(db_path, job_ids=running_ids, status=running_target)
            logger.info(
                f"Shutdown: Step 2 - Marked {len(running_ids)} running job(s) as {running_target}"
            )
            if running_target == "Failed" and respawn_ctx is not None:
                logger.warning(
                    "Shutdown: Step 2 - chain exhausted (respawn reached 0); running "
                    "jobs marked Failed. Manually flip to Restart to continue."
                )

        # Chain continues if anything was reconciled into a runnable state.
        if respawn_ctx is not None and reason == "walltime" and respawn_ctx["respawn"] > 0:
            restart_marked = len(running_ids) + reverted
    except Exception as e:
        logger.error(f"Shutdown: Step 2 - Failed to reconcile jobs: {e}")

    # STEP 3: Parsl cleanup (only when we have runway).
    if cleanup_parsl:
        try:
            if parsl_loaded_flag.get('loaded', False):
                cleanup_start = time.time()
                parsl.dfk().cleanup()
                logger.info(f"Shutdown: Step 3 - Parsl cleanup complete in "
                            f"{time.time() - cleanup_start:.3f}s")
            else:
                logger.info("Shutdown: Step 3 - Parsl not loaded, skipping cleanup")
        except Exception as e:
            logger.error(f"Shutdown: Step 3 - Parsl cleanup FAILED: {e}")
    else:
        logger.info("Shutdown: Step 3 - skipping Parsl cleanup (signal path)")

    # STEP 4: Respawn auto-resubmit.
    if (respawn_ctx is not None and reason == "walltime"
            and respawn_ctx["respawn"] > 0 and restart_marked > 0):
        # Cancel the alarm before subprocess.run; respawn submission can take a
        # few seconds and we don't want to be force-killed mid-submission.
        signal.alarm(0)
        _auto_resubmit(respawn_ctx, restart_marked, logger)
    else:
        signal.alarm(0)

    total_duration = time.time() - start_time
    logger.warning(f"=" * 80)
    logger.warning(f"Shutdown complete in {total_duration:.3f}s")
    logger.warning(f"=" * 80)


def _auto_resubmit(respawn_ctx, restart_marked: int, logger):
    """Build the next per-link script from respawn_template.sh and submit it.

    Lives inside run_cmd_helpers (not restart_helpers) only so perform_shutdown
    can call it without growing its arg list — the heavy logic is in
    `restart_helpers.build_respawn_link_script` / `submit_respawn_link`.
    """
    from parslbox.commands.helpers import restart_helpers
    from parslbox.commands.helpers.resource_estimate import compute_required_nodes

    run_dir = respawn_ctx["run_dir"]
    db_path = respawn_ctx["db_path"]
    system_config = respawn_ctx["system_config"]
    current_respawn = respawn_ctx["respawn"]
    app_filter = respawn_ctx.get("app_filter")
    tag_filter = respawn_ctx.get("tag_filter")

    template_path = run_dir / "respawn_template.sh"
    submit_path = run_dir / "submit.sh"

    scheduler_command = restart_helpers.detect_scheduler_command()
    if scheduler_command is None:
        logger.error(
            "Auto-resubmission failed: cannot detect scheduler (neither "
            "PBS_JOBID nor SLURM_JOB_ID is set). Chain stopped."
        )
        return
    scheduler_type = "pbs" if scheduler_command == "qsub" else "slurm"

    # Pull the remaining runnable jobs (Restart we just marked + any Ready).
    remaining = database.get_jobs(db_path, status="Restart")
    remaining += database.get_jobs(db_path, status="Ready")
    if app_filter:
        remaining = [j for j in remaining if j['app'] in app_filter]
    if tag_filter:
        remaining = [j for j in remaining if j['tag'] in tag_filter]

    computed_nodes = compute_required_nodes(remaining, system_config)
    if computed_nodes <= 0:
        logger.info(
            f"Respawn: no remaining runnable jobs after filters. "
            f"Skipping resubmission (chain ends cleanly)."
        )
        return

    logger.info(
        f"Respawn: marked {restart_marked} jobs as Restart, "
        f"{len(remaining)} total runnable; "
        f"computed optimal nodes = {computed_nodes}; "
        f"current respawn count = {current_respawn}."
    )

    try:
        link_path = restart_helpers.build_respawn_link_script(
            template_path=template_path,
            run_dir=run_dir,
            current_respawn=current_respawn,
            scheduler_type=scheduler_type,
            computed_nodes=computed_nodes,
            submit_file_path=submit_path,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Auto-resubmission failed: {e}")
        return

    logger.info(f"Auto-resubmission: wrote {link_path.name}, submitting via {scheduler_command}.")
    restart_helpers.submit_respawn_link(link_path, scheduler_command, run_dir, logger)


def create_alarm_handler(logger):
    """
    Create a handler for SIGALRM that forces immediate exit.
    
    This handler is triggered by signal.alarm() if the shutdown sequence
    takes longer than the specified timeout (10 seconds). It forces an
    immediate exit to prevent the process from being killed by SIGKILL.
    
    Args:
        logger: Logger instance for logging
    
    Returns:
        SIGALRM handler function
    """
    def handler(signum, frame):
        import time
        logger.error("!" * 80)
        logger.error(f"ALARM TRIGGERED: Shutdown timeout exceeded (10 seconds)")
        logger.error(f"Forcing immediate exit at timestamp {time.time()}")
        logger.error("This prevents SIGKILL from terminating the process uncleanly")
        logger.error("!" * 80)
        sys.exit(143)  # Exit code 143 = 128 + 15 (SIGTERM timeout)
    
    return handler


def create_atexit_handler(status_buffer, logger):
    """
    Create an atexit handler that flushes the status buffer on normal termination.
    
    This handler is called when the Python interpreter exits normally (not via signals).
    It ensures buffered status updates are written to the database on normal program exit.
    
    Args:
        status_buffer: StatusBuffer instance to flush
        logger: Logger instance for logging
    
    Returns:
        atexit handler function
    """
    def handler():
        try:
            count = status_buffer.flush_all()
            if count > 0:
                logger.info(f"Exit cleanup: Flushed {count} job(s) to database")
        except Exception as e:
            logger.error(f"Exit cleanup: Failed to flush status buffer - {e}")
    
    return handler
