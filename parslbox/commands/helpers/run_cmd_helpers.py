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
VALID_JOB_STATUSES = ["ready", "done", "failed", "killed", "restart", "running", "submitted", "warning"]


_RESTART_BANNER_TEMPLATE = (
    "\n{bar}\n=== Restart {ts}\n{bar}\n\n"
)


def choose_output_mode(
    job_id: int,
    current_status: str,
    restarting_job_ids: Set[int],
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
        bar=bar, ts=datetime.now().isoformat()
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


def should_re_dispatch_known_job(tracker_status: str, db_status: str) -> bool:
    """Decide whether a job already in the dispatch loop's `known_job_ids`
    should be re-dispatched when dynamic discovery sees it back in a
    runnable status in the DB.

    Re-dispatch when ALL three hold:
      - tracker shows a settled state (Done / Failed / Warning / Killed / Ready) —
        no live Parsl future for this job
      - DB shows it back in a runnable status (Ready / Restart) — the user
        (or an external script) set it to be run again
      - NOT in flight (Submitted / Running) — a live future exists; let it
        finish on its own path, the DB flip is a no-op for this link

    Re-discovery routes the job into `new_jobs`, which is what makes the
    per-job `apply_restart_for_job` call inside create_parsl_future fire on
    `Restart`-status entries (so they get added to `restarting_job_ids` and
    their stdout/stderr opens in 'a'+banner mode).

    Background: the previous implementation only handled the single combo
    (tracker=Failed AND db=Ready), silently dropping every other valid
    re-dispatch intent — including the common case where the user flips a
    Ready/Done/Killed job back to Restart mid-run to resume from a
    checkpoint. See test_run_dynamic_rediscovery.py for the full table.
    """
    tracker_settled = tracker_status in ('Done', 'Failed', 'Warning', 'Killed', 'Ready')
    db_runnable = db_status in ('Ready', 'Restart')
    in_flight = tracker_status in ('Submitted', 'Running')
    return tracker_settled and db_runnable and not in_flight


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
                            respawn_ctx=None):
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
        )
        sys.exit(130)

    return handler


def perform_shutdown(status_buffer, job_tracker, parsl_loaded_flag, logger,
                     reason: str, cleanup_parsl: bool, respawn_ctx=None):
    """
    Shared shutdown sequence used by both the signal handler and the
    walltime-triggered check in the main loop.

    Ordering:
      STEP 0: snapshot active job IDs from in-memory JobTracker.
      STEP 1: flush the status buffer (writes any pending Done/Failed).
      STEP 2: mark active jobs.
              - respawn_ctx with respawn > 0 and reason=="walltime"
                → mark `Restart` (chain continues).
              - respawn_ctx with respawn == 0 and reason=="walltime"
                → mark `Failed` ("chain exhausted: respawn count reached 0").
              - otherwise → mark `Killed` (existing behavior).
      STEP 3: Parsl cleanup — only when cleanup_parsl=True (walltime path).
              Skipped for the signal path because the scheduler will tear
              down the cgroup anyway and HTE shutdown can be slow.
      STEP 4: Respawn auto-resubmit — only when respawn_ctx with respawn > 0
              and we just marked >= 1 job as Restart.

    Args:
        status_buffer: StatusBuffer instance.
        job_tracker: JobTracker instance (source of active job IDs).
        parsl_loaded_flag: Dict with 'loaded' key.
        logger: Logger instance.
        reason: Human-readable trigger (e.g. "signal SIGTERM", "walltime").
        cleanup_parsl: If True, call parsl.dfk().cleanup().
        respawn_ctx: Optional dict with keys: respawn (int), run_dir (Path),
            system_config (object), db_path (Path), app_filter (set or None),
            tag_filter (set or None). When present AND reason == "walltime",
            triggers the respawn branching for Step 2 / Step 4.
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

    # STEP 0: snapshot active job IDs.
    active_ids = []
    if job_tracker:
        try:
            active_jobs = (
                job_tracker.get_jobs_by_status("Running")
                + job_tracker.get_jobs_by_status("Submitted")
            )
            active_ids = [j['job_id'] for j in active_jobs]
            logger.info(f"Shutdown: Step 0 - Found {len(active_ids)} active jobs")
        except Exception as e:
            logger.error(f"Shutdown: Step 0 - Failed to get active jobs: {e}")

    # STEP 1: flush status buffer (preserves any pending Done/Failed).
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

    # STEP 2: mark active jobs (Killed by default; Restart or Failed under
    # respawn + walltime).
    restart_marked = 0
    if active_ids:
        if respawn_ctx is not None and reason == "walltime":
            if respawn_ctx["respawn"] > 0:
                target_status = "Restart"
            else:
                target_status = "Failed"
        else:
            target_status = "Killed"
        try:
            database.update_jobs(status_buffer.db_path,
                                 job_ids=active_ids, status=target_status)
            logger.info(f"Shutdown: Step 2 - Marked {len(active_ids)} "
                        f"active jobs as {target_status}")
            if target_status == "Restart":
                restart_marked = len(active_ids)
            elif target_status == "Failed" and respawn_ctx is not None:
                logger.warning(
                    "Shutdown: Step 2 - chain exhausted (respawn reached 0); "
                    "these jobs were marked Failed. Manually flip them to Restart "
                    "and resubmit if you want to continue."
                )
        except Exception as e:
            logger.error(f"Shutdown: Step 2 - Failed to mark active jobs: {e}")

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
