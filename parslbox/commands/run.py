import typer
import parsl
import logging
import os
import time
import signal
import atexit
import sys
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated
from concurrent.futures import as_completed, TimeoutError as FutureTimeoutError

from parslbox.system_configs.loader import load_config, get_system_config
from parslbox.utils import path_utils
from parslbox.utils.logging_utils import setup_logging, validate_log_level
from parslbox.utils.pbx_config_utils import load_app_config, is_app_configured, load_full_config
from parslbox.database import database
from parslbox.resource_manager.mpi_config import load_mpi_config
from parslbox.resource_manager.job_tracker import JobTracker

# Import helper functions
from parslbox.commands.helpers.run_cmd_helpers import (
    get_default_run_dir,
    create_shutdown_handler,
    create_alarm_handler,
    create_atexit_handler,
    perform_shutdown,
)
from parslbox.commands.helpers.schedule_helpers import (
    SchedulerContext,
    dispatch_dynamic,
    dispatch_static,
    handle_completion,
)
from parslbox.database.status_buffer import StatusBuffer

app = typer.Typer()

# Valid job status values (stored in lowercase for comparison)
VALID_JOB_STATUSES = ["ready", "done", "failed", "killed", "restart", "running", "submitted", "resubmitted", "warning"]


def get_scheduler_job_id(scheduler):
    """Get scheduler job ID from environment variables."""
    if scheduler == "PBS":
        return os.environ.get('PBS_JOBID', f'local_{int(time.time())}')
    elif scheduler == "SLURM":
        return os.environ.get('SLURM_JOB_ID', f'local_{int(time.time())}')
    else:
        return f'local_{int(time.time())}'


# Main command function

@app.command()
def run(
    config_name: Annotated[
        str,
        typer.Option("--config", "-c", help="The name of the configuration to use (e.g., 'polaris').")
    ],
    run_dir: Annotated[
        Path,
        typer.Option("--run-dir", help="The directory for Parsl run files.")
    ] = None,
    apps: Annotated[
        Optional[str],
        typer.Option("--apps", "-a", help="Comma-separated list of apps to run (e.g., 'lammps-kk,vasp').")
    ] = None,
    tags: Annotated[
        Optional[str],
        typer.Option("--tags", "-t", help="Comma-separated list of tags to run (e.g., 'run1,run2').")
    ] = None,
    retries: Annotated[
        int,
        typer.Option("--retries", help="Number of retries for failed tasks.")
    ] = 0,
    flush_interval: Annotated[
        int,
        typer.Option("--flush-interval", help="Interval in seconds for periodic status buffer flush (default: 150).")
    ] = 150,
    loglevel: Annotated[
        str,
        typer.Option("--loglevel", help="Logging level (debug, info, warning, error, critical)")
    ] = "info",
    dynamic: Annotated[
        bool,
        typer.Option("--dynamic/--static", help="Dynamically discover new jobs during run (default: dynamic).")
    ] = True,
    walltime_seconds: Annotated[
        int,
        typer.Option(
            "--walltime-seconds",
            help=(
                "Batch job walltime in seconds. Required. The orchestrator triggers "
                "graceful shutdown before this elapses so in-flight jobs can be "
                "marked Killed cleanly (30s grace, or 90s under --respawn). "
                "Set automatically by `pbx qsub`/`pbx sbatch`."
            ),
        )
    ] = ...,
    respawn: Annotated[
        Optional[int],
        typer.Option(
            "--respawn",
            help=(
                "Internal: enable the self-respawn chain. Set automatically by "
                "`pbx qsub --respawn N` / `pbx sbatch --respawn N`. The integer "
                "is the number of remaining auto-resubmissions; decremented at "
                "every link by the orchestrator. At walltime, in-flight jobs are "
                "marked Restart and the next link is auto-submitted (when > 0); "
                "when 0, the chain ends and walltime-killed jobs go Failed."
            ),
        )
    ] = None,
):
    """
    Run Parsl workflows by discovering and executing application plugins.
    """
    # Initialization
    # Use default run directory if not provided
    if run_dir is None:
        run_dir = get_default_run_dir()

    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "log.pbx"

    # Validate and convert log level
    try:
        log_level_int = validate_log_level(loglevel)
    except ValueError as e:
        typer.secho(f"❌ Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    setup_logging(log_file=log_file, log_level=log_level_int)
    logger = logging.getLogger(__name__)

    db_path = path_utils.DB_FILE

    logger.info("--- parslbox orchestrator starting ---")
    logger.info(f"Using Parsl run directory: {run_dir.resolve()}")

    # Log environment variable status for debugging
    import os
    if os.getenv("PBX_DB_PATH"):
        logger.info(f"Using custom database path: {db_path}")
    else:
        logger.info(f"Using default database path: {db_path}")

    if os.getenv("PBX_CONFIG_PATH"):
        logger.info(f"Using custom config path: {path_utils.PBX_CONFIG_FILE}")
    else:
        logger.info(f"Using default config path: {path_utils.PBX_CONFIG_FILE}")

    logger.info(f"Job submission delay: {os.getenv('PBX_RUN_DELAY', '0.2')}s (PBX_RUN_DELAY)")

    # Walltime-aware graceful shutdown: trigger before walltime so the main
    # loop can flush state and mark in-flight jobs Killed cleanly. Respawn-mode
    # needs more runway (mark Restart + generate next-link script + qsub/sbatch
    # from the compute node), so we widen the grace from 30s to 90s.
    SHUTDOWN_GRACE_SECONDS = 90 if respawn is not None else 30
    process_start = time.time()
    shutdown_at = process_start + walltime_seconds - SHUTDOWN_GRACE_SECONDS
    logger.info(
        f"Walltime: {walltime_seconds}s; graceful shutdown will trigger at "
        f"~{walltime_seconds - SHUTDOWN_GRACE_SECONDS}s ({SHUTDOWN_GRACE_SECONDS}s grace)"
    )

    # Job Fetching and Filtering
    # The job count will be passed to the config for dynamic worker allocation
    app_filter = set(apps.split(',')) if apps else None
    tag_filter = set(tags.split(',')) if tags else None

    # Tracks every job this `pbx run` invocation treats as a restart-continuation
    # — populated lazily inside `schedule_helpers.create_parsl_future` when
    # `apply_restart_for_job` runs successfully. Used by `choose_output_mode` to
    # decide per-job stdout/stderr file mode (continuation → 'a'+banner; fresh →
    # 'w'). Cleared once a job reaches a terminal status.
    restarting_job_ids: set[int] = set()

    # Restart jobs get priority over Ready jobs — they've already consumed
    # resources once and are mid-workflow; we want to clear them before
    # starting fresh work. Within each group, order is FIFO by job_id.
    all_runnable_jobs = database.get_jobs(db_path, status='Restart')
    all_runnable_jobs += database.get_jobs(db_path, status='Ready')

    filtered_jobs = []
    for job in all_runnable_jobs:
        passes_app_filter = not app_filter or job['app'] in app_filter
        passes_tag_filter = not tag_filter or job['tag'] in tag_filter
        if passes_app_filter and passes_tag_filter:
            filtered_jobs.append(job)

    if not filtered_jobs:
        logger.info("No runnable jobs found matching the specified filters. Exiting.")
        return

    njobs = len(filtered_jobs)
    logger.info(f"Found {njobs} jobs to execute.")

    # Track Parsl loading state for signal handler (must be defined before use)
    parsl_loaded_flag = {'loaded': False}

    # Load Parsl configuration with dynamic worker count based on number of jobs
    try:
        parsl_config, scheduler = load_config(
            name=config_name,
            run_dir=run_dir,
            retries=retries,
            max_workers=njobs  # Pass job count to limit worker spawning
        )
        parsl.load(parsl_config)
        parsl_loaded_flag['loaded'] = True  # Mark Parsl as loaded for signal handler
        logger.info(f"Successfully loaded Parsl config '{config_name}' with max_workers={njobs}.")
    except (FileNotFoundError, ValueError, parsl.errors.ConfigurationError) as e:
        logger.error(f"Failed to load Parsl configuration: {e}")
        raise typer.Exit(code=1)

    # Initialize JobTracker with filtered jobs and database path
    job_tracker = JobTracker(filtered_jobs, db_path)

    # Initialize Resource Manager with JobTracker
    system_config = get_system_config(config_name)
    resource_manager = system_config.create_resource_manager(job_tracker)
    logger.info(f"Initialized resource manager for {config_name}")

    # Initialize Status Buffer for batching database updates
    status_buffer = StatusBuffer(db_path)
    logger.info("Initialized StatusBuffer for batching database updates")

    # Build respawn-mode context (consumed by perform_shutdown at walltime).
    # None when --respawn is unset → existing Killed-and-exit behavior.
    respawn_ctx = None
    if respawn is not None:
        respawn_ctx = {
            "respawn": respawn,
            "run_dir": run_dir,
            "system_config": system_config,
            "db_path": db_path,
            "app_filter": app_filter,
            "tag_filter": tag_filter,
        }
        logger.info(
            f"Respawn chain active: respawn={respawn}. "
            f"At walltime, in-flight jobs will be marked "
            f"{'Restart and chain will resubmit' if respawn > 0 else 'Failed (chain end)'}."
        )

    # This run's claim owner token (its scheduler batch id). Reconciliation and
    # claiming are scoped to it so concurrent runs never touch each other's jobs.
    owner = get_scheduler_job_id(scheduler)

    # Register signal handlers for graceful shutdown on walltime exceeded
    shutdown_handler = create_shutdown_handler(
        status_buffer, logger, parsl_loaded_flag, job_tracker, owner=owner
    )
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)
    logger.info("Registered signal handlers for SIGTERM and SIGINT")

    # Register alarm handler for timeout protection during shutdown
    alarm_handler = create_alarm_handler(logger)
    signal.signal(signal.SIGALRM, alarm_handler)
    logger.info("Registered SIGALRM handler for shutdown timeout protection (10s)")

    # Register atexit handler for normal termination
    atexit_handler = create_atexit_handler(status_buffer, logger)
    atexit.register(atexit_handler)
    logger.info(f"Registered cleanup handlers (periodic flush interval: {flush_interval}s)")

    # Pre-load full YAML config for MPI settings
    try:
        full_yaml_config = load_full_config()
    except FileNotFoundError:
        full_yaml_config = {}
        logger.warning("No config.yaml found, using default MPI settings")

    # Pre-load App Contexts and MPI configs
    unique_app_names = list(set(job['app'] for job in filtered_jobs))
    app_instances = {}
    app_configs = {}
    mpi_configs = {}

    for app_name in unique_app_names:
        try:
            from parslbox.apps.app_registry import get_app_instance
            app_instance = get_app_instance(app_name)
            app_config = load_app_config(app_name=app_name, system_name=config_name)

            # Load MPI config with hierarchy: system-level -> app-level
            mpi_config = load_mpi_config(
                system_name=config_name,
                app_name=app_name,
                system_config_class=system_config,
                yaml_config=full_yaml_config
            )
            mpi_configs[app_name] = mpi_config
            logger.info(f"App '{app_name}' MPI config: backend={mpi_config.backend.value}, "
                       f"gpu_wrapper={mpi_config.use_gpu_wrapper}, cpu_bind={mpi_config.cpu_bind_method}")

            app_instances[app_name] = app_instance
            app_configs[app_name] = app_config

            # Log configuration status
            if len(app_config) == 0:
                if is_app_configured(app_name):
                    logger.warning(f"App '{app_name}' has configuration but no config found for system '{config_name}'. Running with empty configuration.")
                else:
                    logger.info(f"App '{app_name}' has no configuration defined. Running with empty configuration.")
            else:
                logger.info(f"App '{app_name}' loaded configuration for system '{config_name}'.")

            logger.info(f"Loaded app '{app_name}' successfully")

        except Exception as e:
            logger.error(f"Failed to load app '{app_name}': {e}")
            # Mark all jobs of this app type as failed
            failed_job_ids = [job['job_id'] for job in filtered_jobs if job['app'] == app_name]
            database.update_jobs(db_path, job_ids=failed_job_ids, status="Failed")
            # Remove these jobs from processing
            filtered_jobs = [job for job in filtered_jobs if job['app'] != app_name]

    # Build the scheduling context shared by the dispatch engine.
    fut_to_item: dict = {}
    sched_ctx = SchedulerContext(
        db_path=db_path,
        scheduler=scheduler,
        owner=owner,
        config_name=config_name,
        resource_manager=resource_manager,
        job_tracker=job_tracker,
        status_buffer=status_buffer,
        system_config=system_config,
        app_instances=app_instances,
        app_configs=app_configs,
        mpi_configs=mpi_configs,
        restarting_job_ids=restarting_job_ids,
        fut_to_item=fut_to_item,
        shutdown_at=shutdown_at,
        full_yaml_config=full_yaml_config,
    )

    apps_list = sorted(app_filter) if app_filter else None
    tags_list = sorted(tag_filter) if tag_filter else None

    # Static mode: one owner claims every runnable job up front (single batched
    # write) and drains the resource manager's in-memory backlog. Dynamic mode
    # holds no backlog — it re-queries and claims a capacity-sized pick per pass.
    if not dynamic:
        ready_ids = [j['job_id'] for j in filtered_jobs if j['status'] == 'Ready']
        restart_ids = [j['job_id'] for j in filtered_jobs if j['status'] == 'Restart']
        owned = set(database.claim_jobs(db_path, ready_ids, restart_ids, owner))
        for job in filtered_jobs:
            if job['job_id'] in owned:
                resource_manager.add_to_backlog(job['job_id'])
        logger.info(f"Static mode: claimed {len(owned)} job(s) up front into the backlog.")

    spinup_minutes = (time.time() - process_start) / 60
    logger.info(
        f"Starting dispatch in {'dynamic' if dynamic else 'static'} mode "
        f"(elapsed: {spinup_minutes:.2f} minutes)..."
    )

    # Periodic maintenance timers
    last_recovery_attempt = time.time()
    recovery_interval = 60  # Attempt node recovery every 60 seconds
    last_flush_time = time.time()
    poll_interval = 10  # Max seconds to wait on a future before re-checking state

    while True:
        current_time = time.time()

        # Walltime-aware shutdown: trigger graceful shutdown before the scheduler
        # kills the job. cleanup_parsl=True since we have grace-period runway.
        if current_time >= shutdown_at:
            logger.warning(
                f"Approaching walltime ({SHUTDOWN_GRACE_SECONDS}s grace) - "
                f"initiating graceful shutdown"
            )
            perform_shutdown(
                status_buffer=status_buffer,
                job_tracker=job_tracker,
                parsl_loaded_flag=parsl_loaded_flag,
                logger=logger,
                reason="walltime",
                cleanup_parsl=True,
                respawn_ctx=respawn_ctx,
                owner=owner,
            )
            sys.exit(0)

        # Periodic status buffer flush (safety net for walltime termination)
        if current_time - last_flush_time >= flush_interval:
            flushed = status_buffer.flush_all()
            if flushed > 0:
                logger.info(f"Periodic flush: Updated {flushed} job(s) in database")
            last_flush_time = current_time

        # Periodically attempt to recover quarantined nodes
        if current_time - last_recovery_attempt >= recovery_interval:
            recovered_nodes = resource_manager.attempt_node_recovery()
            if recovered_nodes:
                logger.info(f"Recovered {len(recovered_nodes)} nodes from quarantine: {recovered_nodes}")
            last_recovery_attempt = current_time

        # Dispatch as much as currently fits.
        try:
            if dynamic:
                dispatched = dispatch_dynamic(sched_ctx, apps_list, tags_list)
            else:
                dispatched = dispatch_static(sched_ctx)
        except Exception as e:
            logger.error(f"Error during dispatch: {e}")
            dispatched = 0

        status_buffer.flush_all()

        if dispatched:
            status = resource_manager.get_resource_status()
            logger.info(
                f"Dispatched {dispatched} job(s). running={len(fut_to_item)} | "
                f"free: {status['available_nodes']} nodes, {status['available_gpus']} GPUs, "
                f"backlog={status['backlogged_jobs']}"
            )

        # Exit rule (no idling): if nothing is running, this pass placed nothing,
        # so there is no runnable+fitting work for this run right now. Finish.
        if not fut_to_item:
            if not dynamic and resource_manager.get_resource_status()['backlogged_jobs'] > 0:
                # Static orphans — claimed up front but never became runnable
                # (e.g. a failed parent, or too big to ever fit). Return them to
                # the pool rather than stranding them in Submitted/Resubmitted.
                orphans = list(resource_manager._backlogged_jobs_set)
                reverted = database.revert_claims(db_path, orphans, owner=owner)
                logger.info(
                    f"Static: reverted {reverted} un-runnable claimed job(s) to Ready/Restart."
                )
            logger.info("No active futures and no runnable work remaining — run complete.")
            break

        # Wait for a future to complete (frees resources → next pass refills).
        try:
            for fut in as_completed(list(fut_to_item.keys()), timeout=poll_interval):
                if fut not in fut_to_item:
                    continue
                handle_completion(fut, sched_ctx)
                # Break to re-dispatch with freed resources.
                break
        except FutureTimeoutError:
            # No future completed within poll_interval — loop to re-check
            # walltime, flush, and re-query for new work (dynamic).
            pass

    logger.info("All runnable jobs processed.")

    # Final flush before cleanup (safety net)
    final_updated_count = status_buffer.flush_all()
    if final_updated_count > 0:
        logger.info(f"Final flush: Updated status for {final_updated_count} jobs before cleanup")

    # Log final node health summary
    try:
        health_summary = resource_manager.get_node_health_summary()
        system_health = health_summary['system_health']

        logger.info(f"Final System Health Summary:")
        logger.info(f"  Total nodes tracked: {system_health['total_tracked_nodes']}")
        logger.info(f"  Healthy nodes: {system_health['healthy_nodes']}")
        logger.info(f"  Suspected nodes: {system_health['suspected_nodes']}")
        logger.info(f"  Quarantined nodes: {system_health['quarantined_nodes']}")

        if health_summary['quarantined_nodes']:
            logger.warning(f"Quarantined nodes at end of run:")
            for quarantined in health_summary['quarantined_nodes']:
                logger.warning(f"  - {quarantined['node_id']} ({quarantined['hostname']}): "
                              f"{quarantined['consecutive_failures']} failures, "
                              f"last error: {quarantined['last_failure_error']}")
    except Exception as e:
        logger.error(f"Failed to generate final health summary: {e}")

    # Cleanup
    parsl.dfk().cleanup()
    total_minutes = (time.time() - process_start) / 60
    logger.info(f"Total elapsed time: {total_minutes:.2f} minutes")
    logger.info("--- parslbox orchestrator finished ---")
