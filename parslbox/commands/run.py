import typer
import parsl
import logging
import os
import time
import importlib
import signal
import atexit
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated
from concurrent.futures import as_completed

from parslbox.system_configs.loader import load_config, get_system_config
from parslbox.utils import path_utils
from parslbox.utils.logging_utils import setup_logging, validate_log_level
from parslbox.utils.pbx_config_utils import load_app_config, is_app_configured, load_full_config
from parslbox.database import database
from parslbox.resource_manager.node_failure_tracker import NodeHealth
from parslbox.resource_manager.mpi_config import load_mpi_config
from parslbox.resource_manager.mpi_command_builder import build_mpi_command, build_single_rank_launcher
from parslbox.resource_manager.exceptions import InsufficientResources
from parslbox.resource_manager.models import create_job_resource_spec
from parslbox.resource_manager.job_tracker import JobTracker

# Import helper functions
from parslbox.commands.helpers.run_cmd_helpers import (
    validate_and_normalize_status,
    parse_parents,
    get_default_run_dir,
    create_shutdown_handler,
    create_alarm_handler,
    create_atexit_handler,
    perform_shutdown,
    choose_output_mode,
    should_re_dispatch_known_job,
    should_gate_dispatch,
)
from parslbox.commands.helpers.hook_dispatch import dispatch_hook_on_compute
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


def create_parsl_future(job, app_instance, app_config, mpi_config, config_name, db_path, scheduler, resource_manager, futures, system_config, status_buffer, restarting_job_ids, remaining_walltime_s):
    """
    Create a Parsl future for a job with proper preprocessing and error handling.

    Args:
        job: Job dictionary from database
        app_instance: Application instance
        app_config: Application configuration
        mpi_config: MPI configuration for this app/system
        config_name: System configuration name
        db_path: Database path
        scheduler: Scheduler type (PBS, SLURM, etc.)
        resource_manager: Resource manager instance
        futures: List to append the future to
        system_config: System configuration object
        restarting_job_ids: Set of job IDs that are restart-continuations in this
            `pbx run` invocation. Populated lazily inside this function when
            `apply_restart_for_job` returns a successful bucket. Decides
            whether per-job stdout/stderr open in 'a'+banner mode
            (continuation) or 'w' (fresh).

    Returns:
        bool: True if successful, False if failed
    """
    job_id = job['job_id']
    job_path = Path(job['path'])
    logger = logging.getLogger(__name__)

    # Carries restart() patch fields (if any) into the buffered Running write
    # at the end of this function. Empty for fresh Ready jobs and for restart
    # jobs whose app returned None/{} (rerun-as-is).
    pending_patch: dict = {}

    try:
        # Check job status and claim it immediately to prevent race conditions
        current_jobs = database.get_jobs_by_ids(db_path, [job_id])
        if not current_jobs:
            logger.warning(f"Job {job_id}: Job no longer exists in database, skipping")
            return False

        current_status = current_jobs[0]['status']
        if current_status not in ['Ready', 'Restart']:
            logger.info(f"Job {job_id}: Status is '{current_status}', skipping (already processed by another process)")
            return False

        # Immediately claim the job by updating status to "Submitted" and Set scheduler job ID
        sched_job_id = get_scheduler_job_id(scheduler)
        database.update_jobs(db_path, job_ids=[job_id], status="Submitted", sched_job_id=sched_job_id)
        # Also update JobTracker
        resource_manager.job_tracker.update_job_status(job_id, "Submitted")
        logger.info(f"Job {job_id}: Successfully claimed job for processing")
        
        # Get resource assignment (should already exist)
        assignment = resource_manager.get_job_assignment(job_id)
        if not assignment:
            raise ValueError(f"No resource assignment found for job {job_id}")
        
        # Create JobResourceSpec from job data
        job_spec = create_job_resource_spec(job)
        
        # Generate MPI commands using new simplified MPI system
        mpi_commands = build_mpi_command(
            mpi_config=mpi_config,
            system_config=system_config,
            assignment=assignment,
            job_spec=job_spec,
            job_path=str(job_path)
        )
        # Add MPI env_setup to the commands dict so it reaches the bash engine
        if mpi_config.env_setup:
            mpi_commands['PBX_MPI_ENV_SETUP'] = mpi_config.env_setup
        # Pass backend so downstream code can adapt (e.g., skip GPU env vars for srun)
        mpi_commands['PBX_MPI_BACKEND'] = mpi_config.backend.value
        # Tile mode flag for Intel tile-mode systems (e.g., Aurora tile) — used
        # by env-var generation to format ZE_AFFINITY_MASK as "physical.tile".
        mpi_commands['PBX_GPU_TILE_MODE'] = 'tile' in type(system_config).__name__.lower()

        # Expose host/core info for apps that manage their own MPI (e.g., ORCA)
        hostnames = list(assignment.hostnames)
        if mpi_config.use_short_hostnames:
            hostnames = [h.split('.')[0] for h in hostnames]
        mpi_commands['PBX_HOSTNAMES'] = ','.join(hostnames)
        mpi_commands['PBX_CORES_PER_NODE'] = str(system_config.CORES_PER_NODE)
        mpi_commands['PBX_RANKS_PER_NODE'] = str(job_spec.ranks_per_node)

        logger.info(f"Job {job_id}: Generated MPI command - {mpi_commands.get('PBX_MPI_PREFIX', 'None')}")

        # Build the single-rank launcher + per-job GPU env vars when the app
        # opts into RUN_HOOKS_ON_COMPUTE. The launcher lands the hook subprocess
        # on the assigned compute node; the GPU env vars (forwarded explicitly
        # in the bash inner cmd) constrain GPU visibility for that subprocess.
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

        # Lazy per-job restart() — called only after resources are assigned,
        # only for jobs that survived the dispatch gate. Symmetric with
        # preprocess(): runs per-job, on-demand, not eagerly at startup.
        # Patches are buffered into the Running write below (single batched
        # DB call); the in-memory `job` dict is the source of truth for the
        # rest of this function (preprocess, parsl_app build).
        if current_status == 'Restart':
            from parslbox.commands.helpers.restart_helpers import apply_restart_for_job
            bucket, patch, err = apply_restart_for_job(job, app_instance, logger)
            if bucket == 'failed':
                # Re-raise so the generic except below handles resource cleanup
                # + buffered Failed write uniformly with other dispatch failures.
                raise RuntimeError(f"restart() failed: {err}")
            restarting_job_ids.add(job_id)
            if bucket == 'patched':
                job.update(patch)
                pending_patch = patch
                logger.info(
                    f"Job {job_id}: restart() patched fields {sorted(patch)}"
                )
            else:
                logger.info(f"Job {job_id}: restart() returned no patch (rerun as-is)")

        # Run preprocessing — either in-process on the head node (default) or
        # dispatched to the assigned compute node when the app opts in.
        logger.info(f"Running preprocessing for Job ID {job_id}...")
        if app_instance.RUN_HOOKS_ON_COMPUTE:
            dispatch_hook_on_compute(
                app_name=job['app'],
                method_name='preprocess',
                single_rank_launcher=mpi_commands['PBX_SINGLE_RANK_LAUNCHER'],
                env_file=job.get('env_file'),
                gpu_env_vars=hook_gpu_env_vars,
                # method kwargs (forwarded to preprocess)
                job_id=job_id,
                job_path=job_path,
                db_path=db_path,
                app_config=app_config,
                config_name=config_name,
            )
        else:
            app_instance.preprocess(
                job_id=job_id,
                job_path=job_path,
                db_path=db_path,
                app_config=app_config,
                config_name=config_name,
            )
        
        
        # Decide per-job stdout/stderr file mode. Restart-continuation runs
        # (DB status was 'Restart' on entry, or restart() just ran successfully
        # above) append + write a banner so the boundary between links /
        # retries is visible; fresh Ready runs clobber prior content.
        stdout_path = job_path / f"pbx_job_{job_id}.out"
        stderr_path = job_path / f"pbx_job_{job_id}.err"
        file_mode = choose_output_mode(
            job_id, current_status, restarting_job_ids,
            remaining_walltime_s,
            stdout_path, stderr_path,
        )

        # Create Parsl future
        fut = app_instance.parsl_app(
            job_id=job_id,
            job_path=job_path,
            db_path=db_path,
            assignment=assignment,
            mpi_commands=mpi_commands,
            app_config=app_config,
            config_name=config_name,
            in_file=job['in_file'],
            mpi_opts=job['mpi_opts'],
            env_file=job.get('env_file'),
            stdout=(str(stdout_path), file_mode),
            stderr=(str(stderr_path), file_mode)
        )
        
        # Add to futures list. Stash single_rank_launcher, env_file, and
        # gpu_env_vars so the postprocess wrap below can dispatch to the same
        # assigned compute node — by the time the future completes, the local
        # mpi_commands dict here is out of scope.
        futures.append({
            'future': fut,
            'job': job,
            'app_instance': app_instance,
            'assignment': assignment,
            'resource_manager': resource_manager,
            'single_rank_launcher': mpi_commands.get('PBX_SINGLE_RANK_LAUNCHER'),
            'env_file': job.get('env_file'),
            'gpu_env_vars': hook_gpu_env_vars,
        })
        
        logger.info(f"Job {job_id}: Successfully created Parsl future")
        
        # Buffer the "Running" status update and update JobTracker.
        # If restart() patched any fields (in_file/env_file/tag), they ride
        # along on this same buffered write — single batched DB call covers
        # both the status flip and the patch.
        resource_manager.job_tracker.update_job_status(job_id, 'Running')
        status_buffer.add_status_update(job_id, status='Running', **pending_patch)

        return True
        
    except Exception as e:
        logger.error(f"Job {job_id}: Failed to submit Parsl app: {e}")
        # Free resources if job submission failed (with health tracking)
        resource_manager.free_resources_with_health_check(job_id, job_succeeded=False, error_message=str(e))
        # Buffer the "Failed" status update and update JobTracker
        resource_manager.job_tracker.update_job_status(job_id, "Failed")
        status_buffer.add_status_update(job_id, status="Failed")
        return False


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

    # Tracks every job that this `pbx run` invocation considers a
    # restart-continuation — populated lazily inside `create_parsl_future`
    # when `apply_restart_for_job` runs successfully (Scene A patched or
    # Scene B rerun-as-is). Used by `choose_output_mode` to decide per-job
    # stdout/stderr file mode (continuation → 'a'+banner; fresh → 'w').
    # Cleared for any job that reaches a terminal status (Done/Failed/Warning)
    # so a Failed→Ready re-discovery within the same invocation gets a
    # fresh write.
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
    known_job_ids = set(job['job_id'] for job in filtered_jobs)
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

    # Register signal handlers for graceful shutdown on walltime exceeded
    shutdown_handler = create_shutdown_handler(status_buffer, logger, parsl_loaded_flag, job_tracker)
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

    # Process Jobs Individually with dependency checking
    # First attempt to create futures. Jobs without resource assignment will be put in the backlogged queue
    # After this, futures will be created for backlogged jobs as resource becomes available
    futures = []
    
    for job in filtered_jobs:
        job_id = job['job_id']
        app_name = job['app']

        # Skip if app failed to load (already marked as failed above)
        if app_name not in app_instances:
            continue

        if should_gate_dispatch(
            app_instances[app_name], job, shutdown_at - time.time(),
        ):
            floor = app_instances[app_name].min_remaining_walltime(job)
            logger.info(
                f"Job {job_id}: skipping dispatch — remaining walltime "
                f"below {app_name} floor ({int(shutdown_at - time.time())}s "
                f"< {floor}s); will be picked up by next pbx run."
            )
            continue

        logger.info(f"Submitting Job ID {job_id}...")

        # Check dependencies first using JobTracker
        try:
            parents_done = job_tracker.are_parents_done(job_id)
            
            if not parents_done:
                logger.info(f"Job {job_id}: Parent dependencies not satisfied, adding to backlog")
                resource_manager.add_to_backlog(job_id)
                continue
                
        except Exception as e:
            logger.error(f"Job {job_id}: Error checking parent dependencies: {e}")
            job_tracker.update_job_status(job_id, "Failed")
            database.update_jobs(db_path, job_ids=[job_id], status="Failed")
            continue
        
        # Allocate resources
        try:
            assignment = resource_manager.assign_resources(job)

            # Create Parsl future with pre-loaded contexts
            success = create_parsl_future(
                job, app_instances[app_name], app_configs[app_name], mpi_configs[app_name],
                config_name, db_path, scheduler, resource_manager, futures, system_config, status_buffer,
                restarting_job_ids,
                shutdown_at - time.time(),
            )
            
            # Future created successfully - no additional tracking needed
            # Dependencies are now checked via database
            
            # Throttle job submissions. Tunable via PBX_RUN_DELAY env var
            # (e.g., bump up on systems like Perlmutter where rapid srun calls
            # can overload slurmctld).
            time.sleep(float(os.getenv("PBX_RUN_DELAY", "0.2")))
            
        except InsufficientResources as e:
            # This is NOT an error - just temporary resource unavailability
            logger.info(f"Job {job_id}: Resources temporarily unavailable, added to backlog: {e}")
            # The resource manager has already added the job to the backlog queue
            # Don't mark as failed - the job will be scheduled when resources become available
            continue
            
        except Exception as e:
            # This IS an actual error (invalid spec, system error, etc.)
            logger.error(f"Job {job_id}: Failed to allocate resources: {e}")
            job_tracker.update_job_status(job_id, "Failed")
            database.update_jobs(db_path, job_ids=[job_id], status="Failed")
            continue

    # FLUSH POINT 1: After all initial futures are created
    # Batch update all "Running" and "Failed" status updates from job creation
    updated_count = status_buffer.flush_all()
    if updated_count > 0:
        logger.info(f"Batch updated status for {updated_count} initial jobs")

    # Await and Process Results (Dynamic future handling)
    # Create mapping from futures to their metadata
    # Create new futures for rescheduled backlogged jobs in a while loop
    fut_to_item = {item['future']: item for item in futures}

    spinup_minutes = (time.time() - process_start) / 60
    logger.info(f"Starting to process {len(fut_to_item)} initial jobs (elapsed time: {spinup_minutes:.2f} minutes)...")
    if dynamic:
        logger.info("Dynamic job discovery enabled (poll every 60s)")

    # Track last recovery attempt time for periodic node recovery
    last_recovery_attempt = time.time()
    recovery_interval = 60  # Attempt recovery every 60 seconds

    # Track last flush time for periodic status buffer flush
    last_flush_time = time.time()

    # Track last dynamic discovery time
    last_discovery_time = time.time()
    discovery_interval = 60  # Check for new jobs every 60 seconds

    def discover_new_jobs():
        """
        Check DB for new Ready/Restart jobs matching the same app/tag filters.

        Discovers jobs added after the initial collection (e.g., by orchestrator
        scripts or users running `pbx add` from another terminal). New jobs go
        through the standard pipeline: dependency check -> resource allocation
        -> submit or backlog.

        Returns:
            int: Number of new jobs discovered
        """
        # Restart jobs first — same priority rule as the initial collection.
        new_runnable = database.get_jobs(db_path, status='Restart')
        new_runnable += database.get_jobs(db_path, status='Ready')

        new_jobs = []
        rediscovered_count = 0
        newly_added_count = 0
        for job in new_runnable:
            job_id = job['job_id']
            if job_id in known_job_ids:
                # Job was already in this invocation's filtered set. Re-dispatch
                # whenever the tracker considers it settled (Done/Failed/Warning/
                # Killed/Ready-but-not-running) AND the DB now shows it back in a
                # runnable status (Ready/Restart) — that's the user (or an
                # external script) saying "run this again." A job whose tracker
                # status is Submitted/Running has a live future; the DB flip is
                # a no-op for this link, leave the future alone and let the
                # walltime/result path settle it.
                #
                # Routing re-discovered Restart-status jobs through new_jobs is
                # what lets the per-job `apply_restart_for_job` call inside
                # create_parsl_future fire on them and add them to
                # restarting_job_ids, so their stdout/stderr opens in
                # 'a'+banner mode.
                tracked = job_tracker.get_job(job_id)
                if not tracked:
                    logger.warning(f"Job {job_id} in known_job_ids but missing from JobTracker — data inconsistency")
                    continue
                if should_re_dispatch_known_job(tracked['status'], job['status']):
                    job_tracker.update_job_status(job_id, job['status'])
                    new_jobs.append(job)
                    rediscovered_count += 1
                    logger.info(
                        f"Dynamic discovery: Re-discovered job {job_id} "
                        f"(tracker={tracked['status']} → DB={job['status']})"
                    )
                continue
            passes_app = not app_filter or job['app'] in app_filter
            passes_tag = not tag_filter or job['tag'] in tag_filter
            if passes_app and passes_tag:
                new_jobs.append(job)
                known_job_ids.add(job_id)
                newly_added_count += 1

        if not new_jobs:
            return 0

        logger.info(
            f"Dynamic discovery: {rediscovered_count} re-discovered, "
            f"{newly_added_count} newly added"
        )

        # Register new jobs with JobTracker
        job_tracker.register_jobs(new_jobs)

        # Load app contexts for any new app types
        new_app_names = set(job['app'] for job in new_jobs) - set(app_instances.keys())
        for app_name in new_app_names:
            try:
                from parslbox.apps.app_registry import get_app_instance
                app_instances[app_name] = get_app_instance(app_name)
                app_configs[app_name] = load_app_config(app_name=app_name, system_name=config_name)
                mpi_configs[app_name] = load_mpi_config(
                    system_name=config_name, app_name=app_name,
                    system_config_class=system_config, yaml_config=full_yaml_config
                )
                logger.info(f"Loaded app context for dynamically discovered app '{app_name}'")
            except Exception as e:
                logger.error(f"Failed to load app '{app_name}' for dynamic jobs: {e}")
                for job in new_jobs:
                    if job['app'] == app_name:
                        job_tracker.update_job_status(job['job_id'], "Failed")
                        database.update_jobs(db_path, job_ids=[job['job_id']], status="Failed")
                new_jobs = [j for j in new_jobs if j['app'] != app_name]

        # Restart-status jobs in `new_jobs` are not flipped here — the per-job
        # `apply_restart_for_job` call inside create_parsl_future handles them
        # lazily, after resources are assigned. See create_parsl_future for the
        # contract.

        # Process new jobs: dependency check -> resource allocation -> submit/backlog
        new_futures = []
        for job in new_jobs:
            job_id = job['job_id']
            app_name = job['app']

            if app_name not in app_instances:
                continue

            if should_gate_dispatch(
                app_instances[app_name], job, shutdown_at - time.time(),
            ):
                floor = app_instances[app_name].min_remaining_walltime(job)
                logger.info(
                    f"Job {job_id}: skipping dispatch (dynamic discovery) — "
                    f"remaining walltime below {app_name} floor "
                    f"({int(shutdown_at - time.time())}s < {floor}s); "
                    f"will be picked up by next pbx run."
                )
                continue

            # Check dependencies
            try:
                if not job_tracker.are_parents_done(job_id):
                    logger.info(f"Job {job_id}: Parent dependencies not satisfied, adding to backlog")
                    resource_manager.add_to_backlog(job_id)
                    continue
            except Exception as e:
                logger.error(f"Job {job_id}: Error checking dependencies: {e}")
                job_tracker.update_job_status(job_id, "Failed")
                database.update_jobs(db_path, job_ids=[job_id], status="Failed")
                continue

            # Allocate resources
            try:
                assignment = resource_manager.assign_resources(job)

                create_parsl_future(
                    job, app_instances[app_name], app_configs[app_name], mpi_configs[app_name],
                    config_name, db_path, scheduler, resource_manager, new_futures, system_config, status_buffer,
                    restarting_job_ids,
                    shutdown_at - time.time(),
                )
            except InsufficientResources as e:
                logger.info(f"Job {job_id}: Resources unavailable, added to backlog: {e}")
                continue
            except Exception as e:
                logger.error(f"Job {job_id}: Failed to allocate resources: {e}")
                job_tracker.update_job_status(job_id, "Failed")
                database.update_jobs(db_path, job_ids=[job_id], status="Failed")
                continue

        # Add new futures to tracking
        for item in new_futures:
            fut_to_item[item['future']] = item

        # Flush status updates for new jobs
        flushed = status_buffer.flush_all()
        if flushed > 0:
            logger.info(f"Dynamic discovery: Flushed {flushed} status updates")

        return len(new_jobs)

    while True:
        try:
            current_time = time.time()

            # Walltime-aware shutdown: trigger graceful shutdown before
            # the scheduler kills the job. cleanup_parsl=True because we
            # have the configured grace period of runway.
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
                )
                sys.exit(0)

            # Periodic status buffer flush (safety net for walltime termination)
            if current_time - last_flush_time >= flush_interval:
                flushed = status_buffer.flush_all()
                if flushed > 0:
                    logger.info(f"Periodic flush: Updated {flushed} job(s) in database")
                else:
                    logger.debug(f"Periodic flush: No pending updates")
                last_flush_time = current_time

            # Periodically attempt to recover quarantined nodes
            if current_time - last_recovery_attempt >= recovery_interval:
                recovered_nodes = resource_manager.attempt_node_recovery()
                if recovered_nodes:
                    logger.info(f"Recovered {len(recovered_nodes)} nodes from quarantine: {recovered_nodes}")
                last_recovery_attempt = current_time

            # Periodic dynamic job discovery
            if dynamic and current_time - last_discovery_time >= discovery_interval:
                discover_new_jobs()
                last_discovery_time = current_time

            # Wait for any future to complete
            for fut in as_completed(list(fut_to_item.keys()), timeout=10):
                # Check if future still exists (might have been processed already)
                if fut not in fut_to_item:
                    continue

                item = fut_to_item.pop(fut)
                job = item['job']
                app_instance = item['app_instance']
                assignment = item['assignment']
                resource_manager = item['resource_manager']
                job_id = job['job_id']
                job_path = Path(job['path'])

                # Capture execution result and any errors
                error_message = None
                try:
                    fut.result()  # Get result (may raise exception)
                    logger.info(f"Job {job_id}: Execution completed without errors.")
                except Exception as e:
                    error_message = str(e)
                    logger.error(f"Job {job_id}: Execution error occurred: {error_message}")

                # Always call check_success, let app decide based on error_message
                job_status = app_instance.check_success(
                    job_id=job_id,
                    job_path=job_path,
                    db_path=db_path,
                    error_message=error_message
                )

                # Validate and normalize the status from check_success
                job_status = validate_and_normalize_status(job_status, job_id)

                # If job succeeded, run post-processing — either in-process on
                # the head node (default) or dispatched to the assigned compute
                # node when the app opts in via RUN_HOOKS_ON_COMPUTE.
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
                                # method kwargs (forwarded to postprocess)
                                job_id=job_id,
                                job_path=job_path,
                                db_path=db_path,
                            )
                        else:
                            final_status = app_instance.postprocess(
                                job_id=job_id, job_path=job_path, db_path=db_path,
                            )

                        # Validate and use postprocess result if it returns a valid status
                        if final_status:
                            validated_final_status = validate_and_normalize_status(final_status, job_id)
                            job_status = validated_final_status
                        # If postprocess returns None/empty, keep the check_success result

                    except Exception as e:
                        logger.error(f"Job {job_id}: Post-processing failed: {e}")
                        job_status = "Failed"

                # Update JobTracker and buffer final status update
                job_tracker.update_job_status(job_id, job_status)
                status_buffer.add_status_update(job_id, status=job_status)
                logger.info(f"Job {job_id}: Final status = {job_status} (buffered)")

                # Clear restart-continuation membership now that this job has
                # reached a terminal status, so a Failed→Ready re-discovery
                # within the same `pbx run` invocation gets a fresh write.
                if job_status in ('Done', 'Failed', 'Warning'):
                    restarting_job_ids.discard(job_id)

                # Free resources with health tracking based on job outcome
                try:
                    job_succeeded = (job_status == "Done")

                    # For failed jobs, read stderr to get actual error context for node health tracking
                    stderr_error = None
                    if not job_succeeded:
                        stderr_file = job_path / f"pbx_job_{job_id}.err"
                        try:
                            if stderr_file.is_file():
                                stderr_content = stderr_file.read_text(errors='replace')
                                # Get last 50 lines — sufficient to capture error signatures
                                stderr_tail = '\n'.join(stderr_content.splitlines()[-50:])
                                if stderr_tail.strip():
                                    stderr_error = stderr_tail
                        except Exception as e:
                            logger.debug(f"Job {job_id}: Could not read stderr file: {e}")

                    resource_manager.free_resources_with_health_check(
                        job_id=job_id,
                        job_succeeded=job_succeeded,
                        error_message=stderr_error
                    )

                    # Log status after freeing resources
                    status = resource_manager.get_resource_status()
                    cpu_frac = status['available_cpu_capacity'] / status['available_nodes'] if status['available_nodes'] > 0 else 0
                    logger.info(
                        f"Status: running={len(fut_to_item)}, backlog={status['backlogged_jobs']} | "
                        f"free: {status['available_nodes']} nodes, {status['available_gpus']} GPUs, "
                        f"{cpu_frac:.0%} CPU"
                    )

                except Exception as e:
                    logger.error(f"Job {job_id}: Failed to free resources: {e}")

                # Break to refresh as_completed() with new futures
                break

        except TimeoutError:
            # No futures completed in timeout period
            pass

        # Schedule dependency-ready backlog jobs (runs every iteration)
        try:
            dependency_ready_jobs = resource_manager.get_dependency_ready_jobs_from_backlog()
            remaining_s = shutdown_at - time.time()
            eligible = []
            gated_ids = []
            for j in dependency_ready_jobs:
                app_obj = app_instances.get(j['app'])
                if app_obj is None:
                    eligible.append(j)
                    continue
                if should_gate_dispatch(app_obj, j, remaining_s):
                    gated_ids.append(j['job_id'])
                else:
                    eligible.append(j)
            if gated_ids:
                logger.info(
                    f"Walltime-floor gate: holding {len(gated_ids)} backlog "
                    f"job(s) {gated_ids} — remaining={int(remaining_s)}s "
                    f"< app floor; will be picked up by next pbx run."
                )
            rescheduled_jobs = resource_manager.schedule_backlog(eligible)
        except Exception as e:
            logger.error(f"Error during backlog scheduling: {e}")
            dependency_ready_jobs = []
            rescheduled_jobs = []

        # Create Parsl futures for rescheduled jobs
        for rescheduled_job in rescheduled_jobs:
            rescheduled_app_name = rescheduled_job['app']
            rescheduled_job_id = rescheduled_job['job_id']

            if rescheduled_app_name in app_instances:
                logger.info(f"Creating Parsl future for rescheduled job {rescheduled_job_id}")
                new_futures = []
                success = create_parsl_future(
                    rescheduled_job,
                    app_instances[rescheduled_app_name],
                    app_configs[rescheduled_app_name],
                    mpi_configs[rescheduled_app_name],
                    config_name, db_path, scheduler, resource_manager,
                    new_futures, system_config, status_buffer,
                    restarting_job_ids,
                    shutdown_at - time.time(),
                )

                # Add each new future to tracking dict
                for new_item in new_futures:
                    new_fut = new_item['future']
                    fut_to_item[new_fut] = new_item
                    logger.info(f"Added rescheduled job {rescheduled_job_id} to tracking (total active: {len(fut_to_item)})")
            else:
                logger.error(f"App context not found for rescheduled job {rescheduled_job_id}")
                job_tracker.update_job_status(rescheduled_job_id, "Failed")
                database.update_jobs(db_path, job_ids=[rescheduled_job_id], status="Failed")

        # Log status after scheduling, only if jobs were rescheduled
        if rescheduled_jobs:
            status = resource_manager.get_resource_status()
            # Recount dep-ready jobs — scheduled jobs were removed from backlog
            dependency_ready_count = len(resource_manager.get_dependency_ready_jobs_from_backlog())
            cpu_frac = status['available_cpu_capacity'] / status['available_nodes'] if status['available_nodes'] > 0 else 0
            logger.info(
                f"Status: running={len(fut_to_item)}, backlog={status['backlogged_jobs']}, "
                f"dep_ready={dependency_ready_count} | free: {status['available_nodes']} nodes, "
                f"{status['available_gpus']} GPUs, {cpu_frac:.0%} CPU"
            )

        # Flush status buffer (runs every iteration as safety net)
        updated_count = status_buffer.flush_all()
        if updated_count > 0:
            logger.debug(f"Batch updated status for {updated_count} jobs")

        # Exit check: no active futures — should we keep waiting or exit?
        if not fut_to_item:
            has_quarantined = any(
                n.health_tracker.health_status == NodeHealth.QUARANTINED
                for n in resource_manager.nodes
            )
            dep_ready_jobs = resource_manager.get_dependency_ready_jobs_from_backlog()
            backlogged = resource_manager.get_resource_status()['backlogged_jobs']

            # Dynamic discovery on its cadence. Previously this branch
            # called discover_new_jobs() unconditionally on every iteration,
            # which combined with a spurious-rediscovery bug in
            # should_re_dispatch_known_job spun the loop thousands of times
            # per second. Gating on discovery_interval is the structural fix.
            if dynamic and time.time() - last_discovery_time >= discovery_interval:
                new_count = discover_new_jobs()
                last_discovery_time = time.time()
                if new_count > 0:
                    continue
                # Discovery may have added work — refresh state.
                dep_ready_jobs = resource_manager.get_dependency_ready_jobs_from_backlog()
                backlogged = resource_manager.get_resource_status()['backlogged_jobs']

            # Truly idle → exit. Applies in both modes.
            if not backlogged and not dep_ready_jobs and not has_quarantined:
                logger.info("No active futures and no recoverable work remaining — run complete.")
                break

            # Dep-ready jobs waiting on quarantined node recovery — keep looping.
            if dep_ready_jobs and has_quarantined:
                logger.debug(
                    f"No active futures but {len(dep_ready_jobs)} dep-ready jobs "
                    f"waiting on quarantined node recovery"
                )
                time.sleep(10)
                continue

            # Non-dynamic mode: any remaining state (e.g., orphan backlog with
            # never-to-resolve dependencies) → exit. Matches prior behavior.
            if not dynamic:
                logger.info("No active futures and no recoverable work remaining — run complete.")
                break

            # Dynamic mode with pending work but no futures — sleep until the
            # next discovery tick instead of busy-looping. Cap at 10s so the
            # other periodic ticks (recovery, flush) stay responsive.
            sleep_for = min(
                10.0,
                max(0.5, discovery_interval - (time.time() - last_discovery_time)),
            )
            time.sleep(sleep_for)
            continue

    logger.info("All jobs completed, including rescheduled ones")
    
    # FLUSH POINT 3: Final flush before cleanup (safety net)
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

    # 7. Cleanup
    parsl.dfk().cleanup()
    total_minutes = (time.time() - process_start) / 60
    logger.info(f"Total elapsed time: {total_minutes:.2f} minutes")
    logger.info("--- parslbox orchestrator finished ---")
