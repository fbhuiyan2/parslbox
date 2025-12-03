import typer
import parsl
import logging
import os
import time
import importlib
from datetime import datetime
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated
from concurrent.futures import as_completed

from parslbox.configs.loader import load_config, get_system_config
from parslbox.helpers.logging_utils import setup_logging
from parslbox.helpers.config_utils import load_app_config, is_app_configured
from parslbox.helpers import database, path_utils
from parslbox.resource_manager.mpi_launcher import compose_mpi_command
from parslbox.resource_manager.exceptions import InsufficientResources
from parslbox.resource_manager.models import create_job_resource_spec

app = typer.Typer()

# Valid job status values (stored in lowercase for comparison)
VALID_JOB_STATUSES = ["ready", "done", "failed", "restart", "running", "submitted", "warning"]


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
    
    import json
    return [int(x) for x in json.loads(parents_str)]


def are_parents_done(job, db_path):
    """
    Check if job's parent dependencies are satisfied.
    
    Args:
        job: Job dictionary from database
        db_path: Path to database file
    
    Returns:
        bool: True if all parents are done, False otherwise
    """
    parents_str = job.get('parents')
    if not parents_str:
        return True
    
    parent_ids = parse_parents(parents_str)
    
    for parent_id in parent_ids:
        parent_job = database.get_jobs_by_ids(db_path, [parent_id])
        if not parent_job:
            raise ValueError(f"Job {job['job_id']} has non-existent parent {parent_id}")
        
        if parent_job[0]['status'] != 'Done':
            return False  # Can't submit yet
    
    return True


def get_dependency_ready_jobs(backlog_jobs, db_path):
    """Filter jobs whose parents are done."""
    ready_jobs = []
    logger = logging.getLogger(__name__)
    
    for job in backlog_jobs:
        try:
            parents_done = are_parents_done(job, db_path)
            if parents_done:
                ready_jobs.append(job)
        except Exception as e:
            logger.error(f"Error when checking dependencies for job {job['job_id']}: {e}")
            # Don't add to ready_jobs, but don't fail either - job stays in backlog
            continue
    return ready_jobs


def create_parsl_future(job, app_instance, app_config, config_name, db_path, scheduler, resource_manager, futures, system_config):
    """
    Create a Parsl future for a job with proper preprocessing and error handling.
    
    Args:
        job: Job dictionary from database
        app_instance: Application instance
        app_config: Application configuration
        config_name: System configuration name
        db_path: Database path
        scheduler: Scheduler type (PBS, SLURM, etc.)
        resource_manager: Resource manager instance
        futures: List to append the future to
        system_config: System configuration object
    
    Returns:
        bool: True if successful, False if failed
    """
    job_id = job['job_id']
    job_path = Path(job['path'])
    logger = logging.getLogger(__name__)
    
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
        
        # Immediately claim the job by updating status to "Submitted"
        database.update_jobs(db_path, job_ids=[job_id], status="Submitted")
        logger.info(f"Job {job_id}: Successfully claimed job for processing")
        
        # Get resource assignment (should already exist)
        assignment = resource_manager.get_job_assignment(job_id)
        if not assignment:
            raise ValueError(f"No resource assignment found for job {job_id}")
        
        # Create JobResourceSpec from job data
        job_spec = create_job_resource_spec(job)
        
        # Generate MPI commands with app-specific overrides
        mpi_overrides = app_config.get('mpi_overrides')
        mpi_commands = compose_mpi_command(assignment, system_config, job_spec, job_path, mpi_overrides)
        logger.info(f"Job {job_id}: Generated MPI command - {mpi_commands.get('PBX_MPI_PREFIX', 'None')}")
        
        # Run preprocessing
        logger.info(f"Running preprocessing for Job ID {job_id}...")
        app_instance.preprocess(
            job_id=job_id, 
            job_path=job_path, 
            db_path=db_path, 
            app_config=app_config, 
            config_name=config_name
        )
        
        # Set scheduler job ID
        if scheduler == "PBS":
            PBS_JOB_ID = os.environ.get('PBS_JOBID', f'local_{int(time.time())}')
            database.update_jobs(db_path, job_ids=[job_id], sched_job_id=PBS_JOB_ID)
        elif scheduler == "SLURM":
            SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID', f'local_{int(time.time())}')
            database.update_jobs(db_path, job_ids=[job_id], sched_job_id=SLURM_JOB_ID)
        
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
            stdout=(str(job_path / f"pbx_job_{job_id}.out"), 'w'),
            stderr=(str(job_path / f"pbx_job_{job_id}.err"), 'w')
        )
        
        # Add to futures list
        futures.append({
            'future': fut, 
            'job': job, 
            'app_instance': app_instance,
            'assignment': assignment,
            'resource_manager': resource_manager
        })
        
        logger.info(f"Job {job_id}: Successfully created Parsl future")
        return True
        
    except Exception as e:
        logger.error(f"Job {job_id}: Failed to submit Parsl app: {e}")
        # Free resources if job submission failed (with health tracking)
        resource_manager.free_resources_with_health_check(job_id, job_succeeded=False, error_message=str(e))
        database.update_jobs(db_path, job_ids=[job_id], status="Failed")
        return False


def get_default_run_dir() -> Path:
    """Generate default run directory with current time and date in hhmmss_ddmmyy format."""
    now = datetime.now()
    time_str = now.strftime("%H%M%S")  # hhmmss format (hours + minutes + seconds)
    date_str = now.strftime("%d%m%y")  # ddmmyy format
    dir_name = f"{time_str}_{date_str}"
    return Path.home() / ".parslbox" / "runs" / dir_name / "log.pbx"



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
        typer.Option("--apps", "-a", help="Comma-separated list of apps to run (e.g., 'lammps,vasp').")
    ] = None,
    tags: Annotated[
        Optional[str],
        typer.Option("--tags", "-t", help="Comma-separated list of tags to run (e.g., 'run1,run2').")
    ] = None,
    retries: Annotated[
        int,
        typer.Option("--retries", help="Number of retries for failed tasks.")
    ] = 0,
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
    setup_logging(log_file=log_file)
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

    try:
        parsl_config, scheduler = load_config(name=config_name, run_dir=run_dir, retries=retries)
        parsl.load(parsl_config)
        logger.info(f"Successfully loaded Parsl config '{config_name}'.")
    except (FileNotFoundError, ValueError, parsl.errors.ConfigurationError) as e:
        logger.error(f"Failed to load Parsl configuration: {e}")
        raise typer.Exit(code=1)

    # Job Fetching and Filtering
    app_filter = set(apps.split(',')) if apps else None
    tag_filter = set(tags.split(',')) if tags else None
    
    all_runnable_jobs = database.get_jobs(db_path, status='Ready')
    all_runnable_jobs += database.get_jobs(db_path, status='Restart')

    filtered_jobs = []
    for job in all_runnable_jobs:
        passes_app_filter = not app_filter or job['app'] in app_filter
        passes_tag_filter = not tag_filter or job['tag'] in tag_filter
        if passes_app_filter and passes_tag_filter:
            filtered_jobs.append(job)

    if not filtered_jobs:
        logger.info("No runnable jobs found matching the specified filters. Exiting.")
        parsl.dfk().cleanup()
        return

    logger.info(f"Found {len(filtered_jobs)} jobs to execute.")

    # Initialize Resource Manager
    system_config = get_system_config(config_name)
    resource_manager = system_config.create_resource_manager()
    logger.info(f"Initialized resource manager for {config_name}")

    # Pre-load App Contexts
    unique_app_names = list(set(job['app'] for job in filtered_jobs))
    app_instances = {}
    app_configs = {}

    for app_name in unique_app_names:
        try:
            from parslbox.apps.app_registry import get_app_instance
            app_instance = get_app_instance(app_name)
            app_config = load_app_config(app_name=app_name, system_name=config_name)
            
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
            
        logger.info(f"Submitting Job ID {job_id}...")
        
        # Check dependencies first
        try:
            parents_done = are_parents_done(job, db_path)
            
            if not parents_done:
                logger.info(f"Job {job_id}: Parent dependencies not satisfied, adding to backlog")
                resource_manager.add_to_backlog(job_id)
                continue
                
        except Exception as e:
            logger.error(f"Job {job_id}: Error checking parent dependencies: {e}")
            database.update_jobs(db_path, job_ids=[job_id], status="Failed")
            continue
        
        # Allocate resources
        try:
            assignment = resource_manager.assign_resources(job)
            logger.info(f"Job {job_id}: Allocated resources - {assignment.get_summary()}")
            
            # Create Parsl future with pre-loaded contexts
            success = create_parsl_future(
                job, app_instances[app_name], app_configs[app_name], 
                config_name, db_path, scheduler, resource_manager, futures, system_config
            )
            
            # Future created successfully - no additional tracking needed
            # Dependencies are now checked via database
            
            # Add throttling to prevent database write storms
            time.sleep(0.1)
            
        except InsufficientResources as e:
            # This is NOT an error - just temporary resource unavailability
            logger.info(f"Job {job_id}: Resources temporarily unavailable, added to backlog: {e}")
            # The resource manager has already added the job to the backlog queue
            # Don't mark as failed - the job will be scheduled when resources become available
            continue
            
        except Exception as e:
            # This IS an actual error (invalid spec, system error, etc.)
            logger.error(f"Job {job_id}: Failed to allocate resources: {e}")
            database.update_jobs(db_path, job_ids=[job_id], status="Failed")
            continue

    # Await and Process Results (Dynamic future handling)
    # Create mapping from futures to their metadata
    # Create new futures for rescheduled backlogged jobs in a while loop
    fut_to_item = {item['future']: item for item in futures}
    
    logger.info(f"Starting to process {len(fut_to_item)} initial jobs...")
    
    # Track last recovery attempt time for periodic node recovery
    last_recovery_attempt = time.time()
    recovery_interval = 60  # Attempt recovery every 60 seconds

    while fut_to_item:
        try:
            # Periodically attempt to recover quarantined nodes
            current_time = time.time()
            if current_time - last_recovery_attempt >= recovery_interval:
                recovered_nodes = resource_manager.attempt_node_recovery()
                if recovered_nodes:
                    logger.info(f"Recovered {len(recovered_nodes)} nodes from quarantine: {recovered_nodes}")
                last_recovery_attempt = current_time
            
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
                
                # If job succeeded, run post-processing
                if job_status == "Done":
                    logger.info(f"Job {job_id}: Success check passed. Running post-processing...")
                    try:
                        final_status = app_instance.postprocess(job_id=job_id, job_path=job_path, db_path=db_path)
                        
                        # Validate and use postprocess result if it returns a valid status
                        if final_status:
                            validated_final_status = validate_and_normalize_status(final_status, job_id)
                            job_status = validated_final_status
                        # If postprocess returns None/empty, keep the check_success result
                        
                    except Exception as e:
                        logger.error(f"Job {job_id}: Post-processing failed: {e}")
                        job_status = "Failed"
                else:
                    logger.info(f"Job {job_id}: Success check failed. Skipping post-processing.")
                
                # Update database with final validated status
                database.update_jobs(db_path, job_ids=[job_id], status=job_status)
                logger.info(f"Job {job_id}: Final status set to '{job_status}'.")
                
                # Free resources with health tracking based on job outcome
                try:
                    job_succeeded = (job_status == "Done")
                    resource_manager.free_resources_with_health_check(
                        job_id=job_id, 
                        job_succeeded=job_succeeded, 
                        error_message=error_message if not job_succeeded else None
                    )
                    logger.info(f"Job {job_id}: Finished running. Freed allocated resources.")
                    
                    # Filter backlog by dependency satisfaction, then schedule
                    try:
                        dependency_ready_jobs = get_dependency_ready_jobs(resource_manager.backlog, db_path)
                        
                        # Schedule only dependency-ready jobs
                        rescheduled_jobs = resource_manager.schedule_backlog(dependency_ready_jobs)
                    except Exception as e:
                        logger.error(f"Error during backlog scheduling: {e}")
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
                                config_name, db_path, scheduler, resource_manager, 
                                new_futures, system_config
                            )
                            
                            # Add each new future to tracking dict
                            for new_item in new_futures:
                                new_fut = new_item['future']
                                fut_to_item[new_fut] = new_item
                                logger.info(f"Added rescheduled job {rescheduled_job_id} to tracking (total active: {len(fut_to_item)})")
                        else:
                            logger.error(f"App context not found for rescheduled job {rescheduled_job_id}")
                            database.update_jobs(db_path, job_ids=[rescheduled_job_id], status="Failed")
                    
                    # Log comprehensive status after rescheduling
                    status = resource_manager.get_resource_status()
                    dependency_ready_count = len(dependency_ready_jobs) if 'dependency_ready_jobs' in locals() else 0
                    logger.info(f"Run Status: jobs running {len(fut_to_item)}, jobs backlogged {status['backlogged_jobs']}, dependency ready jobs {dependency_ready_count}")
                    logger.info(f"Resource Status: Total {status['available_gpus']} GPUs and {status['available_cpu_capacity']:.1f} cores available on {status['available_nodes']} nodes")
                            
                except Exception as e:
                    logger.error(f"Job {job_id}: Failed to free resources or schedule backlog: {e}")
                
                # Break to refresh as_completed() with new futures
                break
                        
        except TimeoutError:
            # No futures completed in timeout period, continue waiting
            logger.debug(f"Waiting for {len(fut_to_item)} jobs to complete...")
            continue

    logger.info("All jobs completed, including rescheduled ones")
    
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
    logger.info("--- parslbox orchestrator finished ---")
