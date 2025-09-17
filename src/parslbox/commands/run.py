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

from parslbox.configs.loader import load_config
from parslbox.helpers.logging_utils import setup_logging
from parslbox.helpers.config_utils import load_app_config
from parslbox.helpers import database, path_utils


app = typer.Typer()

def get_default_run_dir() -> Path:
    """Generate default run directory with current time and date in hhmmss_ddmmyy format."""
    now = datetime.now()
    time_str = now.strftime("%H%M%S")  # hhmmss format (hours + minutes + seconds)
    date_str = now.strftime("%d%m%y")  # ddmmyy format
    dir_name = f"{time_str}_{date_str}"
    return Path.home() / ".parslbox" / "runs" / dir_name / "log.pbx"

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
    # 1. Initialization
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

    try:
        parsl_config, scheduler = load_config(name=config_name, run_dir=run_dir, retries=retries)
        parsl.load(parsl_config)
        logger.info(f"Successfully loaded Parsl config '{config_name}'.")
    except (FileNotFoundError, ValueError, parsl.errors.ConfigurationError) as e:
        logger.error(f"Failed to load Parsl configuration: {e}")
        raise typer.Exit(code=1)

    # 2. Job Fetching and Filtering
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

    # 3. Group Jobs by Application
    grouped_jobs = {}
    for job in filtered_jobs:
        app_name = job['app']
        if app_name not in grouped_jobs:
            grouped_jobs[app_name] = []
        grouped_jobs[app_name].append(job)

    # 4. Dynamic Plugin Loading and Execution Loop
    futures = []
    for app_name, jobs_list in grouped_jobs.items():
        logger.info(f"Processing {len(jobs_list)} jobs for application: '{app_name}'")
        try:
            # Dynamically import the app module (the "plugin")
            app_module = importlib.import_module(f"parslbox.apps.{app_name}")
            parsl_app_func = getattr(app_module, 'parsl_app')
            check_success_func = getattr(app_module, 'check_success')
            postprocess_func = getattr(app_module, 'postprocess')
            app_config = load_app_config(app_name=app_name, system_name=config_name)
        except (ModuleNotFoundError, AttributeError, ValueError, FileNotFoundError) as e:
            logger.error(f"Could not load plugin for app '{app_name}': {e}. Skipping these jobs.")
            job_ids_to_fail = [j['job_id'] for j in jobs_list]
            database.update_jobs(db_path, job_ids=job_ids_to_fail, status="Failed")
            continue

        # Inner submission loop for this app
        for job in jobs_list:
            job_id = job['job_id']
            logger.info(f"Submitting Job ID {job_id}...")
            database.update_jobs(db_path, job_ids=[job_id], status="Submitted")
            if scheduler == "PBS":
                PBS_JOB_ID = os.environ.get('PBS_JOBID', f'local_{int(time.time())}')
                database.update_jobs(db_path, job_ids=[job_id], sched_job_id=PBS_JOB_ID)
            # Need to add clause for "SLURM" too
            
            fut = parsl_app_func(
                job_id=job_id,
                job_path=Path(job['path']),
                db_path=db_path,
                ngpus=job['ngpus'],
                app_config=app_config,
                config_name=config_name,
                stdout=(str(Path(job['path']) / "pbx.out"), 'w'),
                stderr=(str(Path(job['path']) / "pbx.out"), 'a')
            )
            futures.append({'future': fut, 'job': job, 'check_success': check_success_func, 'postprocessor': postprocess_func})

    # 5. Await and Process Results
    logger.info(f"Waiting for {len(futures)} submitted jobs to complete...")

    for item in futures:
        fut, job, check_success, postprocessor = item['future'], item['job'], item['check_success'], item['postprocessor']
        job_id, job_path = job['job_id'], Path(job['path'])
        
        try:
            fut.result()  # Wait for the Parsl app to finish
            job_status = check_success(job_id=job_id, job_path=job_path, db_path=db_path)
            if job_status and job_status != 'Failed':
                database.update_jobs(db_path, job_ids=[job_id], status=job_status)
                logger.info(f"Job {job_id} execution finished. Running post-processing...")
                postprocessor(job_id=job_id, job_path=job_path, db_path=db_path)
            elif job_status == 'Failed':
                database.update_jobs(db_path, job_ids=[job_id], status="Failed")
        
        except Exception as e:      
            logger.error(f"Job {job_id} hit following error: {e}")
            # Sometimes apps can exit ungracefully even after a good run
            logger.info(f"Job {job_id} checking job success...")
            job_status = check_success(job_id=job_id, job_path=job_path, db_path=db_path)
            if job_status and job_status != 'Failed':
                database.update_jobs(db_path, job_ids=[job_id], status=job_status)
                logger.info(f"Job {job_id} completed successfully. Running post-processing...")
                postprocessor(job_id=job_id, job_path=job_path, db_path=db_path)
            elif job_status == 'Failed':
                logger.error(f"Job {job_id} Failed.")
                database.update_jobs(db_path, job_ids=[job_id], status="Failed")

    # 6. Cleanup
    parsl.dfk().cleanup()
    logger.info("--- parslbox orchestrator finished ---")
