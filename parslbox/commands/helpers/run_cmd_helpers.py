import logging
import json
import signal
import sys
from datetime import datetime
from pathlib import Path

import parsl
from parslbox.database import database

# Valid job status values (stored in lowercase for comparison)
VALID_JOB_STATUSES = ["ready", "done", "failed", "restart", "running", "submitted", "warning"]


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


def create_shutdown_handler(status_buffer, logger, parsl_loaded_flag):
    """
    Create a signal handler that flushes the status buffer on termination.
    
    This handler is called when the batch job receives SIGTERM (typically when
    walltime is exceeded) or SIGINT (manual interruption). It ensures that all
    buffered status updates are written to the database before the process exits.
    
    Args:
        status_buffer: StatusBuffer instance to flush
        logger: Logger instance for logging
        parsl_loaded_flag: Dict with 'loaded' key tracking Parsl state
    
    Returns:
        Signal handler function
    """
    def handler(signum, frame):
        signal_name = signal.Signals(signum).name
        logger.warning(f"Received {signal_name} signal - initiating emergency shutdown")
        
        try:
            # Flush status buffer first (most critical operation)
            logger.info("Emergency flush: Writing buffered job statuses to database...")
            count = status_buffer.flush_all()
            if count > 0:
                logger.info(f"Emergency flush: Successfully updated {count} job(s)")
            else:
                logger.info("Emergency flush: No pending updates to write")
        except Exception as e:
            logger.error(f"Emergency flush: Failed to write status updates - {e}")
        
        try:
            # Clean up Parsl if it was loaded
            if parsl_loaded_flag.get('loaded', False):
                logger.info("Emergency shutdown: Cleaning up Parsl resources...")
                parsl.dfk().cleanup()
                logger.info("Emergency shutdown: Parsl cleanup complete")
        except Exception as e:
            logger.error(f"Emergency shutdown: Parsl cleanup failed - {e}")
        
        logger.warning(f"Emergency shutdown complete - exiting with code 130")
        sys.exit(130)
    
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
