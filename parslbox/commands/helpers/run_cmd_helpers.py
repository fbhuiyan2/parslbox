import logging
import json
from datetime import datetime
from pathlib import Path

from parslbox.database import database

# Valid job status values (stored in lowercase for comparison)
VALID_JOB_STATUSES = ["ready", "done", "failed", "restart", "running", "submitted", "warning"]


def get_default_run_dir() -> Path:
    """Generate default run directory with current time and date in hhmmss_ddmmyy format."""
    now = datetime.now()
    time_str = now.strftime("%H%M%S")  # hhmmss format (hours + minutes + seconds)
    date_str = now.strftime("%d%m%y")  # ddmmyy format
    dir_name = f"{time_str}_{date_str}"
    return Path.home() / ".parslbox" / "runs" / dir_name / "log.pbx"

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
