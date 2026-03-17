import logging
import json
import signal
import sys
from datetime import datetime
from pathlib import Path

import parsl
from parslbox.database import database

# Valid job status values (stored in lowercase for comparison)
VALID_JOB_STATUSES = ["ready", "done", "failed", "killed", "restart", "running", "submitted", "warning"]


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


def create_shutdown_handler(status_buffer, logger, parsl_loaded_flag, job_tracker=None):
    """
    Create a signal handler that flushes the status buffer on termination.
    
    This handler is called when the batch job receives SIGTERM (typically when
    walltime is exceeded) or SIGINT (manual interruption). It ensures that all
    buffered status updates are written to the database before the process exits.
    
    Includes timeout protection via signal.alarm() to force exit if cleanup hangs.
    
    Args:
        status_buffer: StatusBuffer instance to flush
        logger: Logger instance for logging
        parsl_loaded_flag: Dict with 'loaded' key tracking Parsl state
    
    Returns:
        Signal handler function
    """
    def handler(signum, frame):
        import time
        
        signal_name = signal.Signals(signum).name
        start_time = time.time()
        
        logger.warning(f"=" * 80)
        logger.warning(f"Received {signal_name} signal - initiating emergency shutdown")
        logger.warning(f"Handler START at timestamp {start_time}")
        logger.warning(f"=" * 80)
        
        # Set 10-second alarm to force exit if cleanup hangs
        # This prevents the process from being killed by SIGKILL before status updates are written
        logger.info("Setting 10-second alarm for forced exit protection")
        signal.alarm(10)
        
        # STEP 0: Get active job IDs from in-memory JobTracker.
        # Only tracks THIS instance's jobs, so other PBX batch jobs
        # sharing the same database are unaffected.
        active_ids = []
        if job_tracker:
            try:
                active_jobs = (
                    job_tracker.get_jobs_by_status("Running")
                    + job_tracker.get_jobs_by_status("Submitted")
                )
                active_ids = [j['job_id'] for j in active_jobs]
                logger.info(f"Emergency shutdown: Step 0 - Found {len(active_ids)} active jobs")
            except Exception as e:
                logger.error(f"Emergency shutdown: Step 0 - Failed to get active jobs: {e}")

        try:
            # STEP 1: Flush status buffer (clears any stale buffered entries to DB)
            logger.info(f"Emergency shutdown: Step 1/3 - Flushing status buffer...")
            flush_start = time.time()

            count = status_buffer.flush_all()

            flush_duration = time.time() - flush_start
            if count > 0:
                logger.info(f"Emergency shutdown: Step 1/3 - Successfully flushed {count} job(s) in {flush_duration:.3f}s")
            else:
                logger.info(f"Emergency shutdown: Step 1/3 - No pending updates to flush ({flush_duration:.3f}s)")

        except Exception as e:
            logger.error(f"Emergency shutdown: Step 1/3 - Status buffer flush FAILED: {e}")
            logger.error(f"Emergency shutdown: Failed at timestamp {time.time()}")

        # STEP 2: Mark active jobs as Killed (direct DB write, AFTER flush
        # so nothing can overwrite it)
        if active_ids:
            try:
                database.update_jobs(status_buffer.db_path, job_ids=active_ids, status='Killed')
                logger.info(f"Emergency shutdown: Step 2/3 - Marked {len(active_ids)} active jobs as Killed")
            except Exception as e:
                logger.error(f"Emergency shutdown: Step 2/3 - Failed to mark active jobs: {e}")

        try:
            # STEP 3: Clean up Parsl resources
            if parsl_loaded_flag.get('loaded', False):
                logger.info(f"Emergency shutdown: Step 3/3 - Cleaning up Parsl resources...")
                cleanup_start = time.time()
                
                parsl.dfk().cleanup()
                
                cleanup_duration = time.time() - cleanup_start
                logger.info(f"Emergency shutdown: Step 3/3 - Parsl cleanup complete in {cleanup_duration:.3f}s")
            else:
                logger.info("Emergency shutdown: Step 3/3 - Parsl not loaded, skipping cleanup")
                
        except Exception as e:
            logger.error(f"Emergency shutdown: Step 3/3 - Parsl cleanup FAILED: {e}")
            logger.error(f"Emergency shutdown: Failed at timestamp {time.time()}")
            # Don't re-raise - we want to exit cleanly even if Parsl cleanup fails
        
        finally:
            # Cancel the alarm since we completed successfully
            signal.alarm(0)
            
            total_duration = time.time() - start_time
            logger.warning(f"=" * 80)
            logger.warning(f"Emergency shutdown complete in {total_duration:.3f}s - exiting with code 130")
            logger.warning(f"Exit timestamp: {time.time()}")
            logger.warning(f"=" * 80)
            sys.exit(130)
    
    return handler


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
