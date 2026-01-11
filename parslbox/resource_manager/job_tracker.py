"""
JobTracker for ParslBox Resource Manager

In-memory job registry for efficient dependency checking and status tracking.
Eliminates database reads during job orchestration.
"""

import logging
from typing import Dict, List, Optional
from parslbox.commands.helpers.run_cmd_helpers import parse_parents

logger = logging.getLogger(__name__)


class JobTracker:
    """
    In-memory job registry for efficient dependency checking and status tracking.
    
    This class maintains a complete job registry in memory to eliminate database
    reads during job orchestration. It provides fast lookups for job information
    and dependency checking.
    """
    
    def __init__(self, initial_jobs: List[dict]):
        """
        Initialize JobTracker with initial job list.
        
        Args:
            initial_jobs: List of job dictionaries from database
        """
        # Main job registry - key: job_id, value: job dict
        self.jobs = {job['job_id']: job.copy() for job in initial_jobs}
        
        logger.info(f"Initialized JobTracker with {len(self.jobs)} jobs")
    
    def update_job_status(self, job_id: int, status: str) -> None:
        """
        Update job status in the registry.
        
        Args:
            job_id: Job ID to update
            status: New status value
        """
        if job_id in self.jobs:
            old_status = self.jobs[job_id]['status']
            self.jobs[job_id]['status'] = status
            logger.debug(f"Updated job {job_id} status: {old_status} -> {status}")
        else:
            logger.warning(f"Attempted to update status for unknown job {job_id}")
    
    def get_job(self, job_id: int) -> Optional[dict]:
        """
        Get job by ID.
        
        Args:
            job_id: Job ID to retrieve
            
        Returns:
            Job dictionary if found, None otherwise
        """
        return self.jobs.get(job_id)
    
    def get_jobs_by_ids(self, job_ids: List[int]) -> List[dict]:
        """
        Get multiple jobs by IDs.
        
        Args:
            job_ids: List of job IDs to retrieve
            
        Returns:
            List of job dictionaries for found jobs
        """
        return [self.jobs[job_id] for job_id in job_ids if job_id in self.jobs]
    
    def are_parents_done(self, job_id: int) -> bool:
        """
        Check if all parent dependencies are satisfied for a job.
        
        Args:
            job_id: Job ID to check dependencies for
            
        Returns:
            True if all parents are done, False otherwise
        """
        job = self.jobs.get(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found in JobTracker")
            return False
            
        parents_str = job.get('parents')
        if not parents_str:
            return True  # No dependencies
        
        try:
            parent_ids = parse_parents(parents_str)
        except Exception as e:
            logger.error(f"Error parsing parents for job {job_id}: {e}")
            return False
        
        for parent_id in parent_ids:
            parent_job = self.jobs.get(parent_id)
            if not parent_job:
                logger.error(f"Job {job_id} has non-existent parent {parent_id}")
                return False
            
            if parent_job['status'] != 'Done':
                return False  # At least one parent is not done
        
        return True  # All parents are done
    
    def get_dependency_ready_jobs(self, job_ids: List[int]) -> List[dict]:
        """
        Filter jobs whose dependencies are satisfied.
        
        Args:
            job_ids: List of job IDs to check
            
        Returns:
            List of job dictionaries for jobs whose dependencies are satisfied
        """
        ready_jobs = []
        
        for job_id in job_ids:
            try:
                if self.are_parents_done(job_id):
                    job = self.jobs.get(job_id)
                    if job:
                        ready_jobs.append(job)
            except Exception as e:
                logger.error(f"Error checking dependencies for job {job_id}: {e}")
                continue
        
        logger.debug(f"Found {len(ready_jobs)} dependency-ready jobs out of {len(job_ids)} checked")
        return ready_jobs
    
    def get_job_count(self) -> int:
        """Get total number of jobs in tracker."""
        return len(self.jobs)
    
    def get_jobs_by_status(self, status: str) -> List[dict]:
        """
        Get all jobs with a specific status.
        
        Args:
            status: Status to filter by
            
        Returns:
            List of job dictionaries with the specified status
        """
        return [job for job in self.jobs.values() if job['status'] == status]
    
    def get_status_summary(self) -> Dict[str, int]:
        """
        Get summary of job statuses.
        
        Returns:
            Dictionary mapping status -> count
        """
        status_counts = {}
        for job in self.jobs.values():
            status = job['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return status_counts
