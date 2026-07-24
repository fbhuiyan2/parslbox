"""
StatusBuffer for batching database status updates to reduce contention.

This module provides a simple buffer that collects status updates and 
writes them to the database in batches, significantly reducing the number
of individual database calls.
"""

import logging
from typing import Dict, List, Any
from pathlib import Path

from parslbox.database import database


class StatusBuffer:
    """
    Buffer for batching database status updates.
    
    Collects status updates in memory and writes them to the database
    in batches when flush_all() is called. This reduces database contention
    by minimizing the number of individual database transactions.
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize the status buffer.
        
        Args:
            db_path: Path to the database file
        """
        self.db_path = db_path
        self.status_updates: Dict[int, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_status_update(self, job_id: int, **kwargs) -> None:
        """
        Buffer a status update instead of immediate database write.
        
        Args:
            job_id: Job ID to update
            **kwargs: Status update fields (status, sched_job_id, etc.)
        """
        if job_id not in self.status_updates:
            self.status_updates[job_id] = {}
        
        # Update with new fields, overwriting any existing values
        self.status_updates[job_id].update(kwargs)
        
        self.logger.debug(f"Buffered status update for job {job_id}: {kwargs}")
    
    def flush_all(self) -> int:
        """
        Batch write all pending updates to database.
        
        Groups updates by their field combinations to maximize batching efficiency.
        For example, all jobs being updated to "Running" status will be batched together.
        
        Returns:
            Number of jobs that were updated
        """
        if not self.status_updates:
            return 0
        
        total_jobs_attempted = len(self.status_updates)
        total_updated = 0
        
        # Group updates by their field combinations for efficient batching
        update_groups = self._group_updates_by_fields()
        
        for group_key, job_updates in update_groups.items():
            job_ids = list(job_updates.keys())
            
            # Extract the field combination (handle both tuple and tuple-with-index cases)
            if isinstance(group_key, tuple) and len(group_key) == 2 and isinstance(group_key[1], int):
                field_combo = group_key[0]  # Extract field combo from (field_combo, index) tuple
            else:
                field_combo = group_key  # Direct field combo tuple
            
            # Extract the common field values for this group
            sample_update = next(iter(job_updates.values()))
            update_kwargs = {field: sample_update[field] for field in field_combo}
            
            try:
                rows_updated = database.update_jobs(
                    self.db_path, 
                    job_ids=job_ids, 
                    **update_kwargs
                )
                total_updated += len(job_ids)
                
                fields_str = ", ".join(f"{k}={v}" for k, v in update_kwargs.items())
                self.logger.info(
                    f"StatusBuffer: updated {len(job_ids)} job(s) with {fields_str}"
                )
                
            except Exception as e:
                self.logger.error(
                    f"Failed to batch update {len(job_ids)} jobs with fields {field_combo}: {e}"
                )
                # Fallback to individual updates
                self._fallback_individual_updates(job_ids, job_updates)
                total_updated += len(job_ids)
        
        # Clear buffer after successful flush
        self.status_updates.clear()

        return total_updated
    
    def _group_updates_by_fields(self) -> Dict[tuple, Dict[int, Dict[str, Any]]]:
        """
        Group updates by their field combinations.
        
        Jobs with identical field combinations can be batched together.
        For example, all jobs with only 'status' field can be grouped by status value.
        
        Returns:
            Dictionary mapping field combinations to job updates
        """
        groups = {}
        
        for job_id, updates in self.status_updates.items():
            # Create a tuple of field names as the grouping key
            field_combo = tuple(sorted(updates.keys()))
            
            if field_combo not in groups:
                groups[field_combo] = {}
            
            groups[field_combo][job_id] = updates
        
        # Further group by field values within each field combination
        refined_groups = {}
        
        for field_combo, job_updates in groups.items():
            # Group by the actual values of the fields
            value_groups = {}
            
            for job_id, updates in job_updates.items():
                # Create a tuple of field values as the sub-grouping key
                value_combo = tuple(updates[field] for field in field_combo)
                
                if value_combo not in value_groups:
                    value_groups[value_combo] = {}
                
                value_groups[value_combo][job_id] = updates
            
            # Add each value group as a separate batch
            for i, (value_combo, jobs) in enumerate(value_groups.items()):
                # Use the field_combo tuple directly as the key
                group_key = field_combo
                if len(value_groups) > 1:
                    # If multiple value groups, append index to make unique
                    group_key = (field_combo, i)
                refined_groups[group_key] = jobs
        
        return refined_groups
    
    def _fallback_individual_updates(self, job_ids: List[int], job_updates: Dict[int, Dict[str, Any]]) -> None:
        """
        Fallback to individual database updates if batch update fails.
        
        Args:
            job_ids: List of job IDs that failed to batch update
            job_updates: Dictionary of job updates that failed
        """
        self.logger.warning(f"Falling back to individual updates for {len(job_ids)} jobs")
        
        for job_id in job_ids:
            try:
                updates = job_updates[job_id]
                database.update_jobs(self.db_path, job_ids=[job_id], **updates)
                self.logger.debug(f"Individual update successful for job {job_id}")
            except Exception as e:
                self.logger.error(f"Individual update failed for job {job_id}: {e}")
    
    def get_buffer_size(self) -> int:
        """
        Get the number of jobs currently in the buffer.
        
        Returns:
            Number of jobs with pending updates
        """
        return len(self.status_updates)
    
    def clear_buffer(self) -> None:
        """
        Clear all pending updates without writing to database.
        
        This should only be used in error scenarios where you want to
        discard pending updates.
        """
        cleared_count = len(self.status_updates)
        self.status_updates.clear()
        
        if cleared_count > 0:
            self.logger.warning(f"Cleared {cleared_count} pending status updates without writing to database")
