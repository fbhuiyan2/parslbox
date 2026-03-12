"""
Programmatic API for ParslBox

This module provides a clean Python API for all ParslBox functionality,
allowing programmatic access to job management, execution, and monitoring.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sqlite3
import logging
import os
import time
import subprocess
import yaml
from datetime import datetime
from concurrent.futures import as_completed

from parslbox.database import database
from parslbox.utils import path_utils
from parslbox.utils import pbx_config_utils as config_utils
from parslbox.apps.app_registry import (
    get_app_config,
    is_app_registered,
    get_app_instance,
    get_registered_apps,
)
from parslbox.system_configs.loader import get_system_config, load_config
from parslbox.resource_manager.exceptions import InsufficientResources
from parslbox.resource_manager.models import create_job_resource_spec
from parslbox.utils.logging_utils import setup_logging
from parslbox.commands.add import add_jobs
from parslbox.commands.update import update_jobs
from parslbox.commands.qsub import submit_to_scheduler
from parslbox.commands.sbatch import submit_to_slurm
import parsl


class ParslBoxError(Exception):
    """Base exception for ParslBox API errors."""

    pass


class ValidationError(ParslBoxError):
    """Exception raised for validation errors."""

    pass


class JobNotFoundError(ParslBoxError):
    """Exception raised when a job is not found."""

    pass


class ParslBox:
    """
    Main API class for ParslBox functionality.

    This class provides programmatic access to all ParslBox features including
    job management, execution, and monitoring.

    Example:
        >>> from parslbox.api import ParslBox
        >>> pbx = ParslBox()
        >>> job_id = pbx.add_job(
        ...     path="/path/to/simulation",
        ...     app="lammps-kk",
        ...     config="polaris",
        ...     ngpus=2,
        ...     tag="production"
        ... )
        >>> jobs = pbx.list_jobs(status="Ready", app="lammps-kk")
        >>> pbx.update_job(job_id, status="Submitted")
    """

    def __init__(
        self, db_path: Optional[Path] = None, config_path: Optional[Path] = None
    ):
        """
        Initialize ParslBox API.

        Args:
            db_path: Optional path to database file. If None, uses default.
            config_path: Optional path to config file. If None, uses default.
        
        Raises:
            FileNotFoundError: If config file does not exist.
        """
        # Set paths
        if db_path is None:
            db_path = path_utils.DB_FILE
        if config_path is None:
            config_path = path_utils.PBX_CONFIG_FILE

        self.db_path = db_path
        self.config_path = config_path

        # Initialize database (creates if doesn't exist, keeps if exists)
        database.initialize_database(self.db_path)
        
        # Check that config exists (API requires config to exist)
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found at: {self.config_path}\n"
                f"Please run 'pbx config' to create a configuration file, "
                f"or provide config_path parameter to an existing config."
            )

    # ==================== Job Management Methods ====================

    def add_jobs(
        self,
        paths: List[str],
        app: str,
        config: str,
        tag: Optional[str] = None,
        input_file: Optional[str] = None,
        ngpus: int = 0,
        nnodes: int = 1,
        node_occupancy: Optional[float] = None,
        ranks_per_node: Optional[int] = None,
        mpi_opts: Optional[str] = None,
        env_file: Optional[str] = None,
        parents: Optional[List[int]] = None,
        parent_tag: Optional[str] = None,
        status: str = "Ready",
    ) -> Tuple[List[int], List[Tuple[str, str]], Dict[str, List[str]]]:
        """
        Add one or more jobs to the database.

        Args:
            paths: List of paths to job directories (or single path as list)
            app: Application type (e.g., 'lammps-kk', 'vasp', 'python')
            config: System configuration name (e.g., 'polaris', 'sophia')
            tag: Optional tag to categorize the job(s)
            input_file: Input filename for the job(s)
            ngpus: Number of GPUs required (default: 0)
            nnodes: Number of nodes required (default: 1)
            node_occupancy: Node occupancy fraction for CPU-only jobs (0.0-1.0)
            ranks_per_node: Number of MPI ranks per node
            mpi_opts: Additional MPI options
            env_file: Path to environment setup file
            parents: List of parent job IDs
            parent_tag: Tag to wait for (all jobs with this tag must be Done)
            status: Initial job status (default: 'Ready')

        Returns:
            Tuple of (successful_job_ids, failed_jobs, msg_log) where:
            - successful_job_ids: List of created job IDs
            - failed_jobs: List of tuples (path, error_message) for failed jobs
            - msg_log: Dictionary with 'warnings' and 'info' lists

        Examples:
            # Add multiple jobs with partial failure handling
            job_ids, failures, msg_log = pbx.add_jobs(["/path/job1", "/invalid/path"], app="lammps-kk", config="polaris")
            print(f"Successfully added {len(job_ids)} jobs")
            if failures:
                print(f"Failed to add {len(failures)} jobs:")
                for path, error in failures:
                    print(f"  {path}: {error}")
            for warning in msg_log["warnings"]:
                print(f"Warning: {warning}")
            for info in msg_log["info"]:
                print(f"Info: {info}")
        """
        
        try:
            # Use the core function directly
            successful_job_ids, failed_jobs, msg_log = add_jobs(
                paths=paths,
                app=app,
                config_name=config,
                tag=tag,
                input_file=input_file,
                ngpus=ngpus,
                nnodes=nnodes,
                node_occupancy=node_occupancy,
                ranks_per_node=ranks_per_node,
                mpi_opts=mpi_opts,
                env_file=env_file,
                parents=parents,
                parent_tag=parent_tag,
                status=status,
                db_path=self.db_path,
            )
            return successful_job_ids, failed_jobs, msg_log
        except Exception as e:
            # Convert command ValidationError to API ValidationError if needed
            if "ValidationError" in str(type(e)):
                raise ValidationError(str(e))
            else:
                raise

    def remove_job(self, job_id: int) -> bool:
        """
        Remove a single job from the database.

        Args:
            job_id: Job ID to remove

        Returns:
            True if job was removed, False if not found
        """
        count = database.remove_jobs_by_id(self.db_path, [job_id])
        return count > 0

    def remove_jobs(self, job_ids: List[int]) -> int:
        """
        Remove multiple jobs from the database.

        Args:
            job_ids: List of job IDs to remove

        Returns:
            Number of jobs removed
        """
        return database.remove_jobs_by_id(self.db_path, job_ids)

    def remove_all_jobs(self) -> int:
        """
        Remove all jobs from the database.

        Returns:
            Number of jobs removed
        """
        return database.remove_all_jobs(self.db_path)

    def update_jobs(
        self,
        job_ids: List[int],
        status: Optional[str] = None,
        tag: Optional[str] = None,
        input_file: Optional[str] = None,
        ngpus: Optional[int] = None,
        env_file: Optional[str] = None,
        nnodes: Optional[int] = None,
        node_occupancy: Optional[float] = None,
        ranks_per_node: Optional[int] = None,
        add_deps: Optional[List[int]] = None,
        rm_deps: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[Tuple[int, str]], Dict[str, List[str]]]:
        """
        Update one or more jobs' fields.

        Args:
            job_ids: List of job IDs to update
            status: New status
            tag: New tag
            input_file: New input file
            ngpus: New number of GPUs
            env_file: New environment file path
            nnodes: New number of nodes
            node_occupancy: New node occupancy
            ranks_per_node: New ranks per node
            add_deps: Parent job IDs to add
            rm_deps: Parent job IDs to remove

        Returns:
            Tuple of (successful_job_ids, failed_jobs, msg_log) where:
            - successful_job_ids: List of job IDs that were successfully updated
            - failed_jobs: List of tuples (job_id, error_message) for failed jobs
            - msg_log: Dictionary with 'warnings' and 'info' lists

        Raises:
            ValidationError: If validation fails

        Examples:
            # Update a single job
            updated_ids, failures, msg_log = pbx.update_jobs([123], status="Submitted")
            
            # Update multiple jobs
            updated_ids, failures, msg_log = pbx.update_jobs([123, 124, 125], status="Running")
            for warning in msg_log["warnings"]:
                print(f"Warning: {warning}")
            for info in msg_log["info"]:
                print(f"Info: {info}")
            for job_id, error in failures:
                print(f"Failed job {job_id}: {error}")
        """
        try:
            # Use the core function with API-specific settings
            updated_job_ids, failed_jobs, msg_log = update_jobs(
                job_ids=job_ids,
                status=status,
                tag=tag,
                input_file=input_file,
                ngpus=ngpus,
                env_file=env_file,
                nnodes=nnodes,
                node_occupancy=node_occupancy,
                ranks_per_node=ranks_per_node,
                add_deps=add_deps,
                rm_deps=rm_deps,
                db_path=self.db_path,
            )
            return updated_job_ids, failed_jobs, msg_log
        except Exception as e:
            # Convert command ValidationError to API ValidationError if needed
            if "ValidationError" in str(type(e)):
                raise ValidationError(str(e))
            else:
                raise

    def list_jobs(
        self,
        status: Optional[str] = None,
        app: Optional[str] = None,
        tag: Optional[str] = None,
        path: Optional[str] = None,
        in_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List jobs from the database with optional filtering.

        Args:
            status: Filter by status
            app: Filter by application
            tag: Filter by tag
            path: Filter by path (partial match)
            in_file: Filter by input file (partial match)

        Returns:
            List of job dictionaries
        """
        return database.get_jobs(
            self.db_path, status=status, app=app, tag=tag, path=path, in_file=in_file
        )
    
    def filter_jobs(
        self,
        status: Optional[str] = None,
        app: Optional[str] = None,
        tag: Optional[str] = None,
        path: Optional[str] = None,
        in_file: Optional[str] = None,
    ) -> List[int]:
        """
        Filter jobs and return their IDs.

        Args:
            status: Filter by status
            app: Filter by application
            tag: Filter by tag
            path: Filter by path (partial match)
            in_file: Filter by input file (partial match)

        Returns:
            List of job IDs matching the filters
        """
        jobs = self.list_jobs(
            status=status, app=app, tag=tag, path=path, in_file=in_file
        )
        return [job["job_id"] for job in jobs]
    

    def get_job(self, job_id: int) -> Dict[str, Any]:
        """
        Get a single job by ID.

        Args:
            job_id: Job ID

        Returns:
            Job dictionary

        Raises:
            JobNotFoundError: If job is not found
        """
        jobs = database.get_jobs_by_ids(self.db_path, [job_id])
        if not jobs:
            raise JobNotFoundError(f"Job {job_id} not found")
        return jobs[0]

    def get_jobs_by_ids(self, job_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Get multiple jobs by their IDs.

        Args:
            job_ids: List of job IDs

        Returns:
            List of job dictionaries
        """
        return database.get_jobs_by_ids(self.db_path, job_ids)

    # ==================== Job Execution Methods ====================

    def run(
        self,
        config: str,
        run_dir: Optional[Path] = None,
        apps: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        retries: int = 0,
    ) -> Dict[str, Any]:
        """
        Run Parsl workflows by discovering and executing jobs.

        This is a complex method that handles the full execution lifecycle.
        For detailed execution, see the run command implementation.

        Args:
            config: System configuration name
            run_dir: Directory for Parsl run files (default: timestamped)
            apps: List of apps to run (None for all)
            tags: List of tags to run (None for all)
            retries: Number of retries for failed tasks

        Returns:
            Dictionary with execution summary

        Note:
            This method is a simplified wrapper. For full control, use the
            run command directly or extend this method.
        """
        # This is a placeholder - full implementation would mirror run.py
        # For now, we'll provide a basic structure
        raise NotImplementedError(
            "Full run() implementation requires complex async execution. "
            "Use pbx run command or extend this method."
        )

    def qsub(
        self,
        config: str,
        job_name: str,
        queue: str,
        select: str,
        walltime: int,
        project: str,
        run_dir: Optional[Path] = None,
        apps: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        retries: int = 0,
        loglevel: str = "info",
        sched_opts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate and submit a PBS job script.

        Args:
            config: System configuration name
            job_name: PBS job name
            queue: PBS queue name
            select: PBS select specification (e.g., '4', '2:ncpus=32:ngpus=4', '1:ncpus=16+2:ncpus=32:ngpus=2')
            walltime: Wall time in minutes
            project: Project/account name
            run_dir: Custom run directory (default: timestamped)
            apps: List of apps to run
            tags: List of tags to run
            retries: Number of retries for failed tasks
            loglevel: Logging level
            sched_opts: List of extra PBS directive strings

        Returns:
            Dictionary with submission details including job_id and run_dir

        Raises:
            ValidationError: If configuration is invalid
            FileNotFoundError: If qsub command is not found
        """
        try:
            # Use the core function with API-specific settings
            result = submit_to_scheduler(
                config_name=config,
                job_name=job_name,
                queue=queue,
                select=select,
                walltime=walltime,
                project=project,
                run_dir=run_dir,
                apps=apps,
                tags=tags,
                retries=retries,
                loglevel=loglevel,
                config_path=self.config_path,
                sched_opts=sched_opts,
            )
            return result
        except Exception as e:
            # Convert command ValidationError to API ValidationError if needed
            if "ValidationError" in str(type(e)):
                raise ValidationError(str(e))
            else:
                raise

    def sbatch(
        self,
        config: str,
        job_name: str,
        queue: str,
        select: str,
        walltime: int,
        project: str,
        run_dir: Optional[Path] = None,
        apps: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        retries: int = 0,
        loglevel: str = "info",
        sched_opts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate and submit a SLURM job script.

        Args:
            config: System configuration name
            job_name: SLURM job name
            queue: SLURM partition name
            select: Number of nodes
            walltime: Wall time in minutes
            project: Project/account name
            run_dir: Custom run directory (default: timestamped)
            apps: List of apps to run
            tags: List of tags to run
            retries: Number of retries for failed tasks
            loglevel: Logging level
            sched_opts: List of extra SLURM directive strings

        Returns:
            Dictionary with submission details including job_id and run_dir

        Raises:
            ValidationError: If configuration is invalid
            FileNotFoundError: If sbatch command is not found
        """
        try:
            result = submit_to_slurm(
                config_name=config,
                job_name=job_name,
                queue=queue,
                select=select,
                walltime=walltime,
                project=project,
                run_dir=run_dir,
                apps=apps,
                tags=tags,
                retries=retries,
                loglevel=loglevel,
                config_path=self.config_path,
                sched_opts=sched_opts,
            )
            return result
        except Exception as e:
            if "ValidationError" in str(type(e)):
                raise ValidationError(str(e))
            else:
                raise
