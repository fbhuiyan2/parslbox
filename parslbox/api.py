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

from parslbox.helpers import database, path_utils
from parslbox.helpers import pbx_config_utils as config_utils
from parslbox.apps.app_registry import (
    get_app_config,
    is_app_registered,
    get_app_instance,
    get_registered_apps,
)
from parslbox.system_configs.loader import get_system_config, load_config
from parslbox.resource_manager.mpi_launcher import compose_mpi_command
from parslbox.resource_manager.exceptions import InsufficientResources
from parslbox.resource_manager.models import create_job_resource_spec
from parslbox.helpers.logging_utils import setup_logging
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
        ...     app="lammps",
        ...     config="polaris",
        ...     ngpus=2,
        ...     tag="production"
        ... )
        >>> jobs = pbx.list_jobs(status="Ready", app="lammps")
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
        """
        # Initialize database and config
        if db_path is None:
            db_path = path_utils.DB_FILE
        if config_path is None:
            config_path = path_utils.PBX_CONFIG_FILE

        self.db_path = db_path
        self.config_path = config_path

        # Ensure database and config are initialized
        database.initialize_database(self.db_path)
        config_utils.initialize_config_file()

    # ==================== Job Management Methods ====================

    def add_job(
        self,
        path: str,
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
    ) -> int:
        """
        Add a new job to the database.

        Args:
            path: Path to the job directory
            app: Application type (e.g., 'lammps', 'vasp', 'python')
            config: System configuration name (e.g., 'polaris', 'sophia')
            tag: Optional tag to categorize the job
            input_file: Input filename for the job
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
            int: The job ID of the newly created job

        Raises:
            ValidationError: If validation fails
            sqlite3.IntegrityError: If job already exists
        """
        # Validate application
        if not is_app_registered(app):
            available_apps = ", ".join(get_registered_apps())
            raise ValidationError(
                f"Unknown application '{app}'. "
                f"Available applications: {available_apps}"
            )

        # Handle input file logic
        try:
            app_config = get_app_config(app)
            input_required = app_config["INPUT_REQUIRED"]
            default_input = app_config["DFLT_INPUT"]

            final_input_file = None

            if input_required and default_input is None:
                # Allow None for programmatic use - validation can happen at runtime
                final_input_file = input_file
            elif input_required and default_input is not None:
                final_input_file = (
                    input_file if input_file is not None else default_input
                )
            else:
                final_input_file = default_input

        except ValueError as e:
            raise ValidationError(str(e))

        # Validate resource parameters
        if node_occupancy is not None and not (0.0 < node_occupancy <= 1.0):
            raise ValidationError("node_occupancy must be between 0.0 and 1.0")

        if ranks_per_node is not None and ranks_per_node < 1:
            raise ValidationError("ranks_per_node must be a positive integer")

        if nnodes < 1:
            raise ValidationError("nnodes must be at least 1")

        # Get system configuration
        try:
            system_config = get_system_config(config)
            gpus_per_node = system_config.GPUS_PER_NODE
        except Exception as e:
            raise ValidationError(f"Could not load system configuration: {e}")

        # Determine final resource parameters
        if nnodes > 1:
            # Multi-node job
            final_num_nodes = nnodes
            final_ngpus = nnodes * gpus_per_node
            final_node_occupancy = 1.0
        else:
            # Single-node job
            if ngpus > 0:
                if ngpus > gpus_per_node:
                    raise ValidationError(
                        f"Requested {ngpus} GPUs but only {gpus_per_node} available per node"
                    )
                final_num_nodes = 1
                final_ngpus = ngpus
                final_node_occupancy = 1.0
            else:
                final_num_nodes = 1
                final_ngpus = 0
                final_node_occupancy = (
                    node_occupancy if node_occupancy is not None else 1.0
                )

        # Calculate ranks_per_node
        if ranks_per_node is None:
            if final_ngpus > 0:
                final_ranks_per_node = 1
            else:
                calculated_ranks = int(
                    system_config.CORES_PER_NODE * final_node_occupancy
                )
                final_ranks_per_node = max(1, calculated_ranks)
        else:
            if final_ngpus > 0:
                final_ranks_per_node = 1
            else:
                final_ranks_per_node = ranks_per_node

        # Handle environment file
        final_env_file = None
        if env_file:
            env_file_path = Path(env_file)
            if not env_file_path.is_absolute():
                env_file_path = Path.cwd() / env_file_path

            if not env_file_path.exists():
                raise ValidationError(f"Environment file '{env_file}' does not exist")

            if not env_file_path.is_file():
                raise ValidationError(f"Environment file '{env_file}' is not a file")

            final_env_file = str(env_file_path.resolve())

        # Handle parent dependencies
        final_parents = []

        if parents:
            final_parents = parents

        if parent_tag:
            tag_jobs = database.get_jobs(self.db_path, tag=parent_tag, status="Done")
            tag_parent_ids = [job["job_id"] for job in tag_jobs]
            final_parents.extend(tag_parent_ids)

        # Validate parent job IDs
        if final_parents:
            existing_jobs = database.get_jobs_by_ids(self.db_path, final_parents)
            existing_ids = {job["job_id"] for job in existing_jobs}
            missing_ids = set(final_parents) - existing_ids

            if missing_ids:
                raise ValidationError(
                    f"Parent job IDs do not exist: {sorted(missing_ids)}"
                )

        # Validate path
        path_obj = Path(path)
        if not path_obj.exists():
            raise ValidationError(f"Path '{path}' does not exist")

        if not path_obj.is_dir():
            raise ValidationError(f"Path '{path}' is not a directory")

        # Add job to database
        try:
            job_id = database.add_job(
                db_path=self.db_path,
                path=str(path_obj.resolve()),
                app=app,
                num_nodes=final_num_nodes,
                ngpus=final_ngpus,
                node_occupancy=final_node_occupancy,
                ranks_per_node=final_ranks_per_node,
                tag=tag,
                in_file=final_input_file,
                mpi_opts=mpi_opts,
                env_file=final_env_file,
                parents=final_parents,
                status=status,
            )
            return job_id
        except sqlite3.IntegrityError:
            raise ValidationError(
                f"Job with path '{path}' and input file '{final_input_file}' already exists"
            )

    def add_jobs(
        self, paths: List[str], app: str, config: str, **kwargs
    ) -> List[Tuple[int, Optional[Exception]]]:
        """
        Add multiple jobs to the database.

        Args:
            paths: List of paths to job directories
            app: Application type
            config: System configuration name
            **kwargs: Additional arguments passed to add_job()

        Returns:
            List of tuples (job_id, error) where error is None if successful
        """
        results = []
        for path in paths:
            try:
                job_id = self.add_job(path=path, app=app, config=config, **kwargs)
                results.append((job_id, None))
            except Exception as e:
                results.append((None, e))
        return results

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

    def remove_job(self, job_id: int) -> bool:
        """
        Remove a job from the database.

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

    def update_job(
        self,
        job_id: int,
        status: Optional[str] = None,
        app: Optional[str] = None,
        tag: Optional[str] = None,
        input_file: Optional[str] = None,
        ngpus: Optional[int] = None,
        env_file: Optional[str] = None,
        nnodes: Optional[int] = None,
        node_occupancy: Optional[float] = None,
        ranks_per_node: Optional[int] = None,
        add_deps: Optional[List[int]] = None,
        rm_deps: Optional[List[int]] = None,
    ) -> bool:
        """
        Update a job's fields.

        Args:
            job_id: Job ID to update
            status: New status
            app: New application
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
            True if job was updated, False if not found

        Raises:
            ValidationError: If validation fails
        """
        # Validate parameters
        if nnodes is not None and nnodes < 1:
            raise ValidationError("nnodes must be at least 1")

        if node_occupancy is not None and not (0.0 < node_occupancy <= 1.0):
            raise ValidationError("node_occupancy must be between 0.0 and 1.0")

        if ranks_per_node is not None and ranks_per_node < 1:
            raise ValidationError("ranks_per_node must be a positive integer")

        # Handle environment file
        final_env_file = None
        if env_file:
            env_file_path = Path(env_file)
            if not env_file_path.is_absolute():
                env_file_path = Path.cwd() / env_file_path

            if not env_file_path.exists():
                raise ValidationError(f"Environment file '{env_file}' does not exist")

            if not env_file_path.is_file():
                raise ValidationError(f"Environment file '{env_file}' is not a file")

            final_env_file = str(env_file_path.resolve())

        # Handle dependencies
        final_parents = None
        if add_deps is not None or rm_deps is not None:
            current_job = self.get_job(job_id)
            existing_parents = database.parse_existing_parents(
                current_job.get("parents")
            )
            updated_parents = existing_parents.copy()

            # Validate parent IDs
            all_parent_ids = []
            if add_deps:
                all_parent_ids.extend(add_deps)
            if rm_deps:
                all_parent_ids.extend(rm_deps)

            if all_parent_ids:
                invalid_ids = database.validate_parent_job_ids(
                    self.db_path, all_parent_ids
                )
                if invalid_ids:
                    raise ValidationError(f"Parent job IDs do not exist: {invalid_ids}")

            # Prevent circular dependencies
            if add_deps and job_id in add_deps:
                raise ValidationError("Job cannot be a parent of itself")

            # Process removals
            if rm_deps:
                updated_parents, _ = database.remove_dependencies(
                    updated_parents, rm_deps
                )

            # Process additions
            if add_deps:
                updated_parents = database.add_dependencies(updated_parents, add_deps)

            final_parents = updated_parents

        # Handle node occupancy vs ngpus conflict
        final_ngpus = ngpus
        final_node_occupancy = node_occupancy

        if node_occupancy is not None:
            current_job = self.get_job(job_id)
            if current_job.get("ngpus", 0) > 0:
                # Setting node occupancy for GPU job - set ngpus to 0
                final_ngpus = 0

        # Update job
        count = database.update_jobs(
            db_path=self.db_path,
            job_ids=[job_id],
            status=status,
            app=app,
            tag=tag,
            in_file=input_file,
            ngpus=final_ngpus,
            env_file=final_env_file,
            num_nodes=nnodes,
            node_occupancy=final_node_occupancy,
            ranks_per_node=ranks_per_node,
            parents=final_parents,
        )

        return count > 0

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
        select: int,
        walltime: int,
        project: str,
        filesystems: Optional[str] = None,
        run_dir: Optional[Path] = None,
        apps: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        retries: int = 0,
    ) -> Dict[str, Any]:
        """
        Generate and submit a PBS job script.

        Args:
            config: System configuration name
            job_name: PBS job name
            queue: PBS queue name
            select: Number of nodes to request
            walltime: Wall time in minutes
            project: Project/account name
            filesystems: Comma-separated list of filesystems
            run_dir: Custom run directory (default: timestamped)
            apps: List of apps to run
            tags: List of tags to run
            retries: Number of retries for failed tasks

        Returns:
            Dictionary with submission details including job_id and run_dir

        Raises:
            ValidationError: If configuration is invalid
            FileNotFoundError: If qsub command is not found
        """
        # Load configuration
        try:
            with open(self.config_path, "r") as f:
                config_data = yaml.safe_load(f)
        except FileNotFoundError:
            raise ValidationError(f"Configuration file not found: {self.config_path}")

        # Validate scheduler template
        if "schedulers" not in config_data or "pbs" not in config_data["schedulers"]:
            raise ValidationError("PBS scheduler template not found in configuration")

        # Validate system configuration
        if config not in config_data:
            raise ValidationError(f"System '{config}' not found in configuration")

        # Get system-specific python environment setup
        system_config = config_data[config]
        pbx_python_env_setup = system_config.get("pbx_python_env_setup", "")

        # Determine run directory
        if run_dir is None:
            now = datetime.now()
            time_str = now.strftime("%H%M%S")
            date_str = now.strftime("%d%m%y")
            dir_name = f"{time_str}_{date_str}"
            run_dir = Path.home() / ".parslbox" / "runs" / dir_name

        # Create run directory
        run_dir.mkdir(parents=True, exist_ok=True)

        # Convert walltime to HH:MM:SS format
        hours = walltime // 60
        mins = walltime % 60
        walltime_formatted = f"{hours:02d}:{mins:02d}:00"

        # Build run options string
        run_options = []
        if apps:
            run_options.append(f"--apps {','.join(apps)}")
        if tags:
            run_options.append(f"--tags {','.join(tags)}")
        if retries > 0:
            run_options.append(f"--retries {retries}")

        run_options_str = " ".join(run_options)

        # Capture environment variables
        pbx_env_vars = ""
        if os.getenv("PBX_DB_PATH"):
            pbx_env_vars += f'export PBX_DB_PATH="{os.getenv("PBX_DB_PATH")}"\n'
        if os.getenv("PBX_CONFIG_PATH"):
            pbx_env_vars += f'export PBX_CONFIG_PATH="{os.getenv("PBX_CONFIG_PATH")}"\n'

        # Prepare template variables
        template_vars = {
            "job_name": job_name,
            "queue": queue,
            "select": select,
            "walltime": walltime_formatted,
            "filesystems": filesystems or "",
            "project": project,
            "pbx_python_env_setup": pbx_python_env_setup,
            "pbx_env_vars": pbx_env_vars,
            "config": config,
            "run_dir": "./",
            "run_options": run_options_str,
        }

        # Get PBS template and format it
        pbs_template = config_data["schedulers"]["pbs"]["template"]

        # Handle optional filesystems directive
        if not filesystems:
            pbs_template = "\n".join(
                line
                for line in pbs_template.split("\n")
                if "#PBS -l filesystems=" not in line
            )

        submit_script = pbs_template.format(**template_vars)

        # Write submit script
        submit_file = run_dir / "submit.sh"
        with open(submit_file, "w") as f:
            f.write(submit_script)

        # Submit the job
        try:
            result = subprocess.run(
                ["qsub", "submit.sh"],
                cwd=run_dir,
                capture_output=True,
                text=True,
                check=True,
            )

            job_id = result.stdout.strip()

            return {
                "success": True,
                "pbs_job_id": job_id,
                "run_dir": str(run_dir),
                "submit_file": str(submit_file),
            }

        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": e.stderr,
                "run_dir": str(run_dir),
                "submit_file": str(submit_file),
            }
        except FileNotFoundError:
            raise ValidationError("qsub command not found. Make sure PBS is available.")
