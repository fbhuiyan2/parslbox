"""
Programmatic API for ParslBox

This module provides a clean Python API for all ParslBox functionality,
allowing programmatic access to job management, execution, and monitoring.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
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
from parslbox.commands.helpers.qsub_cmd_helpers import parse_walltime
from parslbox.commands.helpers.cancel_helpers import cancel_pbs_job, cancel_slurm_job
from parslbox.local.project import check_db_name
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
            PbxPathError: If PBX_DB_PATH / PBX_CONFIG_PATH, or an explicitly
                passed db_path / config_path, is a relative path.
            FileNotFoundError: If config file does not exist.
        """
        # Set paths
        path_utils.validate_env_paths()

        if db_path is None:
            db_path = path_utils.DB_FILE
        else:
            db_path = path_utils.require_absolute("db_path", db_path)
        if config_path is None:
            config_path = path_utils.PBX_CONFIG_FILE
        else:
            config_path = path_utils.require_absolute("config_path", config_path)

        self.db_path = db_path
        self.config_path = config_path

        # A local project's database is the -local one; creating the ordinary
        # one beside it would silently start an unrelated, empty project.
        check_db_name(self.db_path)

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
        app_args: Optional[str] = None,
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
            app_args: Extra arguments appended to the application command (requires input_file)

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
                app_args=app_args,
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
        app_args: Optional[str] = None,
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
            app_args: Extra arguments appended to the application command; rebuilds
                each job's in_file from its existing base script

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
                app_args=app_args,
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
        num_nodes: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List jobs from the database with optional filtering.

        Args:
            status: Filter by status
            app: Filter by application
            tag: Filter by tag
            path: Filter by path (partial match)
            in_file: Filter by input file (partial match)
            num_nodes: Filter by number of nodes (exact match)

        Returns:
            List of job dictionaries
        """
        return database.get_jobs(
            self.db_path, status=status, app=app, tag=tag, path=path,
            in_file=in_file, num_nodes=num_nodes
        )
    
    def filter_jobs(
        self,
        status: Optional[str] = None,
        app: Optional[str] = None,
        tag: Optional[str] = None,
        path: Optional[str] = None,
        in_file: Optional[str] = None,
        num_nodes: Optional[int] = None,
        exclude_status: Optional[str] = None,
        exclude_app: Optional[str] = None,
        exclude_tag: Optional[str] = None,
    ) -> List[int]:
        """
        Filter jobs and return their IDs.

        Args:
            status: Filter by status
            app: Filter by application
            tag: Filter by tag (supports `*` glob)
            path: Filter by path (partial match)
            in_file: Filter by input file (partial match)
            num_nodes: Filter by number of nodes (exact match)
            exclude_status: Drop jobs with this status
            exclude_app: Drop jobs with this app
            exclude_tag: Drop jobs with this tag (supports `*` glob)

        Returns:
            List of job IDs matching the filters
        """
        from parslbox.commands.helpers.filter_helpers import apply_excludes
        jobs = self.list_jobs(
            status=status, app=app, tag=tag, path=path, in_file=in_file,
            num_nodes=num_nodes
        )
        jobs = apply_excludes(
            jobs,
            exclude_status=exclude_status,
            exclude_app=exclude_app,
            exclude_tag=exclude_tag,
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
        walltime: Union[int, float, str],
        project: Optional[str] = None,
        run_dir: Optional[Path] = None,
        apps: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        retries: int = 0,
        loglevel: str = "info",
        sched_opts: Optional[List[str]] = None,
        respawn: Optional[int] = None,
        no_local: bool = False,
        push_dirs: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate and submit a PBS job script.

        Args:
            config: System configuration name
            job_name: PBS job name
            queue: PBS queue name
            select: PBS select specification (e.g., '4', '2:ncpus=32:ngpus=4', '1:ncpus=16+2:ncpus=32:ngpus=2')
            walltime: Wall time (default: minutes). Supports h/d suffixes (e.g., 90, 4.25h, 3.5d)
            project: Project/account name
            run_dir: Custom run directory (default: timestamped)
            apps: List of apps to run
            tags: List of tags to run
            retries: Number of retries for failed tasks
            loglevel: Logging level
            sched_opts: List of extra PBS directive strings
            respawn: Enable the self-respawn chain. When set, at walltime the
                orchestrator marks in-flight jobs Restart and auto-resubmits the
                next link. The integer is the number of remaining auto-resubmissions
                in the chain (decremented per link; 0 = no resubmit, chain ends).
                Pass None (default) to disable the chain entirely.
            no_local: Submit from this machine even when the database belongs
                to a local project. By default a local project's submission is
                pushed to the remote machine and run there.
            push_dirs: In a local project, also send the matching jobs'
                directories over Globus Transfer before submitting.

        Returns:
            Dictionary with submission details including job_id and run_dir.
            When respawn is set, also includes 'respawn_template_file'.

        Raises:
            ValidationError: If configuration is invalid, or if respawn is
                negative.
            FileNotFoundError: If qsub command is not found
        """
        try:
            walltime_minutes = parse_walltime(str(walltime))
            # Use the core function with API-specific settings
            result = submit_to_scheduler(
                config_name=config,
                job_name=job_name,
                queue=queue,
                select=select,
                walltime=walltime_minutes,
                project=project,
                run_dir=run_dir,
                apps=apps,
                tags=tags,
                retries=retries,
                loglevel=loglevel,
                config_path=self.config_path,
                sched_opts=sched_opts,
                respawn=respawn,
                db_path=self.db_path,
                no_local=no_local,
                push_dirs=push_dirs,
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
        walltime: Union[int, float, str],
        project: Optional[str] = None,
        run_dir: Optional[Path] = None,
        apps: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        retries: int = 0,
        loglevel: str = "info",
        sched_opts: Optional[List[str]] = None,
        respawn: Optional[int] = None,
        no_local: bool = False,
        push_dirs: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate and submit a SLURM job script.

        Args:
            config: System configuration name
            job_name: SLURM job name
            queue: SLURM partition name
            select: Number of nodes
            walltime: Wall time (default: minutes). Supports h/d suffixes (e.g., 90, 4.25h, 3.5d)
            project: Project/account name
            run_dir: Custom run directory (default: timestamped)
            apps: List of apps to run
            tags: List of tags to run
            retries: Number of retries for failed tasks
            loglevel: Logging level
            sched_opts: List of extra SLURM directive strings
            respawn: Enable the self-respawn chain. When set, at walltime the
                orchestrator marks in-flight jobs Restart and auto-resubmits the
                next link. The integer is the number of remaining auto-resubmissions
                in the chain (decremented per link; 0 = no resubmit, chain ends).
                Pass None (default) to disable the chain entirely.
            no_local: Submit from this machine even when the database belongs
                to a local project. By default a local project's submission is
                pushed to the remote machine and run there.
            push_dirs: In a local project, also send the matching jobs'
                directories over Globus Transfer before submitting.

        Returns:
            Dictionary with submission details including job_id and run_dir.
            When respawn is set, also includes 'respawn_template_file'.

        Raises:
            ValidationError: If configuration is invalid, or if respawn is
                negative.
            FileNotFoundError: If sbatch command is not found
        """
        try:
            walltime_minutes = parse_walltime(str(walltime))
            result = submit_to_slurm(
                config_name=config,
                job_name=job_name,
                queue=queue,
                select=select,
                walltime=walltime_minutes,
                project=project,
                run_dir=run_dir,
                apps=apps,
                tags=tags,
                retries=retries,
                loglevel=loglevel,
                config_path=self.config_path,
                sched_opts=sched_opts,
                respawn=respawn,
                db_path=self.db_path,
                no_local=no_local,
                push_dirs=push_dirs,
            )
            return result
        except Exception as e:
            if "ValidationError" in str(type(e)):
                raise ValidationError(str(e))
            else:
                raise

    # ==================== Local Project Methods ====================

    def local_init(
        self,
        directory: Union[str, Path],
        remote_root: str,
        compute_endpoint: Optional[str] = None,
        transfer_local: Optional[str] = None,
        transfer_remote: Optional[str] = None,
        transfer_remote_root: Optional[str] = None,
        remote_config: Optional[str] = None,
        nested_ok: bool = False,
    ) -> Dict[str, Any]:
        """
        Create or adopt a local project in `directory`.

        A local project's database stores remote paths from its first row, so
        jobs authored here can be run on the remote machine without anything
        being rewritten at handover.

        Args:
            directory: Where the project lives on this machine.
            remote_root: Absolute project root on the remote machine.
            compute_endpoint: Globus Compute endpoint UUID on the remote.
            transfer_local: Globus Transfer collection UUID for this machine.
            transfer_remote: Globus Transfer collection UUID for the remote.
            transfer_remote_root: Override for the path the remote collection
                exposes as its own root, e.g. '/lus/flare/projects'. Normally
                left unset: the first directory push detects and records it.
            remote_config: Path to config.yaml on the remote machine.
            nested_ok: Allow creating a project inside another one. False by
                default, because doing it accidentally is more likely than
                doing it on purpose.

        Returns:
            Dict with 'action' ('created', 'adopted' or 'existing'),
            'db_path', 'export_line', and any 'notes'.

        Raises:
            ValidationError: The directory cannot hold a project -- a
                populated database with no project file, a half-deleted
                project, or a parent project with nested_ok left False.
        """
        from parslbox.local import project as project_mod
        try:
            outcome = project_mod.init_project(
                directory,
                remote_root=remote_root,
                compute_endpoint=compute_endpoint,
                transfer_local=transfer_local,
                transfer_remote=transfer_remote,
                transfer_remote_root=transfer_remote_root,
                remote_config=remote_config,
                nested_ok=nested_ok,
            )
        except project_mod.LocalProjectError as e:
            raise ValidationError(str(e))
        proj = outcome["project"]
        return {
            "action": outcome["action"],
            "local_root": str(proj.local_root),
            "remote_root": proj.remote_root,
            "db_path": str(proj.db_path),
            "compute_endpoint": proj.compute_endpoint,
            "export_line": proj.export_line,
            "notes": outcome["notes"],
        }

    def local_status(self, check_remote: bool = True) -> Dict[str, Any]:
        """
        Report which side of a local project is ahead. Changes nothing.

        Args:
            check_remote: Ping the Compute endpoint and fingerprint the remote
                database. Set False to stay offline.

        Returns:
            Dict with 'is_local_project'. When True, also the two roots, the
            job count, local and remote drift since the last sync, endpoint
            reachability, and the PBX_DB_PATH export line.
        """
        from parslbox.local import sync as local_sync
        return local_sync.status(db_path=self.db_path, check_remote=check_remote)

    def local_push(
        self,
        force: bool = False,
        with_dirs: bool = False,
        apps: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        wait_for_dirs: bool = True,
        sync_level: str = "checksum",
    ) -> Dict[str, Any]:
        """
        Send the local database up to the remote machine.

        Args:
            force: Overwrite the remote even if it changed since the last sync.
            with_dirs: Also send matching jobs' directories over Globus Transfer.
            apps: Restrict the directories sent to these apps.
            tags: Restrict the directories sent to these tags.
            wait_for_dirs: Block until the directory transfer finishes.
            sync_level: How Globus decides a file is already at the destination
                -- 'exists', 'size', 'mtime' or 'checksum'. checksum is safest
                but reads every file on both ends first.

        Returns:
            Dict with 'bytes', the remote 'fingerprint', and 'last_sync'. When
            with_dirs is set and the remote collection's root had not been
            recorded yet, also 'transfer_remote_root' and
            'transfer_remote_root_detected'.

        Raises:
            ValidationError: Not a local project; the remote has moved since
                the last sync and force was not set; the Compute endpoint
                could not be reached; or Globus Transfer refused the
                directories, including when no prefix of remote_root resolves
                to the database just written over Compute.
        """
        from parslbox.local import sync as local_sync
        from parslbox.local.compute import ComputeError
        from parslbox.local.project import LocalProjectError
        from parslbox.local.guard import SyncConflict
        from parslbox.local.transfer import TransferError
        try:
            project = local_sync.require_project(self.db_path)
            dirs = None
            report = None
            if with_dirs:
                found = local_sync.job_dirs(project, apps=apps, tags=tags)
                report = found
                dirs = found["present"]
            result = local_sync.push(db_path=self.db_path, force=force, dirs=dirs,
                                     wait_for_dirs=wait_for_dirs,
                                     sync_level=sync_level)
            if with_dirs:
                result["dirs"] = report
            return result
        except (LocalProjectError, SyncConflict, TransferError, ComputeError,
                FileNotFoundError) as e:
            raise ValidationError(str(e))

    def local_pull(self, force: bool = False) -> Dict[str, Any]:
        """
        Bring the remote database down over the local one.

        Args:
            force: Overwrite the local database even if it changed since the
                last sync.

        Returns:
            Dict with 'bytes', the new local 'fingerprint', and 'last_sync'.

        Raises:
            ValidationError: Not a local project, nothing pushed yet, the
                local database has moved since the last sync and force was not
                set, or the Compute endpoint could not be reached.
        """
        from parslbox.local import sync as local_sync
        from parslbox.local.compute import ComputeError
        from parslbox.local.project import LocalProjectError
        from parslbox.local.guard import SyncConflict
        try:
            return local_sync.pull(db_path=self.db_path, force=force)
        except (LocalProjectError, SyncConflict, ComputeError,
                FileNotFoundError) as e:
            raise ValidationError(str(e))

    def qdel(self, jobid: str, grace: int = 30) -> Dict[str, Any]:
        """
        Gracefully cancel a ParslBox PBS job.

        Sends SIGTERM via `qsig`, waits `grace` seconds for the orchestrator
        to reconcile in-flight jobs in the database, then runs `qdel`. After
        the scheduler-level kill, any non-terminal jobs under this batch
        (signal handler did not complete cleanly) are reconciled per state:
        Running→Killed, Submitted→Ready, Resubmitted→Restart.

        Args:
            jobid: PBS job ID.
            grace: Seconds between SIGTERM and hard kill (default 30).

        Returns:
            Dict with keys: success (bool), jobid (str), grace (int),
            reconciled_count (int) on success; success (False), jobid, stage,
            error, reconciled_count on failure.
        """
        return cancel_pbs_job(jobid, grace=grace, db_path=self.db_path)

    def scancel(self, jobid: str, grace: int = 30) -> Dict[str, Any]:
        """
        Gracefully cancel a ParslBox SLURM job.

        Sends SIGTERM to the batch script via `scancel --signal=TERM --batch`,
        waits `grace` seconds for the orchestrator to reconcile in-flight jobs
        in the database, then runs `scancel` to terminate. After the
        scheduler-level kill, any non-terminal jobs under this batch (signal
        handler did not complete cleanly) are reconciled per state:
        Running→Killed, Submitted→Ready, Resubmitted→Restart.

        Args:
            jobid: SLURM job ID.
            grace: Seconds between SIGTERM and hard cancel (default 30).

        Returns:
            Dict with keys: success (bool), jobid (str), grace (int),
            reconciled_count (int) on success; success (False), jobid, stage,
            error, reconciled_count on failure.
        """
        return cancel_slurm_job(jobid, grace=grace, db_path=self.db_path)
