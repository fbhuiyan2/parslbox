import logging
from pathlib import Path
from parslbox.apps.appbase import AppBase


class JuliaApp(AppBase):
    """
    Julia application implementation for ParslBox.

    Executes user-provided Julia scripts. The Julia project environment
    should be activated via the env_file (export JULIA_PROJECT=...)
    or via the --project flag configured in the app config.

    Usage:
        pbx add --app julia --input run_ace_fit.jl --config polaris --envfile env_setup.sh
    """

    # App configuration
    INPUT_REQUIRED = True
    DFLT_INPUT = None
    USES_MPI = True   # Launches via srun/mpirun so scripts are constrained
                      # to assigned nodes. Works with or without MPI.jl.

    @classmethod
    def get_default_ranks_per_node(cls, ngpus, num_nodes, system_config):
        return 1

    def get_command_template(self, **kwargs) -> str:
        """
        Construct Julia execution command.

        Julia command format: {mpi_prefix} {executable} {in_file}

        Args:
            **kwargs: Contains mpi_prefix, in_file, and other parameters

        Returns:
            str: Julia execution command
        """
        mpi_prefix = kwargs['mpi_prefix']
        mpi_opts_str = kwargs['mpi_opts_str']
        in_file = kwargs['in_file']
        executable = kwargs.get('executable') or 'julia'

        logger = logging.getLogger(__name__)
        cmd = f"{mpi_prefix} {mpi_opts_str} {executable} {in_file}"
        logger.info(f"Julia command: {cmd}")

        return cmd

    def get_additional_setup(self, **kwargs) -> str:
        """
        Julia-specific setup: Export MPI command for scripts that need it.

        Args:
            **kwargs: Contains mpi_prefix and other parameters

        Returns:
            str: Environment variable export command
        """
        mpi_prefix = kwargs.get('mpi_prefix', '')

        if mpi_prefix:
            return f"export PBX_MPI_PREFIX='{mpi_prefix}'"

        return ""

    def check_success(self, job_id: int, job_path: Path, db_path: Path,
                      error_message: str = None) -> str:
        """
        Check if Julia job completed successfully.

        Reads the PBX_JOB_STATUS_REPORT file written by the user script.
        If the file is not found, the job is marked as Failed.
        The status file is deleted after reading.

        Julia scripts should write the status file directly:
            open("PBX_JOB_STATUS_REPORT", "w") do f
                write(f, "Done")
            end

        Args:
            job_id: The job ID
            job_path: Path to the job directory
            db_path: Path to the database file
            error_message: Error message from fut.result() if any

        Returns:
            str: Final job status ('Done' or 'Failed')
        """
        logger = logging.getLogger(__name__)

        if error_message:
            logger.error(f"Job {job_id}: Julia script execution failed: {error_message}")
            return "Failed"

        # Read status report file written by user script
        from parslbox.apps.utils import PBX_STATUS_FILE
        status_file = job_path / PBX_STATUS_FILE
        if not status_file.is_file():
            logger.warning(f"Job {job_id}: {PBX_STATUS_FILE} not found. Marking as Failed.")
            return "Failed"
        try:
            status = status_file.read_text().strip()
            logger.info(f"Job {job_id}: Status report: {status}")
            return status
        finally:
            status_file.unlink(missing_ok=True)

    def postprocess(self, job_id: int, job_path: Path, db_path: Path) -> str:
        """
        Post-processing for a Julia job.

        Args:
            job_id: The job ID
            job_path: Path to the job directory
            db_path: Path to the database file

        Returns:
            str: Status after post-processing ('Done')
        """
        # logger = logging.getLogger(__name__)
        # logger.info(f"Job {job_id}: Post-processing started.")
        # logger.info(f"Job {job_id}: Post-processing completed.")
        # return "Done"
        pass
