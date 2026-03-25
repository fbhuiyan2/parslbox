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
    USES_MPI = False  # Julia scripts don't use MPI directly.
                      # PBX wraps the command with a resource launcher to
                      # constrain execution to assigned resources.

    def get_command_template(self, **kwargs) -> str:
        """
        Construct Julia execution command.

        Julia command format: julia {in_file}
        Users can configure a custom executable path (e.g., with --project flag)
        via executable_path in config.yaml, or set JULIA_PROJECT in env_file.

        Args:
            **kwargs: Contains in_file and other parameters

        Returns:
            str: Julia execution command
        """
        in_file = kwargs['in_file']
        executable = kwargs.get('executable', 'julia')

        logger = logging.getLogger(__name__)
        cmd = f"{executable} {in_file}"
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
        else:
            logger.info(f"Job {job_id}: Julia script completed successfully.")
            return "Done"

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
        logger = logging.getLogger(__name__)
        logger.info(f"Job {job_id}: Post-processing started.")
        logger.info(f"Job {job_id}: Post-processing completed.")
        return "Done"
