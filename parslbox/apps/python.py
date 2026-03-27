import logging
from pathlib import Path
from parslbox.database import database
from parslbox.apps.appbase import AppBase


class PythonApp(AppBase):
    """
    Python application implementation for ParslBox.
    
    This app executes user-provided Python scripts with complete flexibility.
    The script can access MPI commands via the PBX_MPI_PREFIX environment variable.
    """
    
    # App configuration
    INPUT_REQUIRED = True
    DFLT_INPUT = None
    USES_MPI = False  # Python scripts don't use MPI for parallelization.
                      # PBX wraps the command with a resource launcher to
                      # constrain execution to assigned resources.
    
    def get_command_template(self, **kwargs) -> str:
        """
        Construct Python execution command.

        Python command format: {executable} {in_file}
        Users can configure a custom executable path (e.g., "python3 -u")
        via executable_path in config.yaml, or rely on the environment.

        Args:
            **kwargs: Contains in_file and other parameters

        Returns:
            str: Python execution command
        """
        in_file = kwargs['in_file']
        executable = kwargs.get('executable') or 'python'

        # Log the command being constructed
        logger = logging.getLogger(__name__)
        cmd = f"{executable} {in_file}"
        logger.info(f"Python command: {cmd}")

        return cmd
    
    def get_additional_setup(self, **kwargs) -> str:
        """
        Python-specific setup: Export MPI command for the script to use.
        
        This allows Python scripts to access the MPI command via the
        PBX_MPI_PREFIX environment variable if they need to launch
        MPI processes themselves.
        
        Args:
            **kwargs: Contains mpi_prefix and other parameters
        
        Returns:
            str: Environment variable export command
        """
        mpi_prefix = kwargs.get('mpi_prefix', '')
        
        if mpi_prefix:
            return f"export PBX_MPI_PREFIX='{mpi_prefix}'"
        
        return ""
    
    def check_success(self, job_id: int, job_path: Path, db_path: Path, error_message: str = None) -> str:
        """
        Check if Python job completed successfully.

        Reads the PBX_JOB_STATUS_REPORT file written by the user script
        via report_status(). If the file is not found, the job is marked
        as Failed. The status file is deleted after reading.

        User scripts should call report_status() to report their outcome:
            from parslbox.apps.utils import report_status
            report_status("done")    # or report_status("failed")

        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file
            error_message (str, optional): Error message from fut.result()

        Returns:
            str: Final job status ('Done' or 'Failed')
        """
        logger = logging.getLogger(__name__)

        if error_message:
            logger.error(f"Job {job_id}: Python script execution failed: {error_message}")
            return "Failed"

        # Read status report file written by user script via report_status()
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
        Post-processing for a Python job.

        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file

        Returns:
            str: Status after post-processing ('Done')
        """
        # logger = logging.getLogger(__name__)
        # logger.info(f"Job {job_id}: Post-processing started.")
        # logger.info(f"Job {job_id}: Post-processing completed.")
        # return "Done"
        pass
