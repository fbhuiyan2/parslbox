import logging
from pathlib import Path
from parslbox.helpers import database
from parslbox.apps.base import AppBase


class PythonApp(AppBase):
    """
    Python application implementation for ParslBox.
    
    This app executes user-provided Python scripts with complete flexibility.
    The script can access MPI commands via the PBX_MPI_PREFIX environment variable.
    """
    
    # App configuration
    INPUT_REQUIRED = True
    DFLT_INPUT = None
    
    def get_command_template(self, **kwargs) -> str:
        """
        Construct Python execution command.
        
        Python command format: python {in_file}
        
        Args:
            **kwargs: Contains in_file and other parameters
        
        Returns:
            str: Python execution command
        """
        in_file = kwargs['in_file']
        
        # Log the command being constructed
        logger = logging.getLogger(__name__)
        logger.info(f"Python command: python {in_file}")
        
        return f"python {in_file}"
    
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
    
    def check_success(self, job_id: int, job_path: Path, db_path: Path) -> str:
        """
        Check if Python job completed successfully.
        
        For Python jobs, we assume success if the script exits with code 0.
        Users can implement their own success checking within their scripts.
        
        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file
            
        Returns:
            str: Final job status ('Done')
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Job {job_id}: Python job completed. Assuming success based on exit code.")
        
        # Since we don't have a specific output file to check for Python jobs,
        # we rely on the bash script's exit code (handled by Parsl)
        final_status = "Done"
        
        database.update_jobs(db_path, job_ids=[job_id], status=final_status)
        logger.info(f"Job {job_id}: Final status set to '{final_status}'.")
        
        return final_status

    def postprocess(self, job_id: int, job_path: Path, db_path: Path):
        """
        Post-processing for a Python job.
        
        Simple implementation that assumes the job was successful if
        the Parsl app future completed without an exception.
        
        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Job {job_id}: Python job post-processing started.")

        # Since there's no complex check, we assume success and set status to 'Done'.
        final_status = "Done"

        database.update_jobs(db_path, job_ids=[job_id], status=final_status)
        logger.info(f"Job {job_id}: Final status set to '{final_status}'.")
