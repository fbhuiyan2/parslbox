import logging
from pathlib import Path
from parsl import bash_app
from parslbox.helpers import database
from parslbox.apps.base import AppBase

# ===================================================================================
#  PYTHON APPLICATION-SPECIFIC IMPLEMENTATION
# ===================================================================================
#  This module provides the necessary functions for the `pbx run` orchestrator
#  to execute Python jobs via bash scripts.
# ===================================================================================

class PythonApp(AppBase):
    """
    Python application implementation for ParslBox.
    
    This app executes user-provided bash scripts that can activate any Python/conda
    environment and run Python scripts with complete flexibility.
    """
    
    # App configuration
    INPUT_REQUIRED = True
    DFLT_INPUT = None
    
    def preprocess(self, job_id: int, job_path: Path, db_path: Path, app_config: dict, config_name: str):
        """
        Preprocessing for Python jobs.
        Currently no preprocessing is needed for Python jobs.
        """
        pass

    @bash_app
    def parsl_app(self, job_id: int, job_path: Path, db_path: Path, ngpus: int, app_config: dict, config_name: str, in_file: str, mpi_opts: str, stdout: str, stderr: str):
        """
        Parsl app for running Python scripts via bash scripts.
        
        This function executes a user-provided bash script that can:
        - Activate any Python/conda environment
        - Set environment variables
        - Run Python scripts with any arguments
        - Perform any other bash operations
        
        The user is responsible for creating the bash script with all necessary
        environment setup and Python execution commands.
        """
        # 1. Unpack app configuration from the YAML file (if any)
        env_setup = app_config.get('environment_setup', '')

        # 2. Command to update status to 'Running' on the worker node
        update_status_cmd = f"python -c \"from parslbox.helpers import database; database.update_jobs('{db_path}', job_ids=[{job_id}], status='Running')\""

        # 3. Construct the full command string
        return f"""
cd {job_path}

# Environment Setup (from config.yaml, if any)
{env_setup}

# Execution
echo "INFO: Updating job status to Running for job ID {job_id}..."
{update_status_cmd}

echo "INFO: Executing bash script {in_file} for job ID {job_id}..."
bash {in_file}
"""

    def check_success(self, job_id: int, job_path: Path, db_path: Path) -> str:
        """
        Checks for success and updates the database with the final status.
        
        For Python jobs, we assume success if the bash script exits with code 0.
        Users can implement their own success checking within their bash scripts.
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
        
        This is a simple implementation that assumes the job was successful if
        the Parsl app future completed without an exception.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Job {job_id}: Python job post-processing started.")

        # Since there's no complex check, we assume success and set status to 'Done'.
        final_status = "Done"

        database.update_jobs(db_path, job_ids=[job_id], status=final_status)
        logger.info(f"Job {job_id}: Final status set to '{final_status}'.")
