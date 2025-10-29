import logging
from pathlib import Path
from parsl import bash_app
from parslbox.helpers import database
from parslbox.apps.base import AppBase
from parslbox.configs.loader import get_system_config
import logging

# ===================================================================================
#  STANDALONE PARSL APP FUNCTION
# ===================================================================================

@bash_app
def python_parsl_app(job_id: int, job_path: str, db_path: str, env_vars: dict, total_gpus: int,
                     assignment_summary: str, mpi_commands: dict, app_config: dict,
                     config_name: str, in_file: str, mpi_opts: str, env_file: str,
                     stdout: str, stderr: str):
    """
    Standalone Parsl app for running Python scripts directly.
    This function dynamically constructs the entire shell command using resource-aware MPI commands.
    """
    # Get MPI command prefix
    mpi_prefix = mpi_commands.get('PBX_MPI_PREFIX', '')
    
    # Start with base environment setup from app config (if any)
    env_setup = app_config.get('environment_setup', '')
    
    # Environment setup from env_file gets appended to env_setup from app_config
    # Append additional environment setup from env_file (if provided)
    if env_file:
        try:
            with open(env_file, 'r') as f:
                env_file_content = f.read()
                env_setup += "\n" + env_file_content  # Append to existing setup
        except Exception as e:
            env_setup += f"\necho 'Warning: Could not read env_file {env_file}: {e}'"
    
    # Handle mpi_opts - use empty string if None
    mpi_opts_str = mpi_opts if mpi_opts is not None else ''

    # Command to update status to 'Running' on the worker node
    update_status_cmd = f"python -c \"from parslbox.helpers import database; database.update_jobs('{db_path}', job_ids=[{job_id}], status='Running')\""

    # Format environment variables for GPU assignment (inline implementation)
    env_exports = ""
    if env_vars:
        exports = []
        for key, value in env_vars.items():
            exports.append(f"export {key}={value}")
        env_exports = "\n".join(exports)

    # Export MPI command as environment variable for the script to use
    mpi_env_export = f"export PBX_MPI_PREFIX='{mpi_prefix}'" if mpi_prefix else ""

    return f"""
cd {job_path}

echo "INFO: Environment Setup {env_setup}"

# Environment Setup (from config.yaml + env_file if provided)
{env_setup}

# Resource-specific environment variables (GPU assignments, etc.)
{env_exports}

# Export MPI command for the script to use if needed
{mpi_env_export}

# Execution
echo "INFO: Updating job status to Running for job ID {job_id}..."
{update_status_cmd}

echo "INFO: Executing Python script {in_file} for job ID {job_id}..."
echo "INFO: Resource assignment: {assignment_summary}"
echo "INFO: Available MPI command: {mpi_prefix}"

python {in_file}
"""


# ===================================================================================
#  PYTHON APPLICATION-SPECIFIC IMPLEMENTATION
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
    
    def parsl_app(self, job_id: int, job_path: Path, db_path: Path, assignment, mpi_commands: dict,
                  app_config: dict, config_name: str, in_file: str, mpi_opts: str, env_file: str, stdout: str, stderr: str):
        """
        Wrapper method that calls the standalone Parsl app function.
        Extracts serializable data from assignment object before passing to Parsl.
        """

        logger = logging.getLogger(__name__)  
        
        try:
            # Extract serializable data from assignment object
            env_vars = assignment.get_env_vars()
            total_gpus = assignment.get_total_gpus()
            assignment_summary = assignment.get_summary()
            
            return python_parsl_app(
                job_id=job_id,
                job_path=str(job_path),  # Convert Path to string for serialization
                db_path=str(db_path),    # Convert Path to string for serialization
                env_vars=env_vars,
                total_gpus=total_gpus,
                assignment_summary=assignment_summary,
                mpi_commands=mpi_commands,
                app_config=app_config,
                config_name=config_name,
                in_file=in_file,
                mpi_opts=mpi_opts,
                env_file=env_file,
                stdout=stdout,
                stderr=stderr
            )
        except Exception as e:
            import traceback
            logger.error(f"Job {job_id}: Failed to submit Parsl app: {e}")
            print("Python bash_app construction failed:", e)
            traceback.print_exc()
            raise

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
