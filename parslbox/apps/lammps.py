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
def lammps_parsl_app(job_id: int, job_path: str, db_path: str, env_vars: dict, total_gpus: int,
                     assignment_summary: str, mpi_commands: dict, app_config: dict,
                     config_name: str, in_file: str, mpi_opts: str, 
                     stdout: str, stderr: str):
    """
    Standalone Parsl app for running a single LAMMPS simulation.
    This function dynamically constructs the entire shell command using resource-aware MPI commands.
    """
    # Get MPI command prefix
    mpi_prefix = mpi_commands.get('PBX_MPI_PREFIX', '')
    
    # Unpack app configuration from the YAML file
    executable = app_config.get('executable_path')
    mpi_extra_tags = app_config.get('mpi_extra')
    env_setup = app_config.get('environment_setup', '')
    
    # Handle mpi_opts - use empty string if None
    mpi_opts_str = mpi_opts if mpi_opts is not None else ''
    mpi_extra_tags = mpi_extra_tags if mpi_extra_tags is not None else ''

    # Command to update status to 'Running' on the worker node
    update_status_cmd = f"python -c \"from parslbox.helpers import database; database.update_jobs('{db_path}', job_ids=[{job_id}], status='Running')\""

    # Format environment variables for GPU assignment (inline implementation)
    env_exports = ""
    if env_vars:
        exports = []
        for key, value in env_vars.items():
            exports.append(f"export {key}={value}")
        env_exports = "\n".join(exports)

    # Construct the full command string
    if total_gpus > 0:
        # GPU-enabled LAMMPS command
        lammps_args = f"-k on g {total_gpus} -sf kk -pk kokkos newton on neigh half -in {in_file}"
    else:
        # CPU-only LAMMPS command
        lammps_args = f"-in {in_file}"

    return f"""
cd {job_path}

# Environment Setup (from config.yaml)
{env_setup}

# Resource-specific environment variables (GPU assignments, etc.)
{env_exports}

# Execution
echo "INFO: Updating job status to Running for job ID {job_id}..."
{update_status_cmd}

echo "INFO: Starting LAMMPS for job ID {job_id} with input file {in_file}..."
echo "INFO: Using MPI command: {mpi_prefix} {mpi_opts_str} {mpi_extra_tags} {executable} {lammps_args}"
echo "INFO: Resource assignment: {assignment_summary}"

{mpi_prefix} {mpi_opts_str} {mpi_extra_tags} {executable} {lammps_args}
"""


# ===================================================================================
#  LAMMPS APPLICATION-SPECIFIC IMPLEMENTATION
# ===================================================================================

class LammpsApp(AppBase):
    """
    LAMMPS application implementation for ParslBox.
    
    LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) is a
    classical molecular dynamics code with a focus on materials modeling.
    """
    
    # App configuration
    INPUT_REQUIRED = True
    DFLT_INPUT = "in.lammps"
    
    def preprocess(self, job_id: int, job_path: Path, db_path: Path, app_config: dict, config_name: str):
        """
        Preprocessing for LAMMPS jobs.
        Currently no preprocessing is needed for LAMMPS.
        """
        pass
    
    def parsl_app(self, job_id: int, job_path: Path, db_path: Path, assignment, mpi_commands: dict,
                  app_config: dict, config_name: str, in_file: str, mpi_opts: str, stdout: str, stderr: str):
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
            
            return lammps_parsl_app(
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
                stdout=stdout,
                stderr=stderr
            )
        except Exception as e:
            import traceback
            logger.error(f"Job {job_id}: Failed to submit Parsl app: {e}")
            print("LAMMPS bash_app construction failed:", e)
            traceback.print_exc()
            raise

    def check_success(self, job_id: int, job_path: Path, db_path: Path) -> str:
        """
        Checks for success and updates the database with the final status.
        """
        logger = logging.getLogger(__name__)
        log_file = job_path / "log.lammps"
        final_status = "Failed"  # Assume failure unless proven otherwise

        if not log_file.is_file():
            logger.warning(f"Job {job_id}: Post-processing failed. log.lammps not found.")
        else:
            try:
                with open(log_file, 'r') as f:
                    if "Total wall time:" in f.read():
                        logger.info(f"Job {job_id}: Success marker found in log.lammps.")
                        final_status = "Done"
                    else:
                        logger.warning(f"Job {job_id}: Finished but success marker not found in log.lammps.")
            except Exception as e:
                logger.error(f"Job {job_id}: Error reading log.lammps during post-processing: {e}")

        # Update the database with the final determined status
        database.update_jobs(db_path, job_ids=[job_id], status=final_status)
        logger.info(f"Job {job_id}: Final status set to '{final_status}'.")

        return final_status

    def postprocess(self, job_id: int, job_path: Path, db_path: Path):
        """
        Post-processing for a LAMMPS job.
        Checks for success and updates the database with the final status.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Job {job_id}: Post-processing started.")

        # Since there's no complex check, we assume success and set status to 'Done'.
        final_status = "Done"

        database.update_jobs(db_path, job_ids=[job_id], status=final_status)
        logger.info(f"Job {job_id}: Final status set to '{final_status}'.")
