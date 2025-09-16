import logging
from pathlib import Path
from parsl import bash_app
from parslbox.helpers import database

# ===================================================================================
#  LAMMPS APPLICATION-SPECIFIC IMPLEMENTATION
# ===================================================================================

@bash_app
def parsl_app(job_id: int, job_path: Path, db_path: Path, ngpus: int, app_config: dict, config_name: str, stdout: str, stderr: str):
    """
    Parsl app for running a single LAMMPS simulation.
    This function dynamically constructs the entire shell command.
    """
    # 1. Derive per-job parameters
    ntotranks = ngpus  # For LAMMPS, 1 GPU per rank
    # System-dependent core/thread count. This can be made more sophisticated later.
    # nthreads = 16 if config_name == 'sophia' else 8

    # 2. Unpack app configuration from the YAML file
    env_setup = app_config.get('environment_setup', '') # Use .get for safety
    mpi_env = app_config.get('mpi_env', '') 
    executable = app_config.get('executable_path', 'lmp') # Default to 'lmp' if not found

    # 3. Command to update status to 'Running' on the worker node
    update_status_cmd = f"python -c \"from parslbox.helpers import database; database.update_jobs('{db_path}', job_ids=[{job_id}], status='Running')\""

    # 4. Construct the full command string
    return f"""
cd {job_path}

# Environment Setup (from flow_config.yaml)
{env_setup}

#export OMP_NUM_THREADS=nthreads

# Execution
echo "INFO: Updating job status to Running for job ID {job_id}..."
{update_status_cmd}

echo "INFO: Starting mpirun for job ID {job_id}..."
mpirun -np {ntotranks} {mpi_env} {executable} -k on g {ngpus} -sf kk -pk kokkos newton on neigh half -in in.lammps
"""

def check_success(job_id: int, job_path: Path, db_path: Path):
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


def postprocess(job_id: int, job_path: Path, db_path: Path):
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
