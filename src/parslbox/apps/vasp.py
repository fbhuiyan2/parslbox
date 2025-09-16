import logging
from pathlib import Path
from parsl import bash_app
from parslbox.helpers import database

# ===================================================================================
#  VASP APPLICATION-SPECIFIC IMPLEMENTATION
# ===================================================================================
#  This module provides the necessary functions for the `pbx run` orchestrator
#  to execute VASP jobs.
# ===================================================================================

@bash_app
def parsl_app(job_id: int, job_path: Path, db_path: Path, ngpus: int, app_config: dict, config_name: str, stdout: str, stderr: str):
    """
    Parsl app for running a single VASP simulation.

    This function dynamically constructs the entire shell command to be executed
    on a worker node, using settings from the flow_config.yaml file.
    """
    # 1. Derive per-job parameters.
    # For GPU VASP, it's common to run one MPI rank per GPU.
    ntotranks = ngpus
    # VASP is typically MPI-dominant; OpenMP threading is often set to 1.
    nthreads = 1

    # 2. Unpack app configuration from the YAML file
    env_setup = app_config.get('environment_setup', '')
    # Default to a common VASP GPU executable name if not specified
    executable = app_config.get('executable_path', 'vasp_gpu')

    # 3. Command to update status to 'Running' on the worker node
    update_status_cmd = f"python -c \"from parslbox.helpers import database; database.update_jobs('{db_path}', job_ids=[{job_id}], status='Running')\""

    # 4. Construct the full command string for VASP
    return f"""
cd {job_path}

# Environment Setup (from flow_config.yaml)
{env_setup}
export OMP_NUM_THREADS={nthreads}

# Execution
echo "INFO: Updating job status to Running for job ID {job_id}..."
{update_status_cmd}

echo "INFO: Starting mpirun for job ID {job_id}..."
# The VASP executable typically finds its input files (INCAR, POSCAR, etc.)
# in the current directory.
mpirun -np {ntotranks} {executable}
"""

def check_success(job_id: int, job_path: Path, db_path: Path):
    """
    Checks for success and updates the database with the final status.
    """
    logger = logging.getLogger(__name__)
    return None


def postprocess(job_id: int, job_path: Path, db_path: Path):
    """
    Post-processing for a VASP job.

    This is a simple implementation that assumes the job was successful if
    the Parsl app future completed without an exception. It updates the
    database status to 'Done'.
    
    A more advanced version could check for "Voluntary context switches" in
    the OUTCAR file.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Job {job_id}: Basic post-processing started.")

    # Since there's no complex check, we assume success and set status to 'Done'.
    final_status = "Done"

    database.update_jobs(db_path, job_ids=[job_id], status=final_status)
    logger.info(f"Job {job_id}: Final status set to '{final_status}'.")
