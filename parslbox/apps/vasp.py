import logging
from pathlib import Path
from parsl import bash_app
from parslbox.helpers import database
from parslbox.apps.base import AppBase

# ===================================================================================
#  VASP APPLICATION-SPECIFIC IMPLEMENTATION
# ===================================================================================
#  This module provides the necessary functions for the `pbx run` orchestrator
#  to execute VASP jobs.
# ===================================================================================

class VaspApp(AppBase):
    """
    VASP application implementation for ParslBox.
    
    VASP (Vienna Ab initio Simulation Package) is a computer program for atomic
    scale materials modelling, e.g. electronic structure calculations and
    quantum-mechanical molecular dynamics.
    """
    
    # App configuration
    INPUT_REQUIRED = False
    DFLT_INPUT = None
    
    def preprocess(self, job_id: int, job_path: Path, db_path: Path, app_config: dict, config_name: str):
        """
        Preprocessing for VASP jobs.
        Currently no preprocessing is needed for VASP.
        """
        pass

    @bash_app
    def parsl_app(self, job_id: int, job_path: Path, db_path: Path, ngpus: int, app_config: dict, config_name: str, in_file: str, mpi_opts: str, stdout: str, stderr: str):
        """
        Parsl app for running a single VASP simulation.

        This function dynamically constructs the entire shell command to be executed
        on a worker node, using settings from the config.yaml file.
        Note: in_file parameter is accepted for consistency but VASP typically uses
        standard input files (INCAR, POSCAR, etc.) found in the job directory.
        """
        # 1. Derive per-job parameters.
        # For GPU VASP, it's common to run one MPI rank per GPU.
        ntotranks = ngpus
        # VASP is typically MPI-dominant; OpenMP threading is often set to 1.
        nthreads = 1

        # 2. Unpack app configuration from the YAML file
        env_setup = app_config.get('environment_setup', '')
        mpi_cmd = app_config.get('mpi_cmd', 'mpiexec') # Default to 'mpiexec'
        mpi_args = app_config.get('mpi_args', f'-n {ntotranks}') # Default args
        mpi_env = app_config.get('mpi_env', '') 
        # Default to a common VASP GPU executable name if not specified
        executable = app_config.get('executable_path', 'vasp_gpu')
        
        # Handle mpi_opts - use empty string if None
        mpi_opts_str = mpi_opts if mpi_opts is not None else ''

        # 3. Command to update status to 'Running' on the worker node
        update_status_cmd = f"python -c \"from parslbox.helpers import database; database.update_jobs('{db_path}', job_ids=[{job_id}], status='Running')\""

        # 4. Construct the full command string for VASP
        return f"""
cd {job_path}

# Environment Setup (from config.yaml)
{env_setup}
export OMP_NUM_THREADS={nthreads}

# Execution
echo "INFO: Updating job status to Running for job ID {job_id}..."
{update_status_cmd}

echo "INFO: Starting mpirun for job ID {job_id}..."
# The VASP executable typically finds its input files (INCAR, POSCAR, etc.)
# in the current directory.
{mpi_cmd} {mpi_args} {mpi_env} {mpi_opts_str} {executable}
"""

    def check_success(self, job_id: int, job_path: Path, db_path: Path) -> str:
        """
        Checks for success and updates the database with the final status.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Job {job_id}: VASP job completed. Assuming success based on exit code.")
        
        # Since we don't have a specific output file to check for VASP jobs,
        # we rely on the bash script's exit code (handled by Parsl)
        final_status = "Done"
        
        database.update_jobs(db_path, job_ids=[job_id], status=final_status)
        logger.info(f"Job {job_id}: Final status set to '{final_status}'.")
        
        return final_status

    def postprocess(self, job_id: int, job_path: Path, db_path: Path):
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
