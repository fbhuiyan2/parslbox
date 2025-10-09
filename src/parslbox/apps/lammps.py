import logging
from pathlib import Path
from parsl import bash_app
from parslbox.helpers import database
from parslbox.apps.base import AppBase
from parslbox.configs.loader import get_system_config

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
    
    @bash_app
    def parsl_app(self, job_id: int, job_path: Path, db_path: Path, ngpus: int, app_config: dict, config_name: str, in_file: str, mpi_opts: str, stdout: str, stderr: str):
        """
        Parsl app for running a single LAMMPS simulation.
        This function dynamically constructs the entire shell command.
        """
        # Derive per-job parameters
        ntotranks = ngpus  # For LAMMPS, 1 GPU per rank
        
        # Get system configuration to determine GPUs per node
        system_config = get_system_config(config_name)
        gpus_per_node = system_config.GPUS_PER_NODE
        
        # Calculate ranks per node based on system and job requirements
        rankspernode = min(ngpus, gpus_per_node)

        # Unpack app configuration from the YAML file
        executable = app_config.get('executable_path')
        mpi_cmd = app_config.get('mpi_cmd', 'mpiexec') # Default to 'mpiexec'
        mpi_args = app_config.get('mpi_args')
        env_setup = app_config.get('environment_setup', '')
        mpi_env = app_config.get('mpi_env', '') 
        
        # Handle mpi_opts - use empty string if None
        mpi_opts_str = mpi_opts if mpi_opts is not None else ''

        # Command to update status to 'Running' on the worker node
        update_status_cmd = f"python -c \"from parslbox.helpers import database; database.update_jobs('{db_path}', job_ids=[{job_id}], status='Running')\""

        # Construct the full command string
        return f"""
cd {job_path}

# Environment Setup (from config.yaml)
{env_setup}

#export OMP_NUM_THREADS=nthreads
export NTOTRANKS={ntotranks}
export NRANKS_PER_NODE={rankspernode}

# Execution
echo "INFO: Updating job status to Running for job ID {job_id}..."
{update_status_cmd}

echo "INFO: Starting mpirun for job ID {job_id} with input file {in_file}..."
{mpi_cmd} {mpi_args} {mpi_env} {mpi_opts_str} {executable} -k on g {ngpus} -sf kk -pk kokkos newton on neigh half -in {in_file}
"""

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
