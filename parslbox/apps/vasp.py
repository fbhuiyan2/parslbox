import logging
from pathlib import Path
from parslbox.helpers import database
from parslbox.apps.base import AppBase


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
    
    def get_command_template(self, **kwargs) -> str:
        """
        Construct VASP execution command.
        
        VASP command format: {mpi_prefix} {mpi_opts} {executable}
        
        VASP typically finds its input files (INCAR, POSCAR, POTCAR, KPOINTS)
        in the current directory.
        
        Args:
            **kwargs: Contains mpi_prefix, mpi_opts_str, executable, total_gpus,
                     app_config, and other parameters
        
        Returns:
            str: VASP execution command
        """
        mpi_prefix = kwargs['mpi_prefix']
        mpi_opts_str = kwargs['mpi_opts_str']
        executable = kwargs['executable']
        total_gpus = kwargs['total_gpus']
        
        # VASP-specific: Default executable based on GPU availability
        if not executable:
            executable = 'vasp_gpu' if total_gpus > 0 else 'vasp_std'
        
        # Log the command being constructed
        logger = logging.getLogger(__name__)
        logger.info(f"VASP command: {mpi_prefix} {mpi_opts_str} {executable}")
        
        return f"{mpi_prefix} {mpi_opts_str} {executable}"
    
    def get_additional_setup(self, **kwargs) -> str:
        """
        VASP-specific setup: Set OpenMP thread count.
        
        VASP is typically MPI-dominant; OpenMP threading is often set to 1.
        
        Args:
            **kwargs: Contains various parameters (not used for VASP)
        
        Returns:
            str: Environment variable export command for OMP_NUM_THREADS
        """
        return "export OMP_NUM_THREADS=1"
    
    def check_success(self, job_id: int, job_path: Path, db_path: Path) -> str:
        """
        Check if VASP job completed successfully.
        
        For VASP jobs, we assume success if the script exits with code 0.
        A more advanced version could check for "Voluntary context switches" 
        in the OUTCAR file.
        
        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file
            
        Returns:
            str: Final job status ('Done')
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

        Simple implementation that assumes the job was successful if
        the Parsl app future completed without an exception.
        
        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Job {job_id}: Basic post-processing started.")

        # Since there's no complex check, we assume success and set status to 'Done'.
        final_status = "Done"

        database.update_jobs(db_path, job_ids=[job_id], status=final_status)
        logger.info(f"Job {job_id}: Final status set to '{final_status}'.")
