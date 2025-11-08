import logging
from pathlib import Path
from parslbox.helpers import database
from parslbox.apps.base import AppBase


class LammpsApp(AppBase):
    """
    LAMMPS application implementation for ParslBox.
    
    LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) is a
    classical molecular dynamics code with a focus on materials modeling.
    """
    
    # App configuration
    INPUT_REQUIRED = True
    DFLT_INPUT = "in.lammps"
    
    def get_command_template(self, **kwargs) -> str:
        """
        Construct LAMMPS-specific execution command.
        
        LAMMPS command format:
        {mpi_prefix} {mpi_opts} {mpi_extra_tags} {executable} {lammps_args}
        
        Args:
            **kwargs: Contains mpi_prefix, mpi_opts_str, executable, in_file,
                     total_gpus, app_config, mpi_commands
        
        Returns:
            str: LAMMPS execution command
        """
        mpi_prefix = kwargs['mpi_prefix']
        mpi_opts_str = kwargs['mpi_opts_str']
        executable = kwargs['executable']
        in_file = kwargs['in_file']
        total_gpus = kwargs['total_gpus']
        app_config = kwargs['app_config']
        
        # LAMMPS-specific: GPU vs CPU arguments
        if total_gpus > 0:
            # GPU-enabled LAMMPS command with Kokkos
            lammps_args = f"-k on g {total_gpus} -sf kk -pk kokkos newton on neigh half -in {in_file}"
        else:
            # CPU-only LAMMPS command
            lammps_args = f"-in {in_file}"
        
        # LAMMPS-specific: mpi_extra_tags from app config
        mpi_extra_tags = app_config.get('mpi_extra', '') or ''
        
        # Log the command being constructed
        logger = logging.getLogger(__name__)
        logger.info(f"LAMMPS command: {mpi_prefix} {mpi_opts_str} {mpi_extra_tags} {executable} {lammps_args}")
        
        return f"{mpi_prefix} {mpi_opts_str} {mpi_extra_tags} {executable} {lammps_args}"
    
    def check_success(self, job_id: int, job_path: Path, db_path: Path) -> str:
        """
        Check if LAMMPS job completed successfully.
        
        LAMMPS-specific: Looks for "Total wall time:" in log.lammps file.
        
        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file
            
        Returns:
            str: Final job status ('Done' or 'Failed')
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
        
        Simple implementation that sets status to 'Done'.
        The actual success checking is done in check_success().
        
        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Job {job_id}: Post-processing started.")

        # Since there's no complex post-processing, we assume success
        final_status = "Done"

        database.update_jobs(db_path, job_ids=[job_id], status=final_status)
        logger.info(f"Job {job_id}: Final status set to '{final_status}'.")
