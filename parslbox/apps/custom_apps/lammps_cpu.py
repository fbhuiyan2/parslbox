"""
CPU-only LAMMPS Application for ParslBox

This is a custom app implementation for running LAMMPS without Kokkos.
Useful for CPU-only systems where Kokkos is not installed or not needed.

Usage in pbx_config.yaml:
    custom_apps:
      lammps-cpu:
        module: "parslbox/apps/custom_apps/lammps_cpu.py"
        class: "LammpsBasicApp"
        executable_path: "/path/to/lmp"
        environment_setup: |
          module load lammps
"""

import logging
from pathlib import Path
from parslbox.apps.appbase import AppBase


class LammpsBasicApp(AppBase):
    """
    CPU-only LAMMPS application implementation for ParslBox.
    
    This app runs standard LAMMPS without Kokkos acceleration.
    It's designed for CPU-only systems or cases where Kokkos is not needed.
    
    LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) is a
    classical molecular dynamics code with a focus on materials modeling.
    """
    
    # App configuration
    INPUT_REQUIRED = True
    DFLT_INPUT = "in.lammps"
    USES_MPI = True  # LAMMPS uses MPI for parallelization
    
    def get_command_template(self, **kwargs) -> str:
        """
        Construct CPU-only LAMMPS execution command.
        
        LAMMPS command format (CPU-only):
        {mpi_prefix} {mpi_opts} {executable} -in {in_file}
        
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
        
        # CPU-only LAMMPS command - no Kokkos flags
        lammps_args = f"-in {in_file}"
        
        # Log the command being constructed
        logger = logging.getLogger(__name__)
        logger.info(f"LAMMPS-CPU command: {mpi_prefix} {mpi_opts_str} {executable} {lammps_args}")
        
        return f"{mpi_prefix} {mpi_opts_str} {executable} {lammps_args}"
    
    def check_success(self, job_id: int, job_path: Path, db_path: Path, error_message: str = None) -> str:
        """
        Check if LAMMPS job completed successfully.
        
        LAMMPS-specific: Ignores execution errors and looks for "Total wall time:" in log.lammps file.
        LAMMPS can exit ungracefully even after a successful run.
        
        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file
            error_message (str, optional): Error message from fut.result() (ignored for LAMMPS)
            
        Returns:
            str: Final job status ('Done' or 'Failed')
        """
        logger = logging.getLogger(__name__)
        
        # LAMMPS-specific: Ignore execution errors, check output files instead
        if error_message:
            logger.info(f"Job {job_id}: Execution error occurred but ignoring for LAMMPS: {error_message}")
        
        log_file = job_path / "log.lammps"
        
        if not log_file.is_file():
            logger.warning(f"Job {job_id}: log.lammps not found. Marking as Failed.")
            return "Failed"
        
        try:
            with open(log_file, 'r') as f:
                if "Total wall time:" in f.read():
                    logger.info(f"Job {job_id}: Success marker found in log.lammps.")
                    return "Done"
                else:
                    logger.warning(f"Job {job_id}: Success marker not found in log.lammps. Marking as Failed.")
                    return "Failed"
        except Exception as e:
            logger.error(f"Job {job_id}: Error reading log.lammps: {e}")
            return "Failed"
    
    def postprocess(self, job_id: int, job_path: Path, db_path: Path) -> str:
        """
        Post-processing for a LAMMPS job.
        
        Simple implementation that returns 'Done'.
        The actual success checking is done in check_success().
        
        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file
            
        Returns:
            str: Status after post-processing ('Done')
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Job {job_id}: Post-processing started.")
        
        # No complex post-processing for LAMMPS
        logger.info(f"Job {job_id}: Post-processing completed.")
        return "Done"
