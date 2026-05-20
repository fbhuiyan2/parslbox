import logging
from pathlib import Path
from parslbox.database import database
from parslbox.apps.appbase import AppBase


class LammpsKokkosApp(AppBase):
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
        
        # Determine how many GPUs each rank sees for KOKKOS -k on g N:
        # - map_gpu: 1 GPU per rank (srun binds each rank to 1 GPU)
        # - mask_gpu: multi-GPU per rank (e.g., mask_gpu:0x3,0xc → 2 GPUs/rank)
        # - wrapper script: 1 GPU per rank via CUDA_VISIBLE_DEVICES
        # - none/absent: all GPUs visible to each rank
        from parslbox.system_configs.loader import get_system_config
        config_name = kwargs['config_name']
        gpus_per_node = get_system_config(config_name).GPUS_PER_NODE
        if total_gpus > 0 and ('map_gpu' in mpi_prefix or ('.sh' in mpi_prefix and 'wrapper' in mpi_prefix)):
            lammps_gpu_count = 1
        elif total_gpus > 0 and 'mask_gpu' in mpi_prefix:
            mpi_commands = kwargs.get('mpi_commands', {})
            ranks_per_node = mpi_commands.get('PBX_RANKS_PER_NODE')
            if ranks_per_node:
                lammps_gpu_count = min(total_gpus, gpus_per_node) // int(ranks_per_node)
            else:
                lammps_gpu_count = min(total_gpus, gpus_per_node)
        else:
            lammps_gpu_count = min(total_gpus, gpus_per_node)
        
        # LAMMPS-specific: GPU vs CPU arguments
        if total_gpus > 0:
            # GPU-enabled LAMMPS command with Kokkos
            lammps_args = f"-k on g {lammps_gpu_count} -sf kk -pk kokkos newton on neigh half -in {in_file}"
        else:
            # CPU-only LAMMPS command
            lammps_args = f"-in {in_file}"
        
        ### Legace code, 'mpi_extra' in app config is removed from pbx_config
        ### Instead, this is now handled by mpi_overrides: add: [] option in the pbx_config
        # LAMMPS-specific: mpi_extra_tags from app config
        # mpi_extra_tags = app_config.get('mpi_extra', '') or ''
        
        # Log the command being constructed
        logger = logging.getLogger(__name__)
        logger.info(f"LAMMPS command: {mpi_prefix} {mpi_opts_str} {executable} {lammps_args}")
        
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
