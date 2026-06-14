"""
Base Application Class for ParslBox

This module defines the abstract base class for application implementations,
providing a consistent interface for all supported applications.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from parsl import bash_app
import logging


class AppBase(ABC):
    """
    Abstract base class for ParslBox applications.
    
    Each application (LAMMPS, VASP, Python, etc.) should inherit from this class
    and implement the required methods and class attributes.
    """
    
    # App configuration (must be defined in subclasses)
    INPUT_REQUIRED: bool
    DFLT_INPUT: str | None

    # When True, preprocess() and postprocess() are dispatched on the assigned
    # compute node via a subprocess wrapped with the resource launcher, instead
    # of running in-process in the pbx run orchestrator on the head node.
    # Default False preserves prior behavior. Does NOT affect restart().
    # Apps opting in MUST have a side-effect-free __init__ (the dispatcher
    # re-instantiates the app on the compute node via get_app_instance()).
    RUN_HOOKS_ON_COMPUTE: bool = False

    @classmethod
    def get_default_ranks_per_node(cls, ngpus, num_nodes, system_config):
        """Return default ranks_per_node when user doesn't specify --ranks-per-node.

        Default: 1 rank per GPU for GPU jobs, 1 rank per available core for CPU jobs.
        Override in subclasses for different behavior (e.g., Python → 1 rank per node).
        """
        if ngpus > 0:
            if num_nodes > 1:
                return system_config.GPUS_PER_NODE
            return ngpus
        excluded = getattr(system_config, 'EXCLUDE_CORES', None) or []
        return system_config.CORES_PER_NODE - len(excluded)

    @abstractmethod
    def get_command_template(self, **kwargs) -> str:
        """
        Get the command template for executing the application.
        
        This method should return the main execution command for the app.
        It receives all relevant parameters as kwargs and should construct
        the appropriate command string.
        
        Args:
            **kwargs: All parameters needed for command construction including:
                - mpi_prefix: MPI command prefix
                - mpi_opts_str: MPI options string
                - executable: Path to executable
                - in_file: Input file name
                - total_gpus: Number of GPUs assigned
                - app_config: Application configuration dict
                - mpi_commands: Full MPI commands dict
                
        Returns:
            str: The command to execute the application
        """
        pass
    
    def get_additional_setup(self, **kwargs) -> str:
        """
        Get additional setup commands to run before the main command.
        
        Override this method to add app-specific setup like environment
        variables, thread settings, etc.
        
        Args:
            **kwargs: All parameters that might be needed including:
                - total_gpus: Number of GPUs assigned
                - app_config: Application configuration dict
                - mpi_prefix: MPI command prefix
                - mpi_commands: Full MPI commands dict
            
        Returns:
            str: Additional setup commands (empty string by default)
        """
        return ""
    
    def preprocess(self, job_id: int, job_path: Path, db_path: Path, app_config: dict, config_name: str):
        """
        Preprocessing before job execution.

        This method is called before the main parsl_app execution.
        Default implementation does nothing.

        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file
            app_config (dict): Application configuration from YAML
            config_name (str): Name of the system configuration (e.g., 'polaris')
        """
        pass

    def restart(self, job_dict: dict) -> dict | None:
        """
        Restart hook called at `pbx run` startup for every job in `Restart`
        status, before the run loop touches it. Fires whether the Restart
        status came from the previous link's walltime kill (under
        `--respawn`) or was set manually by the user via
        `pbx update --status Restart`.

        Three scenes, encoded by whether the subclass overrides this method:

        - Scene A (real restart): override to return a dict of changed job fields
          (e.g., {"in_file": "in.restart"}). Orchestrator patches those fields in
          the DB, flips status to `Ready`, and re-runs with `preprocess` skipped.
        - Scene B (re-run as-is): override to return None or {}. Orchestrator flips
          status to `Ready` without patches; re-runs unchanged, `preprocess` skipped.
        - Scene C (no restart capability): do NOT override. Base class raises
          NotImplementedError; orchestrator marks the job `Failed` with reason
          "app does not support restart".

        Args:
            job_dict (dict): Full job row from the DB.

        Returns:
            dict | None: Partial mapping of DB column names to new values, or
            None/{} to re-run as-is.

        Raises:
            NotImplementedError: Base default; signals Scene C.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support restart. "
            f"Override restart() to enable Restart-status job handling for this app."
        )

    def min_remaining_walltime(self, job_dict: dict) -> int:
        """Minimum remaining batch walltime (seconds) required to dispatch a
        fresh job for this app. When `remaining < this value`, the orchestrator
        skips the job at every dispatch site (initial, backlog, dynamic-discovery);
        the job stays in DB as Ready/Restart and is picked up by the next pbx run.

        Default 0 = no gate (preserves existing behavior). Restart-continuations
        (jobs already in `restarting_job_ids`) are exempt regardless of this
        value — they have checkpoint state and brief runtime still advances them.

        Override for apps where a fresh job needs meaningful runtime to be useful
        (long MD runs, etc.).

        Args:
            job_dict (dict): Full job row from the DB. Available so subclasses
                can scale the floor by input size, GPU count, etc. if desired.

        Returns:
            int: Minimum remaining walltime in seconds. 0 (default) disables
            the gate for this app.
        """
        return 0

    def parsl_app(self, job_id: int, job_path: Path, db_path: Path, assignment, mpi_commands: dict,
                  app_config: dict, config_name: str, in_file: str, mpi_opts: str, env_file: str, 
                  stdout: str, stderr: str):
        """
        Main Parsl app function for executing the application.
        
        This is a template method that handles common logic and delegates
        app-specific command construction to get_command_template().
        
        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file
            assignment: NodeAssignment object with resource allocation details
            mpi_commands (dict): Dictionary of MPI command prefixes
            app_config (dict): Application configuration from YAML
            config_name (str): Name of the system configuration (e.g., 'polaris')
            in_file (str): Input filename for the job
            mpi_opts (str): Additional MPI options to append to the MPI command
            env_file (str): Path to environment file
            stdout (str): Standard output file path
            stderr (str): Standard error file path
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Extract serializable data from assignment object
            mpi_backend = mpi_commands.get('PBX_MPI_BACKEND')
            tile_mode = mpi_commands.get('PBX_GPU_TILE_MODE', False)
            env_vars = assignment.get_env_vars(mpi_backend=mpi_backend, tile_mode=tile_mode)
            total_gpus = assignment.get_total_gpus()
            assignment_summary = assignment.get_summary()
            
            # Call the generic bash app with all parameters
            # Note: _general_bash_app_engine is decorated with @bash_app, so it's a standalone function
            return AppBase._general_bash_app_engine(
                self,  # Pass self as first argument since it's still a method
                job_id=job_id,
                job_path=str(job_path),
                db_path=str(db_path),
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
            print(f"{self.__class__.__name__} bash_app construction failed:", e)
            traceback.print_exc()
            raise
    
    @bash_app
    def _general_bash_app_engine(self, job_id: int, job_path: str, db_path: str, env_vars: dict, 
                                total_gpus: int, assignment_summary: str, mpi_commands: dict, 
                                app_config: dict, config_name: str, in_file: str, mpi_opts: str, 
                                env_file: str, stdout: str, stderr: str):
        """
        Generic Parsl bash app engine that constructs commands using the template pattern.
        
        This method handles all common logic and calls get_command_template()
        to get the app-specific command.
        """
        # Get MPI command prefix and MPI env setup
        mpi_prefix = mpi_commands.get('PBX_MPI_PREFIX', '')
        mpi_env_setup = mpi_commands.get('PBX_MPI_ENV_SETUP', '')

        # Get executable from app config
        executable = app_config.get('executable_path', '')

        # Environment setup from app config (config.yaml)
        env_setup_frm_appconfig = app_config.get('environment_setup', '')

        # Environment setup from env_file
        env_setup_frm_envfile = ''
        if env_file:
            try:
                with open(env_file, 'r') as f:
                    env_setup_frm_envfile = f.read()
            except Exception as e:
                env_setup_frm_envfile = f"echo 'Warning: Could not read env_file {env_file}: {e}'"

        # Handle mpi_opts - use empty string if None
        mpi_opts_str = mpi_opts if mpi_opts is not None else ''

        # GPU/resource env vars from resource manager
        env_exports_frm_rsrc_mgr = self._format_env_vars(env_vars)

        # Get app-specific command template
        command = self.get_command_template(
            mpi_prefix=mpi_prefix,
            mpi_opts_str=mpi_opts_str,
            executable=executable,
            in_file=in_file,
            total_gpus=total_gpus,
            config_name=config_name,
            app_config=app_config,
            mpi_commands=mpi_commands
        )

        # App-specific setup (e.g., OMP_NUM_THREADS)
        setup_frm_app = self.get_additional_setup(
            total_gpus=total_gpus,
            app_config=app_config,
            mpi_prefix=mpi_prefix,
            mpi_commands=mpi_commands
        )

        # Construct the full bash script
        return f"""
cd {job_path}

# MPI Environment Setup (from mpi.env_setup in config.yaml)
{mpi_env_setup}

# App Environment Setup (from app config in config.yaml)
{env_setup_frm_appconfig}

# App Environment Setup (from env_file)
{env_setup_frm_envfile}

# Resource environment variables (GPU assignments from resource manager)
{env_exports_frm_rsrc_mgr}

# App-specific setup (from app's get_additional_setup)
{setup_frm_app}

# Execution
echo "INFO: Starting {self.__class__.__name__} for job ID {job_id}..."
echo "INFO: Resource assignment: {assignment_summary}"

{command}
"""
    
    def check_success(self, job_id: int, job_path: Path, db_path: Path, error_message: str = None) -> str:
        """
        Check if the job completed successfully and return final status.
        
        Default implementation assumes success if no error_message is provided.
        Override this method for app-specific success checking.
        
        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file
            error_message (str, optional): Error message from fut.result() if any
            
        Returns:
            str: Final job status ('Done', 'Failed', etc.)
        """
        logger = logging.getLogger(__name__)
        
        if error_message:
            logger.warning(f"Job {job_id}: Execution error occurred: {error_message}")
            logger.info(f"Job {job_id}: Default behavior - marking as Failed due to execution error.")
            return "Failed"
        else:
            logger.info(f"Job {job_id}: Job completed without execution errors. Assuming success.")
            return "Done"
    
    def postprocess(self, job_id: int, job_path: Path, db_path: Path) -> str:
        """
        Perform any post-processing after job completion.
        
        Default implementation does basic post-processing and returns 'Done'.
        Override this method for app-specific post-processing.
        
        Args:
            job_id (int): The job ID
            job_path (Path): Path to the job directory
            db_path (Path): Path to the database file
            
        Returns:
            str: Final job status after post-processing ('Done', 'Failed', etc.)
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Job {job_id}: Post-processing started.")
        
        # Default post-processing - just return Done
        logger.info(f"Job {job_id}: Post-processing completed successfully.")
        return "Done"
    
    def _format_env_vars(self, env_vars: dict) -> str:
        """
        Format environment variables for shell export.
        
        Args:
            env_vars (dict): Dictionary of environment variable name -> value
            
        Returns:
            str: Formatted export statements
        """
        if not env_vars:
            return ""
        
        exports = []
        for key, value in env_vars.items():
            exports.append(f"export {key}={value}")
        
        return "\n".join(exports)
