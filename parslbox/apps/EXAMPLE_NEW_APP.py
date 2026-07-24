"""
EXAMPLE: How to Create a New App

This file demonstrates how simple it is to create a new application
using the refactored template-based system.

To create a new app, you only need to:
1. Inherit from AppBase
2. Set INPUT_REQUIRED and DFLT_INPUT class attributes
3. Implement get_command_template() method
4. Optionally set RUN_HOOKS_ON_COMPUTE = True to dispatch preprocess/postprocess
   on the assigned compute node (default False runs them on the head node)
5. Optionally override get_additional_setup(), check_success(), or postprocess()

That's it! All the common logic (environment setup, resource handling, 
bash script construction) is handled by the base class.
"""

import logging
from pathlib import Path
from parslbox.database import database
from parslbox.apps.appbase import AppBase


class ExampleApp(AppBase):
    """
    Example application implementation for ParslBox.
    
    This demonstrates the minimal code needed to create a new app.
    """
    
    # App configuration (REQUIRED)
    INPUT_REQUIRED = True  # Does this app require an input file?
    DFLT_INPUT = "input.txt"  # Default input filename (or None)
    # RUN_HOOKS_ON_COMPUTE = False  # (default) Set to True to run
    #     preprocess()/postprocess() on the assigned compute node instead of
    #     the head node. Requires a side-effect-free __init__ (the app is
    #     re-instantiated on the compute node).
    
    def get_command_template(self, **kwargs) -> str:
        """
        Define how to execute your application (REQUIRED).
        
        This is the ONLY method you MUST implement.
        
        Args:
            **kwargs: Contains all parameters you might need:
                - mpi_prefix: MPI command prefix (e.g., "mpiexec -n 4 --host node1")
                - mpi_opts_str: additional MPI options from job config
                - executable: path to executable from app config
                - in_file: input file name
                - total_gpus: number of GPUs assigned to this job
                - config_name: system config name (e.g., 'polaris')
                - app_config: full app config dict from YAML
                - mpi_commands: full MPI commands dict from mpi_launcher
        
        Returns:
            str: The command to execute your application
        """
        # Extract what you need from kwargs
        mpi_prefix = kwargs['mpi_prefix']
        mpi_opts_str = kwargs['mpi_opts_str']
        executable = kwargs['executable']
        in_file = kwargs['in_file']
        
        # Build the command with optional MPI options
        if mpi_opts_str:
            return f"{mpi_prefix} {mpi_opts_str} {executable} --input {in_file}"
        else:
            return f"{mpi_prefix} {executable} --input {in_file}"
    
    # OPTIONAL: Override if you need app-specific setup
    def get_additional_setup(self, **kwargs) -> str:
        """
        Add any app-specific environment setup (OPTIONAL).
        
        For example, setting environment variables, loading modules, etc.
        Default returns empty string (no additional setup).
        """
        # Example: Set an environment variable
        return "export MY_APP_THREADS=4"
    
    # OPTIONAL: Override if you need custom success checking
    def check_success(self, job_id: int, job_path: Path, db_path: Path, error_message: str = None) -> str:
        """
        Check if job completed successfully (OPTIONAL).
        
        Default assumes success based on exit code and error_message.
        Override to check output files, logs, etc.
        
        Args:
            job_id: The job ID
            job_path: Path to the job directory
            db_path: Path to the database file
            error_message: Error message from fut.result() if any
            
        Returns:
            str: Final job status ('Done', 'Failed', etc.)
            
        Note:
            This method should only return the status. The caller handles database updates.
        """
        logger = logging.getLogger(__name__)
        
        # Check for execution errors first
        if error_message:
            logger.warning(f"Job {job_id}: Execution error occurred: {error_message}")
            return "Failed"
        
        # Example: Check for a specific output file
        output_file = job_path / "output.dat"
        
        if output_file.exists():
            logger.info(f"Job {job_id}: Output file found, marking as Done.")
            return "Done"
        else:
            logger.warning(f"Job {job_id}: Output file not found, marking as Failed.")
            return "Failed"
    
    # OPTIONAL: Override if you need custom post-processing
    def postprocess(self, job_id: int, job_path: Path, db_path: Path) -> str:
        """
        Perform post-processing after job completion (OPTIONAL).
        
        Default returns 'Done'.
        Override to do cleanup, data processing, etc.
        
        Args:
            job_id: The job ID
            job_path: Path to the job directory
            db_path: Path to the database file
            
        Returns:
            str: Final job status after post-processing ('Done', 'Failed', etc.)
            
        Note:
            This method should return the final status. The caller handles database updates.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Job {job_id}: Running custom post-processing...")
        
        # Example: Do some custom processing
        # process_output_files(job_path)
        
        logger.info(f"Job {job_id}: Post-processing completed successfully.")
        return "Done"


# ============================================================================
# MINIMAL EXAMPLE: If you don't need any custom behavior
# ============================================================================

class MinimalApp(AppBase):
    """
    Absolute minimal app - just define the command template.
    Everything else uses defaults from base class.
    """

    INPUT_REQUIRED = True
    DFLT_INPUT = "input.txt"

    def get_command_template(self, **kwargs) -> str:
        """Just run a simple command with MPI support."""
        mpi_prefix = kwargs['mpi_prefix']
        mpi_opts_str = kwargs['mpi_opts_str']
        executable = kwargs['executable']
        in_file = kwargs['in_file']
        
        # Build command with MPI prefix and optional MPI options
        if mpi_opts_str:
            return f"{mpi_prefix} {mpi_opts_str} {executable} {in_file}"
        else:
            return f"{mpi_prefix} {executable} {in_file}"
    
    # That's it! No other methods needed.
    # The base class handles everything else:
    # - Environment setup from config.yaml and env_file
    # - Resource allocation and MPI command generation
    # - Status updates and database management
    # - Success checking (assumes exit code 0 = success)
    # - Post-processing (returns 'Done' by default)


# ============================================================================
# ADVANCED: Using MPI Overrides
# ============================================================================

class AdvancedMPIApp(AppBase):
    """
    Example showing how to use MPI overrides for custom MPI behavior.
    
    MPI overrides allow you to disable certain MPI flags or add custom ones.
    This is useful when your app has specific MPI requirements.
    """
    
    INPUT_REQUIRED = True
    DFLT_INPUT = "input.txt"
    
    def get_command_template(self, **kwargs) -> str:
        """Command template with MPI overrides consideration."""
        mpi_prefix = kwargs['mpi_prefix']
        executable = kwargs['executable']
        in_file = kwargs['in_file']
        
        # The mpi_prefix already includes any overrides applied by mpi_launcher.py
        return f"{mpi_prefix} {executable} --input {in_file}"
    
    # To use MPI overrides, configure them in your app's YAML config:
    # 
    # your_system_name:
    #   apps:
    #     advanced_mpi:
    #       executable_path: "/path/to/your/app"
    #       mpi_overrides:
    #         disable:
    #           - "--bind-to"      # Remove specific flags
    #           - "core"           # Remove flags containing this substring
    #         add:
    #           - "--bind-to socket"  # Add custom flags
    #           - "--report-bindings" # Add debugging flags


# ============================================================================
# HOW TO REGISTER YOUR CUSTOM APP
# ============================================================================
#
# METHOD 1: Config-based Registration (RECOMMENDED)
# ================================================
#
# After creating your app class, register it in ~/.parslbox/pbx_config.yaml:
#
# 1. Add to the custom_apps section:
#
#    custom_apps:
#      my_app:
#        module: "/path/to/this/file.py"    # Path to your Python file
#        class: "ExampleApp"                # Name of your class (e.g., ExampleApp, MinimalApp)
#
# 2. Configure it like any built-in app:
#
#    my_app:
#      polaris:
#        executable_path: "/path/to/executable"
#        environment_setup: |
#          module load my_modules
#
# 3. Use it:
#
#    pbx add --app my_app --input input.txt
#
# That's it! No need to modify ParslBox source code.
#
# ============================================================================
# COMPLETE EXAMPLE (Config-based)
# ============================================================================
#
# Let's say you save this file as ~/my_custom_lammps.py and want to use
# the MinimalApp class. Your config would look like:
#
# ~/.parslbox/pbx_config.yaml:
# ---
# custom_apps:
#   my_lammps:
#     module: "~/my_custom_lammps.py"
#     class: "MinimalApp"
#
# my_lammps:
#   polaris:
#     executable_path: "/path/to/lammps"
#     environment_setup: |
#       module load lammps
# ---
#
# Then use it:
#   pbx add --app my_lammps --input in.lammps
#
# ============================================================================
# METHOD 2: Source Code Registration (NOT RECOMMENDED)
# ====================================================
#
# Alternative: Modify ParslBox source code (requires package modification):
#
# 1. Add import to parslbox/apps/app_registry.py:
#
#    from parslbox.apps.example_new_app import ExampleApp, MinimalApp, AdvancedMPIApp
#
# 2. Add to APP_FACTORY in parslbox/apps/app_registry.py:
#
#    APP_FACTORY = {
#        "lammps": LammpsApp,
#        "vasp": VaspApp,
#        "python": PythonApp,
#        "example": ExampleApp,        # <-- Add your apps here
#        "minimal": MinimalApp,
#        "advanced_mpi": AdvancedMPIApp,
#    }
#
# WARNING: This method requires modifying the ParslBox package source code,
# which makes it difficult to maintain when updating ParslBox versions.
# Use METHOD 1 (config-based) instead!
#
# ============================================================================
