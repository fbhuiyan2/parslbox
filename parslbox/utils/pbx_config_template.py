DEFAULT_CONFIG_YAML = """
# ---------------------------------------------------------------------------
# parslbox Application Configuration
#
# This file defines settings for different applications (e.g., lammps)
# on different systems (e.g., polaris, sophia).
#
# IMPORTANT: You must edit this file and provide the correct, absolute
#            path to the executables for your environment.
# ---------------------------------------------------------------------------

# Scheduler submission templates
schedulers:
  pbs:
    template: |
      #!/bin/bash
      #PBS -N {job_name}
      #PBS -q {queue}
      #PBS -l select={select}
      #PBS -l walltime={walltime}
      #PBS -l filesystems={filesystems}
      #PBS -A {project}
      #PBS -o pbx_scheduler.out
      #PBS -j oe
      #PBS -m be
      #PBS -M your@email.com
      
      cd $PBS_O_WORKDIR
      > pbx_scheduler.out
      
      NNODES=$(wc -l < $PBS_NODEFILE)
      echo "NNODES = $NNODES"
      echo "Job ID: $PBS_JOBID"
      echo "Job Name: $PBS_JOBNAME"
      
      {pbx_python_env_setup}
      
      # Set ParslBox environment variables if provided
      {pbx_env_vars}
      
      pbx run --config {config} --run-dir {run_dir} {run_options}

  slurm:
    template: |
      #!/bin/bash
      #SBATCH --job-name={job_name}
      #SBATCH --partition={queue}
      #SBATCH --nodes={select}
      #SBATCH --time={walltime}
      #SBATCH --account={project}
      #SBATCH --output=pbx_scheduler.out
      #SBATCH --error=pbx_scheduler.out
      
      > pbx_scheduler.out
      
      echo "NNODES = $SLURM_JOB_NUM_NODES"
      echo "Job ID: $SLURM_JOB_ID"
      echo "Job Name: $SLURM_JOB_NAME"
      
      {pbx_python_env_setup}
      
      # Set ParslBox environment variables if provided
      {pbx_env_vars}
      
      pbx run --config {config} --run-dir {run_dir} {run_options}

# System-specific configurations
sophia:
  pbx_python_env_setup: |
    module load conda
    conda activate parslbox

polaris:
  pbx_python_env_setup: |
    module load conda
    conda activate parslbox


# ---------------------------------------------------------------------------
# Custom Application Registration
#
# Register your own custom applications here. Once registered, they work
# just like built-in apps (lammps, vasp, python).
#
# How it works:
# 1. Create a Python file with your app class (see EXAMPLE_NEW_APP.py)
# 2. Your class must inherit from AppBase: class MyApp(AppBase):
# 3. Register it here by specifying the file/module and class name
#
# Two methods are supported:
# - File path: module: "/path/to/my_app.py"
# - Python module: module: "my_package.my_module"
#
# Example:
# custom_apps:
#   my_app:
#     module: "/path/to/my_app.py"     # Path to your Python file
#     class: "MyApp"                   # Name of your class inside the file
#   
#   another_app:
#     module: "~/my_apps/another.py"   # Supports ~ expansion
#     class: "AnotherApp"              # Name of your class inside the file
# ---------------------------------------------------------------------------

custom_apps: {}
  # Uncomment and modify to add your custom apps:
  # my_custom_app:
  #   module: "/path/to/my_app.py"     # Path to your Python file
  #   class: "MyCustomApp"             # Your class name (must inherit from AppBase)
  #
  # simulation_app:
  #   module: "my_simulation_package.apps"  # Or use installed Python package
  #   class: "SimulationApp"                 # Class name within that module
 
# ==========================================

# Application configurations

#===== Example APP =====#

app_name: # e.g., lammps, vasp
  # Settings for running LAMMPS on ALCF Polaris
  system_name: # e.g., polaris, aurora-tile
    # Full, absolute path to the software executable on the system.
    executable_path: "/path/to/your/software/executable/on/polaris" 

    # Shell commands to set up the environment on a compute node.
    # This block will be executed before the main mpirun command.
    # Use '|' to define a multi-line string in YAML.
    environment_setup: |
      # Add all necessary `module load` and `export` commands here.
      # Example:
      # module load PrgEnv-gnu
      # module load ...
      module restore   # Always restore modules first in apps


#===== LAMMPS APP =====#

lammps:
  # Settings for running LAMMPS on ALCF Polaris
  polaris:
    # Full, absolute path to the LAMMPS executable on the system.
    executable_path: "/path/to/your/lammps/executable/on/polaris" 

    # Shell commands to set up the environment on a compute node.
    # This block will be executed before the main mpirun command.
    # Use '|' to define a multi-line string in YAML.
    environment_setup: |
      # Add all necessary `module load` and `export` commands here.
      # Example:
      # module load PrgEnv-gnu
      # module load ...
      module purge  # # Always purge modules first in apps
      module restore   # Then restore modules if you want

  # Settings for running LAMMPS on ALCF Sophia
  sophia:
    # Full, absolute path to the LAMMPS executable on the system.
    executable_path: "/path/to/your/lammps/executable/on/sophia"

    # Additional mpi tags
    mpi_extra: 

    environment_setup: |
      # Add all necessary `module load` and `export` commands here.
      # Example:
      # module load compilers/openmpi/5.0.3
      # export LD_LIBRARY_PATH=...
      module purge  # # Always purge modules first in apps
      module restore   # Then restore modules if you want
    

      
# --- Add other applications below ---
# vasp:
#   polaris:
#     executable_path: "/path/to/vasp_gpu"
#     environment_setup: |
#       # module load vasp_env
#   sophia:
#     executable_path: "/path/to/vasp_gpu"
#     environment_setup: |
#       # module load vasp_env
#     # Example: VASP on Sophia has issues with rankfile and hostname flags
#     mpi_overrides:
#       disable: ["rankfile", "-H"]
#       # add: ["--mca btl ^openib"]  # Optional: add custom MPI flags

# --- Example for systems with older OpenMPI (4.x) ---
# OpenMPI 4.x uses --rankfile instead of --map-by rankfile:file=...
# Use mpi_overrides with template variables to fix this:
#
# lammps:
#   lcrc-swing:
#     executable_path: "/path/to/lmp"
#     environment_setup: |
#       module load openmpi
#     mpi_overrides:
#       disable: ["--map-by"]                    # Remove OpenMPI 5.x style flag
#       add: ["--rankfile {rankfile_path}"]      # Add OpenMPI 4.x style flag
#
# Available template variables for mpi_overrides.add:
#   {rankfile_path} - Path to the generated rankfile
#   {wrapper_path}  - Path to the GPU wrapper script
#   {hostlist}      - Comma-separated list of hostnames
#   {total_ranks}   - Total number of MPI ranks
#
# --- Example to disable GPU wrapper ---
# If your application handles GPU assignment internally (e.g., via Kokkos),
# you can disable the GPU wrapper script:
#
# lammps:
#   lcrc-swing:
#     executable_path: "/path/to/lmp"
#     environment_setup: |
#       module load openmpi
#     mpi_overrides:
#       disable: ["gpu-wrapper"]                 # Disable GPU wrapper script
#       add: ["--map-by rankfile:file={rankfile_path}:OVERSUBSCRIBE"]
"""
