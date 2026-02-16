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

# ---------------------------------------------------------------------------
# System-specific configurations
#
# Each system can define:
#   - pbx_python_env_setup: Shell commands to set up ParslBox Python environment
#   - mpi: MPI configuration defaults for this system (see MPI Configuration below)
# ---------------------------------------------------------------------------

sophia:
  pbx_python_env_setup: |
    module load conda
    conda activate parslbox
  # MPI defaults for Sophia (OpenMPI)
  mpi:
    backend: openmpi

polaris:
  pbx_python_env_setup: |
    module load conda
    conda activate parslbox
  # MPI defaults for Polaris (PALS)
  mpi:
    backend: pals
    use_gpu_wrapper: true
    cpu_bind_method: depth

lcrc-swing:
  pbx_python_env_setup: |
    module load conda
    conda activate parslbox
  # MPI defaults for LCRC Swing (OpenMPI)
  mpi:
    backend: openmpi
    use_short_hostnames: true
    add: ["--oversubscribe"]


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
    

      
# ---------------------------------------------------------------------------
# MPI Configuration
#
# The mpi: section can be defined at system-level (above) or app-level (below).
# App-level settings override system-level settings.
#
# MPI Configuration Options:
#   backend: openmpi | pals | srun
#   mpi_cmd: Custom MPI command path (optional, overrides backend default)
#   use_gpu_wrapper: true | false (generate GPU assignment wrapper script)
#   use_hostlist: true | false (explicitly pass hostlist to MPI)
#   use_short_hostnames: true | false (strip domain from hostnames)
#   cpu_bind_method: none | rankfile | list | depth | depth <N>
#   disable: [] (list of flags/substrings to remove, applied first)
#   add: [] (list of flags to append, supports templates)
#
# CPU Binding Methods:
#   none      - No CPU binding (simplest, default)
#   rankfile  - Use rankfile for precise per-rank CPU binding
#   list      - Use --cpu-bind list (PALS) or rankfile (OpenMPI)
#   depth     - Auto-calculated cores per rank (--depth for PALS, --map-by core:PE= for OpenMPI)
#   depth 8   - Explicit depth value (e.g., 8 cores per rank)
#
# Template Variables (for use in 'add'):
#   {total_ranks}    - Total number of MPI ranks
#   {ranks_per_node} - Ranks per node
#   {cores_per_rank} - Auto-calculated cores per rank
#   {hostlist}       - Comma-separated hostnames (short if use_short_hostnames: true)
#   {rankfile_path}  - Path to generated rankfile (generated on-demand)
#   {wrapper_path}   - Path to GPU wrapper script (generated on-demand)
# ---------------------------------------------------------------------------

# --- Add other applications below ---

# vasp:
#   polaris:
#     executable_path: "/path/to/vasp_gpu"
#     environment_setup: |
#       module load vasp_env
#     mpi:
#       # Inherits from polaris system defaults (pals, gpu_wrapper, depth binding)
#       # Can override if needed
#
#   sophia:
#     executable_path: "/path/to/vasp_gpu"
#     environment_setup: |
#       module load vasp_env
#     mpi:
#       backend: openmpi              # Override system default if needed
#       use_gpu_wrapper: false        # VASP handles GPU internally
#       disable: ["-H"]               # Remove hostlist flag if it causes issues

# --- Example: LAMMPS on LCRC Swing (OpenMPI) ---
# lammps:
#   lcrc-swing:
#     executable_path: "/path/to/lmp"
#     environment_setup: |
#       module load openmpi cuda
#     mpi:
#       # Inherits from lcrc-swing system defaults (openmpi, short_hostnames, oversubscribe)
#       # Add CPU-GPU affinity if needed:
#       cpu_bind_method: rankfile
#       use_gpu_wrapper: true

# --- Example: CPU-only job with depth binding ---
# my_cpu_app:
#   crux:
#     executable_path: "/path/to/app"
#     environment_setup: |
#       module load intel
#     mpi:
#       cpu_bind_method: depth        # Auto-calculate cores per rank
#       # Or specify explicit depth:
#       # cpu_bind_method: depth 16   # 16 cores per rank

# --- Example: Custom MPI command path ---
# special_app:
#   my_system:
#     executable_path: "/path/to/app"
#     mpi:
#       backend: openmpi
#       mpi_cmd: /opt/openmpi-4.1.6/bin/mpirun  # Use specific MPI version

# --- Example: Power user with custom flags ---
# advanced_app:
#   sophia:
#     executable_path: "/path/to/app"
#     mpi:
#       disable: ["--map-by"]
#       add: ["--map-by rankfile:file={rankfile_path}:OVERSUBSCRIBE", "--mca btl ^openib"]
"""
