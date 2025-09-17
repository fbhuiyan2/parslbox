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
      
      cd $PBS_O_WORKDIR
      > pbx_scheduler.out
      
      NNODES=$(wc -l < $PBS_NODEFILE)
      echo "NNODES = $NNODES"
      echo "Job ID: $PBS_JOBID"
      echo "Job Name: $PBS_JOBNAME"
      
      {python_env_setup}
      
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
      
      {python_env_setup}
      
      pbx run --config {config} --run-dir {run_dir} {run_options}

# System-specific configurations
sophia:
  python_env_setup: |
    module load conda
    conda activate parslbox

polaris:
  python_env_setup: |
    module load conda
    conda activate parslbox

# Application configurations
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

  # Settings for running LAMMPS on ALCF Sophia
  sophia:
    executable_path: "/path/to/your/lammps/executable/on/sophia"
    environment_setup: |
      # Add all necessary `module load` and `export` commands here.
      # Example:
      # module load compilers/openmpi/5.0.3
      # export LD_LIBRARY_PATH=...

# --- Add other applications below ---
# vasp:
#   polaris:
#     executable_path: "/path/to/vasp_gpu"
#     environment_setup: |
#       # module load vasp_env
"""
