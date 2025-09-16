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
