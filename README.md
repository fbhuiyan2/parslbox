# ParslBox

A powerful CLI workflow management tool for computational simulations built on top of [Parsl](https://parsl-project.org/). ParslBox simplifies the management and execution of high-performance computing (HPC) workflows, with built-in support for popular simulation codes like LAMMPS and VASP.

## Features

- **Job Management**: Add, list, remove, update, and run computational simulation jobs
- **Multi-Application Support**: Built-in support for LAMMPS and VASP simulations
- **Database Tracking**: Persistent tracking of job status, metadata, and execution history
- **HPC Integration**: Pre-configured support for major HPC systems (Polaris, Sophia)
- **GPU-Aware Execution**: Intelligent GPU resource allocation and management
- **Flexible Configuration**: YAML-based configuration for different compute environments
- **Rich CLI Interface**: User-friendly command-line interface with colored output and tables

## Installation

ParslBox uses [Poetry](https://python-poetry.org/) for dependency management. Make sure you have Poetry installed, then:

```bash
# Clone the repository
git clone <repository-url>
cd parslbox

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

## Quick Start

After installation, you can use the `pbx` command to manage your simulation workflows:

```bash
# List all jobs
pbx ls

# Add a new job
pbx add --app lammps --path /path/to/simulation --ngpus 2 --tag "my-simulation"

# Run jobs with specific status
pbx run --status "Queued" --config polaris

# Update job status
pbx update --job-ids 1,2,3 --status "Cancelled"

# Remove completed jobs
pbx rm --status "Done"
```

## Commands

### `pbx ls` - List Jobs
Display all jobs in the database with filtering options:

```bash
# List all jobs
pbx ls

# Filter by status
pbx ls --status "Running"

# Filter by application
pbx ls --app "lammps"

# Filter by tag
pbx ls --tag "production"
```

### `pbx add` - Add Jobs
Add new simulation jobs to the database:

```bash
# Add a LAMMPS job
pbx add --app lammps --path /path/to/lammps/simulation --ngpus 4 --tag "md-simulation"

# Add a VASP job
pbx add --app vasp --path /path/to/vasp/calculation --ngpus 2 --tag "dft-calc"
```

### `pbx run` - Execute Jobs
Run queued jobs on HPC systems:

```bash
# Run all queued jobs on Polaris
pbx run --status "Queued" --config polaris

# Run specific jobs
pbx run --job-ids 1,2,3 --config sophia
```

### `pbx update` - Update Jobs
Modify job properties:

```bash
# Update job status
pbx update --job-ids 1,2,3 --status "Cancelled"

# Update job tags
pbx update --job-ids 4,5 --tag "high-priority"
```

### `pbx info` - Show Job Information
Display detailed information about specific jobs:

```bash
# Show all information for specific jobs
pbx info 1 2 7 12

# Show only paths for specific jobs
pbx info 1 2 --path

# Show multiple fields (path and number of GPUs)
pbx info 1 2 --path --ngpus

# Show only status for specific jobs
pbx info 5 --status

# Available flags:
# --path, -p: Show job paths
# --ngpus, -n: Show number of GPUs
# --app, -a: Show application type
# --status, -s: Show job status
# --tag, -t: Show job tags
# --sched-job-id, -j: Show scheduler job ID
# --timestamp, -ts: Show timestamps
```

### `pbx rm` - Remove Jobs
Remove jobs from the database:

```bash
# Remove completed jobs
pbx rm --status "Done"

# Remove specific jobs
pbx rm --job-ids 1,2,3

# Remove jobs by tag
pbx rm --tag "test"
```

## Configuration

ParslBox uses YAML configuration files to define execution environments for different HPC systems. Configuration files are automatically created in your home directory under `.parslbox/`.

### Supported Systems

- **Polaris** (Argonne National Laboratory)
- **Sophia** (Custom HPC configuration)

### Application Configuration

Each supported application (LAMMPS, VASP) can be configured with:

- Environment setup commands
- MPI environment variables
- Executable paths
- System-specific optimizations

Example configuration structure:
```yaml
apps:
  lammps:
    environment_setup: |
      module load lammps
      export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    mpi_env: "-x CUDA_VISIBLE_DEVICES"
    executable_path: "/path/to/lmp"
  
  vasp:
    environment_setup: |
      module load vasp
    executable_path: "/path/to/vasp_gpu"
```

## Job Status Workflow

Jobs progress through the following states:

1. **Queued** - Job added to database, ready for execution
2. **Running** - Job currently executing on compute resources
3. **Done** - Job completed successfully
4. **Failed** - Job encountered an error during execution
5. **Cancelled** - Job manually cancelled by user

## Database Schema

ParslBox maintains a SQLite database with the following job information:

- **Job ID** - Unique identifier
- **Application** - Simulation code (lammps, vasp)
- **Status** - Current job state
- **NGPUs** - Number of GPUs allocated
- **Scheduler Job ID** - HPC scheduler job identifier
- **Tag** - User-defined label for organization
- **Timestamp** - Job creation time
- **Path** - Simulation directory path

## Requirements

- Python ≥ 3.11, < 3.14
- Parsl ≥ 2025.9.8
- Additional scientific computing libraries (NumPy, Pandas, SciPy, etc.)

## Dependencies

ParslBox includes the following key dependencies:

- **Parsl** - Parallel scripting library
- **Typer** - CLI framework
- **Rich** - Terminal formatting and tables
- **PyYAML** - YAML configuration parsing
- **Scientific Stack** - NumPy, Pandas, SciPy, scikit-learn
- **Chemistry Tools** - RDKit, ASE, Pymatgen

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

[Add your license information here]

## Support

For questions, issues, or feature requests, please [open an issue](link-to-issues) on the project repository.
