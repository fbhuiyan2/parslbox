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

ParslBox uses [Poetry](https://python-poetry.org/) for dependency management. Create a Python virtual environment or a Conda envrironment (Python version >=3.11, <3.14) with Poetry installed.

For a Conda envrironment, you can do:

```bash
conda create --name parslbox python=3.11.9
conda activate parslbox
pip install poetry
```

```bash
# Clone the repository
git clone https://github.com/fbhuiyan2/parslbox.git
cd parslbox

# Activate your Python virtual env or Conda env if you have not done so yet

# Install dependencies
poetry install

# Check installation
pbx ls
```

## Quick Start

After installation, you can use the `pbx` command to manage your simulation workflows:

```bash
# List all jobs
pbx ls

# Add a new job, note the --path is the path to the directory where simulation or calculation files are present
pbx add --app lammps --path /path/to/simulation --ngpus 2 --tag my-simulation

# If you have many calculations (like 100 VASP calculations) in sub-dirs inside a dir, then add all
pbx add all -a vasp -n 1 -t ManyVaspCalc

# Run ready jobs on Polaris
pbx qsub \
  --config polaris \
  --job-name first_run \
  --queue debug \
  --select 1 \
  --walltime 30 \
  --project myproject \
  --run-dir /path/to/custom/directory

# pbx qsub command will create a submit.sh file in the run-dir and submit it to the queue
# Inside the submit file, the qsub command runs the `pbx run` command
pbx run --config polaris

# Update job status
pbx update 1 2 3 --status "Failed"

# Remove completed jobs (using filter)
pbx rm $(pbx filter --status done)
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
Run ready jobs on HPC systems:

```bash
# Run all ready jobs on Polaris
pbx run --config polaris

# Run with specific filters
pbx run --config polaris --apps lammps --tags production
```

### `pbx update` - Update Jobs
Modify job properties:

```bash
# Update job status
pbx update 1 2 3 --status "Failed"

# Update job tags
pbx update 4 5 --tag "high-priority"

# Update failed jobs to restart (using filter)
pbx update $(pbx filter --status failed) --status "Ready"
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

### `pbx filter` - Filter Jobs
Get job IDs matching specific criteria for use with other commands:

```bash
# Get IDs of all done jobs
pbx filter --status done

# Get IDs of LAMMPS jobs
pbx filter --app lammps

# Get IDs of jobs with specific tag
pbx filter --tag production

# Combine filters (AND logic)
pbx filter --status failed --app vasp

# Short flags
pbx filter -s done -a lammps -t test
```

### `pbx rm` - Remove Jobs
Remove jobs from the database:

```bash
# Remove completed jobs (using filter)
pbx rm $(pbx filter --status done)

# Remove specific jobs
pbx rm 1 2 3

# Remove all jobs
pbx rm all

# Remove jobs by tag (using filter)
pbx rm $(pbx filter --tag test)

# Remove failed LAMMPS jobs (using filter)
pbx rm $(pbx filter --status failed --app lammps)
```

### `pbx qsub` - Submit PBS Jobs
Generate and submit PBS job scripts for running parslbox workflows:

```bash
# Basic PBS job submission
pbx qsub \
  --config sophia \
  --job-name myrun \
  --queue gpu \
  --select 2 \
  --walltime 90 \
  --project myproject

# With application and tag filters
pbx qsub \
  --config sophia \
  --job-name lammps_production \
  --queue gpu \
  --select 4 \
  --walltime 180 \
  --project myproject \
  --apps lammps \
  --tags production \
  --filesystems home:eagle

# With custom run directory
pbx qsub \
  --config sophia \
  --job-name custom_run \
  --queue gpu \
  --select 1 \
  --walltime 60 \
  --project myproject \
  --run-dir /path/to/custom/directory
```

**Required Parameters:**
- `--config, -c`: System configuration (e.g., 'sophia', 'polaris')
- `--job-name, -N`: PBS job name
- `--queue, -q`: PBS queue name  
- `--select`: Number of nodes to request
- `--walltime, -T`: Wall time in minutes (e.g., 90 for 1.5 hours)
- `--project, -A`: Project/account name

**Optional Parameters:**
- `--filesystems`: Comma-separated list of filesystems (e.g., 'home:eagle')
- `--run-dir`: Custom run directory (default: timestamped directory)
- `--apps, -a`: Comma-separated list of apps to run (e.g., 'lammps,vasp')
- `--tags, -t`: Comma-separated list of tags to run (e.g., 'run1,run2')
- `--retries`: Number of retries for failed tasks (default: 0)

The command automatically:
- Creates timestamped run directories when `--run-dir` is not specified
- Converts walltime from minutes to HH:MM:SS format
- Generates PBS submission script using scheduler templates
- Submits the job with `qsub` and provides the Job ID
- Integrates with existing `pbx run` command parameters

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

1. **Ready** - Job added to database, ready for execution
2. **Restart** - Job marked for re-execution
3. **Submitted** - Job submitted to scheduler
4. **Running** - Job currently executing on compute resources
5. **Done** - Job completed successfully
6. **Failed** - Job encountered an error during execution

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
