# LAMMPS Weak Scaling Analysis for ParslBox

This directory contains scripts for orchestrating and analyzing LAMMPS weak scaling tests using ParslBox.

## Overview

**Weak scaling** tests measure how well a parallel application maintains performance as both the problem size and compute resources scale proportionally. Unlike strong scaling (fixed problem size), weak scaling keeps the work per compute unit constant.

For LAMMPS simulations, this means:
- Each GPU/core processes approximately the same number of atoms
- The total system size grows linearly with compute resources
- Ideal performance remains constant regardless of scale

## Files

- `lammps_weak_scale_orchestrator.py` - Main orchestrator script that creates jobs
- `plot_weak-scale-results.py` - Analysis script that processes results and creates plots
- `python_env-setup.sh` - Environment setup script for the analysis
- `README.md` - This documentation file

## Quick Start

1. **Prepare your input files** in a working directory:
   - Small LAMMPS data file (e.g., `small.lmp`) with <100 atoms, <100 Ų
   - LAMMPS input file (e.g., `in.lammps`)
   - Force field file (e.g., `pair_coeff.dat`, `forcefield.dat`)

2. **Copy the scripts** to your working directory:
   ```bash
   cp /path/to/parslbox/examples/weak_scaling/*.py .
   cp /path/to/parslbox/examples/weak_scaling/*.sh .
   ```

3. **Run the orchestrator** to create jobs:
   ```bash
   # GPU scaling
   python lammps_weak_scale_orchestrator.py polaris --gpus 1 2 4 --fstruct small.lmp --ff pair_coeff.dat
   
   # CPU scaling
   python lammps_weak_scale_orchestrator.py sophia --cores 1 8 64 --fstruct small.lmp --ff pair_coeff.dat
   ```

4. **Submit and monitor jobs**:
   ```bash
   pbx qsub          # Submit all jobs
   pbx ls            # Monitor job status
   ```

5. **Results will be generated automatically** when all jobs complete.

## Detailed Usage

### Orchestrator Script

```bash
python lammps_weak_scale_orchestrator.py <config_name> [--gpus | --cores] <counts> --fstruct <file> --ff <file> [options]
```

**Required Arguments:**
- `config_name` - System configuration (e.g., `polaris`, `crux`, `sophia`)
- `--gpus` OR `--cores` - Space-separated list of GPU/core counts (mutually exclusive)
- `--fstruct` - Small structure/data file name (<100 atoms, <100 Ų)
- `--ff` - Force field file name

**Optional Arguments:**
- `--in` - LAMMPS input file name (default: `in.lammps`)
- `--natoms-per-unit` - Target atoms per GPU/core (default: 250)
- `--rankspercore` - Ranks per core for CPU scaling (default: 1)
- `--tag` - Job tag for organization (default: `weak_scale`)

**Examples:**

```bash
# GPU weak scaling on Polaris
python lammps_weak_scale_orchestrator.py polaris \
    --gpus 1 2 4 8 \
    --fstruct small.lmp \
    --ff pair_coeff.dat

# GPU scaling with custom atoms per GPU
python lammps_weak_scale_orchestrator.py polaris \
    --gpus 1 2 4 \
    --fstruct data.lmp \
    --ff forcefield.dat \
    --natoms-per-unit 500

# CPU weak scaling on Sophia
python lammps_weak_scale_orchestrator.py sophia \
    --cores 1 8 64 128 \
    --fstruct small.lmp \
    --ff pair_coeff.dat

# CPU scaling with oversubscription
python lammps_weak_scale_orchestrator.py sophia \
    --cores 1 8 64 \
    --fstruct data.lmp \
    --ff forcefield.dat \
    --rankspercore 2
```

### What the Orchestrator Does

1. **Parses the data file**: Extracts atom count and box dimensions
2. **Calculates replication factors**: Determines optimal nx, ny, nz to match target atoms per unit
3. **Creates directories**: `1gpu/`, `2gpu/`, etc. OR `1core/`, `8core/`, etc.
4. **Modifies input files**: Inserts `replicate` command after `read_data`
5. **Copies files**: Structure file, force field, and modified input to each directory
6. **Creates jobs**: LAMMPS jobs with proper GPU/core allocation
7. **Sets dependencies**: Jobs run in sequence
8. **Adds analysis job**: Python analysis runs after all LAMMPS jobs complete

### Replication Strategy

The orchestrator uses **non-cubic replication** to get as close as possible to the target atom count:

```
Target atoms = natoms_per_unit × num_units
Scale factor = target_atoms / initial_atoms

# Algorithm finds optimal nx, ny, nz to minimize:
|initial_atoms × nx × ny × nz - target_atoms|
```

**Example:**
- Initial structure: 64 atoms
- Target: 250 atoms/GPU
- 1 GPU: 64 atoms (1×1×1 replication)
- 2 GPUs: 512 atoms (2×2×2 replication, 256 atoms/GPU)
- 4 GPUs: 512 atoms (2×2×2 replication, 128 atoms/GPU) or 4096 atoms (4×4×4, 1024 atoms/GPU)

The algorithm picks the replication that gets closest to the target.

### Analysis Script

The analysis script runs automatically as the final job, but can also be run manually:

```bash
python plot_weak-scale-results.py [--base-dir DIR] [--output-prefix PREFIX] [--mode MODE]
```

**Options:**
- `--base-dir` - Directory containing GPU/core subdirectories (default: current directory)
- `--output-prefix` - Prefix for output files (default: `weak_scaling`)
- `--mode` - Scaling mode: `gpu`, `core`, or `auto` to detect (default: `auto`)

### Generated Output Files

The analysis script generates:

1. **`weak_scaling_performance.png`** - Performance vs GPU/core count
   - Shows timesteps/s and ns/day
   - Includes baseline (ideal constant performance)
2. **`weak_scaling_efficiency.png`** - Weak scaling efficiency plot
   - Efficiency = (Performance_N / Performance_1) × 100%
   - Ideal = 100% (constant performance)
3. **`weak_scaling_system_size.png`** - Total atoms vs GPU/core count
   - Shows actual vs ideal linear scaling
4. **`weak_scaling_data.csv`** - Raw performance and efficiency data
5. **`weak_scaling_summary.txt`** - Summary report with key metrics

All plots are publication-ready with 300 DPI resolution.

## Directory Structure

After running the orchestrator, your directory will look like:

```
working_directory/
├── in.lammps                           # Original input file
├── small.lmp                          # Original small structure file
├── pair_coeff.dat                     # Original force field file
├── lammps_weak_scale_orchestrator.py
├── plot_weak-scale-results.py
├── python_env-setup.sh
├── 1gpu/  (or 1core/)
│   ├── in.lammps                      # Modified with replicate 1 1 1
│   ├── small.lmp                      # Copied original
│   ├── pair_coeff.dat                 # Copied
│   └── log.lammps                     # Generated by LAMMPS
├── 2gpu/
│   ├── in.lammps                      # Modified with replicate 2 2 2
│   ├── small.lmp
│   ├── pair_coeff.dat
│   └── log.lammps
├── 4gpu/
│   └── (same structure)
├── weak_scaling_performance.png       # Generated plots
├── weak_scaling_efficiency.png
├── weak_scaling_system_size.png
├── weak_scaling_data.csv
└── weak_scaling_summary.txt
```

## Performance Metrics

### Weak Scaling Efficiency

For weak scaling, the key metric is:

**Efficiency = (Performance_N / Performance_1) × 100%**

Where:
- Performance_N = Performance at N GPUs/cores
- Performance_1 = Baseline performance at 1 GPU/core

**Interpretation:**
- 100% = Ideal (performance stays constant as system scales)
- 80-95% = Good (typical for real applications)
- <80% = Poor (significant overhead from communication/synchronization)

### Why Performance Should Stay Constant

In weak scaling:
- Each GPU/core processes ~same number of atoms
- Computational work per unit stays constant
- Ideal: No performance degradation as scale increases
- Reality: Some degradation due to:
  - Increased communication overhead
  - Load imbalancing
  - System effects (network, I/O)

## GPU vs CPU Scaling

### GPU Scaling (`--gpus`)

```bash
python lammps_weak_scale_orchestrator.py polaris --gpus 1 2 4 8 --fstruct small.lmp --ff pair_coeff.dat
```

- Uses GPU-accelerated LAMMPS with Kokkos
- Each GPU processes target atoms (default: 250)
- Multi-node jobs automatically allocated for counts > GPUs per node
- Example: On Polaris (4 GPUs/node), `--gpus 8` uses 2 nodes

### CPU Scaling (`--cores`)

```bash
python lammps_weak_scale_orchestrator.py sophia --cores 1 8 64 128 --fstruct small.lmp --ff pair_coeff.dat
```

- Uses CPU-only LAMMPS
- Each core processes target atoms (default: 250)
- Can use `--rankspercore` for oversubscription
- Multi-node jobs automatically allocated for counts > cores per node

## Input File Requirements

### LAMMPS Data File

The small structure file should:
- Contain <100 atoms (recommended)
- Have box size <100 Ų (recommended)
- Use periodic boundary conditions (required for replication)
- Be a valid LAMMPS data file format

**Example header:**
```
LAMMPS data file

64 atoms
2 atom types

0.0 25.0 xlo xhi
0.0 25.0 ylo yhi
0.0 25.0 zlo zhi
```

### LAMMPS Input File

The input file should:
- Contain `read_data <filename>` command
- Not already have a `replicate` command (will be inserted automatically)
- Be compatible with the structure and force field files

**Example:**
```lammps
# LAMMPS input file
units metal
atom_style atomic
boundary p p p

read_data small.lmp
# replicate command will be inserted here automatically

pair_style eam/alloy
pair_coeff * * forcefield.dat Al Cu

timestep 0.001
run 1000
```

## Validation and Error Checking

The orchestrator performs several validation checks:

1. **File existence**: Verifies all input files exist
2. **Data file parsing**: Extracts atom count and box dimensions
3. **Size warnings**: Warns if structure is larger than recommended
4. **Scale count validation**: Ensures multi-node jobs use proper multiples
5. **LAMMPS configuration**: Verifies LAMMPS is configured in ParslBox

## Troubleshooting

**Common Issues:**

1. **"Could not find atom count in LAMMPS data file"**
   - Ensure data file has proper header with "N atoms" line
   - Check file format matches LAMMPS data file specification

2. **"GPU count X is not compatible with multi-node jobs"**
   - For multi-node jobs, use multiples of GPUs per node
   - Example: Polaris (4 GPU/node) requires 4, 8, 12, etc.

3. **"No performance data found"**
   - Check that LAMMPS jobs completed successfully
   - Verify `log.lammps` files contain "Performance:" line

4. **Replication produces unexpected atom counts**
   - This is normal - non-cubic replication gets close but not exact
   - Check the orchestrator output for actual vs target atoms

5. **Analysis script can't detect mode**
   - Manually specify with `--mode gpu` or `--mode core`
   - Ensure directories are named correctly (e.g., `1gpu`, `2gpu`)

## Advanced Usage

### Custom Atom Counts

```bash
# Target 500 atoms per GPU instead of default 250
python lammps_weak_scale_orchestrator.py polaris \
    --gpus 1 2 4 \
    --fstruct small.lmp \
    --ff pair_coeff.dat \
    --natoms-per-unit 500
```

### CPU Oversubscription

```bash
# Run 2 MPI ranks per core
python lammps_weak_scale_orchestrator.py sophia \
    --cores 1 8 64 \
    --fstruct small.lmp \
    --ff pair_coeff.dat \
    --rankspercore 2
```

### Custom Job Tags

```bash
# Use custom tag for job organization
python lammps_weak_scale_orchestrator.py polaris \
    --gpus 1 2 4 \
    --fstruct small.lmp \
    --ff pair_coeff.dat \
    --tag my_weak_scaling_test
```

## Comparison with Strong Scaling

| Aspect | Strong Scaling | Weak Scaling |
|--------|---------------|--------------|
| Problem size | Fixed | Scales with resources |
| Work per unit | Decreases | Constant |
| Ideal speedup | Linear (Nx faster) | Constant performance |
| Efficiency metric | Speedup / N | Performance_N / Performance_1 |
| Use case | Fixed problem, more resources | Larger problems, more resources |

## System Requirements

- ParslBox properly configured for your system
- Python 3.7+ with matplotlib, numpy, pandas
- LAMMPS configured in ParslBox
- Sufficient storage for multiple job directories
- Small initial structure file (<100 atoms recommended)

## Integration with ParslBox

The scripts integrate seamlessly with ParslBox:
- Uses ParslBox API for job creation and management
- Follows ParslBox job dependency patterns
- Compatible with all ParslBox-supported systems
- Leverages existing LAMMPS app configuration

## Performance Considerations

- **Storage**: Each directory contains copies of input files
- **Dependencies**: Jobs run sequentially to ensure proper measurement
- **Analysis**: Runs with minimal CPU resources (10% node occupancy)
- **Plots**: Generated with publication-ready quality (300 DPI)
- **Replication**: LAMMPS handles replication efficiently at runtime

## Example Workflow

```bash
# 1. Set up working directory
mkdir my_weak_scaling_test
cd my_weak_scaling_test

# 2. Copy your LAMMPS input files (small structure!)
cp /path/to/small.lmp .
cp /path/to/in.lammps .
cp /path/to/pair_coeff.dat .

# 3. Copy weak scaling scripts
cp /path/to/parslbox/examples/weak_scaling/*.py .
cp /path/to/parslbox/examples/weak_scaling/*.sh .

# 4. Run orchestrator
python lammps_weak_scale_orchestrator.py polaris \
    --gpus 1 2 4 8 \
    --fstruct small.lmp \
    --ff pair_coeff.dat \
    --natoms-per-unit 250

# 5. Submit jobs
pbx qsub

# 6. Monitor progress
pbx ls

# 7. View results (generated automatically)
ls weak_scaling_*
```

## Expected Results

For a well-optimized LAMMPS simulation:
- **Efficiency**: 80-95% across all scale points
- **Performance**: Relatively constant (within 10-20% of baseline)
- **System size**: Linear growth with compute resources

Lower efficiency may indicate:
- Communication bottlenecks
- Load imbalancing
- System-specific issues (network, I/O)
- Problem size too small (overhead dominates)

## Citation

If you use this weak scaling workflow in your research, please cite ParslBox:

```
[ParslBox citation information]
```

## Support

For issues or questions:
- Check the troubleshooting section above
- Review ParslBox documentation
- Open an issue on the ParslBox GitHub repository
