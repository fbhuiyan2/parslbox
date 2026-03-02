# Changelog

All notable changes to ParslBox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.8.1] - 2026-03-01

### Added

#### Full srun Backend for SLURM Systems
- **Job-type-aware srun command generation** — `_build_srun_flags()` in `mpi_launcher.py` now generates proper CPU binding and GPU wrapper flags for all job types:
  - `fullnode_cpu`: `--ntasks-per-node M --cpus-per-task D --cpu-bind=cores`
  - `subnode_cpu`: `--ntasks-per-node N --cpus-per-task D --cpu-bind=cores --exact`
  - `subnode_gpu`: Same CPU flags as subnode_cpu + GPU wrapper script
  - `fullnode_gpu`: `--ntasks-per-node M --cpus-per-task D --cpu-bind=cores` + GPU wrapper script
- **`--exact` flag for subnode srun steps** — Prevents step from accessing more CPUs than allocated, enabling correct subnode isolation on SLURM
- **GPU wrapper reuse** — srun uses the same `generate_mpiexec_gpu_wrapper()` wrappers as PBS backends; wrappers already include `SLURM_PROCID`/`SLURM_LOCALID` fallback chains for rank detection
- **`--cpus-per-task` in depth binding** — `_build_depth_binding()` srun branch in `mpi_command_builder.py` now emits `--cpus-per-task N --cpu-bind=cores` instead of bare `--cpu-bind=cores`

#### General-purpose `sched_opts` (Scheduler Options) Support
- **Three-layer override chain** for scheduler directives: template → config `sched_opts` → CLI `--sched-opts`
  - Template directives serve as the base layer
  - Per-system persistent overrides via `sched_opts` key in config YAML
  - Per-run overrides via repeatable `--sched-opts` CLI flag
  - Matching keys are replaced in-place; new directives are appended at the `{sched_opts}` placeholder

- **`parslbox/commands/helpers/sched_opts_helpers.py`** — Directive parsing & merging
  - `extract_directive_key()` — Parses `#PBS` and `#SBATCH` directive lines, returns canonical key
  - `merge_sched_opts()` — Merges directives from template, config, and CLI layers

- **`parslbox/commands/helpers/submit_helpers.py`** — Shared submission logic
  - `submit_job()` — Unified function for PBS/SLURM submission with `sched_opts` support
  - `ValidationError` — Shared exception class used by both qsub and sbatch commands

#### New `pbx sbatch` Command
- **SLURM job submission** via `pbx sbatch` — Mirrors `pbx qsub` for SLURM-based systems
  - `submit_to_slurm()` — Thin wrapper calling `submit_job(scheduler_type="slurm")`
  - CLI with SLURM-appropriate terminology (partition, sbatch, squeue)
  - Supports `--sched-opts` for extra `#SBATCH` directives

#### MCP & API
- **`sbatch()` method** added to `ParslBox` API class
- **`SBatchSchema`** added to MCP schemas for SLURM job submission
- **`submit_slurm_job` MCP tool** for submitting SLURM jobs

### Changed

#### PBS Template Cleanup
- **Removed `filesystems` parameter** from `qsub` CLI, API, and MCP schema
  - Filesystems are now handled via `--sched-opts "#PBS -l filesystems=home:eagle"` instead of a dedicated flag
- **Removed special filesystems line-removal logic** from `qsub.py`
- **Removed `#PBS -m be` and `#PBS -M your@email.com`** from default PBS templates
  - Mail notifications can be added via `sched_opts` if needed
- **Added `{sched_opts}` placeholder** to both PBS and SLURM templates in `pbx_config_template.py` and `config_generator.py`

#### Refactored Submission Logic
- **`qsub.py` refactored** — `submit_to_scheduler()` now delegates to `submit_helpers.submit_job()`
- **`--sched-opts` CLI option** added to `pbx qsub` (repeatable flag for extra `#PBS` directives)
- **Generic `job_id` key** in submission results, with backward-compatible `pbs_job_id`/`slurm_job_id` keys

### New Files
- `parslbox/commands/helpers/sched_opts_helpers.py`
- `parslbox/commands/helpers/submit_helpers.py`
- `parslbox/commands/sbatch.py`
- `tests/test_sched_opts.py` (49 tests)

### Modified Files
- `parslbox/commands/qsub.py`
- `parslbox/utils/pbx_config_template.py`
- `parslbox/utils/config_generator.py`
- `parslbox/main.py`
- `parslbox/api.py`
- `parslbox/mcp/schemas.py`
- `parslbox/mcp/mcp_server.py`
- `parslbox/resource_manager/mpi_launcher.py` — Rewrote `_build_srun_flags()`, fixed `build_command()` srun branch
- `parslbox/resource_manager/mpi_command_builder.py` — Added `--cpus-per-task` to srun depth binding
- `tests/test_mpi_launcher.py` — Added srun tests (fullnode_cpu, subnode_gpu, fullnode_gpu, multinode_gpu)
- `tests/test_comprehensive_mpi_wrappers.py` — Added SLURM env var wrapper verification test

---

## [0.7.1] - 2025-02-07


#### New `pbx config` Command
- **Interactive configuration initialization** - New `pbx config` command creates curated config files based on user selections
  - Prompts for systems and applications instead of dumping entire template
  - Supports flexible input formats: space-separated (`1 3 5`) or comma-separated (`1, 3, 5`)
  - Intelligent path handling:
    - `pbx config` → Uses `PBX_CONFIG_PATH` env var if set, else prompts with default home
    - `pbx config ~` → Forces default home directory (`~/.parslbox`)
    - `pbx config .` → Current directory
    - `pbx config /path` → Custom path
  - Smart file conflict resolution with user-provided backup names
  - Optional database creation with conflict handling
  - Environment variable guidance for non-default paths
  - Defaults to `python` app if no apps selected
  - Requires at least one system to be selected
  - API mode for programmatic config generation
  - Automatic invocation from CLI when config is missing (prompts user)

#### MPI Configuration System
- **System-level MPI defaults** - Each system config defines its own MPI defaults via `get_default_mpi_config_yaml()` method
  - `polaris`: MPICH backend with GPU wrapper and depth binding
  - `lcrc-swing`: OpenMPI with short hostnames and oversubscribe flag
  - `aurora-gpu`: MPICH backend with GPU wrapper and depth binding
  - `aurora-tile`: MPICH backend with GPU wrapper and depth binding
  - `crux`: MPICH backend with depth binding (CPU-only)
  - `sophia`: Minimal OpenMPI defaults (base class)
  - Scalable design: new systems automatically define their own defaults

- **Config generation utilities** (`parslbox/utils/config_generator.py`)
  - `ConfigGenerator` class for building curated YAML configs
  - Only includes scheduler templates (PBS/SLURM) needed by selected systems
  - Generates system-specific sections with appropriate MPI defaults
  - Includes example app template for easy custom app addition
  - Adds helpful comments for MPI options (especially `cpu_bind_method`)

### Changed

#### Configuration Management
- **Deprecated automatic config creation** - `initialize_config_file()` is now deprecated and no-op
  - Users must run `pbx config` to create configuration files
  - Main callback now prompts to auto-run `pbx config` if config missing (CLI only)
  - No longer auto-dumps entire template file on first run
  - Better user experience with curated, minimal configs

#### Main Application Flow
- **Updated `main_callback()`** in `main.py`
  - Accepts `typer.Context` to detect current subcommand
  - Skips config file check when running `pbx init`
  - Prompts user to create config interactively when missing
  - Auto-runs `pbx config` setup if user agrees
  - No longer calls deprecated `initialize_config_file()`

### Added

#### Application Arguments (`--args`)
- **New `--args` flag for `pbx add` and `pbx update`** - Pass custom arguments to job executables
  - Arguments are appended to `in_file` before storage (e.g., `python script.py --file afile -o 8 bfile`)
  - `pbx add ... -i script.py --args "--file afile -o 8 bfile"` stores `in_file = "script.py --file afile -o 8 bfile"`
  - `pbx update 1 --args "--new-flag"` reconstructs `in_file` by extracting base script name and appending new args
  - `pbx update 1 -i new_script.py --args "--flag value"` uses new script as base

#### Single-Node GPU Auto-Assignment
- **Auto-assign GPUs for single-node jobs on GPU systems** - `pbx add -n 1` on a GPU system now auto-assigns all GPUs on the node, matching multi-node behavior
  - Previously, `-n 1` without `-g` silently created a CPU-only job even on GPU systems
  - Use `-o` flag to explicitly opt into CPU-only mode on GPU systems

### Fixed

#### MPI Configuration Merging
- **Fixed boolean override bug** in `mpi_config.py`
  - Changed from object-level merging to dict-level merging
  - Now properly handles explicit `false` values at app level overriding system-level `true`
  - Preserves user intent when disabling features (e.g., `use_gpu_wrapper: false`)
  - Prevents loss of information about whether user explicitly set a value

- **Fixed duplicate `--map-by` flags** in MPI command generation
  - `_build_mpi_args()` now checks `cpu_bind_method` before adding default mapping flags
  - Prevents conflicts when using `rankfile` or `depth` binding methods
  - Cleaner, non-conflicting MPI commands

- **Cleaned up redundant code** in `_calculate_ranks_per_node()`
  - Simplified ternary expression that returned same value in both branches
  - Now directly returns `job_spec.ngpus` for GPU-based rank calculation


### Added

#### LAMMPS Scaling Workflow Examples
- **Strong Scaling** (`examples/strong_scaling/`) - Automated orchestration and analysis for LAMMPS strong scaling tests with GPU/CPU support, multi-node validation, and publication-ready plots
- **Weak Scaling** (`examples/weak_scaling/`) - Automated orchestration with intelligent system replication, maintains constant atoms per compute unit, supports GPU/CPU modes with configurable atoms-per-unit


### Technical Details

#### New Files
- `parslbox/commands/config.py` - Interactive config initialization command
- `parslbox/utils/config_generator.py` - Config file generation utilities
- `examples/strong_scaling/lammps_strong_scale_orchestrator.py` - Strong scaling job orchestrator
- `examples/strong_scaling/plot_strong-scale-results.py` - Strong scaling analysis script
- `examples/strong_scaling/python_env-setup.sh` - Python environment setup
- `examples/strong_scaling/README.md` - Strong scaling documentation
- `examples/weak_scaling/lammps_weak_scale_orchestrator.py` - Weak scaling job orchestrator
- `examples/weak_scaling/plot_weak-scale-results.py` - Weak scaling analysis script
- `examples/weak_scaling/python_env-setup.sh` - Python environment setup
- `examples/weak_scaling/README.md` - Weak scaling documentation
- `CHANGELOG.md` - This file

#### Modified Files
- `parslbox/main.py` - Integrated config command, refactored callback to auto-run setup
- `parslbox/api.py` - Updated `ParslBox.__init__()` to check (not create) config
- `parslbox/utils/pbx_config_utils.py` - Deprecated `initialize_config_file()`
- `parslbox/system_configs/base_sysconf.py` - Added `get_default_mpi_config_yaml()` method
- `parslbox/system_configs/polaris.py` - Added MPI defaults
- `parslbox/system_configs/lcrc_swing.py` - Added MPI defaults
- `parslbox/system_configs/aurora_gpu.py` - Added MPI defaults
- `parslbox/system_configs/aurora_tile.py` - Added MPI defaults
- `parslbox/system_configs/crux.py` - Added MPI defaults
- `parslbox/resource_manager/mpi_config.py` - Fixed boolean override merging
- `parslbox/resource_manager/mpi_command_builder.py` - Fixed duplicate flags, cleaned redundant code

#### API Return Value (config_setup)
```python
{
    "config_path": str,           # Path to created config file
    "db_path": str | None,        # Path to database (if created)
    "env_vars_needed": dict | None,  # Environment variables to set (if not default path)
    "systems_selected": List[str],   # Selected systems
    "apps_selected": List[str],      # Selected applications
}
```

### Migration Guide

#### For Users
- **First-time setup**: Run `pbx config` (or will be prompted automatically on first command)
- **Existing users**: No action needed - existing configs continue to work
- **New configs**: Use `pbx config` for interactive, curated config creation

#### For Developers
- **Adding new systems**: Implement `get_default_mpi_config_yaml()` in your system config class
  - Returns dict with MPI defaults for generated configs
  - Falls back to base class (minimal defaults) if not implemented
- **Custom config generation**: Use `ConfigGenerator` class from `config_generator.py`
- **Programmatic init**: Call `config_setup()` with required parameters

---

## [Previous Versions]

_(Version history before 0.7.1 to be added)_
