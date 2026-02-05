# ParslBox Resource Manager (Updated)

Concise reference to the Resource Manager architecture, key concepts, and MPI command generation strategy.

## Scope and Goals

- Allocate and track nodes, CPUs, and GPUs across HPC systems.
- Support sub-node sharing for single-node jobs, exclusive nodes for multi-node jobs.
- Provide deterministic per-rank resource assignments for sub-node CPU/GPU jobs.
- Generate MPI command lines consistently with modular overrides where needed.

## Core Modules

- `models.py`
  - `NodeResource`: tracks per-node GPUs, CPU cores, occupancy, and health.
  - `JobResourceSpec`: job requirements; detects job type (`subnode_cpu`, `subnode_gpu`, `fullnode_cpu`, `fullnode_gpu`).
  - `ResourceAssignment`: per-rank GPU/CPU assignments, hostnames, environment exports.

- `resource_manager.py`
  - `ResourceManager`: initializes nodes, assigns resources based on job type, manages backlog and health.
  - Design decisions:
    - Single-node sub-node jobs can share nodes.
    - Multi-node jobs get exclusive nodes (no sharing).
    - Single-node CPU jobs use `node_occupancy` to determine core count.
    - GPU jobs assign 1 rank per GPU; also assign CPU cores with optional affinity.

- `cpu_affinity.py`
  - `CPUAffinityManager`: optional affinity for CPU selection per GPU.
  - Format: `"list:group1:group2:..."`, with groups defined by comma-separated ranges (e.g., `"0-15,128-143:16-31,144-159"`).
  - If affinity is present, prefer cores in the GPU's affinity group; otherwise fall back to available cores.

- `mpi_launcher.py`
  - `MPICommandBuilder`: modular MPI flag construction for `mpirun`, `mpiexec`, `srun`.
  - Rankfile generation:
    - OpenMPI: `rank <global_rank>=<hostname> slot=<cpu_cores>`
    - MPICH/PALS: `<rank> <host_index> <cpu_cores>`
  - GPU wrappers export `CUDA_VISIBLE_DEVICES` and `ZE_AFFINITY_MASK` (and related env) per rank.

## Key Assumptions

- GPU jobs: one MPI rank per GPU.
- Cores per GPU (balanced CPU distribution) = `CORES_PER_NODE // GPUS_PER_NODE`.
- Sub-node CPU jobs: per-rank core sets assigned explicitly and bound via rankfile or `--cpu-bind list` (depending on launcher).
- Full-node CPU jobs: core distribution handled by MPI (no per-rank core lists in assignment).
- Multi-node jobs require completely free nodes.

## Job Type Detection

`JobResourceSpec.detect_job_type(system_config)` returns one of:
- `subnode_cpu`: single-node, `node_occupancy < 1.0`.
- `subnode_gpu`: single-node, GPU count less than GPUs per node.
- `fullnode_cpu`: single-node full occupancy or multi-node CPU-only.
- `fullnode_gpu`: single-node all GPUs or multi-node GPU jobs.

This classification drives both resource allocation and MPI binding strategy.

## MPI Command Generation Strategy

MPI flags are built modularly and then optionally filtered/extended by overrides.

### mpirun (OpenMPI)
- Common flags:
  - Process count: `-np <total_ranks>`
  - Hostlist: `-H <host1,host2,...>`
- Full-node CPU jobs:
  - `--map-by core:PE=<cores_per_rank> --bind-to core`
  - MPI manages core distribution (no rankfile).
- Sub-node CPU jobs and any GPU job:
  - Use rankfile-based CPU binding: `--map-by rankfile:file=<openmpi_rankfile>`
  - GPU jobs additionally use a wrapper script to set per-rank GPU env:
    - `CUDA_VISIBLE_DEVICES=<gpu_id>`
    - `ZE_AFFINITY_MASK="<gpu_id>.0"` and related Intel GPU env

### mpiexec (MPICH/PALS)
- Common flags:
  - Process count: `-n <total_ranks>`
- Single-node jobs:
  - `-host <hostname>`
  - Sub-node CPU: `--cpu-bind list:<core_sets_per_rank>`
  - GPU jobs: same as above plus a GPU wrapper script; avoid `--gpu-bind`.
- Multi-node full-node CPU jobs:
  - `-ppn <ranks_per_node> -hosts <hostlist> --depth <cores_per_rank> --cpu-bind depth`
- Multi-node sub-node or GPU jobs:
  - Use rankfile for CPU binding: `--rankfile <mpiexec_rankfile>`
  - GPU jobs add wrapper script for GPU env.

### srun (SLURM)
- Basic layout:
  - `--ntasks <total_ranks> --ntasks-per-node <ranks_per_node_or_ngpus> --nodelist <hostlist> --nodes <num_nodes>`
- Binding specifics are left to SLURM/MPI defaults for the target system.

## Overrides (Minimal and Intuitive)

Apps may provide simple overrides in `config.yaml` to disable or add flags. If no overrides are present, defaults apply.

Example (YAML):
```yaml
vasp:
  sophia:
    executable_path: "/path/to/vasp_gpu"
    environment_setup: |
      module load vasp_env
    mpi_overrides:
      disable: ["rankfile", "-H"]   # remove rankfile-related flags and hostlist flag
      # add: ["--bind-to cores"]    # optional: add exact flags
```

Rules:
- `disable`: accepts substrings (e.g., `"rankfile"`) or exact flags (e.g., `"--rankfile"`, `"-H"`). If matched, the flag and its argument (if present) are removed.
- `add`: list of full flag strings to append; each item is split on spaces to preserve flag-argument pairs. Supports template variables (see below).
- Absent or empty `mpi_overrides` means no changes to defaults.

### Template Variables in `add`

The `add` list supports template variables that are substituted at runtime:

| Variable | Description |
|----------|-------------|
| `{rankfile_path}` | Path to the generated rankfile |
| `{wrapper_path}` | Path to the GPU wrapper script |
| `{hostlist}` | Comma-separated list of hostnames |
| `{total_ranks}` | Total number of MPI ranks |

Example: OpenMPI 4.x compatibility (uses `--rankfile` instead of `--map-by rankfile:file=...`):
```yaml
lammps:
  lcrc-swing:
    executable_path: "/path/to/lmp"
    environment_setup: |
      module load openmpi
    mpi_overrides:
      disable: ["--map-by"]                    # Remove OpenMPI 5.x style rankfile flag
      add: ["--rankfile {rankfile_path}"]      # Add OpenMPI 4.x style rankfile flag
```

## Environment Exports for GPU Jobs

- Single-node GPU jobs: `ResourceAssignment.get_env_vars()` aggregates all GPU IDs on the node:
  - `CUDA_VISIBLE_DEVICES="id1,id2,..."` and `ZE_AFFINITY_MASK="id1,id2,..."`.
- Multi-node GPU jobs: per-rank GPU env is set via wrapper scripts during MPI execution.

## Fault Tolerance (Brief)

Nodes track health and can be quarantined after consecutive failures. Resource freeing updates health state via:
- `free_resources_with_health_check(job_id, job_succeeded, error_message)`

System-level summaries are available for recovery and scheduling decisions.

## Summary

- Deterministic per-rank assignments for sub-node CPU/GPU jobs.
- Exclusive nodes for multi-node jobs.
- MPI command lines built modularly; binding differs by job type:
  - Full-node CPU (mpirun): `--map-by core:PE=N --bind-to core`
  - Sub-node CPU/GPU (mpirun): rankfile + GPU wrapper (for GPU)
  - mpiexec uses `--cpu-bind list` (single-node sub-node), `--rankfile` (multi-node sub-node/GPU), and a GPU wrapper (no `--gpu-bind`).
- Simple, opt-in overrides (`disable`, `add`) per app/system to handle site-specific MPI constraints.
