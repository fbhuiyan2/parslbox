# Supported HPC systems

ParslBox ships with 26 system configs. Pass the name to `--config` (`pbx qsub`, `pbx sbatch`, `pbx add`, `pbx run`) or pick it in the `pbx config` wizard. The MCP tool `list_systems_and_apps` prints the same list.

| System | Scheduler | GPUs/Node | GPU Type | Cores/Node | MPI Backend |
|--------|-----------|-----------|----------|------------|-------------|
| **polaris** | PBS | 4 | NVIDIA A100 | 64 | MPICH |
| **aurora-gpu** | PBS | 6 | Intel Max 1550 | 208 | MPICH |
| **aurora-tile** | PBS | 12 (6x2 tiles) | Intel Max 1550 | 208 | MPICH |
| **sophia** | PBS | 8 | NVIDIA A100 (DGX) | 128 | OpenMPI |
| **crux** | PBS | 0 (CPU-only) | — | 256 | MPICH |
| **lcrc-swing** | PBS | 8 | NVIDIA A100 | 64 | OpenMPI |
| **lcrc-improv** | PBS | 0 (CPU-only) | — | 128 | OpenMPI |
| **pinnacles-cenvalarc** | SLURM | 0 or 2 (auto-detected) | NVIDIA L40S / H200 NVL | 64 | srun |
| **perlmutter-gpu** | SLURM | 4 | NVIDIA A100 | 128 | srun |
| **perlmutter-cpu** | SLURM | 0 (CPU-only) | — | 128 | srun |
| **perlmutter-gpu-srun** | SLURM | 4 | NVIDIA A100 | 128 | srun |
| **kestrel-cpu** | SLURM | 0 (CPU-only) | — | 104 | srun |
| **kestrel-gpu** | SLURM | 4 | NVIDIA H100 | 128 | srun |
| **delta-cpu** | SLURM | 0 (CPU-only) | — | 128 | srun |
| **delta-gpu** | SLURM | 4 | NVIDIA A100 / A40 | 64 | srun |
| **delta-h200** | SLURM | 8 | NVIDIA H200 | 96 | srun |
| **anvil-cpu** | SLURM | 0 (CPU-only) | — | 128 | srun |
| **anvil-gpu** | SLURM | 4 | NVIDIA A100 | 128 | srun |
| **anvil-ai** | SLURM | 4 | NVIDIA H100 | 96 | srun |
| **expanse-cpu** | SLURM | 0 (CPU-only) | — | 128 | srun |
| **expanse-gpu** | SLURM | 4 | NVIDIA V100 | 40 | srun |
| **midway3-cpu** | SLURM | 0 (CPU-only) | — | 48 | srun |
| **midway3-gpu** | SLURM | 4 | NVIDIA V100 / RTX 6000 / A100 | 48 | srun |
| **bridges2-cpu** | SLURM | 0 (CPU-only) | — | 128 | srun |
| **bridges2-v100** | SLURM | 8 | NVIDIA V100 | 40 | srun |
| **bridges2-h100** | SLURM | 8 | NVIDIA H100 | 104 | srun |

## Status of the newer SLURM configs

The Kestrel, Delta, Anvil, Expanse, Midway3 and Bridges-2 configs were written from each site's public documentation and have not been run on the systems yet. Treat their defaults as a starting point and adjust through `config.yaml` / `--sched-opts`. GPU counts and types are requested there, not hardcoded, because most of these sites let jobs share a node.

Sub-node (shared-node) allocations on SLURM are untested; use whole-node allocations until they have been checked. The assumptions involved are listed in [`parslbox/resource_manager/README.md`](../parslbox/resource_manager/README.md) under "SLURM sub-node allocations".

## Adding a system

Shipping with these systems does not limit you to them; adding one is a small, self-contained change.

1. **Write a module** in `parslbox/system_configs/`. Copy the closest existing one (`perlmutter_cpu.py` for a SLURM CPU system, `perlmutter_gpu.py` or `polaris.py` for a GPU system) and set the class attributes: `SYSTEM_NAME`, `CORES_PER_NODE`, `GPUS_PER_NODE`, `SCHEDULER` (`PBS` or `SLURM`), `MPI_BACKEND` (`mpich`, `openmpi` or `srun`), `MAX_WORKERS_PER_NODE`, `GPU_TYPE`, and optionally `WORKER_CPU_AFFINITY` and `EXCLUDE_CORES`.
2. **Implement the four methods** from `SystemConfig` (`base_sysconf.py`): `detect_resources()` (nodes and GPUs of the current allocation), `get_config()` (the Parsl executor config), `get_default_sched_opts()` (extra `#PBS` / `#SBATCH` lines, or `None`), and `get_default_mpi_config_yaml()` (MPI defaults written into a fresh `config.yaml`).
3. **Register it** in `parslbox/system_configs/loader.py`: import the class and add one entry to `CONFIG_FACTORIES`. That is the only registration point. `--config` choices, the `pbx config` wizard, the API and the MCP schemas are all built from that registry.
4. **Fill in `config.yaml`** for the new system: `<system>.pbx_python_env_setup` and, per app, `<app>.<system>.executable_path` / `environment_setup`. See [`config-file.md`](config-file.md).

Start with a whole-node job to check the config before trying anything fancier.
