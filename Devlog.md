# Devlog

**Current Version: 0.9.3**

## Purpose
This file tracks bug fixes and development issues encountered during ParslBox development. 
It serves as a quick reference for resolved problems and their solutions, separate from the formal CHANGELOG.md which documents version releases and feature changes.

---

[2026-10-24]
- **Issue**: `ranks_per_node` above the job's GPU or core count produced ranks with empty GPU and core lists — no wrapper block, no rankfile line, no binding — and the job launched anyway. The 0.9.3 note "removed hardcoded 1-rank-per-GPU" removed only the default; the allocator still sliced a resource list by rank index. The `--ranks-per-node` bound also compared against node capacity (`GPUS_PER_NODE` / `CORES_PER_NODE`) rather than the job's allocation, so it rejected valid sharing and admitted the broken cases.
  - **Fix**: New `resource_manager/rank_distribution.py` holds the split; the four assignment paths and `resource_req._synthetic_assignment` (the command preview, which had its own drifted copy) all call it. Ranks round-robin over GPUs and share the GPU's cores; oversubscribed CPU jobs get the whole owned set (delivered by `rankfile`/`list`/srun `mask_cpu`; `depth`/`cores` carry one number and fall back to one exclusive core per rank). Bound is now hardware threads, with a warning when ranks exceed the job's own GPUs/cores. Also fixed `get_total_gpus()` (counted a shared GPU once per rank), `cores_per_rank` (rank-0 extrapolation over-claimed `--depth`/`--cpus-per-task`), the srun `list`/`rankfile` CPU binding (srun has no `list`; it was emitting `map_cpu` with a whole core set per rank, which SLURM reads as one cpu id per task) and the srun full-node `--gpu-bind` branch, which recomputed `GPUS_PER_NODE // ranks_per_node` instead of reading the assignment and so stranded the same remainder GPUs under srun that the allocator fix had just stopped stranding under MPICH/OpenMPI.

[2025-02-15]
- **Issue**: "Text file busy" error when rerunning jobs after updating status to ready - wrapper scripts couldn't be overwritten.
  - **Fix**: Added file deletion with try-except handling before writing new wrapper scripts and rankfiles in mpi_launcher_helpers.py.

[2025-02-15]
- **Issue**: GPU count not automatically updated when changing number of nodes via `pbx update -n` flag for GPU jobs.
  - **Fix**: Added auto-scaling logic in update.py to calculate and update GPU count based on GPUs-per-node ratio. Multi-node GPU jobs auto-scale; single-node→multi-node requires explicit -g flag with error message showing formula (total_gpus = nnodes * gpus_per_node).

[2025-02-15]
- **Issue**: Jobs left in "Running" state when batch job runs out of walltime on Aurora - SIGTERM handler not properly updating job statuses.
  - **Fix**: Enhanced signal handler in run_cmd_helpers.py with: (1) 10-second timeout protection via signal.alarm() to prevent SIGKILL, (2) detailed timestamped logging at each shutdown step, (3) separate try-except blocks for status buffer flush (critical) vs Parsl cleanup (secondary), ensuring status updates always complete even if Parsl cleanup fails. Added SIGALRM handler in run.py to force exit if shutdown exceeds timeout.

[2026-02-27]
- **Issue**: Adding a job with `-n 1` on a GPU system (e.g., Polaris, Aurora Tile) silently treated it as a CPU-only job (`ngpus=0`), inconsistent with multi-node behavior where `-n 2+` auto-calculates GPUs. On Aurora Tile, this produced a nonsensical MPI command with 204 CPU ranks and no GPU binding.
  - **Fix**: Added auto-GPU-assignment in `validate_resource_parameters()` (job_info_validator.py). When `nnodes==1`, `ngpus==0`, `gpus_per_node > 0`, and no `-o` flag is set, all node GPUs are auto-assigned. Users can opt out to CPU-only mode with `-o` flag.

[2026-02-27]
- **Feature**: Added `--args` flag to `pbx add` and `pbx update` commands for passing custom arguments to job executables (e.g., `python script.py --file afile -o 8`). Args are concatenated with `in_file` before storage — no DB schema changes needed. On update, base script name is extracted from current `in_file` and reconstructed with new args.

[2026-03-05]
- **Bug**: `--ppn` set to total GPUs instead of GPUs-per-node for multi-node GPU jobs. For example, 100 nodes with 12 GPUs/node produced `--ppn 1200` instead of `--ppn 12`. `SimpleMPICommandBuilder._calculate_ranks_per_node()` returned `job_spec.ngpus` (total across all nodes) for GPU jobs.
  - **Fix**: Updated `_calculate_ranks_per_node()` in mpi_command_builder.py to divide `ngpus` by `num_nodes` for multi-node GPU jobs (`ngpus // num_nodes`). Single-node jobs unaffected. Added comprehensive test suite (tests/test_simple_mpi_command_builder.py, 39 tests) covering all backends, hostlists, CPU binding, GPU wrappers, and disable/add overrides.

[2026-03-18]
- **Bug**: Empty rankfile generated for full-node CPU jobs with `cpu_bind_method: rankfile` (e.g., lcrc-improv with OpenMPI). `_assign_fullnode_cpu_job()` in resource_manager.py assigned empty `cpu_ids=[]` per rank with the comment "MPI handles distribution." But the rankfile generator (`generate_openmpi_rankfile()`) skips ranks with empty CPU lists, producing an empty file.
  - **Fix**: `_assign_fullnode_cpu_job()` now captures each node's available core IDs before marking the node as occupied (`assign_fullnode_cpu_job()` clears `available_core_ids`), then distributes cores evenly across ranks — same pattern as `_assign_subnode_cpu_job()`. Updated 4 tests in test_comprehensive_resource_manager.py and test_comprehensive_resource_manager_depr.py to reflect non-empty CPU assignments.

[2026-03-17]
- **Feature**: Resource constraining for non-MPI apps. Non-MPI apps (e.g., Python) previously ran on the head node, ignoring resource assignments. Added `USES_MPI` class attribute to `AppBase` (default `True`). Apps with `USES_MPI=False` get a single-process resource launcher (`mpiexec -n 1 --ppn 1 ...`) prepended to their command, placing them on the assigned node with assigned CPU/GPU resources. GPU binding via `CUDA_VISIBLE_DEVICES` env var; CPU binding via the MPI launcher's native mechanism. New `build_resource_launcher()` in mpi_command_builder.py reuses `MPICommandBuilder` with a synthetic single-rank spec. All user MPI config settings respected. Added `NonMPIApp` example in EXAMPLE_NEW_APP.py and 13 new tests in test_resource_launcher.py.

[2026-03-17]
- **Bug**: Jobs stuck in "Running" state after SIGTERM (walltime exceeded). The signal handler called `status_buffer.flush_all()` but the buffer was typically empty — running statuses had already been flushed during normal execution. The handler never marked active jobs with a terminal status.
  - **Fix**: Restructured signal handler to 3 steps: (0) get active job IDs from in-memory `JobTracker` (only this instance's jobs — safe with concurrent PBX batch jobs), (1) flush status buffer to clear any stale entries, (2) direct `database.update_jobs()` to mark active jobs as "Killed". Direct DB write happens AFTER flush so nothing overwrites it. Added "killed" as a new valid job status.

[2026-03-19]
- **Bug (FIXED)**: MPI config `disable` list only applies to `mpi_extra` flags, not core `mpi_args`. In `MPICommandBuilder.build_command()`, the disable/add pipeline only processes flags from `_build_mpi_extra()`, while flags from `_build_mpi_args()` (like `--map-by ppr:N:node` for OpenMPI) bypass filtering entirely. This prevents users from disabling core flags, breaking workarounds for OpenMPI version differences (e.g., 4.1.x uses `--rankfile <path>` vs 5.x uses `--map-by rankfile:file=<path>`).
  - **Fix applied**: `_apply_disable()` and `_apply_add()` now operate on the combined `mpi_args + mpi_extra` list in `build_command()`. Tests added for disabling core args across all three backends (OpenMPI, MPICH, srun).

[2026-04-18]
- **Bug**: Sub-node srun jobs fail with "Memory required by task is not available" because the first srun step reserves all node memory, leaving none for subsequent steps.
  - **Fix**: Added `--mem=0` to srun args in `_build_mpi_args()` (mpi_command_builder.py). `--mem=0` grants each job step access to all available job memory on the node, preventing the default behavior where the first step reserves all memory.
  - **Superseded**: `--mem=0` replaced with `--mem-per-gpu={DRAM_PER_NODE // GPUS_PER_NODE}G` in v0.9.2 for proper per-step memory partitioning.

[2026-05-15]
- **Refactor**: Rewrote srun backend to use native SLURM GPU/memory flags instead of wrapper scripts and `--mem=0`. Added `cores` and `threads` as srun-native `cpu_bind_method` options (replaces misleading `depth` for SLURM).
- **Feature**: Added NERSC Perlmutter GPU and CPU system configs with `DRAM_PER_NODE` base class attribute.

[2026-05-19]
- **Refactor**: Reworked srun sub-node mechanism after Perlmutter testing. `--gpus-per-task` and `--exact` don't partition resources between concurrent steps on Perlmutter. Replaced with `--gpu-bind=map_gpu:{pbx_ids}` + `--cpu-bind=mask_cpu:{hex}` + `--overlap`. Added `-N {nodes}` to all srun jobs. Removed `--mem-per-gpu` (not supported on Perlmutter).
- **Feature**: Configurable `ranks_per_node` for GPU jobs. Removed hardcoded 1-rank-per-GPU. Per-app defaults via `get_default_ranks_per_node()`. Multi-GPU per rank uses `--gpu-bind=mask_gpu`. Simplified `get_total_ranks()` and `_calculate_ranks_per_node()`.
- **Change**: Python and Julia apps switched to `USES_MPI=True` with `ranks_per_node=1` default. Scripts are now launched via srun/mpirun with 1 rank per node, works with or without mpi4py/MPI.jl.
