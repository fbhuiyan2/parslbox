# Devlog

**Current Version: 0.7.1**

## Purpose
This file tracks bug fixes and development issues encountered during ParslBox development. 
It serves as a quick reference for resolved problems and their solutions, separate from the formal CHANGELOG.md which documents version releases and feature changes.

---

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

[2026-03-17]
- **Feature**: Resource constraining for non-MPI apps. Non-MPI apps (e.g., Python) previously ran on the head node, ignoring resource assignments. Added `USES_MPI` class attribute to `AppBase` (default `True`). Apps with `USES_MPI=False` get a single-process resource launcher (`mpiexec -n 1 --ppn 1 ...`) prepended to their command, placing them on the assigned node with assigned CPU/GPU resources. GPU binding via `CUDA_VISIBLE_DEVICES` env var; CPU binding via the MPI launcher's native mechanism. New `build_resource_launcher()` in mpi_command_builder.py reuses `MPICommandBuilder` with a synthetic single-rank spec. All user MPI config settings respected. Added `NonMPIApp` example in EXAMPLE_NEW_APP.py and 13 new tests in test_resource_launcher.py.

[2026-03-17]
- **Bug**: Jobs stuck in "Running" state after SIGTERM (walltime exceeded). The signal handler called `status_buffer.flush_all()` but the buffer was typically empty — running statuses had already been flushed during normal execution. The handler never marked active jobs with a terminal status.
  - **Fix**: Restructured signal handler to 3 steps: (0) get active job IDs from in-memory `JobTracker` (only this instance's jobs — safe with concurrent PBX batch jobs), (1) flush status buffer to clear any stale entries, (2) direct `database.update_jobs()` to mark active jobs as "Killed". Direct DB write happens AFTER flush so nothing overwrites it. Added "killed" as a new valid job status.