# Changelog

All notable changes to ParslBox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed

#### `pbx run` — one DB write per job during completion drain
- **The status buffer never batched on the completion side.** The orchestrator loop handled exactly one finished future per pass and then flushed, so every flush carried a single row. `database.update_jobs` is one `UPDATE ... WHERE job_id IN (...)` — one connection, one commit — regardless of how many IDs it carries, so the cost is per *write*, not per row. A 24,000-job run performed 24,000 connection-opens and WAL commits instead of a handful. Dispatch was unaffected (it buffers in bulk and flushed correctly).
- `schedule_helpers.drain_completions` now processes every future that has already finished before returning to dispatch, so one flush carries the whole batch. Batch size self-tunes: slower flushes let more completions accumulate, which shrinks the flush count. Falls back to the original bounded blocking wait when nothing is ready, so walltime is still re-checked at the poll cadence.
- Trade-off: freed capacity is not re-dispatched until the drain returns. Resources are released progressively inside `handle_completion`, so nothing is lost, only deferred. The drain aborts early if it crosses `shutdown_at`, keeping the graceful-shutdown grace period intact.

#### `pbx run` — no-idle exit can no longer skip a job on buffered parent state
- **Fan-in jobs could be silently dropped.** Dependency checks read parent status from the DB, not the status buffer. If the last running job was a parent, its completion sat in the buffer, its child looked un-ready, and the no-idle exit rule fired and ended the run — reporting success while leaving the child `Ready`. Previously this was prevented only incidentally, by flushing on every single pass.
- `schedule_helpers.check_run_complete` now flushes pending updates before concluding the run is over, and takes one more dispatch pass if it wrote anything. Terminates after at most one extra pass. Static-orphan reverting moved onto its terminal path.

### Changed

#### `pbx run --flush-interval` is inactive
- The timer-based periodic flush was redundant: the unconditional flush before each dispatch pass already bounds buffer residency to one iteration, and nothing between the two produces buffer entries. The block is commented out rather than deleted in case a conditional flush is introduced later. The flag remains accepted but has no effect, and is no longer documented.

#### `pbx ls` — row count is now a positional argument, `-n` means nodes
- **Breaking: `--all` and `-n <count>` removed from `pbx ls`.** Row count moves to an optional positional `COUNT`: `pbx ls 10` (first 10), `pbx ls -20` (last 20), `pbx ls all` (everything). Bare `pbx ls` is unchanged (first 10 + last 10 once more than 25 jobs match). `-n 0` as an alias for "all" is gone — use `all`.
- **`-n` / `--nnodes` on `pbx ls` now filters by node count**, matching its meaning on `add`, `update`, and `info`. This is a silent semantic change: `pbx ls -n 10` used to show the first 10 jobs and now shows jobs requiring 10 nodes.
- `pbx ls` sets `ignore_unknown_options` so `-20` parses as a count rather than a flag; unparseable counts (including mistyped flags) raise a clear `Invalid count` error.

#### Bulk `add` / `update` output no longer scales with job count
- **Job IDs print as compressed ranges** — `✅ Added 5000 job(s), IDs: 1-5000` instead of a 29,000-character comma list. Non-contiguous sets collapse to `1-3 8 14-16`, which pastes straight back into `pbx update`/`rm`/`info`. Applies to `pbx add`, `pbx update`, the `--nocc` warning, and the MCP `add_jobs` response.
- **Failures group by error message** instead of repeating it per job:
  - `pbx update` — `[4990 job(s)] 1-4990: <message>`, one line per distinct error
  - `pbx add` — `[5000 job(s)] <message>:` followed by the affected paths, sorted
- **`update --args` emits one summary line per resulting input file** rather than one per job, and writes each group in a single DB call instead of one call per job.
- **Update failure messages no longer embed the job ID** (it is redundant with the ID list, and prevented grouping). `pbx info` output is unchanged.
- Measured at 5,000 jobs: `add` success 29,100 → 211 bytes, `update` success 28,931 → 46 bytes, duplicate-`add` failures 524,024 → 94,120 bytes.

#### Generated batch scripts pin the resolved DB and config paths
- `submit.sh` now always contains `export PBX_DB_PATH` and `export PBX_CONFIG_PATH`, resolved by `submit_job`. Previously each line was written only when the corresponding environment variable happened to be set in the submitting shell, and carried the raw variable value rather than the resolved path. `pbx run` inside the allocation now opens exactly the database that was validated at submit time instead of re-resolving against whatever environment the batch job inherits.
- No change for the common case: with `PBX_DB_PATH` set, the value is the same; with it unset, the script now states the default (`~/.parslbox/job_database_pbx.db`) that `pbx run` would have resolved on its own. Sites where `$HOME` differs between the submit host and the compute nodes will now get the submit host's path.

### Fixed

- **`ParslBox(db_path=...)` / `ParslBox(config_path=...)` were ignored by `qsub` and `sbatch`.** Every other API method routes through `self.db_path`, but `submit_job` had no `db_path` parameter and read the module-global `path_utils.DB_FILE` directly (`submit_helpers.py:175,178`), so tag-glob expansion, the zero-runnable-jobs guard, and the script's `PBX_DB_PATH` export all used the environment-derived default. `db_path` is now threaded through `submit_to_scheduler` / `submit_to_slurm` into `submit_job`, which falls back to `path_utils.DB_FILE` when unset. Unreachable via CLI or MCP — both construct against the same default — but it blocked any caller wanting per-instance DB selection without mutating the process environment, which `path_utils.py:39` binds at import.
- **`pbx update --args` reported "No jobs were updated" after successfully updating.** `app_args` was missing from the `non_dependency_updates` check in `commands/update.py`, so an args-only update wrote to the DB but returned an empty `updated_job_ids`. The IDs are now counted, and `update_jobs` returns them sorted rather than in `set()` order.
- **`skills/parslbox-cli/SKILL.md` documented a job-ID syntax that does not work.** Positional ID ranges must be separate shell words (`pbx update 1-5 8 14-20`); quoting a multi-token list (`"1-5 8 14-20"`) fails to parse. `add --parents` is the opposite and does require quotes.

### Added

- **`format_job_ids` / `group_failures`** in `commands/helpers/job_id_parser.py` — range compression (inverse of `parse_job_ids`) and failure grouping, shared by CLI and MCP.
- **`tests/test_job_id_format.py`** — range compression, round-trip through `parse_job_ids`, failure grouping, and the resulting `add`/`update` output.
- **`app_args` exposed on the API and MCP.** It was CLI-only: `pbx add --args` / `pbx update --args` reached the shared core, but `ParslBox.add_jobs`, `ParslBox.update_jobs`, `AddJobSchema` and `UpdateJobSchema` had no such parameter, so no API or MCP caller could set app arguments. Now available on all four.
- **Layer-alignment guard tests** in `tests/test_mcp_server.py` — assert every MCP `AddJobSchema`/`UpdateJobSchema` field exists as an API parameter, so a schema field the API cannot accept fails the suite.
- **`num_nodes` filter across every layer** — `database.get_jobs`, `ParslBox.list_jobs`, `ParslBox.filter_jobs`, the `list_jobs` / `filter_jobs` MCP schemas, and `pbx filter -n/--nnodes`. Exact match.
- **`tests/test_ls_command.py`** — first test coverage for `pbx ls` (count parsing, pagination, node filter, and the shared-layer/API/MCP wiring).

---

## [1.0.1] - 2026-07-24

### Added

#### Restart Mode (`--restart`) — Walltime-Driven Resubmission Chain
- **`--restart` flag on `pbx run`, `pbx qsub`, `pbx sbatch`** — Enables a chained-submission model where the active batch job, upon nearing walltime, submits its own successor that picks up the remaining backlog of `Ready`/`Restart` jobs
  - Startup hook records job context (scheduler job ID, generated script path, walltime) so the orchestrator knows what to resubmit
  - Walltime-aware resubmission: the current job exits cleanly and queues the next link in the chain before the scheduler kills it
  - Works across PBS and SLURM via the unified `submit_helpers.submit_job()` path
  - Plumbed through CLI, `ParslBox` API, MCP schemas/tools, and bash script generation
- **`restart_template` path included in submit responses (MCP + API)** — Callers can inspect/modify the generated restart script before the next chain link fires
- **New helpers:** `parslbox/commands/helpers/restart_helpers.py`, `parslbox/commands/helpers/resource_estimate.py`
- **Docs:** new `docs/pbx-run-details.md` covers the restart chain and runtime behavior in depth

#### Tag System Enhancements
- **Partial tag matching** — Filters and selectors now support partial/substring tag matches via new `parslbox/utils/tag_match.py` helper
- **Tag exclusion** — `pbx filter` and related selectors accept exclude-tag syntax (negated tags) to skip jobs matching given tags; `exclude_tags` parameter added to the API and MCP filter schema
- **Submission summary print** — `pbx qsub`/`pbx sbatch` now print a summary of matched jobs at submission time
- **`pbx info` updated** — improved output and selector handling consistent with the new tag matching rules

#### Job Cancellation Commands
- **`pbx qdel`** — Cancels a PBS-submitted PBX job (wraps `qdel`)
- **`pbx scancel`** — Cancels a SLURM-submitted PBX job (wraps `scancel`)
- **Shared helper:** `parslbox/commands/helpers/cancel_helpers.py`
- Both commands exposed via the `ParslBox` API and MCP server

#### `PBX_RUN_DELAY` Environment Variable
- **Configurable inter-job submission delay** (default `0.2s`) — Throttles the rate at which `pbx run` hands jobs to Parsl, smoothing scheduler/launcher load on bursty submissions

#### Revamped SIGTERM Handling in `pbx run`
- **Restructured shutdown logic** in `parslbox/commands/helpers/run_cmd_helpers.py` — Cleaner separation between walltime-kill, user-cancel, and graceful-exit paths; better integration with `JobTracker` to mark active jobs correctly on termination
- **Elapsed-time logging** — `pbx run` now logs total elapsed time on exit

#### System Configs
- **`aurora-tile` now uses `MpiExecLauncher`** — places one Parsl manager per compute node (workers distributed across the allocation) instead of the previous `SimpleLauncher`, so it scales past the head-node RAM ceiling (~10k workers)
- **`perlmutter-gpu-srun`** (`parslbox/system_configs/perlmutter_gpu_srun.py`) — Perlmutter GPU variant using the `SrunLauncher` for the same per-node placement; registered in `parslbox/system_configs/loader.py`

#### Add Command — Range Syntax for Parents
- **`pbx add --parent` accepts ranges** — e.g., `--parent 1-5,8,12-14`, matching the range syntax already supported by `pbx update`/`rm`/`info`

#### Julia Test Scripts
- **`job_test/hello_affinity_julia.jl`** + `hello_affinity_julia_env.sh` — Julia affinity test for verifying rank/GPU placement
- **`job_test/create_test_jobs.py`** extended to generate Julia jobs alongside Python jobs

#### Documentation Restructure
- **`README.md` slimmed down**, content split into focused docs:
  - `docs/api-details.md` — `ParslBox` API reference
  - `docs/apps.md` — App reference (LAMMPS, VASP, ORCA, Python, Julia)
  - `docs/commands.md` — CLI command reference
  - `docs/pbx-run-details.md` — `pbx run` runtime/restart-chain reference

#### Hooks on Compute (`RUN_HOOKS_ON_COMPUTE`)
- **New opt-in class attribute on `AppBase`** — When set to `True`, an app's `preprocess()` and `postprocess()` Python methods are dispatched on the assigned compute node via a subprocess wrapped with `build_resource_launcher()`, instead of running in-process in the `pbx run` orchestrator on the head node
  - Default `False` preserves prior behavior — no existing app is affected
  - Does **not** affect `restart()` (which runs before any assignment exists)
  - Frees the head node from heavy hook work; lets hooks use the same modules/Python env/GPU access as the job itself
- **New compute-side dispatcher** — `parslbox/apps/_hook_runner.py` invoked as `python -m parslbox.apps._hook_runner <app_name> <method_name> <args_json>`; re-instantiates the app via `app_registry.get_app_instance()`, calls the method, writes its return value to `<job_path>/PBX_HOOK_RETURN` (mirrors the existing `PBX_JOB_STATUS_REPORT` idiom from v0.8.7)
- **New head-side helper** — `parslbox/commands/helpers/hook_dispatch.py` builds the `bash -c` payload with `shlex.quote()`, sources `env_file` if present, captures stdout/stderr, propagates `CalledProcessError` to the orchestrator's existing exception handlers
- **Resource launcher built for MPI apps too** when the flag is True — previously only built for `USES_MPI=False`. The single-rank launcher constrains the hook subprocess to assigned resources

### Fixed

#### Multi-GPU-per-Rank GPU Wrapper
- **GPU wrapper now correctly handles ranks holding multiple GPUs** — `mpi_launcher_helpers.py` and the resource-manager models reworked so wrappers emit the right `CUDA_VISIBLE_DEVICES`/binding mask when `gpus_per_rank > 1`
- New test suite: `tests/test_multi_gpu_per_rank.py`

#### Multinode CPU Jobs on GPU Systems
- **`job_info_validator` now allows CPU-only multinode jobs on GPU systems** — Previously rejected; CPU jobs on GPU partitions are valid use cases (e.g., pre/post-processing on GPU-node CPUs)
- **Sub-node CPU job `node_occupancy < 1` validation fix** — corrects edge cases where small CPU jobs were misclassified

#### srun Subnode `--cpus-per-task` Fix
- Corrected `--cpus-per-task` emission for srun sub-node CPU jobs in `mpi_command_builder.py`; Pinnacles CENVALARC config updated accordingly

#### `sched_opts` Injection
- **Fixed `sched_opts` directive injection** in `submit_helpers.py` / `sched_opts_helpers.py` — corrects an issue where merged directives were not placed at the `{sched_opts}` template slot under certain code paths

### Changed

#### System Config Tuning
- **`perlmutter-cpu`**: `CORES_PER_NODE` and worker counts corrected
- **`perlmutter-gpu`**: max-worker accounting corrected
- **`pinnacles-cenvalarc`**: max-worker accounting corrected

#### Parsl Dependency
- **Parsl bumped to the 2027 release line** (`pyproject.toml`, `poetry.lock`)

#### Scaling Orchestrators
- **CPU-only mode handling corrected** in `lammps_strong_scale_orchestrator.py` and `lammps_weak_scale_orchestrator.py`

### New Files
- `parslbox/commands/qdel.py`
- `parslbox/commands/scancel.py`
- `parslbox/commands/helpers/cancel_helpers.py`
- `parslbox/commands/helpers/restart_helpers.py`
- `parslbox/commands/helpers/resource_estimate.py`
- `parslbox/commands/helpers/hook_dispatch.py`
- `parslbox/apps/_hook_runner.py`
- `parslbox/utils/tag_match.py`
- `parslbox/system_configs/perlmutter_gpu_srun.py`
- `docs/api-details.md`
- `docs/apps.md`
- `docs/commands.md`
- `docs/pbx-run-details.md`
- `job_test/hello_affinity_julia.jl`
- `job_test/hello_affinity_julia_env.sh`
- `tests/test_restart_mode_orchestrator.py`
- `tests/test_restart_flag_plumbing.py`
- `tests/test_qdel_scancel.py`
- `tests/test_shutdown.py`
- `tests/test_tag_match.py`
- `tests/test_filter_command.py`
- `tests/test_info_command.py`
- `tests/test_multi_gpu_per_rank.py`
- `tests/api_tests/test_submission.py`

### Modified Files
- `parslbox/commands/run.py` — restart-mode orchestrator, elapsed-time log, `PBX_RUN_DELAY`, revamped SIGTERM
- `parslbox/commands/qsub.py`, `parslbox/commands/sbatch.py` — `--restart` flag, submission-summary print
- `parslbox/commands/add.py` — `--parent` accepts ranges
- `parslbox/commands/filter.py` — exclude-tag support
- `parslbox/commands/info.py` — selector/tag-matching updates
- `parslbox/commands/helpers/submit_helpers.py` — restart plumbing, `sched_opts` injection fix, `PBX_RUN_DELAY`
- `parslbox/commands/helpers/run_cmd_helpers.py` — restructured shutdown, restart hooks
- `parslbox/commands/helpers/sched_opts_helpers.py` — injection fix
- `parslbox/commands/helpers/filter_helpers.py` — exclude-tag plumbing
- `parslbox/commands/helpers/job_info_validator.py` — multinode CPU on GPU systems, sub-node occupancy fix
- `parslbox/commands/helpers/mpi_launcher_helpers.py` — multi-GPU-per-rank wrapper rework
- `parslbox/resource_manager/mpi_command_builder.py` — srun sub-node `--cpus-per-task` fix
- `parslbox/resource_manager/models.py` — multi-GPU-per-rank fields
- `parslbox/apps/appbase.py` — restart-aware script generation, GPU wrapper fix
- `parslbox/api.py` — `qdel`/`scancel` methods, `--restart` plumbing, exclude-tag filter
- `parslbox/mcp/mcp_server.py`, `parslbox/mcp/schemas.py` — restart, qdel/scancel, exclude-tag, restart_template in responses
- `parslbox/database/database.py` — supporting tag-match query changes
- `parslbox/utils/pbx_config_template.py` — `PBX_RUN_DELAY` and related entries
- `parslbox/system_configs/perlmutter_cpu.py`, `perlmutter_gpu.py`, `pinnacles_cenvalarc.py` — worker-count and CPU-per-node corrections
- `parslbox/system_configs/aurora_tile.py` — switched to `MpiExecLauncher` (per-node manager placement)
- `parslbox/system_configs/loader.py` — registers `perlmutter-gpu-srun`
- `pyproject.toml`, `poetry.lock` — Parsl 2027
- `README.md` — slimmed, links to new `docs/` pages
- `examples/strong_scaling/lammps_strong_scale_orchestrator.py`, `examples/weak_scaling/lammps_weak_scale_orchestrator.py` — CPU-only correction
- `job_test/create_test_jobs.py` — Julia job generation
- `tests/test_add_command.py` — tests for `--parent` range, sub-node occupancy, multinode CPU on GPU systems
- `tests/api_tests/test_job_queries.py` — tests for `exclude_tags` filter
- `tests/test_hooks_on_compute.py` — tests for `RUN_HOOKS_ON_COMPUTE` dispatch path
- `parslbox/apps/appbase.py` — `RUN_HOOKS_ON_COMPUTE` attribute
- `parslbox/apps/EXAMPLE_NEW_APP.py` — documents `RUN_HOOKS_ON_COMPUTE`
- `docs/apps.md` — documents `RUN_HOOKS_ON_COMPUTE` in optional overrides table

---

## [0.9.3] - 2026-05-19

### Added

#### Configurable `ranks_per_node` for GPU Jobs
- **Removed hardcoded 1-rank-per-GPU limitation.** `ranks_per_node` is now configurable via `--ranks-per-node` at job submission time
- **Per-app defaults via `get_default_ranks_per_node()` classmethod:**
  - LAMMPS/VASP/ORCA: 1 rank per GPU (preserves existing behavior)
  - Python/Julia: 1 rank per node
- **Validation:** `ranks_per_node` must not exceed `GPUS_PER_NODE` (GPU jobs) or `CORES_PER_NODE` (CPU jobs)
- **Multi-GPU per rank support:** `--gpu-bind=mask_gpu` used when ranks < GPUs (e.g., 2 ranks with 2 GPUs each)

#### Python and Julia Now Use MPI Launch
- **Python and Julia apps switched to `USES_MPI=True`** — launched via `srun`/`mpirun` with proper rank placement. Works with or without `mpi4py`/`MPI.jl`. Default 1 rank per node ensures scripts are constrained to assigned nodes.

### Changed

#### srun Backend Rework (Verified on Perlmutter)
- **Sub-node GPU isolation:** `--gpu-bind=map_gpu:{pbx_gpu_ids}` with explicit GPU IDs from resource manager + `--cpu-bind=mask_cpu:{hex}` for CPU isolation + `--overlap` for concurrent steps. Replaces `--gpus-per-task`/`--exact` which don't partition on Perlmutter.
- **`-N {nodes}` added** to all srun jobs to prevent SLURM from spreading tasks to unassigned nodes
- **`--mem-per-gpu` removed** — does not work on Perlmutter's SLURM configuration

---

## [0.9.2] - 2026-05-15

### Added

#### NERSC Perlmutter System Configs
- **`perlmutter-gpu`**: 1x AMD EPYC 7763 (128 logical cores), 4x NVIDIA A100 GPUs, 256 GB DRAM, `#SBATCH -C gpu` default constraint (overridable to `gpu&hbm80g` for 80 GB HBM nodes)
- **`perlmutter-cpu`**: 2x AMD EPYC 7763 (256 logical cores), 512 GB DRAM, `#SBATCH -C cpu` default constraint
- **`DRAM_PER_NODE`** attribute added to `SystemConfig` base class for memory-aware job scheduling

#### ORCA Application
- Added ORCA quantum chemistry application support

#### Native SLURM srun Backend
- **Replaced wrapper scripts with native SLURM GPU binding** for srun
- **Removed `--mem=0`**
- **`CUDA_VISIBLE_DEVICES` injection muted** for srun backend
- **New `cpu_bind_method` options**: `cores` and `threads` for srun

---

## [0.9.1] - 2026-04-22

### Added

#### srun GPU Resource Partitioning (`--gres=gpu:N --gpu-bind=none`)
- **Sub-node GPU jobs now declare their GPU share** via `--gres=gpu:N` in srun commands. Without this, the first srun step implicitly claims all job GPUs and concurrent sub-node steps fail with `srun: error: Invalid generic resource (gres) specification`
- **`--gpu-bind=none`** prevents SLURM from overriding `CUDA_VISIBLE_DEVICES`, which would clash with PBX's own GPU assignment via wrappers and env vars
- Only added for sub-node GPU jobs; full-node and multi-node GPU jobs use all node GPUs by default
- **Superseded in v0.9.2** by native SLURM GPU flags (`--gpus-per-task`, `--gpus-per-node`, `--gpu-bind=map_gpu`)

#### `--exact` Flag Restored for srun Sub-node Jobs
- **`--exact` was present in the deprecated `mpi_launcher_depr.py`** (added in v0.8.1) but was not carried over when `mpi_command_builder.py` replaced it. Now restored via `_should_add_srun_exact()` helper that checks job type (`subnode_cpu` or `subnode_gpu`)
- Prevents srun steps from accessing more resources than allocated, enabling correct sub-node isolation on SLURM

### Fixed

#### `build_resource_launcher` Node Occupancy
- **`node_occupancy` now recalculated from actual assigned cores** instead of passing through the original job request. Accounts for excluded cores and rounding, ensuring `_should_add_srun_exact()` correctly identifies sub-node jobs

### Modified Files
- `parslbox/resource_manager/mpi_command_builder.py` — `--gres`/`--gpu-bind=none` for sub-node GPU, `--exact` via `_should_add_srun_exact()`, cores_per_rank from assignment, resource launcher occupancy fix
- `tests/test_comprehensive_resource_manager.py` — Tests for gres flags, `--exact`, depth binding with sub-node GPU and excluded cores
- `tests/test_mpi_command_builder.py` — Unit tests for gres flags, `GPUS_PER_NODE` in mock config

---

## [0.9.0] - 2026-04-10

### Added

#### Dynamic Job Discovery (`--dynamic/--static`)
- **`--dynamic` flag on `pbx run`, `pbx qsub`, `pbx sbatch`** — Enables periodic polling for newly added jobs during an active run session. Jobs added via `pbx add` (from another terminal or from within a running job script) are automatically discovered and executed without restarting the run
  - `--dynamic` (default): polls the database every 60 seconds for new `Ready`/`Restart` jobs matching the same `--apps`/`--tags` filters
  - `--static`: collect-once-and-exit behavior (previous default)
  - When the last active future completes, an immediate discovery check runs before exiting — prevents premature exit when a completing job spawned child jobs
  - Backlog-aware exit: if backlogged jobs exist (waiting on dependencies), the loop stays alive even with no active futures
  - Failed jobs reset to Ready by the user (via `pbx update --status Ready` from another terminal) are re-discovered and re-run
  - New app types discovered dynamically are loaded on-demand (app instance, config, MPI config)

#### MPI Environment Setup (`env_setup` under `mpi:`)
- **`env_setup` key in MPI configuration** — Shell commands (e.g., `module load openmpi`) that run before the app's `environment_setup` in the generated bash script. Useful for non-MPI apps (Python, Julia) that need `mpirun`/`mpiexec` loaded for the resource launcher
  - Participates in the existing merge hierarchy: system-level → app-level override
  - Added to `MPIConfig` dataclass, parsed in `mpi_config_from_dict()`
  - Passed through `mpi_commands` dict as `PBX_MPI_ENV_SETUP`
  - Generated configs include the key (empty, with guiding comments) via `pbx config`

#### Pinnacles CENVALARC System Config (First SLURM System)
- **New `pinnacles-cenvalarc` system config** — UC Merced Pinnacles cluster (CENVALARC partition). ParslBox's first SLURM-based system configuration
  - Single config handles both CPU and GPU partitions: `cenvalarc.compute`, `cenvalarc.bigmem`, `cenvalarc.gpu`
  - All nodes: 64 cores (2× Intel 32-Core Xeon Gold 6530). GPU nodes: 2× NVIDIA (L40S or H200 NVL)
  - Runtime GPU detection via `nvidia-smi -L` — returns 0 on CPU partitions (CPU-only mode), 2 on GPU partition
  - SLURM scheduler with srun MPI backend, depth CPU binding, GPU wrapper support

#### Dynamic Job Test Scripts
- **`job_test/test_dynamic_jobs/`** — Test suite for `--dynamic` feature
  - `spawner.py`: Python job that prints affinity info and adds 5 `hello_affinity` child jobs at runtime
  - `test_dynamic_orchestrator.py`: Orchestrator that creates spawner jobs with configurable resource allocation
  - `spawner_env.sh`: Shared environment file

### Changed

#### Bash Script Variable Naming
- **Renamed variables in `_general_bash_app_engine()`** for clarity about their origin:
  - `env_setup` → split into `env_setup_frm_appconfig` and `env_setup_frm_envfile`
  - `env_exports` → `env_exports_frm_rsrc_mgr`
  - `additional_setup` → `setup_frm_app`

#### Config Generator Improvements
- **`environment_setup` defaults** — Generated configs now include uncommented `module purge` and `module restore` with a comment explaining why (clean module environment for worker subprocesses)

#### Scaling Scripts
- **Improved plot formatting and docstrings** in strong and weak scaling orchestrator and analysis scripts

### New Files
- `parslbox/system_configs/pinnacles_cenvalarc.py`
- `job_test/test_dynamic_jobs/spawner.py`
- `job_test/test_dynamic_jobs/test_dynamic_orchestrator.py`
- `job_test/test_dynamic_jobs/spawner_env.sh`
- `job_test/test_dynamic_jobs/hello_affinity.py`

### Modified Files
- `parslbox/commands/run.py` — `--dynamic/--static` flag, `discover_new_jobs()` function, modified main loop
- `parslbox/commands/qsub.py` — `--dynamic/--static` flag forwarded to `submit_job()`
- `parslbox/commands/sbatch.py` — `--dynamic/--static` flag forwarded to `submit_job()`
- `parslbox/commands/helpers/submit_helpers.py` — `dynamic` parameter, `--static` in run options
- `parslbox/resource_manager/job_tracker.py` — `register_jobs()` method for mid-run job registration
- `parslbox/resource_manager/mpi_config.py` — `env_setup` field in `MPIConfig`
- `parslbox/apps/appbase.py` — Renamed bash script variables, added `mpi_env_setup` injection
- `parslbox/utils/pbx_config_template.py` — `env_setup` in MPI config docs
- `parslbox/utils/config_generator.py` — `env_setup` key, `module purge/restore` defaults
- `examples/strong_scaling/lammps_strong_scale_orchestrator.py` — Updated docstring
- `examples/weak_scaling/lammps_weak_scale_orchestrator.py` — Updated docstring
- `examples/strong_scaling/plot_strong-scale-results.py` — Tick sizes, axis params, legends
- `examples/weak_scaling/plot_weak-scale-results.py` — Tick sizes, axis params, `_calculate_y_axis_params_performance`
- `parslbox/system_configs/loader.py` — Registered `pinnacles-cenvalarc`
- `tests/test_mpi_config_merge.py` — 10 new tests for `env_setup` parsing and merge behavior

---

## [0.8.8] - 2026-04-05

### Added

#### MCP Server Enhancements
- **3 new MCP tools** — `list_jobs`, `get_job`, `get_jobs` for retrieving full job details (all fields) via MCP. Previously, MCP clients could only get job IDs via `filter_jobs` but had no way to see actual job data
- **`GetJobSchema` and `GetJobsByIdsSchema`** added to MCP schemas
- **`ListJobsSchema` updated** — Now matches the `ParslBox.list_jobs()` API with `path` and `in_file` filters. Removed CLI-only `all_jobs` and `n` fields that didn't map to the API
- **stdio transport support** — MCP server now supports `--stdio` flag for Claude Code integration (`python -m parslbox.mcp.mcp_server --stdio`). HTTP mode remains the default when run without the flag
- **`.mcp.json` project config** — Added `.mcp.json` to project root with both stdio and HTTP server configurations for Claude Code auto-discovery

### Fixed

#### MCP Schema Corrections
- **Config names fixed** — `aurora_gpu`/`aurora_tile` → `aurora-gpu`/`aurora-tile` in `AddJobSchema` and `QSubSchema` descriptions, matching actual `CONFIG_FACTORIES` keys. Previous values would cause `ValueError` at runtime
- **`select` type fixed** — Changed from `int` to `str` in both `QSubSchema` and `SBatchSchema` to match the API, which accepts complex PBS select specs (e.g., `'2:ncpus=32:ngpus=4'`)
- **`tag` default fixed** — Changed `AddJobSchema.tag` default from `"test"` to `None`, matching the `ParslBox.add_jobs()` API default
- **`AddJobSchema` constraints aligned** — Added `ge=0` to `ngpus`, `ge=1` to `nnodes` and `ranks_per_node`, changed `node_occupancy` from `ge=0.0` to `gt=0.0`. Now consistent with `UpdateJobSchema` and core validators
- **`RemoveJobsSchema` type annotation** — Changed `list[int]` to `List[int]` for consistency with all other schemas
- **Valid status values documented** — Both `AddJobSchema` and `UpdateJobSchema` now list valid statuses (Ready, Done, Failed, Restart, Running, Submitted, Warning) in field descriptions
- **App names updated** — Schema descriptions now list `lammps-kk`, `vasp`, `python`, `julia` (was `lammps`, `vasp`, `python`)

### Modified Files
- `parslbox/mcp/schemas.py` — All fixes above, plus new `GetJobSchema` and `GetJobsByIdsSchema`
- `parslbox/mcp/mcp_server.py` — Added `list_jobs`, `get_job`, `get_jobs` tools; stdio transport support; updated imports and instructions
- `.mcp.json` — New file for Claude Code MCP auto-discovery

---

## [0.8.7] - 2026-03-27

### Added

#### Julia App
- **New `JuliaApp`** (`parslbox/apps/julia.py`) — Executes user-provided Julia scripts via PBX. Set to `USES_MPI = False` with resource launcher constraining, matching the Python app pattern. Julia project environments can be activated via `env_file` or `--project` flag in `executable_path`

#### Range Syntax for Job IDs
- **`pbx update`, `pbx rm`, `pbx info` now support ranges** — e.g., `pbx update 1-5 8 14-20 --status Restart`. New `parse_job_ids()` utility in `parslbox/commands/helpers/job_id_parser.py` handles expansion, deduplication, and validation

#### Script-Driven Status Reporting for Python and Julia Apps
- **New `report_status()` utility** (`parslbox/apps/utils.py`) — Allows user scripts to report job success or failure from within the script itself. Creates a `PBX_JOB_STATUS_REPORT` file in the job directory containing the status
  - Usage: `from parslbox.apps.utils import report_status; report_status("done")`
  - Validates against supported statuses: `done`, `failed` (case-insensitive)
- **`PythonApp.check_success` reads status report file** — After execution, reads `PBX_JOB_STATUS_REPORT` from the job directory, returns the reported status, and deletes the file. If the file is not found, the job is marked as Failed
- **`JuliaApp.check_success`** — Same status file checking logic. Julia scripts write the file directly: `open("PBX_JOB_STATUS_REPORT", "w") do f; write(f, "Done"); end`
- **Scaling example scripts updated** — `plot_strong-scale-results.py` and `plot_weak-scale-results.py` now call `report_status("done")` after verifying `.png` output files exist, or `report_status("failed")` on error

### Fixed

#### Missing Python/Julia Executable in Command
- **Fixed empty executable bug** in `PythonApp` and `JuliaApp` — When `executable_path` is not set in config, the base class passes `executable=''` (empty string) to `get_command_template()`. The previous `kwargs.get('executable', 'python')` default only applied when the key was absent, not when it was empty. Changed to `kwargs.get('executable') or 'python'` (and `'julia'`) so the default correctly activates for empty strings

### New Files
- `parslbox/apps/julia.py`
- `parslbox/apps/utils.py`
- `parslbox/commands/helpers/job_id_parser.py`

### Modified Files
- `parslbox/apps/python.py` — Fixed executable default, added status file reading in `check_success`, commented out `postprocess` body
- `parslbox/commands/update.py` — Job IDs argument accepts ranges
- `parslbox/commands/rm.py` — Job IDs argument accepts ranges
- `parslbox/commands/info.py` — Job IDs argument accepts ranges
- `examples/strong_scaling/plot_strong-scale-results.py` — Added `report_status` import and calls
- `examples/weak_scaling/plot_weak-scale-results.py` — Added `report_status` import and calls

---

## [0.8.5] - 2026-03-17

### Added

#### Resource Constraining for Non-MPI Apps
- **New `USES_MPI` class attribute on `AppBase`** (default `True`) — Apps that don't use MPI for parallelization set `USES_MPI = False` to get a resource launcher prepended to their command, ensuring execution on the assigned node with assigned CPU/GPU resources instead of the head node
- **New `build_resource_launcher()` function** in `mpi_command_builder.py` — Generates a single-process MPI launcher (`mpiexec -n 1 --ppn 1 ...`) by reusing `MPICommandBuilder` with a synthetic single-rank spec. Consolidates all assigned CPU cores into a single binding. GPU binding via `CUDA_VISIBLE_DEVICES` env var. All user MPI config settings respected
- **`PythonApp` set to `USES_MPI = False`** — Python scripts now run on assigned resources instead of the head node
- **`NonMPIApp` example** added to `EXAMPLE_NEW_APP.py` with documentation
- **13 new tests** in `tests/test_resource_launcher.py` covering all backends, CPU consolidation, hostlist, GPU wrapper exclusion, and `USES_MPI` attribute verification

#### Resource Requirement Estimator (`--req` flag)
- **New `--req / -r` flag on `pbx info`** — Calculates resource requirements for selected jobs against a target system (e.g., `pbx info 1 2 3 --req polaris`)
  - Shows simultaneous execution nodes (all jobs at once) vs optimal packing nodes (no idle resources)
  - Handles GPU jobs, CPU jobs, mixed workloads, multi-node jobs, and sub-node sharing
  - Warns about incompatible jobs (GPU jobs on CPU-only systems, oversized single-node jobs)
  - Filters to schedulable jobs (Ready/Restart), warns about skipped non-schedulable jobs
  - Validates system name against available configurations

### Fixed

#### SIGTERM Handler Now Marks Active Jobs as Killed
- **Jobs no longer stuck in "Running" state after walltime exceeded** — Previously, the SIGTERM handler only flushed the status buffer (which was typically empty). Now it identifies active jobs via `JobTracker` (in-memory, only this instance's jobs), flushes the buffer first to clear stale entries, then marks active jobs as "Killed" via direct database write
- **New "Killed" job status** — Added to `VALID_JOB_STATUSES` in both `run.py` and `run_cmd_helpers.py`. Distinguishes walltime-killed jobs from application failures ("Failed")
- **Safe with concurrent PBX instances** — Uses `JobTracker` which only tracks this instance's jobs, so other batch jobs sharing the same database are unaffected

### Modified Files
- `parslbox/apps/appbase.py` — Added `USES_MPI` attribute and resource launcher prepend logic
- `parslbox/apps/python.py` — Set `USES_MPI = False`
- `parslbox/apps/EXAMPLE_NEW_APP.py` — Added `USES_MPI` documentation and `NonMPIApp` example
- `parslbox/resource_manager/mpi_command_builder.py` — Added `build_resource_launcher()` function
- `parslbox/commands/run.py` — Added `build_resource_launcher` import, resource launcher generation for non-MPI apps, `job_tracker` passed to shutdown handler, added "killed" to valid statuses
- `parslbox/commands/info.py` — Added `--req / -r` flag with resource analysis functions
- `parslbox/commands/helpers/run_cmd_helpers.py` — Restructured signal handler (3-step: get active IDs → flush → mark Killed), added `job_tracker` parameter, added "killed" to valid statuses

### New Files
- `tests/test_resource_launcher.py` (13 tests)

---

## [0.8.2] - 2026-03-02

### Changed
- **Renamed `mpiexec` helper functions to `mpich`** — `generate_mpiexec_rankfile`, `generate_mpiexec_gpu_wrapper`, `mpiexec_cuda_gpu_wrapper`, and `mpiexec_intel_gpu_wrapper` are now `generate_mpich_*` / `mpich_*` to match the backend name and avoid confusion with the `mpiexec` command
- **Added dedicated srun wrapper functions** — `generate_srun_gpu_wrapper()` and `generate_srun_rankfile()` give srun its own entry points (delegating to MPICH internally) for future srun-specific customization

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
- **GPU wrapper reuse** — srun now uses dedicated `generate_srun_gpu_wrapper()` wrappers (delegating to MPICH internally); wrappers already include `SLURM_PROCID`/`SLURM_LOCALID` fallback chains for rank detection
- **`--cpus-per-task` in depth binding** — `_build_depth_binding()` srun branch in `mpi_command_builder.py` now emits `--cpus-per-task N --cpu-bind=cores` instead of bare `--cpu-bind=cores`

#### General-purpose `sched_opts` (Scheduler Options) Support
- **Three-layer override chain** for scheduler directives: template → config `sched_opts` → CLI `--sched-opts`
  - Template directives serve as the base layer
  - Per-system persistent overrides via `sched_opts` key in config YAML
  - Per-run overrides via repeatable `--sched-opts` CLI flag
  - Matching keys are replaced in-place; new directives are appended at the `{sched_opts}` placeholder

- **System-level default `sched_opts`** — Each system config can now define default scheduler directives via `get_default_sched_opts()`
  - Polaris: `#PBS -l filesystems=home:eagle`
  - Aurora GPU/Tile: `#PBS -l filesystems=home:flare`
  - Crux: `#PBS -l filesystems=home:eagle`
  - Sophia: `#PBS -l filesystems=home:eagle`
  - Base class returns `None` (no defaults); systems override as needed
  - Defaults are automatically included in generated configs via `pbx config`
  - Priority chain: system defaults → config `sched_opts` → CLI `--sched-opts`

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
