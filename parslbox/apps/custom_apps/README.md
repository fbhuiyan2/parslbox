# Custom Applications for ParslBox

This directory contains custom application implementations that can be loaded via the `custom_apps` section in `config.yaml`.

## Available Custom Apps

### lammps-kk-restart (LammpsKkRestart)

Extends `lammps-kk` with chain-restart support for use with `pbx qsub --respawn N` / `pbx sbatch --respawn N` (or any time a job is set to `Restart` status).

**Per job dir, the user provides:**
- `in.lammps` — link 0 input.
- `in.restart_template.lammps` — restart template containing:
  - `variable sim_total_steps equal <int>`
  - `variable link equal <<PBX_LINK>>`
  - `variable remaining_steps_2_run equal <<PBX_REMAINING_STEPS>>`
  - `variable restart_file string <<PBX_RESTART_FILE>>`
  - the usual `read_restart ${restart_file}`, dumps named like `dump_link-${link}.lammpsdump`, and `run ${remaining_steps_2_run}`.

**Behavior:**
- `restart()` (per link): renames `log.lammps` → `log_link-<N-1>.lammps`, picks the highest-step `restart.<step>` checkpoint, computes `remaining = sim_total_steps - latest_step` (clamped to 0), substitutes the three placeholders into the template, writes `in.restart_link-<N>.lammps`, and patches the job's `in_file`.
- `postprocess()` (final Done): appends `dump_link-*.lammpsdump` (numeric order) onto `dump.lammpsdump`. No files are deleted.
- `RUN_HOOKS_ON_COMPUTE = True` — dump merging runs on the assigned compute node.

**config.yaml:**
```yaml
custom_apps:
  lammps-kk-restart:
    module: "parslbox/apps/custom_apps/lammps_kk_restart.py"
    class: "LammpsKkRestart"

lammps-kk-restart:
  <system_name>:
    executable_path: "/path/to/lmp"
    environment_setup: |
      module load lammps/kokkos
```

### lammps-cpu (LammpsBasicApp)

A CPU-only LAMMPS application without Kokkos acceleration. Useful for:
- CPU-only systems where Kokkos is not installed
- Systems where standard LAMMPS (without Kokkos) is preferred
- Testing and development on systems without GPU support

**Features:**
- Simple MPI-based execution without Kokkos flags
- Same success checking as lammps-kk (looks for "Total wall time:" in log.lammps)
- Ignores ungraceful exits if simulation completed successfully
- Minimal overhead for CPU-only workloads

**Usage in config.yaml:**

```yaml
# Step 1: Register the custom app (tells ParslBox where to find the code)
custom_apps:
  lammps-cpu:
    module: "parslbox/apps/custom_apps/lammps_cpu.py"
    class: "LammpsBasicApp"

# Step 2: Configure the app for your system(s) (like built-in apps)
lammps-cpu:
  lcrc-improv:  # Your system name
    executable_path: "lmp"
    environment_setup: |
      module load lammps
      export OMP_NUM_THREADS=1
      export OMP_PROC_BIND=spread
      export OMP_PLACES=cores
  
  # You can configure for multiple systems
  polaris:
    executable_path: "/path/to/lmp"
    environment_setup: |
      module load lammps/kokkos
```

**Example: adding a job** (jobs are added via `pbx add`, not via YAML):

```bash
pbx add /path/to/job_dir --app lammps-cpu -i in.lammps -n 2 --ranks-per-node 64
```

## Creating Your Own Custom Apps

To create a new custom app:

1. **Create a Python file** in this directory (e.g., `my_app.py`)

2. **Inherit from AppBase** and implement required methods:
   ```python
   from parslbox.apps.appbase import AppBase
   
   class MyApp(AppBase):
       INPUT_REQUIRED = True  # or False
       DFLT_INPUT = "input.txt"  # default input filename

       def get_command_template(self, **kwargs) -> str:
           # Return the command to execute your app
           pass
   ```

3. **Add to config.yaml** in two sections — registration (`custom_apps:`) and per-system config (top-level, keyed by app name):
   ```yaml
   # Step 1: Register the custom app (only `module` and `class` are read here)
   custom_apps:
     my-app:
       module: "parslbox/apps/custom_apps/my_app.py"
       class: "MyApp"

   # Step 2: Per-system config at the top level, keyed by app name
   my-app:
     <system_name>:
       executable_path: "/path/to/executable"
       environment_setup: |
         # Setup commands
   ```

4. **Optional: Override additional methods** for custom behavior:
   - `get_additional_setup()` - Add environment variables, thread settings, etc.
   - `preprocess()` - Run before job execution
   - `check_success()` - Custom success checking logic
   - `postprocess()` - Run after job completion
   - `restart(job_dict)` - Handle Restart-status jobs. Called lazily per-job inside `create_parsl_future` after resources are assigned. May return a dict patching `in_file`, `env_file`, or `tag` (other non-resource keys are warned + dropped; returning resource fields like `ngpus`/`num_nodes` is a contract violation → job marked Failed). See `pbx-run-details.md` for the full three-scene contract.
   - `min_remaining_walltime(job_dict) -> int` - Minimum remaining batch walltime (seconds) required to dispatch a fresh job. Default 0 (no gate); restart-continuations exempt. See `lammps_kk_restart.py` for an example.

See `lammps_cpu.py` for a complete example implementation.

## Notes

- Custom apps are loaded at runtime from the config file
- Custom app names cannot conflict with built-in apps (lammps-kk, vasp, python, julia, orca)
- The module path can be absolute, relative to config directory, or use `~` for home directory
- All custom apps must properly implement the AppBase interface
