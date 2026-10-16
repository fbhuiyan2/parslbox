# The ParslBox config file

`config.yaml` holds per-system and per-app settings: scheduler templates, the
environment that makes `pbx` available inside a batch job, executable paths,
and MPI defaults. It does **not** hold pbx's own environment variables
(`PBX_CONFIG_PATH`, `PBX_DB_PATH`, `PBX_RUN_DELAY`) — those are shell
variables.

Create one with `pbx config` (interactive) or `ParslBox.create_config(...)`
(programmatic). Both write the same file.

---

## Where it lives, and where the database lives

| | Config | Database |
|---|---|---|
| Default | `~/.parslbox/config.yaml` | `~/.parslbox/job_database_pbx.db` |
| Override | `PBX_CONFIG_PATH` (dir or file) | `PBX_DB_PATH` (dir or file) |

**The database does not follow the config.** Creating a config in
`/scratch/myproject` still initializes the database at `PBX_DB_PATH`, or in
`~/.parslbox` when that is unset. If you want a project-local database, set
`PBX_DB_PATH` as well:

```bash
export PBX_CONFIG_PATH=/scratch/myproject
export PBX_DB_PATH=/scratch/myproject
```

Both must be absolute paths. A module-level constant captures them at import,
so changing them mid-process has no effect — pass `config_path=` / `db_path=`
to `ParslBox()` instead.

---

## What a freshly generated config leaves blank

`pbx config` writes a scaffold, not a finished config. Three things are
placeholders, and the first one is fatal.

### 1. `<system>.pbx_python_env_setup` — required

Shell commands that make the `pbx` command available on the compute node.
They are substituted into the generated `submit.sh` ahead of the final
`exec pbx run` (the exported `PBX_*` variables sit between the two).

Generated as comments only:

```yaml
polaris:
  pbx_python_env_setup: |
    # Add your module load and conda activate commands here
    # Example:
    # module load conda
    # conda activate your_env
```

**Leave it as-is and your batch job dies with `pbx: command not found`** — the
scheduler starts the job, the script reaches `exec pbx run`, and there is no
`pbx` on `PATH`. Fill it with whatever activates the environment you installed
parslbox into on that machine:

```yaml
polaris:
  pbx_python_env_setup: |
    module load conda
    conda activate parslbox
```

This is per system, and it describes the **target** machine. A config written
on a laptop for Polaris must carry Polaris's activation commands.

### 2. `<app>.<system>.executable_path` — required for compiled apps

```yaml
vasp:
  polaris:
    executable_path: "/path/to/your/vasp/executable/on/polaris"
```

Replace with the absolute path to the binary on that system. `python` and
`julia` have no `executable_path` — they use the interpreter from the
environment.

### 3. `<app>.<system>.environment_setup` — usually needed

Shell commands run on the compute node before the application launches.
Generated with `module purge` / `module restore` already active:

```yaml
vasp:
  polaris:
    environment_setup: |
      # Start with purge & restore so worker subprocesses
      # get a clean module environment (avoids subtle errors).
      module purge
      module restore
      # Add app-specific environment setup below
      # module load ...
```

**Keep `module purge` and `module restore` at the top** and add the app's
module loads below them. The purge/restore pair gives worker subprocesses a
clean module environment; dropping it causes subtle, hard-to-trace failures.

---

## Finding what is still blank

Every unfilled executable path contains the literal string `/path/to/your/`.
Anchor the pattern to a real (uncommented) key — the generated file also
carries a commented example-app template containing the same string:

```bash
grep -nE '^[[:space:]]*executable_path: "/path/to/your/' ~/.parslbox/config.yaml
```

An unfilled `pbx_python_env_setup` block contains only comment lines.

---

## Other keys

| Key | Scope | Purpose |
|---|---|---|
| `schedulers.pbs.template` / `schedulers.slurm.template` | global | The batch script skeleton. `{...}` fields are filled at submit time. |
| `<system>.sched_opts` | system | Extra scheduler directives (filesystems, placement, mail). Overridable per run with `--sched-opts`. |
| `<system>.mpi` | system | MPI backend defaults — `cpu_bind_method`, `use_gpu_wrapper`, `add`, `disable`. |
| `<app>.<system>.mpi` | app | Per-app override of the system MPI settings. |
| `custom_apps` | global | Register an app class of your own. See [apps.md](apps.md). |

Full MPI and `sched_opts` reference is written into the generated file as
comments.
