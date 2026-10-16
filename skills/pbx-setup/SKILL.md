---
name: pbx-setup
description: Set up ParslBox from scratch — install it into a Python environment if needed, then create and fully fill in a config.yaml for the chosen HPC systems and apps. Covers first-time setup, an additional project-local config, reconfiguring an existing one, and writing a config on a laptop for a remote HPC. Use when the user says they want to set up, install, or configure parslbox/pbx, when `pbx` is missing, or when a config is missing, incomplete, or still has placeholder paths.
---

# Setting up ParslBox

Reference: [docs/config-file.md](docs/config-file.md) — what each key means and
which blanks are fatal. [docs/commands.md](docs/commands.md) for `pbx` usage
afterwards.

A generated config is a **scaffold, not a working setup**. Writing the file is
the middle of this job, not the end. Do not tell the user setup is complete
while placeholders remain.

## Rules

- **Never guess which machine you are on.** State your evidence, ask for
  confirmation.
- **Never pass `force=True` to `create_config`.** It renames the user's
  existing config. If one exists, show it and let the user decide.
- **Never remove `module purge` / `module restore`** from an
  `environment_setup` block. Add below them.
- Ask before writing outside the directory the user named.

---

## Phase 0 — Establish what you are setting up

Three questions. Resolve all three before touching anything.

**0a. Which machine will run the jobs?** Gather evidence first:

```bash
hostname; echo "---"
command -v qstat sbatch squeue 2>/dev/null; echo "---"
env | grep -E '^(PBS|SLURM)_' | head; echo "---"
ls -d /lus /eagle /grand /pscratch /scratch 2>/dev/null
```

A scheduler binary on `PATH` plus a recognizable hostname means you are
probably on an HPC login node. Nothing matching means a local machine. Say
which you think it is and what made you think so, then ask. Nothing in
parslbox maps hostnames to system names, so this is always a guess.

**0b. If the jobs will run elsewhere**, ask which of these the user wants:

- **A config to copy over** — you write `config.yaml` here, they copy it to
  `~/.parslbox/` on the HPC. Everything machine-specific has to be *asked*,
  not detected.
- **`pbx local`** — author jobs on this machine, run them on the remote one
  over Globus. That is a different feature; see
  [docs/remote-workflow.md](docs/remote-workflow.md) and stop following this
  skill for the transfer parts.

**0c. Where does the config go?**

- `~/.parslbox/config.yaml` — the default, used when nothing is set.
- A project directory, e.g. `/scratch/myproject` — requires
  `export PBX_CONFIG_PATH=/scratch/myproject`.

If project-local, say this plainly: **setting `PBX_CONFIG_PATH` does not move
the database.** It is created at `PBX_DB_PATH`, or `~/.parslbox` if that is
unset. Ask whether they also want `PBX_DB_PATH` set to the project directory.

Both are ordinary shell variables — they vanish with the shell. Tell the user
to add them to their shell profile if the project is not a one-off, and give
them the exact lines (Phase 7).

Check what already exists before proposing anything:

```bash
echo "PBX_CONFIG_PATH=${PBX_CONFIG_PATH:-unset}  PBX_DB_PATH=${PBX_DB_PATH:-unset}"
ls -l ~/.parslbox/config.yaml ./config.yaml 2>/dev/null
```

---

## Phase 1 — Make sure pbx is usable here

```bash
command -v pbx || echo "pbx not on PATH"
cd /tmp && python -c "import parslbox; print(parslbox.__file__)" 2>/dev/null || echo "not importable"
```

There is no `pbx --version`; `command -v` is the check. Run the import from
`/tmp`, not from the repo — inside the repo directory `import parslbox`
succeeds from the source tree even when nothing is installed, so the same
check there would tell you everything is fine when it is not.

Already working → go to Phase 2.

Not working → ask whether parslbox is installed in an environment that is not
active, and if so how to activate it.

Once you have the activation command, **prefix every later command in this
skill with it**, in the same shell invocation — each bash call starts fresh,
so activating once does not carry over:

```bash
source /abs/path/.uvenv/bin/activate && python -c "..."
```

Say which prefix you are using so the user can reproduce it by hand.

No environment at all → install. You need the repo path (`git clone
https://github.com/fbhuiyan2/parslbox.git` if they do not have it). Check what
is available and prefer uv:

```bash
command -v uv conda python3 2>/dev/null
```

**Check for an active conda environment first.** If `$CONDA_PREFIX` is set
and you are about to create a uv or venv environment, Poetry will ignore the
environment you create and install into its own under
`~/.cache/pypoetry/virtualenvs/` — `pbx` then never lands on `PATH`, and the
failure looks nothing like its cause. Conda's `base` is often auto-activated
in a login shell, so check rather than assume:

```bash
echo "CONDA_PREFIX=${CONDA_PREFIX:-unset}"
```

If it is set and the user wants a uv or venv install, clear it **for the
install commands** by prefixing them:

```bash
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV sh -c '<install command>'
```

Do not use `conda deactivate` for this. It is a shell function that only
exists in a shell where `conda init` has run, so in the non-interactive shell
you get per bash call it fails with
`CondaError: Run 'conda init' before 'conda deactivate'` and leaves
`CONDA_PREFIX` set — you would carry on believing you had cleared it.

If the user wants a conda install instead, an active conda env is fine as is.

Chain the steps in a single shell invocation — a fresh bash call does not
inherit the previous one's activation.

**uv** (preferred):

```bash
cd <repo> && env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV uv venv -p 3.11.9 .uvenv
cd <repo> && env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV sh -c '. .uvenv/bin/activate && uv pip install poetry && poetry install'
```

(Drop the `env -u ...` prefix if `CONDA_PREFIX` was already unset.)

**conda**:

```bash
conda create -y -n parslbox python=3.11.9
cd <repo> && conda run -n parslbox pip install poetry
cd <repo> && conda run -n parslbox poetry install
```

Extras, added to `poetry install` as `--extras "..."` — `agentic` (MCP
server), `remote` (Globus), `simulation` (ase and pymatgen, for the user's
own scripts; no built-in app imports them), or `--all-extras`. Core is right
if the user is unsure; extras can be added later.

Then confirm it worked, from outside the repo. If `pbx` is missing here, the
install went into Poetry's own virtualenv — re-run it with the `env -u` prefix
above rather than hunting under `~/.cache/pypoetry/`:

```bash
cd /tmp && <activate> && command -v pbx && python -c "import parslbox"
```

Record the exact activation command — Phase 4 writes it into the config so
the compute node can find `pbx`.

**If the target is a different machine**, this phase installs parslbox *here*
only so you can generate the config. Generating a config requires parslbox to
be importable, so either do that, or ssh to the HPC and run this skill there
instead. Say which you are doing.

---

## Phase 2 — Choose systems and apps

```bash
python -c "
from parslbox.api import ParslBox
for s in ParslBox.available_systems():
    gpu = f\"{s['gpus_per_node']}x {s['gpu_type']}\" if s['gpus_per_node'] else 'no gpu'
    print(f\"{s['name']:20} {s['scheduler']:6} {gpu:12} {s['cores_per_node']} cores\")
print('apps:', ', '.join(ParslBox.builtin_apps()))
"
```

Show the table, not just the names — a user who does not know these machines
cannot choose from names alone. Ask which systems they have accounts on, and
which apps they will run.

---

## Phase 3 — Write the config

If a config already exists at the target path, show it and ask: keep it, write
somewhere else, or replace it. Only if the user explicitly chooses replace,
pass `force=True` — it renames the old file to
`config.yaml.backup.<timestamp>` and reports the name.

```bash
python -c "
from parslbox.api import ParslBox
r = ParslBox.create_config('<dir>', systems=['<sys>'], apps=['<app>'])
print(r)
"
```

Pass an absolute path. A relative one resolves against the current working
directory, which is rarely where the user meant.

Report `config_path` and `db_path` from the result.

---

## Phase 4 — Fill `pbx_python_env_setup` (required)

This is the one that breaks jobs silently. It is substituted into
`submit.sh` ahead of the final `exec pbx run`; left blank, the batch job
dies with `pbx: command not found`.

Edit the config so each selected system has the real activation commands:

```yaml
polaris:
  pbx_python_env_setup: |
    module load conda
    conda activate parslbox
```

- **Setting up on the target machine** — use the environment from Phase 1. If
  it is a conda env, `module load conda` then `conda activate <name>`. If it
  is a uv/venv, `source /abs/path/.uvenv/bin/activate`. Confirm the path with
  the user; a relative path will not work from a compute node.
  Then check the commands actually work before writing them in:

  ```bash
  conda env list            # the env you are about to name is in here
  ls /abs/path/.uvenv/bin/activate
  ```

- **Writing for a remote machine** — you cannot know this, and you cannot
  check it. Ask the user for the exact activation commands for the HPC.

  If parslbox is not installed there yet, that is a prerequisite, not a
  detail: the config is useless until it is. Tell them so, and offer the two
  routes — run this same skill on the HPC login node (simplest), or install
  it there by hand with the Phase 1 commands and bring the activation lines
  back here. Do not invent plausible-looking module commands to fill the gap.

---

## Phase 5 — Checkpoint

Tell the user plainly:

> The base config is written and the pbx environment is set. The **app
> configs are not** — each app still has a placeholder executable path.

Show what is left:

```bash
grep -nE '^[[:space:]]*executable_path: "/path/to/your/' <config_path>
```

Anchor the pattern like that. Every generated config carries a commented
example-app template that also contains `/path/to/your/`; a bare grep reports
it forever and you would tell the user a finished config is unfinished.

Ask whether to set up the app configs now. If they decline, stop here and tell
them jobs for those apps will fail until the paths are filled.

---

## Phase 6 — App configs

Work through them grouped by system, not by app — the user is recalling one
machine's layout at a time, and the module names come from the same place as
the binary path.

For each app × system pair that appears in the grep above:

**First ask for an existing submission script.** A working LAMMPS/VASP/ORCA
job script on that machine already contains both things you need. Read it and
extract:

- the absolute path to the binary → `executable_path`
- the `module load` / `export` lines → `environment_setup`

Show the user what you extracted and confirm before writing it.

**If there is no script**, ask directly for the binary path (an absolute path
on that system) and the modules the app needs.

If the user does not know where the binary is, help them look — on the target
machine:

```bash
which vasp_std; command -v lmp orca
module avail 2>&1 | grep -i vasp
ls /soft /opt/software /usr/local/apps 2>/dev/null
```

If it is a remote machine, these are commands for the user to run there. Do
not guess a path: a wrong `executable_path` fails at run time, long after
setup looked successful.

Write into the existing block, keeping the first two lines:

```yaml
vasp:
  polaris:
    executable_path: "/soft/applications/vasp/6.4.2/bin/vasp_std"
    environment_setup: |
      module purge
      module restore
      module load PrgEnv-nvhpc
      module load cray-libsci
```

`python` and `julia` have no `executable_path` — they use the interpreter from
the environment. They still take an `environment_setup` block if the user
needs modules loaded.

---

## Phase 7 — Verify and report

On the target machine:

```bash
grep -cE '^[[:space:]]*executable_path: "/path/to/your/' <config_path>   # expect 0
python -c "
from parslbox.api import ParslBox
p = ParslBox(config_path='<config_path>')
print('config ok:', p.config_path)
print('apps:', p.available_apps())
"
pbx ls
```

For a config written for a **different** machine, none of this can be checked
here. Give the user the copy command and tell them to run the checks above
once they are there:

```bash
scp <config_path> <user>@<host>:~/.parslbox/config.yaml
```

Final report:

- config path, database path
- the env vars to export, as a block they can paste — and say to add it to
  `~/.bashrc` (or equivalent) if this is not a one-off:

  ```bash
  export PBX_CONFIG_PATH=/scratch/myproject
  export PBX_DB_PATH=/scratch/myproject
  ```

  Omit the block entirely when the config is at the default `~/.parslbox`;
  nothing needs exporting there.
- anything still unfilled, named explicitly
- next step: `pbx add <dir> --app <app> --config <system>`, then `pbx qsub` or
  `pbx sbatch`
