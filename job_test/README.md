# `job_test/` — quick test-drive kit for pbx

This directory holds tiny throwaway workloads for exercising a fresh pbx install end-to-end. Use it to verify that your `pbx config`, MPI setup, and scheduler wiring all work — before you invest time preparing real jobs.

Everything here is intentionally minimal (seconds to run, no scientific meaning). If you want realistic orchestrations, see [`../examples/`](../examples/) instead.

## Quick start

From this directory:

```bash
# Generate 3 Python jobs that print per-rank CPU/GPU affinity, then submit them
python create_test_jobs.py <config_name> --python hello_affinity.py 3 --tag firstrun

pbx ls -t firstrun                                     # confirm 3 jobs got registered
pbx qsub -c <config_name> -N firstrun -q <queue> --select 1 -T 15 -A <project> -t firstrun
```

The generated jobs land under `job_test/tests/python/` (self-contained; delete when done). Each job's `.out` file should contain a line like `Hello from host x4009c1s7b0n0: GPU ID(s): 0, CPU affinity: 26 CPUs available to this process` — that confirms binding is working.

## Contents

### `create_test_jobs.py`

Auto-generates a batch of test jobs and registers them with pbx. Supports LAMMPS, Python, Julia, and VASP with varied resource requests and dependency patterns. Run `python create_test_jobs.py --help` for the full flag list; the docstring at the top of the file also has a usage summary and concrete examples.

Notes:
- LAMMPS jobs need a working LAMMPS install on the target system — the script derives the friction example path from your `config.yaml`'s `executable_path`. Override with `--lmp-exm-dir` if auto-detection misses.
- Python/Julia jobs use the accompanying `*_env.sh` for their environment (must exist alongside the `.py` / `.jl` script — hence the "run from this directory" note above).
- VASP jobs use the input files in `vasp_original_files/`.
- Jobs are created under `tests/<app>/<job_name>/` (relative to CWD).

### `hello_affinity.py` + `hello_affinity_env.sh`

A ~30-line Python script that sleeps 5s, then prints its hostname, `CUDA_VISIBLE_DEVICES` / `ZE_AFFINITY_MASK`, and `sched_getaffinity` (CPU cores this rank can run on). Uses `report_status("done")` at the end so the `python` app marks the job Done. Drop-in `--input` target for `pbx add --app python`.

### `hello_affinity_julia.jl` + `hello_affinity_julia_env.sh`

Same idea, in Julia. Drop-in `--input` target for `pbx add --app julia`.

### `vasp_original_files/`

Minimal VASP input set (`INCAR`, `KPOINTS`, `POSCAR`, `POTCAR`) used by `create_test_jobs.py --vasp N` to seed each generated VASP job directory. Do not use for real DFT work — it's a smoke test.

### `test_dynamic_jobs/`

Exercises the `--dynamic` job-discovery flow of `pbx run`:

- `spawner.py` — a Python job that prints affinity info, then dynamically adds 5 `hello_affinity.py` children to the DB via the `ParslBox` API while it runs.
- `test_dynamic_orchestrator.py` — creates N spawner jobs (configurable GPUs / node-occupancy). Run its `--help` for details.
- `spawner_env.sh`, `hello_affinity.py` — support files.

Only under `pbx run --dynamic` (the default) will the children get picked up in the same run — with `--static` the run session exits after the spawners complete and the children stay `Ready` for the next run.

## Cleanup

`create_test_jobs.py` writes into `job_test/tests/` (never touches the source files). Delete that subdirectory when you're done to reset. To also drop the registered jobs from the pbx DB:

```bash
# Remove only the jobs tagged 'firstrun' (or whatever --tag you used):
pbx rm $(pbx filter -t firstrun)

# Or nuke every job in the DB (destructive — also removes non-test jobs):
pbx rm all
```
