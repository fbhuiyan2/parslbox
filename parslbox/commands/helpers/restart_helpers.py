"""
Helpers for `pbx run --restart-mode`: the startup hook that calls each app's
`restart()` for `Restart`-status jobs, and the walltime-time resubmit logic
that generates the next per-link script from `restart_template.sh` and
hands it to qsub/sbatch.

See docs/restart.md for the end-to-end design.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

from parslbox.database import database


# Fields the orchestrator allows an app's restart() to patch. Any other keys
# returned in the dict are ignored with a logged warning.
_PATCHABLE_FIELDS = {
    "in_file", "env_file", "ngpus", "num_nodes", "node_occupancy",
    "ranks_per_node", "mpi_opts", "tag",
}

# Required args that MUST be present in a restart_template.sh's `pbx run` line.
# If any of these are missing, the chain stops with a clear error.
_REQUIRED_RUN_ARGS = ("--restart-mode", "--max-restarts", "--config", "--run-dir")


# ============================================================
# Startup hook: app.restart() per Restart-status job
# ============================================================


def apply_restart_hook(
    restart_jobs: List[Dict[str, Any]],
    app_instances: Dict[str, Any],
    db_path: Path,
    logger,
) -> Dict[str, List[int]]:
    """
    For each job in `restart_jobs`, call `app_instances[job['app']].restart(job)`
    and partition the outcomes into three buckets (the three-scene contract):

      - "patched":     dict returned → patches applied via bulk DB update, status → Ready
      - "rerun":       None or {} returned → status → Ready, no field changes
      - "failed":      NotImplementedError raised (or app missing) → status → Failed

    Three bulk `database.update_jobs` calls are issued (one per bucket).

    Returns:
        dict mapping bucket name to list of job IDs in that bucket.
    """
    patched_ids: List[int] = []
    rerun_ids: List[int] = []
    failed_ids: List[int] = []

    # Patches need per-(job_id, patch) detail since each job may patch
    # different fields; we group them at the end by (column_set) for batched
    # updates. v1 keeps it simple: one update per patched job. Bulk DB
    # update by status is still done in one call below.
    patches_per_job: Dict[int, Dict[str, Any]] = {}

    for job in restart_jobs:
        job_id = job["job_id"]
        app_name = job["app"]
        app = app_instances.get(app_name)
        if app is None:
            logger.warning(
                f"Restart hook: job {job_id} app '{app_name}' not loaded; "
                f"marking Failed."
            )
            failed_ids.append(job_id)
            continue
        try:
            result = app.restart(job)
        except NotImplementedError as e:
            logger.info(
                f"Restart hook: job {job_id} app '{app_name}' does not support "
                f"restart ({e}); marking Failed."
            )
            failed_ids.append(job_id)
            continue
        except Exception as e:
            logger.error(
                f"Restart hook: job {job_id} app '{app_name}' restart() raised "
                f"{type(e).__name__}: {e}; marking Failed."
            )
            failed_ids.append(job_id)
            continue

        if not result:
            rerun_ids.append(job_id)
            continue

        # Scene A: filter unknown keys, warn but don't fail.
        unknown = set(result) - _PATCHABLE_FIELDS
        if unknown:
            logger.warning(
                f"Restart hook: job {job_id} restart() returned unknown "
                f"fields {sorted(unknown)}; ignoring those."
            )
        clean_patch = {k: v for k, v in result.items() if k in _PATCHABLE_FIELDS}
        patches_per_job[job_id] = clean_patch
        patched_ids.append(job_id)

    # Apply bulk DB updates per bucket.
    if rerun_ids:
        database.update_jobs(db_path, job_ids=rerun_ids, status="Ready")
        logger.info(f"Restart hook: {len(rerun_ids)} job(s) flipped to Ready as-is.")

    if patched_ids:
        # Per-job update for patched ones (each patch is potentially unique).
        for jid in patched_ids:
            patch = patches_per_job[jid]
            database.update_jobs(db_path, job_ids=[jid], status="Ready", **patch)
        logger.info(
            f"Restart hook: {len(patched_ids)} job(s) patched and flipped to Ready."
        )

    if failed_ids:
        database.update_jobs(db_path, job_ids=failed_ids, status="Failed")
        logger.info(f"Restart hook: {len(failed_ids)} job(s) marked Failed.")

    return {"patched": patched_ids, "rerun": rerun_ids, "failed": failed_ids}


# ============================================================
# Walltime-time resubmit: build the next per-link script and qsub/sbatch it
# ============================================================


def detect_scheduler_command() -> Optional[str]:
    """Return 'qsub' if running under PBS, 'sbatch' under SLURM, else None."""
    if os.environ.get("PBS_JOBID"):
        return "qsub"
    if os.environ.get("SLURM_JOB_ID"):
        return "sbatch"
    return None


def validate_pbx_run_line(template_text: str) -> str:
    """
    Find the `exec pbx run ...` line and confirm every required arg is present.

    Returns the run line (unchanged) on success. Raises ValueError with a
    clear message on failure.
    """
    run_lines = [
        line for line in template_text.splitlines()
        if "pbx run" in line and "--restart-mode" in line
    ]
    if not run_lines:
        raise ValueError(
            "restart_template.sh has no `pbx run --restart-mode ...` line. "
            "Restore the line or regenerate the template via a fresh "
            "`pbx qsub --restart`."
        )
    # If the template has multiple, pick the last (matches the template structure).
    run_line = run_lines[-1]
    missing = [arg for arg in _REQUIRED_RUN_ARGS if arg not in run_line]
    if missing:
        raise ValueError(
            f"restart_template.sh's `pbx run` line is missing required arg(s): "
            f"{', '.join(missing)}. Chain stopped."
        )
    return run_line


def extract_original_allocation(
    submit_file_path: Path, scheduler_type: str
) -> Optional[int]:
    """
    Read the original submit.sh and return the user's originally-requested
    integer node count. Returns None if the value can't be parsed as a plain
    integer (e.g., complex PBS select strings like '2:ncpus=32:ngpus=4') so
    the caller can skip capping in that case.
    """
    if not submit_file_path.exists():
        return None
    text = submit_file_path.read_text()
    if scheduler_type == "pbs":
        m = re.search(r"^\s*#PBS\s+-l\s+select=(\S+)", text, re.MULTILINE)
    else:
        m = re.search(r"^\s*#SBATCH\s+--nodes=(\S+)", text, re.MULTILINE)
    if not m:
        return None
    raw = m.group(1)
    try:
        return int(raw)
    except ValueError:
        return None


def _substitute_resource_placeholders(
    template_text: str, computed_nodes: int, original_cap: Optional[int]
) -> str:
    """Replace <<PBX_AUTO_SELECT>> / <<PBX_AUTO_NODES>> with min(computed, cap)."""
    if original_cap is not None:
        value = min(computed_nodes, original_cap)
    else:
        value = computed_nodes
    value_str = str(max(1, value))  # never go below 1 node
    out = template_text.replace("<<PBX_AUTO_SELECT>>", value_str)
    out = out.replace("<<PBX_AUTO_NODES>>", value_str)
    return out


def _decrement_max_restarts(text: str, new_value: int) -> str:
    """Rewrite `--max-restarts <anything>` to `--max-restarts <new_value>`."""
    return re.sub(
        r"(--max-restarts)\s+\S+",
        f"--max-restarts {new_value}",
        text,
    )


def _next_link_index(run_dir: Path) -> int:
    """Return next per-link index (1, 2, 3, ...) based on existing files."""
    existing = list(run_dir.glob("restart_link_*.sh"))
    return len(existing) + 1


def build_restart_link_script(
    template_path: Path,
    run_dir: Path,
    current_max_restarts: int,
    scheduler_type: str,
    computed_nodes: int,
    submit_file_path: Path,
) -> Path:
    """
    Generate the next chain link's submit script from `restart_template.sh`.

    Reads the template, validates its `pbx run` line, substitutes resource
    placeholders (capped at the original allocation), rewrites the
    `--max-restarts` value to `current_max_restarts - 1`, and writes the
    result to `restart_link_<idx>.sh` in `run_dir`.

    Returns the path to the new link script.

    Raises:
        FileNotFoundError: if `template_path` is missing.
        ValueError: if the template's `pbx run` line is broken or missing
            required args.
    """
    if not template_path.exists():
        raise FileNotFoundError(
            f"restart_template.sh not found at {template_path}. "
            "Cannot resubmit."
        )
    text = template_path.read_text()
    validate_pbx_run_line(text)  # raises ValueError on broken template

    original_cap = extract_original_allocation(submit_file_path, scheduler_type)
    text = _substitute_resource_placeholders(text, computed_nodes, original_cap)
    text = _decrement_max_restarts(text, current_max_restarts - 1)

    idx = _next_link_index(run_dir)
    link_path = run_dir / f"restart_link_{idx}.sh"
    link_path.write_text(text)
    return link_path


def submit_restart_link(
    link_path: Path, scheduler_command: str, run_dir: Path, logger
) -> bool:
    """
    Hand the per-link script to qsub/sbatch from inside the compute-node run_dir.

    Returns True if the submission succeeded (exit code 0). Logs a clear
    error and returns False on any failure — the caller stops the chain.
    """
    try:
        result = subprocess.run(
            [scheduler_command, link_path.name],
            cwd=run_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except FileNotFoundError:
        logger.error(
            f"Auto-resubmission failed: `{scheduler_command}` not found on this "
            f"compute node. Auto-restart requires qsub/sbatch availability from "
            f"compute nodes. Chain stopped. Manually rerun `pbx {scheduler_command}` "
            f"to continue."
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error(
            f"Auto-resubmission failed: `{scheduler_command}` timed out after 60s. "
            f"Chain stopped."
        )
        return False
    except Exception as e:
        logger.error(f"Auto-resubmission failed: {type(e).__name__}: {e}. Chain stopped.")
        return False

    if result.returncode != 0:
        logger.error(
            f"Auto-resubmission failed: `{scheduler_command} {link_path.name}` "
            f"returned {result.returncode}.\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}\n"
            f"Chain stopped. Manually rerun `pbx {scheduler_command}` to continue."
        )
        return False

    logger.info(
        f"Auto-resubmission OK: {scheduler_command} accepted "
        f"{link_path.name} → {result.stdout.strip()}"
    )
    return True
