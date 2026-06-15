"""
Helpers for `pbx run` Restart-status handling and the `--respawn` chain:
the per-job `apply_restart_for_job` call (used lazily inside
`create_parsl_future` for each Restart-status job), and the walltime-time
resubmit logic that generates the next per-link script from
`respawn_template.sh` and hands it to qsub/sbatch (gated on --respawn).
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from parslbox.database import database


# Fields the orchestrator allows an app's restart() to patch. Resource fields
# (ngpus, num_nodes, node_occupancy, ranks_per_node, mpi_opts) are intentionally
# excluded — they affect resource allocation, which happens BEFORE restart() runs
# in the lazy per-job design. An app returning any of those is a contract
# violation handled in `apply_restart_for_job` (job marked Failed). Other unknown
# keys are warned-and-dropped without failing the job.
_PATCHABLE_FIELDS = {"in_file", "env_file", "tag"}

# If restart() returns any of these, it's a contract violation — resources are
# decided before restart() is called, so patches here would be silently ignored
# and that would be confusing. Fail loud instead.
_FORBIDDEN_RESTART_PATCH_FIELDS = {
    "ngpus", "num_nodes", "node_occupancy", "ranks_per_node", "mpi_opts",
}

# Required args that MUST be present in a respawn_template.sh's `pbx run` line.
# If any of these are missing, the chain stops with a clear error.
_REQUIRED_RUN_ARGS = ("--respawn", "--config", "--run-dir")


# ============================================================
# Per-job restart() invocation (called lazily inside create_parsl_future)
# ============================================================


def apply_restart_for_job(
    job: Dict[str, Any],
    app: Any,
    logger,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """Run `app.restart(job)` for one Restart-status job and classify the result.

    Returns:
        (bucket, patch, err) where:
          - bucket == "patched": app returned a non-empty valid patch dict.
            `patch` is the cleaned (only `_PATCHABLE_FIELDS` keys) dict to
            apply to the in-memory job dict + buffer with the Running write.
          - bucket == "rerun":   app returned None or {}. `patch` is None.
          - bucket == "failed":  restart() raised (NotImplementedError,
            generic exception, or contract violation by returning a
            forbidden resource field). `err` is a human-readable reason.

    The caller is responsible for the DB write (Failed status, or Running +
    patch via status_buffer) and for adding the job to `restarting_job_ids`
    on success.
    """
    job_id = job["job_id"]
    app_name = job["app"]

    if app is None:
        return ("failed", None, f"app '{app_name}' not loaded")

    try:
        result = app.restart(job)
    except NotImplementedError as e:
        return ("failed", None, f"app '{app_name}' does not support restart ({e})")
    except Exception as e:
        return ("failed", None, f"restart() raised {type(e).__name__}: {e}")

    if not result:
        return ("rerun", None, None)

    # Contract violation: app tried to patch a resource field. Resources are
    # already allocated before restart() runs, so we can't honor these.
    forbidden = set(result) & _FORBIDDEN_RESTART_PATCH_FIELDS
    if forbidden:
        return (
            "failed", None,
            f"restart() returned forbidden resource field(s) {sorted(forbidden)}; "
            f"resources cannot be patched by restart()"
        )

    # Unknown non-resource keys: warn and drop, but don't fail the job.
    unknown = set(result) - _PATCHABLE_FIELDS - _FORBIDDEN_RESTART_PATCH_FIELDS
    if unknown:
        logger.warning(
            f"Job {job_id}: restart() returned unknown field(s) "
            f"{sorted(unknown)}; ignoring those."
        )
    clean_patch = {k: v for k, v in result.items() if k in _PATCHABLE_FIELDS}
    if not clean_patch:
        # Only unknown fields were returned — treat as rerun (nothing to patch).
        return ("rerun", None, None)
    return ("patched", clean_patch, None)


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
        if "pbx run" in line and "--respawn" in line
    ]
    if not run_lines:
        raise ValueError(
            "respawn_template.sh has no `pbx run --respawn ...` line. "
            "Restore the line or regenerate the template via a fresh "
            "`pbx qsub --respawn N`."
        )
    # If the template has multiple, pick the last (matches the template structure).
    run_line = run_lines[-1]
    missing = [arg for arg in _REQUIRED_RUN_ARGS if arg not in run_line]
    if missing:
        raise ValueError(
            f"respawn_template.sh's `pbx run` line is missing required arg(s): "
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


def _decrement_respawn(text: str, new_value: int) -> str:
    """Rewrite `--respawn <anything>` to `--respawn <new_value>`."""
    return re.sub(
        r"(--respawn)\s+\S+",
        f"--respawn {new_value}",
        text,
    )


def _next_link_index(run_dir: Path) -> int:
    """Return next per-link index (1, 2, 3, ...) based on existing files."""
    existing = list(run_dir.glob("respawn_link_*.sh"))
    return len(existing) + 1


def build_respawn_link_script(
    template_path: Path,
    run_dir: Path,
    current_respawn: int,
    scheduler_type: str,
    computed_nodes: int,
    submit_file_path: Path,
) -> Path:
    """
    Generate the next chain link's submit script from `respawn_template.sh`.

    Reads the template, validates its `pbx run` line, substitutes resource
    placeholders (capped at the original allocation), rewrites the
    `--respawn` value to `current_respawn - 1`, and writes the result to
    `respawn_link_<idx>.sh` in `run_dir`.

    Returns the path to the new link script.

    Raises:
        FileNotFoundError: if `template_path` is missing.
        ValueError: if the template's `pbx run` line is broken or missing
            required args.
    """
    if not template_path.exists():
        raise FileNotFoundError(
            f"respawn_template.sh not found at {template_path}. "
            "Cannot resubmit."
        )
    text = template_path.read_text()
    validate_pbx_run_line(text)  # raises ValueError on broken template

    original_cap = extract_original_allocation(submit_file_path, scheduler_type)
    text = _substitute_resource_placeholders(text, computed_nodes, original_cap)
    text = _decrement_respawn(text, current_respawn - 1)

    idx = _next_link_index(run_dir)
    # Roll the scheduler log to a per-link filename so it isn't overwritten.
    text = re.sub(
        r"pbx_scheduler_link-\d+\.out",
        f"pbx_scheduler_link-{idx}.out",
        text,
    )
    link_path = run_dir / f"respawn_link_{idx}.sh"
    link_path.write_text(text)
    return link_path


def submit_respawn_link(
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
