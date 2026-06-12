"""
Head-node-side helper that dispatches an app's preprocess/postprocess on the
assigned compute node when the app sets `RUN_HOOKS_ON_COMPUTE = True`.

The actual work is done by parslbox.apps._hook_runner inside a subprocess
launched via the single-rank launcher built by build_single_rank_launcher(). The
runner writes its return value to <job_path>/PBX_HOOK_RETURN, which this
helper reads and deletes before returning to the orchestrator.
"""

import json
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Optional


PBX_HOOK_RETURN_FILE = "PBX_HOOK_RETURN"
ALLOWED_METHODS = {"preprocess", "postprocess"}

# GPU-visibility env vars that the orchestrator's outer allocation may have
# set and that would otherwise leak into the hook subprocess. Always unset
# before exporting whatever ResourceAssignment.get_env_vars() returned for
# this specific job — so CPU-only jobs see no GPUs and GPU jobs see exactly
# their assigned GPUs. Keep in sync with get_env_vars() in
# parslbox/resource_manager/models.py.
_GPU_VISIBILITY_VARS = ("CUDA_VISIBLE_DEVICES", "ZE_AFFINITY_MASK")


def _json_safe(value):
    # json can't serialize Path; convert to str. Other types pass through.
    if isinstance(value, Path):
        return str(value)
    return value


def dispatch_hook_on_compute(
    app_name: str,
    method_name: str,
    single_rank_launcher: str,
    env_file: Optional[str],
    gpu_env_vars: Optional[Dict[str, str]] = None,
    **method_kwargs,
) -> Optional[str]:
    """
    Run an app hook on the assigned compute node via subprocess.

    Args:
        app_name: Job's `app_name` field. Resolved by app_registry.get_app_instance()
                  on the compute node.
        method_name: "preprocess" or "postprocess".
        single_rank_launcher: PBX_SINGLE_RANK_LAUNCHER prefix string from build_single_rank_launcher().
        env_file: Optional env file to `source` before invoking the runner.
                  None/empty skips the source step.
        gpu_env_vars: Optional dict of GPU-visibility env vars (typically
            CUDA_VISIBLE_DEVICES / ZE_AFFINITY_MASK) computed for THIS job
            by ResourceAssignment.get_env_vars(). Always exported in the
            bash inner command — after a defensive `unset` of all known
            GPU-visibility vars — so the orchestrator's allocation-wide
            values cannot leak into the hook. Empty/None → no exports, but
            the unset still runs (CPU-only jobs see no GPUs).
        **method_kwargs: Keyword args forwarded to the hook method. MUST include
            `job_path` (used as the location of the args JSON and return files).

    Returns:
        The hook's return value as a string, or None if the hook returned None
        (empty PBX_HOOK_RETURN file). An empty string is treated as None so the
        caller's `if final_status:` check at run.py:787 behaves the same as today.

    Raises:
        ValueError: invalid method_name or missing job_path.
        subprocess.CalledProcessError: subprocess exited non-zero. The orchestrator's
            existing exception handlers (run.py:190 for preprocess, run.py:792 for
            postprocess) convert this to a Failed job — same as an in-process raise.
    """
    if method_name not in ALLOWED_METHODS:
        raise ValueError(
            f"method_name must be one of {sorted(ALLOWED_METHODS)}, got {method_name!r}"
        )

    job_path = method_kwargs.get("job_path")
    if job_path is None:
        raise ValueError("method_kwargs must include 'job_path'")
    job_path = Path(job_path)

    logger = logging.getLogger(__name__)
    job_id = method_kwargs.get("job_id", "?")

    # Serialize kwargs to a JSON file in job_path. Overwrite any stale file from
    # a prior retry. The runner deletes this file in its `finally`.
    args_file = job_path / f"PBX_HOOK_ARGS_{method_name}.json"
    args_file.write_text(json.dumps({k: _json_safe(v) for k, v in method_kwargs.items()}))

    # The runner writes its return value here on success. Always clear before
    # invoking so a leftover file from a previous run cannot poison the read.
    return_file = job_path / PBX_HOOK_RETURN_FILE
    return_file.unlink(missing_ok=True)

    # Build inner shell command. Order matters:
    #   1. source env_file       — user modules/conda first
    #   2. unset GPU vars        — defensive clear of orchestrator's leaked values
    #   3. export gpu_env_vars   — set ONLY this job's GPU visibility
    #   4. <launcher> python ... — mpiexec/srun propagates the shell env to compute
    # The single_rank_launcher string is trusted as-is (built by our own code).
    # Interpolated paths/values are shlex-quoted to handle spaces/quotes.
    parts = []
    if env_file:
        parts.append(f"source {shlex.quote(env_file)}")
    parts.append("unset " + " ".join(_GPU_VISIBILITY_VARS))
    for k, v in (gpu_env_vars or {}).items():
        parts.append(f"export {k}={shlex.quote(v)}")
    parts.append(
        f"{single_rank_launcher} python -m parslbox.apps._hook_runner "
        f"{shlex.quote(app_name)} {shlex.quote(method_name)} {shlex.quote(str(args_file))}"
    )
    inner = " && ".join(parts)
    cmd = ["bash", "-c", inner]

    # Run from the job directory so the launcher and any relative paths
    # the hook uses resolve correctly — same convention as the bash_app,
    # which `cd`s to job_path at the top of its generated script.
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(job_path),
        )
        logger.info(
            f"Job {job_id}: hook {method_name} returned rc={proc.returncode}"
        )
        if proc.stdout:
            logger.debug(f"Job {job_id}: hook {method_name} stdout:\n{proc.stdout}")
        if proc.stderr:
            logger.debug(f"Job {job_id}: hook {method_name} stderr:\n{proc.stderr}")
        if return_file.is_file():
            return return_file.read_text().strip() or None
        return None
    finally:
        # Clear the return file regardless of success/failure so a subsequent
        # retry sees a clean slate. The args file is cleared by the runner
        # itself; clear it here too in case the runner never got to start.
        return_file.unlink(missing_ok=True)
        args_file.unlink(missing_ok=True)
