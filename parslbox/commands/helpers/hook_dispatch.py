"""
Head-node-side helper that dispatches an app's preprocess/postprocess on the
assigned compute node when the app sets `RUN_HOOKS_ON_COMPUTE = True`.

The actual work is done by parslbox.apps._hook_runner inside a subprocess
launched via the resource launcher built by build_resource_launcher(). The
runner writes its return value to <job_path>/PBX_HOOK_RETURN, which this
helper reads and deletes before returning to the orchestrator.
"""

import json
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Optional


PBX_HOOK_RETURN_FILE = "PBX_HOOK_RETURN"
ALLOWED_METHODS = {"preprocess", "postprocess"}


def _json_safe(value):
    # json can't serialize Path; convert to str. Other types pass through.
    if isinstance(value, Path):
        return str(value)
    return value


def dispatch_hook_on_compute(
    app_name: str,
    method_name: str,
    resource_launcher: str,
    env_file: Optional[str],
    **method_kwargs,
) -> Optional[str]:
    """
    Run an app hook on the assigned compute node via subprocess.

    Args:
        app_name: Job's `app_name` field. Resolved by app_registry.get_app_instance()
                  on the compute node.
        method_name: "preprocess" or "postprocess".
        resource_launcher: PBX_RESOURCE_LAUNCHER prefix string from build_resource_launcher().
        env_file: Optional env file to `source` before invoking the runner.
                  None/empty skips the source step.
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

    # Build inner shell command. The resource_launcher string is trusted as-is
    # (it's built by our own code, not user-supplied). Other interpolated paths
    # are shlex-quoted to handle spaces/quotes.
    inner = (
        f"{resource_launcher} python -m parslbox.apps._hook_runner "
        f"{shlex.quote(app_name)} {shlex.quote(method_name)} {shlex.quote(str(args_file))}"
    )
    if env_file:
        inner = f"source {shlex.quote(env_file)} && {inner}"
    cmd = ["bash", "-c", inner]

    # Run from the job directory so relative paths in resource_launcher
    # (e.g. mpiexec's `--rankfile ./rankfile_pbx_mpich_N.txt`, GPU wrapper
    # scripts) resolve correctly — same convention as the bash_app, which
    # `cd`s to job_path at the top of its generated script.
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
