"""
Shared helpers for `pbx qdel` and `pbx scancel`.

The pattern in both cases is: send SIGTERM first, wait a user-controlled
grace period to let the running `pbx run` orchestrator mark in-flight jobs
as Killed in the database, then hard-terminate. Used by both the CLI
commands and the ParslBox API methods.

After the scheduler-level kill, an optional DB reconciliation step queries
for any jobs still owned by this batch job (matching `sched_job_id`) that
remain in a non-terminal status (`Running` / `Submitted`) — these are jobs
whose orchestrator signal handler never got to STEP 2 (mark Killed) before
the process exited (DB contention, alarm timeout, exception, etc.). The
reconciliation force-flips them to `Killed` so the DB matches reality.

Reconciliation is SAFE for concurrent multi-batch use: `sched_job_id` is
unique per claim (overwritten on each `create_parsl_future` call), so other
batch jobs' active jobs carry different ids and are not touched. Stale
`sched_job_id` values can only persist for jobs that were never re-claimed
— exactly the stuck set we want to clean.

Reconciliation is SKIPPED when the initial SIGTERM-delivery step fails: if
the signal was never delivered, the batch is still running normally and
flipping its claimed jobs to Killed would corrupt state.
"""

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from parslbox.database import database


def _reconcile_stuck_jobs(db_path: Optional[Path], jobid: str) -> int:
    """Force-flip jobs left in Running/Submitted by the killed batch to Killed.

    Returns the number of jobs reconciled (0 if none stuck, or if db_path is
    None). Never raises — reconciliation failures log to stderr but do not
    propagate so the cancel call returns its primary result cleanly.
    """
    if db_path is None:
        return 0
    try:
        stuck = database.get_jobs_by_sched_id(
            db_path, jobid, statuses=["Running", "Submitted"]
        )
        if not stuck:
            return 0
        stuck_ids = [j["job_id"] for j in stuck]
        database.update_jobs(db_path, job_ids=stuck_ids, status="Killed")
        return len(stuck_ids)
    except Exception:
        # Best-effort; primary cancel result must not be masked by a DB hiccup.
        return 0


def cancel_pbs_job(
    jobid: str,
    grace: int = 30,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Gracefully cancel a PBS job.

    Sends SIGTERM via `qsig`, sleeps for `grace` seconds, then runs `qdel`.
    The SIGTERM lets the orchestrator's signal handler run and mark active
    jobs as Killed in the database before SIGKILL.

    When `db_path` is provided, after the scheduler-level kill (whether qdel
    succeeded or failed) the DB is reconciled: jobs still in Running/Submitted
    with `sched_job_id == jobid` are force-flipped to Killed. This handles
    the case where the orchestrator's signal handler did not complete its
    cleanup before the process exited.
    """
    try:
        subprocess.run(["qsig", "-s", "SIGTERM", jobid], check=True,
                       capture_output=True, text=True)
    except FileNotFoundError:
        # Signal never delivered — do NOT reconcile (batch may still be running).
        return {"success": False, "jobid": jobid, "stage": "qsig",
                "error": "qsig command not found", "reconciled_count": 0}
    except subprocess.CalledProcessError as e:
        return {"success": False, "jobid": jobid, "stage": "qsig",
                "error": (e.stderr or str(e)).strip(), "reconciled_count": 0}

    time.sleep(grace)

    qdel_error: Optional[str] = None
    qdel_stage: Optional[str] = None
    try:
        subprocess.run(["qdel", jobid], check=True,
                       capture_output=True, text=True)
    except FileNotFoundError:
        qdel_error = "qdel command not found"
        qdel_stage = "qdel"
    except subprocess.CalledProcessError as e:
        qdel_error = (e.stderr or str(e)).strip()
        qdel_stage = "qdel"

    # Reconcile DB regardless of qdel outcome — qsig succeeded so the batch
    # got the signal; whatever happens at qdel time, stuck jobs need cleanup.
    reconciled_count = _reconcile_stuck_jobs(db_path, jobid)

    if qdel_error is not None:
        return {"success": False, "jobid": jobid, "stage": qdel_stage,
                "error": qdel_error, "reconciled_count": reconciled_count}

    return {"success": True, "jobid": jobid, "grace": grace,
            "reconciled_count": reconciled_count}


def cancel_slurm_job(
    jobid: str,
    grace: int = 30,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Gracefully cancel a SLURM job.

    Sends SIGTERM to the batch script (without marking the job for
    cancellation) via `scancel --signal=TERM --batch`, sleeps for `grace`
    seconds, then runs `scancel` to terminate.

    When `db_path` is provided, after the scheduler-level kill the DB is
    reconciled (see `cancel_pbs_job` for details).
    """
    try:
        subprocess.run(["scancel", "--signal=TERM", "--batch", jobid],
                       check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return {"success": False, "jobid": jobid, "stage": "scancel-signal",
                "error": "scancel command not found", "reconciled_count": 0}
    except subprocess.CalledProcessError as e:
        return {"success": False, "jobid": jobid, "stage": "scancel-signal",
                "error": (e.stderr or str(e)).strip(), "reconciled_count": 0}

    time.sleep(grace)

    scancel_error: Optional[str] = None
    scancel_stage: Optional[str] = None
    try:
        subprocess.run(["scancel", jobid], check=True,
                       capture_output=True, text=True)
    except FileNotFoundError:
        scancel_error = "scancel command not found"
        scancel_stage = "scancel"
    except subprocess.CalledProcessError as e:
        scancel_error = (e.stderr or str(e)).strip()
        scancel_stage = "scancel"

    reconciled_count = _reconcile_stuck_jobs(db_path, jobid)

    if scancel_error is not None:
        return {"success": False, "jobid": jobid, "stage": scancel_stage,
                "error": scancel_error, "reconciled_count": reconciled_count}

    return {"success": True, "jobid": jobid, "grace": grace,
            "reconciled_count": reconciled_count}
