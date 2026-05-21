"""
Shared helpers for `pbx qdel` and `pbx scancel`.

The pattern in both cases is: send SIGTERM first, wait a user-controlled
grace period to let the running `pbx run` orchestrator mark in-flight jobs
as Killed in the database, then hard-terminate. Used by both the CLI
commands and the ParslBox API methods.
"""

import subprocess
import time
from typing import Any, Dict


def cancel_pbs_job(jobid: str, grace: int = 30) -> Dict[str, Any]:
    """
    Gracefully cancel a PBS job.

    Sends SIGTERM via `qsig`, sleeps for `grace` seconds, then runs `qdel`.
    The SIGTERM lets the orchestrator's signal handler run and mark active
    jobs as Killed in the database before SIGKILL.
    """
    try:
        subprocess.run(["qsig", "-s", "SIGTERM", jobid], check=True,
                       capture_output=True, text=True)
    except FileNotFoundError:
        return {"success": False, "jobid": jobid, "stage": "qsig",
                "error": "qsig command not found"}
    except subprocess.CalledProcessError as e:
        return {"success": False, "jobid": jobid, "stage": "qsig",
                "error": (e.stderr or str(e)).strip()}

    time.sleep(grace)

    try:
        subprocess.run(["qdel", jobid], check=True,
                       capture_output=True, text=True)
    except FileNotFoundError:
        return {"success": False, "jobid": jobid, "stage": "qdel",
                "error": "qdel command not found"}
    except subprocess.CalledProcessError as e:
        return {"success": False, "jobid": jobid, "stage": "qdel",
                "error": (e.stderr or str(e)).strip()}

    return {"success": True, "jobid": jobid, "grace": grace}


def cancel_slurm_job(jobid: str, grace: int = 30) -> Dict[str, Any]:
    """
    Gracefully cancel a SLURM job.

    Sends SIGTERM to the batch script (without marking the job for
    cancellation) via `scancel --signal=TERM --batch`, sleeps for `grace`
    seconds, then runs `scancel` to terminate.
    """
    try:
        subprocess.run(["scancel", "--signal=TERM", "--batch", jobid],
                       check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return {"success": False, "jobid": jobid, "stage": "scancel-signal",
                "error": "scancel command not found"}
    except subprocess.CalledProcessError as e:
        return {"success": False, "jobid": jobid, "stage": "scancel-signal",
                "error": (e.stderr or str(e)).strip()}

    time.sleep(grace)

    try:
        subprocess.run(["scancel", jobid], check=True,
                       capture_output=True, text=True)
    except FileNotFoundError:
        return {"success": False, "jobid": jobid, "stage": "scancel",
                "error": "scancel command not found"}
    except subprocess.CalledProcessError as e:
        return {"success": False, "jobid": jobid, "stage": "scancel",
                "error": (e.stderr or str(e)).strip()}

    return {"success": True, "jobid": jobid, "grace": grace}
