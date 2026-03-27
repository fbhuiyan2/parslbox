from pathlib import Path


PBX_STATUS_FILE = "PBX_JOB_STATUS_REPORT"
VALID_REPORT_STATUSES = {"done", "failed"}


def report_status(status: str):
    """Report job status to PBX. Call from within a user script.

    Creates a PBX_JOB_STATUS_REPORT file in the current directory
    containing the normalized status. PBX reads this file during
    check_success to determine job outcome.

    Usage:
        from parslbox.apps.utils import report_status
        report_status("done")    # or report_status("failed")

    Args:
        status: "done" or "failed" (case-insensitive)
    """
    normalized = status.strip().lower()
    if normalized not in VALID_REPORT_STATUSES:
        raise ValueError(
            f"Invalid status '{status}'. Must be one of: {', '.join(VALID_REPORT_STATUSES)}"
        )
    Path(PBX_STATUS_FILE).write_text(normalized.capitalize())
