from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def format_job_ids(job_ids: Iterable[int]) -> str:
    """Compress job IDs into range notation. Inverse of `parse_job_ids`.

    [1,2,3,4,5]        -> "1-5"
    [1,2,3,8,14,15,16] -> "1-3 8 14-16"

    Space-separated so the result can be pasted straight back into
    `pbx update`/`rm`/`info` (which take ranges as separate shell words).
    """
    ids = sorted(set(job_ids))
    if not ids:
        return ""

    tokens = []
    start = prev = ids[0]
    for jid in ids[1:]:
        if jid == prev + 1:
            prev = jid
            continue
        tokens.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = jid
    tokens.append(str(start) if start == prev else f"{start}-{prev}")
    return " ".join(tokens)


def group_failures(failed: List[Tuple[object, str]]) -> Dict[str, list]:
    """Group (key, error_message) failure tuples by error message.

    Preserves first-seen message order. Callers render the keys themselves —
    job IDs compress via `format_job_ids`, paths are listed verbatim.
    """
    grouped: Dict[str, list] = {}
    for key, msg in failed:
        grouped.setdefault(msg, []).append(key)
    return grouped


MAX_JOB_ID = 2 ** 63 - 1
"""Largest value SQLite stores as an INTEGER.

Past it the driver raises OverflowError, which no CLI command catches, so a
bare id with a few extra digits crashed with a traceback instead of the
message every other bad id gets. Unlike the database-height ceiling this
applies to bare ids too: a number this size is not a job id under any
database.
"""


def _token_bounds(token: str) -> Tuple[int, int]:
    """(first, last) id a token covers, without materialising anything between.

    Separated out so a range can be rejected on its endpoints alone. '1-5'
    and '1-100000000' cost the same to inspect here; only the second one is
    expensive to expand, and by then it is too late to refuse it.
    """
    if token.startswith("-"):
        raise ValueError(f"Invalid job ID '{token}'")
    if "-" in token:
        first, last = token.split("-", 1)
        start, end = int(first), int(last)
        if start > end:
            raise ValueError(f"Invalid range '{token}': start must be <= end")
        if end > MAX_JOB_ID:
            raise ValueError(f"Job ID '{token}' is out of range.")
        return start, end
    jid = int(token)
    if jid > MAX_JOB_ID:
        raise ValueError(f"Job ID '{token}' is out of range.")
    return jid, jid


def _beyond_database(token: str, max_range_end: int) -> str:
    if max_range_end == 0:
        return f"Range '{token}': the database has no jobs."
    return (f"Range '{token}' goes past the highest job id in the database "
            f"({max_range_end}).")


def validate_job_id_tokens(raw_ids: List[str]) -> None:
    """Reject malformed tokens without expanding any range.

    For callers that must check syntax somewhere the database is not in
    reach — an MCP schema validator — and expand later against a known DB.

    Raises:
        ValueError: If a token is invalid (negative, non-numeric, bad range)
    """
    for token in raw_ids:
        token = token.strip()
        if token:
            _token_bounds(token)


def parse_job_ids(raw_ids: List[str], max_range_end: Optional[int] = None) -> List[int]:
    """Parse job ID arguments supporting ranges.

    Accepts individual IDs and ranges (e.g., ["1-5", "8", "14-20"]).
    Returns a sorted, deduplicated list of integer job IDs.

    Args:
        raw_ids: List of string tokens from typer arguments
                 (e.g., ["1-5", "8", "14-20"])
        max_range_end: Furthest id a *range* may reach, normally the
                 database's highest job id. The endpoint is compared before
                 the range is expanded, so '1-100000000' costs nothing to
                 refuse. A bare id is never refused here: one id is cheap to
                 carry, and each command already reports the ones it could
                 not find. Omit where no database is in reach.

    Returns:
        Sorted list of unique integer job IDs

    Raises:
        ValueError: If a token is invalid (negative, non-numeric, bad range)
                 or is a range reaching past `max_range_end`
    """
    result = set()
    for token in raw_ids:
        token = str(token).strip()
        if not token:
            continue
        start, end = _token_bounds(token)
        if start != end and max_range_end is not None and end > max_range_end:
            raise ValueError(_beyond_database(token, max_range_end))
        result.update(range(start, end + 1))
    return sorted(result)


def resolve_job_ids(raw_ids: List[str], db_path: Path) -> List[int]:
    """`parse_job_ids`, with ranges bounded by what `db_path` actually holds.

    The route every command should take: a slipped keystroke turning '1-10'
    into '1-100000000' is answered in microseconds instead of building a
    hundred million integers first.

    Raises:
        ValueError: As `parse_job_ids`.
    """
    from parslbox.database import database

    return parse_job_ids(raw_ids, max_range_end=database.get_max_job_id(db_path))
