from typing import Dict, Iterable, List, Tuple


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


def parse_job_ids(raw_ids: List[str]) -> List[int]:
    """Parse job ID arguments supporting ranges.

    Accepts individual IDs and ranges (e.g., ["1-5", "8", "14-20"]).
    Returns a sorted, deduplicated list of integer job IDs.

    Args:
        raw_ids: List of string tokens from typer arguments
                 (e.g., ["1-5", "8", "14-20"])

    Returns:
        Sorted list of unique integer job IDs

    Raises:
        ValueError: If a token is invalid (negative, non-numeric, bad range)
    """
    result = set()
    for token in raw_ids:
        token = token.strip()
        if not token:
            continue
        if token.startswith("-"):
            raise ValueError(f"Invalid job ID '{token}'")
        if "-" in token:
            parts = token.split("-", 1)
            start, end = int(parts[0]), int(parts[1])
            if start > end:
                raise ValueError(f"Invalid range '{token}': start must be <= end")
            result.update(range(start, end + 1))
        else:
            result.add(int(token))
    return sorted(result)
