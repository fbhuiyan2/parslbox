from typing import List


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
