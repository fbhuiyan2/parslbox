from typing import Optional, List, Tuple, Dict, Any

def truncate_path(path: str, first_dirs: int = 2, last_dirs: int = 3) -> str:
    """
    Truncate a path to show first N and last M directories with '...' in between.
    
    Args:
        path: The full path to truncate
        first_dirs: Number of directories to show from the beginning
        last_dirs: Number of directories to show from the end
    
    Returns:
        Truncated path in format: /first/dirs/.../last/dirs
        
    Example:
        /lus/eagle/projects/CSTEELML/fbhuiyan/testruns/parslbox_test/polaris/lammps-kk/friction_1
        -> /lus/eagle/.../polaris/lammps-kk/friction_1
    """
    if not path or not isinstance(path, str):
        return path or ""
    
    # Handle both Unix and Windows paths
    separator = '/' if '/' in path else '\\'
    parts = [p for p in path.split(separator) if p]  # Remove empty parts
    
    # If path is short enough, return as-is
    if len(parts) <= first_dirs + last_dirs:
        return path
    
    # Build truncated path
    first_part = separator.join(parts[:first_dirs])
    last_part = separator.join(parts[-last_dirs:])
    
    # Handle absolute paths (starting with /)
    if path.startswith(separator):
        return f"{separator}{first_part}{separator}...{separator}{last_part}"
    else:
        return f"{first_part}{separator}...{separator}{last_part}"


def truncate_sched_job_id(sched_job_id: str, max_length: int = 11) -> str:
    """
    Truncate scheduler job ID to specified length with '...' suffix.
    
    Args:
        sched_job_id: The scheduler job ID to truncate
        max_length: Maximum length before truncation
    
    Returns:
        Truncated job ID in format: first_chars...
        
    Example:
        6586495.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov -> 6586495.pol...
    """
    if not sched_job_id or sched_job_id == "None":
        return sched_job_id or "None"
    
    if len(sched_job_id) <= max_length:
        return sched_job_id
    
    return sched_job_id[:max_length] + "..."


def parse_parents(parents_str: str) -> List[int]:
    """Parse JSON parent string to list of integers"""
    if not parents_str:
        return []
    import json
    return [int(x) for x in json.loads(parents_str)]


def format_job_id_with_parents(job_id: int, parents: List[int]) -> str:
    """Format job ID with parent dependencies"""
    if not parents:
        return str(job_id)
    
    if len(parents) <= 4:
        parents_str = ",".join(str(p) for p in parents)
        return f"{job_id} ({parents_str})"
    else:
        # Truncate after 4 parents: "100 (1,2,...,5)"
        first_parents = ",".join(str(p) for p in parents[:2])
        last_parent = parents[-1]
        return f"{job_id} ({first_parents},...,{last_parent})"


def parse_ls_count(raw: Optional[str]) -> Tuple[bool, Optional[int]]:
    """
    Parse the optional positional count argument of `pbx ls`.

    Accepts 'all', a positive integer (first N) or a negative integer (last N).
    Returns: (show_all, count)

    Raises ValueError with a user-facing message on anything else. Because the
    command enables `ignore_unknown_options` (so `-20` parses as an argument
    rather than a flag), a mistyped flag lands here as the count — rejecting
    unparseable values is what turns it back into a clear error.
    """
    if raw is None:
        return False, None

    token = raw.strip()
    if token.lower() == "all":
        return True, None

    try:
        count = int(token)
    except ValueError:
        raise ValueError(
            f"Invalid count '{raw}'. Expected an integer (e.g. 10, -20) or 'all'."
        )

    if count == 0:
        raise ValueError("Invalid count '0'. Use 'all' to show every job.")

    return False, count


def select_jobs_to_display(job_list: List[Dict[str, Any]], show_all: bool, count: Optional[int]) -> Tuple[List[Any], List[str]]:
    """
    Select which jobs to display and return info messages.
    Returns: (jobs_to_display, info_messages)

    The jobs_to_display list may contain job dictionaries or the string "SEPARATOR"
    to indicate where to show "..." in the table.
    """
    total_jobs = len(job_list)
    info_messages = []

    if show_all:
        return job_list, info_messages

    if count is not None:
        if count > 0:
            if count >= total_jobs:
                info_messages.append(f"Found {total_jobs} jobs in the database")
                return job_list, info_messages
            else:
                info_messages.append(f"Showing the first {count} jobs")
                return job_list[:count], info_messages
        else:  # negative count
            abs_n = abs(count)
            if abs_n >= total_jobs:
                info_messages.append(f"Found {total_jobs} jobs in the database")
                return job_list, info_messages
            else:
                info_messages.append(f"Showing the last {abs_n} jobs")
                return job_list[-abs_n:], info_messages

    # Default behavior (no count given)
    if total_jobs <= 25:
        return job_list, info_messages
    else:
        # Show first 10 + last 10 with separator
        first_10 = job_list[:10]
        last_10 = job_list[-10:]
        return first_10 + ["SEPARATOR"] + last_10, info_messages