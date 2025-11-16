import typer
from typing import Optional, List, Tuple, Dict, Any
from rich.console import Console
from rich.table import Table

from parslbox.helpers import database, path_utils

console = Console()

app = typer.Typer()


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
        /lus/eagle/projects/CSTEELML/fbhuiyan/testruns/parslbox_test/polaris/lammps/friction_1
        -> /lus/eagle/.../polaris/lammps/friction_1
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


def select_jobs_to_display(job_list: List[Dict[str, Any]], all_jobs_flag: bool, n_flag: Optional[int]) -> Tuple[List[Any], List[str]]:
    """
    Select which jobs to display based on flags and return info messages.
    Returns: (jobs_to_display, info_messages)
    
    The jobs_to_display list may contain job dictionaries or the string "SEPARATOR"
    to indicate where to show "..." in the table.
    """
    total_jobs = len(job_list)
    info_messages = []
    
    # Handle --all flag
    if all_jobs_flag:
        return job_list, info_messages
    
    # Handle -n flag
    if n_flag is not None:
        if n_flag == 0:
            return job_list, info_messages
        elif n_flag > 0:
            if n_flag >= total_jobs:
                info_messages.append(f"Found {total_jobs} jobs in the database")
                return job_list, info_messages
            else:
                info_messages.append(f"Showing the first {n_flag} jobs")
                return job_list[:n_flag], info_messages
        else:  # negative n_flag
            abs_n = abs(n_flag)
            if abs_n >= total_jobs:
                info_messages.append(f"Found {total_jobs} jobs in the database")
                return job_list, info_messages
            else:
                info_messages.append(f"Showing the last {abs_n} jobs")
                return job_list[-abs_n:], info_messages
    
    # Default behavior (no flags)
    if total_jobs <= 25:
        return job_list, info_messages
    else:
        # Show first 10 + last 10 with separator
        first_10 = job_list[:10]
        last_10 = job_list[-10:]
        return first_10 + ["SEPARATOR"] + last_10, info_messages


@app.command()
def ls(
    status: Optional[str] = typer.Option(
        None, "--status", "-s", help="Filter jobs by status."
        ),
    app: Optional[str] = typer.Option(
        None, "--app", "-a", help="Filter jobs by app."
        ),
    tag: Optional[str] = typer.Option(
        None, "--tag", "-t", help="Filter jobs by tag."
    ),
    all_jobs: bool = typer.Option(
        False, "--all", help="Show all jobs regardless of count."
    ),
    n: Optional[int] = typer.Option(
        None, "-n", help="Number of jobs to show. Negative for last N jobs, 0 for all."
    ),
):
    """
    Lists jobs in the database with various display options.
    """
    # Check for conflicting flags
    if all_jobs and n is not None:
        console.print("[red]❌ Error: Cannot use both --all and -n flags together. Please use only one.[/red]")
        raise typer.Exit(code=1)
    
    job_list = database.get_jobs(path_utils.DB_FILE, status=status, app=app, tag=tag)
    
    if not job_list:
        console.print("[yellow]ℹ️ No jobs found in the database.[/yellow]")
        return
    
    # Select which jobs to display based on flags
    jobs_to_display, info_messages = select_jobs_to_display(job_list, all_jobs, n)
    
    # Print job count and any info messages
    console.print(f"[green]# of jobs in the database: {len(job_list)}[/green]")
    for message in info_messages:
        console.print(f"[blue]ℹ️ {message}[/blue]")

    # Create table
    table = Table("ID", "App", "Status", "Resources", "Sched Job ID", "Tag", "Input", "Timestamp", "Path")
    
    for item in jobs_to_display:
        if item == "SEPARATOR":
            # Add separator row
            table.add_row("...", "", "", "", "", "", "", "", "")
        else:
            job = item
            # Generate resource display string
            num_nodes = job.get('num_nodes', 1)
            ngpus = job.get('ngpus', 0)
            node_occupancy = job.get('node_occupancy', 1.0)
            ranks_per_node = job.get('ranks_per_node', 1)
            
            # Calculate total ranks
            if ngpus > 0:
                total_ranks = num_nodes * ngpus
            else:
                total_ranks = num_nodes * ranks_per_node
            
            if num_nodes > 1:
                resources_display = f"n:{num_nodes}-r:{total_ranks}-g:auto-nocc:NA"
            elif ngpus > 0:
                resources_display = f"n:1-r:{total_ranks}-g:{ngpus}-nocc:NA"
            else:
                resources_display = f"n:1-r:{total_ranks}-g:0-nocc:{node_occupancy}"
            
            # Format job ID with parent dependencies
            parents = parse_parents(job.get('parents'))
            formatted_id = format_job_id_with_parents(job['job_id'], parents)
            
            table.add_row(
                formatted_id,
                job['app'],
                job['status'],
                resources_display,
                truncate_sched_job_id(job['sched_job_id'] or "None"),
                job['tag'] or "None",  # Display 'None' if tag is None
                job['in_file'] or "None",  # Display 'None' if in_file is None
                job['timestamp'],
                truncate_path(job['path'])
            )
    
    console.print(table)
