import typer
from typing import Optional, List
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
):
    """
    Lists all jobs in the database.
    """
    job_list = database.get_jobs(path_utils.DB_FILE, status=status, app=app, tag=tag)
    
    if not job_list:
        console.print("[yellow]ℹ️ No jobs found in the database.[/yellow]")
        #raise typer.Exit()
    else:
        console.print(f"[green]# of jobs in the database: {len(job_list)}[/green]")

    # Add the new 'App', 'Tag', 'Input', and 'Resources' columns to the table header
    table = Table("ID", "App", "Status", "Resources", "Sched Job ID", "Tag", "Input", "Timestamp", "Path")
    
    for job in job_list:
        # Generate resource display string
        num_nodes = job.get('num_nodes', 1)
        ngpus = job.get('ngpus', 0)
        node_occupancy = job.get('node_occupancy', 1.0)
        
        if num_nodes > 1:
            resources_display = f"n:{num_nodes}-g:auto-nocc:NA"
        elif ngpus > 0:
            resources_display = f"n:1-g:{ngpus}-nocc:NA"
        else:
            resources_display = f"n:1-g:0-nocc:{node_occupancy}"
        
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
    #console.print(f"[green]Found {len(job_list)} job(s).[/green]")
