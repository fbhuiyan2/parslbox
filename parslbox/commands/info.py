import typer
from typing import List
from rich.console import Console
from rich.table import Table

from parslbox.helpers import database, path_utils

console = Console()

app = typer.Typer()


def parse_parents(parents_str: str) -> List[int]:
    """Parse JSON parent string to list of integers"""
    if not parents_str:
        return []
    import json
    return [int(x) for x in json.loads(parents_str)]


def format_job_id_with_parents(job_id: int, parents: List[int], show_all: bool = False) -> str:
    """Format job ID with parent dependencies"""
    if not parents:
        return str(job_id)
    
    if show_all or len(parents) <= 4:
        parents_str = ",".join(str(p) for p in parents)
        return f"{job_id} ({parents_str})"
    else:
        # Truncate after 4 parents: "100 (1,2,...,5)"
        first_parents = ",".join(str(p) for p in parents[:2])
        last_parent = parents[-1]
        return f"{job_id} ({first_parents},...,{last_parent})"


@app.command()
def info(
    job_ids: List[int] = typer.Argument(..., help="ID(s) of the job(s) to get information about."),
    path: bool = typer.Option(False, "--path", "-p", help="Show only the path field."),
    ngpus: bool = typer.Option(False, "--ngpus", "-n", help="Show only the number of GPUs field."),
    app_name: bool = typer.Option(False, "--app", "-a", help="Show only the application field."),
    status: bool = typer.Option(False, "--status", "-s", help="Show only the status field."),
    tag: bool = typer.Option(False, "--tag", "-t", help="Show only the tag field."),
    input_file: bool = typer.Option(False, "--input", "-i", help="Show only the input file field."),
    sched_job_id: bool = typer.Option(False, "--sched-job-id", "-j", help="Show only the scheduler job ID field."),
    timestamp: bool = typer.Option(False, "--timestamp", "-ts", help="Show only the timestamp field."),
    env_file: bool = typer.Option(False, "--envfile", "-e", help="Show only the environment file field."),
    parents: bool = typer.Option(False, "--parents", "-P", help="Show all parent dependencies without truncation."),
):
    """
    Shows detailed information about specific jobs.
    """
    # Get jobs from database
    jobs = database.get_jobs_by_ids(path_utils.DB_FILE, job_ids)
    
    # Check if any jobs were found
    found_job_ids = {job['job_id'] for job in jobs}
    missing_job_ids = set(job_ids) - found_job_ids
    
    if missing_job_ids:
        missing_ids_str = ", ".join(map(str, sorted(missing_job_ids)))
        console.print(f"[yellow]⚠️ Warning: Job ID(s) {missing_ids_str} not found in database.[/yellow]")
    
    if not jobs:
        console.print("[red]❌ No jobs found with the specified IDs.[/red]")
        raise typer.Exit(code=1)
    
    # Determine which fields to show
    selected_fields = []
    if path:
        selected_fields.append(('path', 'Path'))
    if ngpus:
        selected_fields.append(('ngpus', 'NGPUs'))
    if app_name:
        selected_fields.append(('app', 'App'))
    if status:
        selected_fields.append(('status', 'Status'))
    if tag:
        selected_fields.append(('tag', 'Tag'))
    if input_file:
        selected_fields.append(('in_file', 'Input'))
    if sched_job_id:
        selected_fields.append(('sched_job_id', 'Sched Job ID'))
    if timestamp:
        selected_fields.append(('timestamp', 'Timestamp'))
    if env_file:
        selected_fields.append(('env_file', 'Env File'))
    
    # If no specific fields selected, show all fields
    if not selected_fields:
        selected_fields = [
            ('job_id', 'ID'),
            ('app', 'App'),
            ('status', 'Status'),
            ('ngpus', 'NGPUs'),
            ('sched_job_id', 'Sched Job ID'),
            ('tag', 'Tag'),
            ('in_file', 'Input'),
            ('env_file', 'Env File'),
            ('timestamp', 'Timestamp'),
            ('path', 'Path')
        ]
    else:
        # Always include job_id when specific fields are selected
        selected_fields.insert(0, ('job_id', 'ID'))
    
    # Create and populate table
    headers = [field[1] for field in selected_fields]
    table = Table(*headers)
    
    for job in jobs:
        row_data = []
        for field_key, _ in selected_fields:
            value = job[field_key]
            if value is None:
                row_data.append("None")
            elif field_key == 'ngpus':
                row_data.append(str(value))
            elif field_key == 'job_id':
                # Format job ID with parent dependencies
                job_parents = parse_parents(job.get('parents'))
                formatted_id = format_job_id_with_parents(job['job_id'], job_parents, show_all=parents)
                row_data.append(formatted_id)
            else:
                row_data.append(str(value))
        table.add_row(*row_data)
    
    console.print(table)
    
    # Show summary
    if len(jobs) == 1:
        console.print(f"[green]Showing information for 1 job.[/green]")
    else:
        console.print(f"[green]Showing information for {len(jobs)} jobs.[/green]")
