import typer
from typing import Optional
from rich.console import Console
from rich.table import Table

from parslbox.helpers import database, path_utils

console = Console()

app = typer.Typer()

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
        
        table.add_row(
            str(job['job_id']),
            job['app'],
            job['status'],
            resources_display,
            job['sched_job_id'] or "None",
            job['tag'] or "None",  # Display 'None' if tag is None
            job['in_file'] or "None",  # Display 'None' if in_file is None
            job['timestamp'],
            job['path']
        )
    
    console.print(table)
    #console.print(f"[green]Found {len(job_list)} job(s).[/green]")
