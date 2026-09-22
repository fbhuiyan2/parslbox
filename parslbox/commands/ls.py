import typer
from typing import Optional
from rich.console import Console
from rich.table import Table

from parslbox.database import database
from parslbox.utils import path_utils
from parslbox.system_configs.loader import get_system_config

# Import helper functions
from parslbox.commands.helpers.ls_cmd_helpers import (
    parse_ls_count,
    select_jobs_to_display,
    truncate_path,
    parse_parents,
    format_job_id_with_parents,
    truncate_sched_job_id
)

console = Console()

app = typer.Typer()


@app.command(context_settings={"ignore_unknown_options": True})
def ls(
    count: Optional[str] = typer.Argument(
        None, metavar="[COUNT]",
        help="How many jobs to show: N for the first N, -N for the last N, or 'all'. Omit for a truncated summary."
    ),
    status: Optional[str] = typer.Option(
        None, "--status", "-s", help="Filter jobs by status."
        ),
    app: Optional[str] = typer.Option(
        None, "--app", "-a", help="Filter jobs by app."
        ),
    tag: Optional[str] = typer.Option(
        None, "--tag", "-t", help="Filter jobs by tag."
    ),
    nnodes: Optional[int] = typer.Option(
        None, "--nnodes", "-n", help="Filter jobs by number of nodes (exact match)."
    ),
):
    """
    Lists jobs in the database with various display options.
    """
    try:
        show_all, count_n = parse_ls_count(count)
    except ValueError as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(code=1)

    job_list = database.get_jobs(
        path_utils.DB_FILE, status=status, app=app, tag=tag, num_nodes=nnodes
    )

    if not job_list:
        console.print("[yellow]ℹ️ No jobs found in the database.[/yellow]")
        return

    jobs_to_display, info_messages = select_jobs_to_display(job_list, show_all, count_n)
    
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
            
            total_ranks = num_nodes * ranks_per_node
            if num_nodes > 1:
                resources_display = f"n:{num_nodes}-r:{total_ranks}-g:{ngpus}-nocc:NA"
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
