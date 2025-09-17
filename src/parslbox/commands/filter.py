import typer
from typing import Optional

from parslbox.helpers import database, path_utils

app = typer.Typer()

@app.command()
def filter(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter jobs by status."),
    app_name: Optional[str] = typer.Option(None, "--app", "-a", help="Filter jobs by app."),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter jobs by tag."),
):
    """
    Returns space-separated job IDs matching the specified filters.
    Useful for command composition with other pbx commands.
    """
    # Get filtered jobs from database
    jobs = database.get_jobs(path_utils.DB_FILE, status=status, app=app_name, tag=tag)
    
    # Extract job IDs and print as space-separated string
    if jobs:
        job_ids = [str(job['job_id']) for job in jobs]
        print(' '.join(job_ids))
    # If no jobs found, print nothing (silent exit)
