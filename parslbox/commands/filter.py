import typer
from typing import Optional

from parslbox.database import database
from parslbox.utils import path_utils
from parslbox.commands.helpers.filter_helpers import apply_excludes

app = typer.Typer()

@app.command()
def filter(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter jobs by status."),
    app_name: Optional[str] = typer.Option(None, "--app", "-a", help="Filter jobs by app."),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Filter jobs by tag (supports `*` glob)."),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="Filter jobs by path (partial match)."),
    in_file: Optional[str] = typer.Option(None, "--in-file", "-i", help="Filter jobs by input file (partial match)."),
    exclude_status: Optional[str] = typer.Option(None, "--exclude-status", "--xstatus", help="Exclude jobs with this status."),
    exclude_app: Optional[str] = typer.Option(None, "--exclude-app", "--xapp", help="Exclude jobs with this app."),
    exclude_tag: Optional[str] = typer.Option(None, "--exclude-tag", "--xtag", help="Exclude jobs with this tag (supports `*` glob)."),
):
    """
    Returns space-separated job IDs matching the specified filters.
    Useful for command composition with other pbx commands.
    """
    jobs = database.get_jobs(path_utils.DB_FILE, status=status, app=app_name, tag=tag, path=path, in_file=in_file)
    jobs = apply_excludes(
        jobs,
        exclude_status=exclude_status,
        exclude_app=exclude_app,
        exclude_tag=exclude_tag,
    )

    if jobs:
        print(' '.join(str(job['job_id']) for job in jobs))
    # If no jobs found, print nothing (silent exit)
