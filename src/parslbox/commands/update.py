import typer
from typing import List, Optional
from typing_extensions import Annotated
from parslbox.helpers import database, path_utils

app = typer.Typer()

@app.command()
def update(
    job_ids: Annotated[
        List[int],
        typer.Argument(help="ID(s) of the job(s) to update.")
    ],
    status: Annotated[
        Optional[str],
        typer.Option("--status", "-s", help="Update the job status.")
    ] = None,
    app: Annotated[
        Optional[str],
        typer.Option("--app", "-a", help="Update the job application.")
    ] = None,
    tag: Annotated[
        Optional[str],
        typer.Option("--tag", "-t", help="Update the job tag.")
    ] = None,
    ngpus: Annotated[
        Optional[int],
        typer.Option("--ngpus", "-n", help="Update the number of GPUs.")
    ] = None,
):
    """
    Updates one or more fields for a given set of jobs.
    """
    # Validate that at least one update option was provided
    if all(opt is None for opt in [status, app, tag, ngpus]):
        typer.secho("❌ Error: You must provide at least one field to update.", fg=typer.colors.RED)
        typer.echo("Example: pbx update 1 --status Submitted")
        raise typer.Exit(code=1)

    count = database.update_jobs(
        db_path=path_utils.DB_FILE,
        job_ids=job_ids,
        status=status,
        app=app,
        tag=tag,
        ngpus=ngpus
    )
    
    if count > 0:
        typer.secho(f"🔄 Successfully updated {count} job(s).", fg=typer.colors.BLUE)
    else:
        typer.secho("⚠️ No jobs found with the specified IDs to update.", fg=typer.colors.YELLOW)


'''def update(
        status: Optional[str] = typer.Option(
            None, "--status", "-s", help="Update job status."
        ),
        app: Optional[str] = typer.Option(
            None, "--app", "-a", help="Update job app."
        ),
        tag: Optional[str] = typer.Option(
            None, "--tag", "-t", help="Update job tag."
        ),
        ngpus: Optional[str] = typer.Option(
            None, "--ngpus", "-n", help="Update job ngpu."
        ),
        job_ids: List[int] = typer.Argument(..., help="ID(s) of the job(s) to update."),
        status: str = typer.Argument(..., help="The new status (e.g., Submitted, Completed).")
):
    """
    Updates the status of one or more jobs.
    """
    count = database.update_job_status(path_utils.DB_FILE, job_ids, status)
    if count > 0:
        typer.secho(f"🔄 Updated {count} job(s) to status '{status.capitalize()}'.", fg=typer.colors.BLUE)
    else:
        typer.secho("⚠️ No jobs found with the specified IDs to update.", fg=typer.colors.RED)
'''
