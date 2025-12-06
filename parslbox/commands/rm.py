import typer
from typing import List

from parslbox.database import database
from parslbox.utils import path_utils 

app = typer.Typer()

@app.command()
def rm(
    job_ids: List[str] = typer.Argument(..., help="ID(s) of the job(s) to remove, or 'all'.")
):
    """
    Removes one or more jobs from the database.
    """
    if len(job_ids) == 1 and job_ids[0].lower() == 'all':
        if typer.confirm("⚠️ Are you sure you want to delete ALL jobs from the database?"):
            count = database.remove_all_jobs(path_utils.DB_FILE)
            typer.secho(f"🗑️ Removed all {count} job(s) from the database.", fg=typer.colors.YELLOW)
        else:
            typer.echo("❌ Operation cancelled.")
            raise typer.Exit()
    else:
        try:
            int_ids = [int(job_id) for job_id in job_ids]
            count = database.remove_jobs_by_id(path_utils.DB_FILE, int_ids)
            if count > 0:
                typer.secho(f"🗑️ Removed {count} job(s).", fg=typer.colors.YELLOW)
            else:
                typer.secho("⚠️ No jobs found with the specified IDs.", fg=typer.colors.RED)
        except ValueError:
            typer.secho("❌ Error: Job IDs must be integers.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
