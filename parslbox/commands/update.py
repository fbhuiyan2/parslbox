import typer
from pathlib import Path
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
    input_file: Annotated[
        Optional[str],
        typer.Option("--input", "-i", help="Update the input filename.")
    ] = None,
    ngpus: Annotated[
        Optional[int],
        typer.Option("--ngpus", "-n", help="Update the number of GPUs.")
    ] = None,
    env_file: Annotated[
        Optional[str],
        typer.Option("--envfile", "-e", help="Update the environment setup file path.")
    ] = None,
):
    """
    Updates one or more fields for a given set of jobs.
    """
    # Validate that at least one update option was provided
    if all(opt is None for opt in [status, app, tag, input_file, ngpus, env_file]):
        typer.secho("❌ Error: You must provide at least one field to update.", fg=typer.colors.RED)
        typer.echo("Example: pbx update 1 --status Submitted")
        raise typer.Exit(code=1)

    # Handle environment file validation and processing
    final_env_file = None
    if env_file:
        # Convert relative path to absolute path
        env_file_path = Path(env_file)
        if not env_file_path.is_absolute():
            env_file_path = Path.cwd() / env_file_path
        
        # Validate that the environment file exists
        if not env_file_path.exists():
            typer.secho(f"❌ Error: Environment file '{env_file}' does not exist.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        
        if not env_file_path.is_file():
            typer.secho(f"❌ Error: Environment file '{env_file}' is not a file.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        
        final_env_file = str(env_file_path.resolve())
        typer.secho(f"ℹ️  Using environment file: {final_env_file}", fg=typer.colors.BLUE)

    count = database.update_jobs(
        db_path=path_utils.DB_FILE,
        job_ids=job_ids,
        status=status,
        app=app,
        tag=tag,
        in_file=input_file,
        ngpus=ngpus,
        env_file=final_env_file
    )
    
    if count > 0:
        typer.secho(f"🔄 Successfully updated {count} job(s).", fg=typer.colors.BLUE)
    else:
        typer.secho("⚠️ No jobs found with the specified IDs to update.", fg=typer.colors.YELLOW)


