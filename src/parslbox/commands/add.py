import typer
import sqlite3
from pathlib import Path
from typing import List, Optional
from typing_extensions import Annotated
from parslbox.helpers import database, path_utils

app = typer.Typer()

@app.command()
def add(
    paths: Annotated[
        List[str],
        typer.Argument(
            help="One or more paths to job directories, or 'all' to add all subdirectories in the current location."
        ),
    ],
    app: Annotated[
        str,
        typer.Option("--app", "-a", help="The application type (e.g., 'lammps', 'vasp')."),
    ],
    tag: Annotated[
        Optional[str],
        typer.Option("--tag", "-t", help="An optional tag to categorize the job(s)."),
    ] = None,
    ngpus: Annotated[
        int,
        typer.Option("--ngpus", "-n", help="Number of GPUs required for the job(s)."),
    ] = 1,
    status: Annotated[
        str,
        typer.Option("--status", "-s", help="Initial status for the job(s)."),
    ] = "Ready",
):
    """
    Adds one or more new jobs to the database.
    Can add specific directories by path, or all subdirectories with 'all'.
    """
    paths_to_add: List[Path] = []

    # --- Determine the list of paths to process ---
    if len(paths) == 1 and paths[0].lower() == 'all':
        typer.secho("Scanning current directory for subdirectories...", fg=typer.colors.BLUE)
        current_dir = Path.cwd()
        subdirectories = [p for p in current_dir.iterdir() if p.is_dir()]
        
        if not subdirectories:
            typer.secho("No subdirectories found in the current directory.", fg=typer.colors.YELLOW)
            raise typer.Exit()
            
        paths_to_add = [p.resolve() for p in subdirectories]
        typer.secho(f"Found {len(paths_to_add)} directories to add.", fg=typer.colors.BLUE)
    else:
        # Validate user-provided paths
        for path_str in paths:
            path_obj = Path(path_str)
            if not path_obj.exists():
                typer.secho(f"❌ Error: Path '{path_str}' does not exist.", fg=typer.colors.RED)
                raise typer.Exit(code=1)
            if not path_obj.is_dir():
                typer.secho(f"❌ Error: Path '{path_str}' is not a directory.", fg=typer.colors.RED)
                raise typer.Exit(code=1)
            paths_to_add.append(path_obj.resolve())

    # --- Add the determined paths to the database ---
    success_count = 0
    fail_count = 0

    for path in paths_to_add:
        try:
            new_id = database.add_job(
                db_path=path_utils.DB_FILE,
                path=str(path),
                app=app,
                ngpus=ngpus,
                tag=tag,
                status=status
            )
            typer.secho(f"✅ Added job '{path}' with ID {new_id}", fg=typer.colors.GREEN)
            success_count += 1
        except sqlite3.IntegrityError:
            typer.secho(f"⚠️  Skipped: Job path '{path}' already exists in the database.", fg=typer.colors.YELLOW)
            fail_count += 1

    typer.echo("---") # Separator
    if success_count > 0:
        typer.secho(f"Summary: Successfully added {success_count} job(s).", fg=typer.colors.GREEN)
    if fail_count > 0:
        typer.secho(f"Summary: Skipped {fail_count} job(s) that already existed.", fg=typer.colors.YELLOW)


'''
def add(
    paths: Annotated[
        List[Path],
        typer.Argument(
            help="One or more paths to the job directories.",
            exists=True,
            resolve_path=True,
        ),
    ],
    app: Annotated[
        str,
        typer.Option("--app", "-a", help="The application type (e.g., 'lammps', 'vasp')."),
    ],
    tag: Annotated[
        Optional[str],
        typer.Option("--tag", "-t", help="An optional tag to categorize the job(s)."),
    ] = None,
    ngpus: Annotated[
        int,
        typer.Option("--ngpus", "-n", help="Number of GPUs required for the job(s)."),
    ] = 1,
):
    """
    Adds one or more new jobs to the database.
    """
    success_count = 0
    fail_count = 0

    for path in paths:
        try:
            new_id = database.add_job(
                db_path=path_utils.DB_FILE,
                path=str(path),
                app=app,
                ngpus=ngpus,
                tag=tag
            )
            typer.secho(f"✅ Added job '{path}' with ID {new_id}", fg=typer.colors.GREEN)
            success_count += 1
        except sqlite3.IntegrityError:
            typer.secho(f"❌ Error: Job path '{path}' already exists in the database.", fg=typer.colors.RED)
            fail_count += 1
    
    if fail_count > 0:
        typer.secho(f"\nSummary: {success_count} jobs added, {fail_count} failed.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

'''
