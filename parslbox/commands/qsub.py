import typer
from pathlib import Path
from typing import Optional, List
from typing_extensions import Annotated

from parslbox.commands.helpers.submit_helpers import submit_job, ValidationError
from parslbox.commands.helpers.qsub_cmd_helpers import parse_walltime

app = typer.Typer()


def submit_to_scheduler(
    config_name: str,
    job_name: str,
    queue: str,
    select: str,
    walltime: int,
    project: Optional[str] = None,
    run_dir: Optional[Path] = None,
    apps: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    retries: int = 0,
    loglevel: str = "info",
    config_path: Optional[Path] = None,
    sched_opts: Optional[List[str]] = None,
    dynamic: bool = True,
):
    """
    Submit a PBS job via qsub. Thin wrapper around submit_job().
    """
    return submit_job(
        config_name=config_name,
        job_name=job_name,
        queue=queue,
        select=select,
        walltime=walltime,
        project=project,
        run_dir=run_dir,
        apps=apps,
        tags=tags,
        retries=retries,
        loglevel=loglevel,
        config_path=config_path,
        sched_opts=sched_opts,
        scheduler_type="pbs",
        submit_command="qsub",
        dynamic=dynamic,
    )


# Main CLI command

@app.command()
def qsub(
    config_name: Annotated[
        str,
        typer.Option("--config", "-c", help="The name of the configuration to use (e.g., 'sophia').")
    ],
    job_name: Annotated[
        str,
        typer.Option("--job-name", "-N", help="PBS job name.")
    ],
    queue: Annotated[
        str,
        typer.Option("--queue", "-q", help="PBS queue name.")
    ],
    select: Annotated[
        str,
        typer.Option("--select", help="PBS select specification (e.g., '4', '2:ncpus=32:ngpus=4', '1:ncpus=16+2:ncpus=32:ngpus=2').")
    ],
    walltime: Annotated[
        str,
        typer.Option("--walltime", "-T", help="Wall time (default: minutes). Supports h/d suffixes (e.g., 90, 4.25h, 3.5d).")
    ],
    project: Annotated[
        Optional[str],
        typer.Option("--project", "-A", help="Project/account name.")
    ] = None,
    run_dir: Annotated[
        Optional[Path],
        typer.Option("--run-dir", help="Custom run directory (default: timestamped directory).")
    ] = None,
    apps: Annotated[
        Optional[str],
        typer.Option("--apps", "-a", help="Comma-separated list of apps to run (e.g., 'lammps-kk,vasp').")
    ] = None,
    tags: Annotated[
        Optional[str],
        typer.Option("--tags", "-t", help="Comma-separated list of tags to run (e.g., 'run1,run2').")
    ] = None,
    retries: Annotated[
        int,
        typer.Option("--retries", help="Number of retries for failed tasks.")
    ] = 0,
    loglevel: Annotated[
        str,
        typer.Option("--loglevel", help="Logging level (debug, info, warning, error, critical)")
    ] = "info",
    sched_opts: Annotated[
        Optional[List[str]],
        typer.Option("--sched-opts", help="Extra PBS directives (repeatable, e.g., --sched-opts '#PBS -l filesystems=home:eagle').")
    ] = None,
    dynamic: Annotated[
        bool,
        typer.Option("--dynamic/--static", help="Dynamically discover new jobs during run (default: dynamic).")
    ] = True,
):
    """
    Generate and submit a PBS job script for running parslbox workflows.

    Use --sched-opts to pass extra #PBS directives. This flag can be repeated.
    Directives that match a key already in the template will override it;
    new directives are appended.
    """
    try:
        # Convert CLI string arguments to lists for core function
        apps_list = apps.split(',') if apps else None
        tags_list = tags.split(',') if tags else None
        walltime_minutes = parse_walltime(walltime)

        # Call core function
        result = submit_to_scheduler(
            config_name=config_name,
            job_name=job_name,
            queue=queue,
            select=select,
            walltime=walltime_minutes,
            project=project,
            run_dir=run_dir,
            apps=apps_list,
            tags=tags_list,
            retries=retries,
            loglevel=loglevel,
            sched_opts=sched_opts,
            dynamic=dynamic,
        )

        # CLI-specific output formatting
        typer.secho(f"\U0001f4c1 Created run directory: {result['run_dir']}", fg=typer.colors.BLUE)
        typer.secho(f"\U0001f4dd Generated submit script: {result['submit_file']}", fg=typer.colors.GREEN)

        if result["success"]:
            job_id = result.get("pbs_job_id", result.get("job_id", "UNKNOWN"))
            typer.secho(f"\U0001f680 Job submitted successfully! Job ID: {job_id}", fg=typer.colors.GREEN)
            typer.secho(f"\U0001f4ca Monitor with: qstat {job_id}", fg=typer.colors.BLUE)
            typer.secho(f"\U0001f4c1 Run directory: {result['run_dir']}", fg=typer.colors.BLUE)
        else:
            typer.secho(f"\u274c Error submitting job: {result['error']}", fg=typer.colors.RED)
            typer.secho(f"\U0001f4cb Submit script saved at: {result['submit_file']}", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)

    except ValidationError as e:
        typer.secho(f"\u274c Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"\u274c Unexpected error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
