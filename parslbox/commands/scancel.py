import typer
from typing_extensions import Annotated

from parslbox.commands.helpers.cancel_helpers import cancel_slurm_job

app = typer.Typer()


@app.command()
def scancel(
    jobid: Annotated[
        str,
        typer.Argument(help="SLURM job ID to cancel.")
    ],
    grace: Annotated[
        int,
        typer.Option("--grace", "-g",
                     help="Seconds between SIGTERM (scancel --signal=TERM) and hard cancel. Default: 30.")
    ] = 30,
):
    """
    Gracefully cancel a ParslBox SLURM job.

    Sends SIGTERM to the batch script via `scancel --signal=TERM --batch`
    so the running `pbx run` orchestrator can mark in-flight jobs as Killed
    in the database, then waits `grace` seconds and runs `scancel` to
    terminate.

    Prefer this over raw `scancel` so the database stays accurate.
    """
    typer.secho(f"Sending SIGTERM to job {jobid} via scancel --signal=TERM --batch...",
                fg=typer.colors.BLUE)
    typer.secho(f"Waiting {grace}s for graceful shutdown, then running scancel...",
                fg=typer.colors.BLUE)

    result = cancel_slurm_job(jobid, grace=grace)

    if result["success"]:
        typer.secho(f"Job {jobid} cancelled cleanly.", fg=typer.colors.GREEN)
    else:
        stage = result.get("stage", "?")
        err = result.get("error", "unknown error")
        typer.secho(f"Cancel failed at stage '{stage}': {err}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
