import typer
from typing_extensions import Annotated

from parslbox.commands.helpers.cancel_helpers import cancel_pbs_job

app = typer.Typer()


@app.command()
def qdel(
    jobid: Annotated[
        str,
        typer.Argument(help="PBS job ID to cancel.")
    ],
    grace: Annotated[
        int,
        typer.Option("--grace", "-g",
                     help="Seconds between SIGTERM (qsig) and hard kill (qdel). Default: 30.")
    ] = 30,
):
    """
    Gracefully cancel a ParslBox PBS job.

    Sends SIGTERM via `qsig` so the running `pbx run` orchestrator can
    mark in-flight jobs as Killed in the database, then waits `grace`
    seconds and runs `qdel` to terminate.

    Prefer this over raw `qdel` so the database stays accurate.
    """
    typer.secho(f"Sending SIGTERM to job {jobid} via qsig...", fg=typer.colors.BLUE)
    typer.secho(f"Waiting {grace}s for graceful shutdown, then running qdel...",
                fg=typer.colors.BLUE)

    result = cancel_pbs_job(jobid, grace=grace)

    if result["success"]:
        typer.secho(f"Job {jobid} cancelled cleanly.", fg=typer.colors.GREEN)
    else:
        stage = result.get("stage", "?")
        err = result.get("error", "unknown error")
        typer.secho(f"Cancel failed at stage '{stage}': {err}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
