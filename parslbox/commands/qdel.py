import typer
from typing_extensions import Annotated

from parslbox.commands.helpers.cancel_helpers import cancel_pbs_job
from parslbox.utils import path_utils

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
    seconds and runs `qdel` to terminate. After the scheduler kill, any
    jobs still left in Running/Submitted under this batch (because the
    orchestrator's signal handler didn't complete cleanly) are force-
    flipped to Killed.

    Prefer this over raw `qdel` so the database stays accurate.
    """
    typer.secho(f"Sending SIGTERM to job {jobid} via qsig...", fg=typer.colors.BLUE)
    typer.secho(f"Waiting {grace}s for graceful shutdown, then running qdel...",
                fg=typer.colors.BLUE)

    result = cancel_pbs_job(jobid, grace=grace, db_path=path_utils.DB_FILE)

    reconciled = result.get("reconciled_count", 0)

    if result["success"]:
        typer.secho(f"Job {jobid} cancelled cleanly.", fg=typer.colors.GREEN)
        if reconciled:
            typer.secho(
                f"Reconciled {reconciled} job(s) left in Running/Submitted by "
                f"an incomplete signal-handler shutdown — flipped to Killed.",
                fg=typer.colors.YELLOW,
            )
        return

    stage = result.get("stage", "?")
    err = result.get("error", "unknown error")

    # qsig failure → signal never delivered → batch may still be running →
    # reconciliation was skipped → hard fail.
    if stage == "qsig":
        typer.secho(f"Cancel failed at stage '{stage}': {err}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # qdel failure with reconciliation → batch got the signal, DB cleaned up;
    # the user's intent (kill the batch) is satisfied. Warn but exit 0.
    typer.secho(
        f"Hard-kill stage '{stage}' returned an error: {err}", fg=typer.colors.YELLOW
    )
    typer.secho(
        "  (This is usually harmless — the batch typically exited during the "
        "grace window, so qdel had nothing left to kill.)",
        fg=typer.colors.YELLOW,
    )
    if reconciled:
        typer.secho(
            f"Reconciled {reconciled} job(s) left in Running/Submitted — "
            f"flipped to Killed.",
            fg=typer.colors.YELLOW,
        )
