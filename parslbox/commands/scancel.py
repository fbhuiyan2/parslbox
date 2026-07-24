import typer
from typing_extensions import Annotated

from parslbox.commands.helpers.cancel_helpers import cancel_slurm_job
from parslbox.utils import path_utils

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
    so the running `pbx run` orchestrator can reconcile in-flight jobs in
    the database, then waits `grace` seconds and runs `scancel` to
    terminate. After the scheduler kill, any non-terminal jobs under this
    batch (because the orchestrator's signal handler didn't complete
    cleanly) are reconciled per state: Running→Killed, Submitted→Ready,
    Resubmitted→Restart.

    Prefer this over raw `scancel` so the database stays accurate.
    """
    typer.secho(f"Sending SIGTERM to job {jobid} via scancel --signal=TERM --batch...",
                fg=typer.colors.BLUE)
    typer.secho(f"Waiting {grace}s for graceful shutdown, then running scancel...",
                fg=typer.colors.BLUE)

    result = cancel_slurm_job(jobid, grace=grace, db_path=path_utils.DB_FILE)

    reconciled = result.get("reconciled_count", 0)

    if result["success"]:
        typer.secho(f"Job {jobid} cancelled cleanly.", fg=typer.colors.GREEN)
        if reconciled:
            typer.secho(
                f"Reconciled {reconciled} non-terminal job(s) left by an "
                f"incomplete signal-handler shutdown (Running→Killed, "
                f"Submitted→Ready, Resubmitted→Restart).",
                fg=typer.colors.YELLOW,
            )
        return

    stage = result.get("stage", "?")
    err = result.get("error", "unknown error")

    # First-stage failure → signal never delivered → reconciliation skipped → hard fail.
    if stage == "scancel-signal":
        typer.secho(f"Cancel failed at stage '{stage}': {err}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Hard-cancel failure with reconciliation → batch got the signal, DB cleaned up.
    typer.secho(
        f"Hard-cancel stage '{stage}' returned an error: {err}", fg=typer.colors.YELLOW
    )
    typer.secho(
        "  (This is usually harmless — the batch typically exited during the "
        "grace window, so scancel had nothing left to cancel.)",
        fg=typer.colors.YELLOW,
    )
    if reconciled:
        typer.secho(
            f"Reconciled {reconciled} non-terminal job(s) (Running→Killed, "
            f"Submitted→Ready, Resubmitted→Restart).",
            fg=typer.colors.YELLOW,
        )
