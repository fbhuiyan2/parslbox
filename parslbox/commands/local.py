"""`pbx local` -- author jobs here, run them on a remote machine.

Presentation only. Everything this module prints comes from parslbox.local.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from parslbox.local import project as project_mod
from parslbox.local import sync as sync_mod
from parslbox.local.project import (
    LOCAL_DB_NAME,
    LocalProjectError,
    NestedProject,
)

console = Console()
app = typer.Typer(
    help="Author jobs locally, run them on a remote machine.",
    no_args_is_help=True,
)


def _fail(message: str) -> None:
    typer.secho(f"❌ {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _short(value: Optional[str], width: int = 12) -> str:
    if not value:
        return "-"
    return value if len(value) <= width else f"{value[:width - 1]}…"


def _rows(pairs) -> Table:
    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2, 0, 0))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_column(style="dim", overflow="fold")
    for row in pairs:
        table.add_row(*(list(row) + [""] * (3 - len(row))))
    return table


@app.command("init")
def init_cmd(
    remote_root: Optional[str] = typer.Option(
        None, "--remote-root", help="Absolute project root on the remote machine."),
    endpoint: Optional[str] = typer.Option(
        None, "--endpoint", help="Globus Compute endpoint UUID on the remote machine."),
    transfer_local: Optional[str] = typer.Option(
        None, "--transfer-local", help="Globus Transfer collection UUID for this machine."),
    transfer_remote: Optional[str] = typer.Option(
        None, "--transfer-remote", help="Globus Transfer collection UUID for the remote."),
    transfer_remote_root: Optional[str] = typer.Option(
        None, "--transfer-remote-root",
        help="Override the collection's root, e.g. /lus/flare/projects. Detected "
             "on the first directory push, so normally leave this unset."),
    remote_config: Optional[str] = typer.Option(
        None, "--remote-config", help="Path to config.yaml on the remote machine."),
    nested_ok: bool = typer.Option(
        False, "--nested-ok", help="Create the project even if one exists above it."),
    verify: bool = typer.Option(
        False, "--verify", help="Ping the Compute endpoint when done."),
):
    """Create or resume a local project in the current directory."""
    cwd = Path.cwd()
    state = project_mod.inspect_directory(cwd)

    # Resuming an existing project: no prompts, no changes.
    if state["yaml_exists"]:
        try:
            outcome = project_mod.init_project(cwd)
        except LocalProjectError as e:
            _fail(str(e))
        proj = outcome["project"]
        typer.secho(f"✅ A local project already exists in {cwd}", fg=typer.colors.GREEN)
        console.print(_summary(proj))
        _print_export(proj)
        return

    # Refusals come before any prompting: a populated database should not
    # walk the user through a remote root and three UUIDs and then say no.
    if state["db_exists"] and state["row_count"]:
        _fail(
            f"{cwd / LOCAL_DB_NAME} holds {state['row_count']} job(s) but its "
            f".pbxlocal.yaml is gone.\nThose rows are already remote paths, so the "
            f"remote root and endpoint cannot be recovered from them. Restore the "
            f"file from a backup, write it by hand, or move the database aside."
        )

    if state["parent_project"] is not None and not nested_ok:
        typer.secho(
            f"⚠  A local project already exists at {state['parent_project']}",
            fg=typer.colors.YELLOW,
        )
        typer.secho(
            "   Nesting works -- PBX_DB_PATH decides which database you are on -- "
            "but it is usually an accident.",
            fg=typer.colors.YELLOW,
        )
        if not typer.confirm(f"Create another project here, in {cwd}?", default=False):
            raise typer.Exit(code=0)
        nested_ok = True

    if state["db_exists"]:
        typer.secho(
            f"ℹ️  Adopting the empty {LOCAL_DB_NAME} already in this directory.",
            fg=typer.colors.CYAN,
        )
    if state["plain_db_exists"]:
        typer.secho(
            "ℹ️  An ordinary job_database_pbx.db is also here. Different file, "
            "never touched by the local project.",
            fg=typer.colors.CYAN,
        )

    console.print(Panel(
        f"[bold]{cwd}[/bold] becomes the local project root.\n\n"
        "Jobs you add here are stored with the paths they will have on the "
        "remote machine, so nothing is rewritten later. You keep working in "
        "this directory; [bold]pbx qsub[/bold] sends the work over.",
        title="[bold cyan]pbx local init[/bold cyan]", border_style="cyan",
    ))

    # Naming the remote root on the command line means this is a script, not
    # a person: fill in what was given and ask for nothing.
    if not remote_root:
        remote_root = typer.prompt("Remote project root (absolute path)")
        endpoint = endpoint or typer.prompt(
            "Globus Compute endpoint UUID on the remote (blank to add later)",
            default="", show_default=False) or None
        remote_config = remote_config or typer.prompt(
            "Path to config.yaml on the remote (blank for its default)",
            default="", show_default=False) or None
        transfer_local = transfer_local or typer.prompt(
            "Globus Transfer collection UUID for this machine (blank to skip)",
            default="", show_default=False) or None
        transfer_remote = transfer_remote or typer.prompt(
            "Globus Transfer collection UUID for the remote (blank to skip)",
            default="", show_default=False) or None

    try:
        outcome = project_mod.init_project(
            cwd,
            remote_root=remote_root,
            compute_endpoint=endpoint,
            transfer_local=transfer_local,
            transfer_remote=transfer_remote,
            transfer_remote_root=transfer_remote_root,
            remote_config=remote_config,
            nested_ok=nested_ok,
        )
    except NestedProject as e:
        _fail(str(e))
    except LocalProjectError as e:
        _fail(str(e))

    proj = outcome["project"]
    verb = "Created" if outcome["action"] == "created" else "Adopted"
    typer.secho(f"✅ {verb} local project in {proj.local_root}", fg=typer.colors.GREEN)
    for note in outcome["notes"]:
        typer.secho(f"   {note}", fg=typer.colors.CYAN)
    console.print(_summary(proj))

    if not proj.compute_endpoint:
        typer.secho(
            "\n⚠  No Compute endpoint yet. Set one up on the remote machine "
            "(globus-compute-endpoint configure) and add its UUID to "
            f"{proj.yaml_path} as compute_endpoint.",
            fg=typer.colors.YELLOW,
        )
    elif verify:
        _verify_endpoint(proj)

    _print_export(proj)


def _verify_endpoint(proj) -> None:
    from parslbox.local import compute
    typer.secho("… pinging the Compute endpoint", fg=typer.colors.BLUE)
    info = compute.ping(proj.compute_endpoint)
    if info.get("reachable"):
        typer.secho(
            f"✅ Endpoint reachable: {info.get('host')}, python "
            f"{info.get('python')}, parslbox {info.get('parslbox')}",
            fg=typer.colors.GREEN,
        )
    else:
        typer.secho(f"❌ Endpoint did not answer: {info.get('error')}",
                    fg=typer.colors.RED)


def _summary(proj) -> Table:
    return _rows([
        ("project", str(proj.local_root), f"created {proj.created[:10]}"),
        ("remote", proj.remote_root),
        ("database", LOCAL_DB_NAME),
        ("endpoint", proj.compute_endpoint or "not configured"),
    ])


def _print_export(proj) -> None:
    typer.secho("\nPoint your shell at this project:", fg=typer.colors.CYAN)
    typer.secho(f"  {proj.export_line}", fg=typer.colors.GREEN, bold=True)


@app.command("status")
def status_cmd(
    local_only: bool = typer.Option(
        False, "--local-only", help="Skip the Compute call and report only what is here."),
):
    """Show which side is ahead, endpoint health, and the export line."""
    report = sync_mod.status(check_remote=not local_only)

    if not report.get("is_local_project"):
        if report.get("misconfigured"):
            typer.secho(f"❌ {report['message']}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        typer.secho(f"ℹ️  {report['message']}", fg=typer.colors.CYAN)
        typer.secho(f"   PBX_DB_PATH resolves to: {report['db_path']}")
        raise typer.Exit(code=0)

    if report.get("error"):
        typer.secho(f"❌ {report['error']}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    remote = report.get("remote", {})
    if remote.get("reachable"):
        endpoint_note = "reachable"
    elif remote.get("checked"):
        endpoint_note = "no answer"
    else:
        endpoint_note = remote.get("error", "not checked")

    rows = [
        ("project", report["local_root"], f"created {report['created'][:10]}"),
        ("remote", report["remote_root"]),
        ("endpoint", _short(report["compute_endpoint"], 12) if report["compute_endpoint"]
         else "not configured", endpoint_note),
        ("database", LOCAL_DB_NAME, f"{report['job_count']} jobs"),
    ]
    if report.get("transfer_remote"):
        root = report.get("transfer_remote_root")
        rows.append(("transfer", _short(report["transfer_remote"], 12),
                     f"rooted at {root}" if root
                     else "root not detected yet (first --with-dirs push finds it)"))
    console.print(_rows(rows))

    if report.get("moved"):
        typer.secho(
            f"\n⚠  This project was created at {report['recorded_local_root']} "
            f"and has been moved here. Paths translate from the current "
            f"location.", fg=typer.colors.YELLOW)

    last = report.get("last_sync")
    lines = []
    if last:
        lines.append(("last synced", f"{last['at']}  ({last.get('direction', '?')})"))
    else:
        lines.append(("last synced", "never"))
    lines.append(("local", report["local"]["drift"]))
    if remote.get("fingerprint"):
        lines.append(("remote", remote["drift"]))
    elif remote.get("checked") and remote.get("reachable"):
        lines.append(("remote", "no database there yet"))
    elif remote.get("checked"):
        stale = remote.get("last_known")
        detail = (f"last known: {stale.get('count')} jobs (stale)"
                  if stale else "never synced")
        lines.append(("remote", f"endpoint did not answer -- {detail}"))
    console.print("")
    console.print(_rows(lines))

    if remote.get("checked") and not remote.get("reachable"):
        typer.secho(f"\n⚠  {remote.get('error', '').splitlines()[0]}",
                    fg=typer.colors.YELLOW)
    if remote.get("identity_ok") is False:
        typer.secho(
            "\n❌ The database at the remote root belongs to a different "
            "project. Check remote_root before syncing.", fg=typer.colors.RED)
    if report.get("recommendation"):
        typer.secho(f"\n→ {report['recommendation']}", fg=typer.colors.YELLOW)

    endpoint_info = remote.get("endpoint_info") or {}
    if endpoint_info:
        typer.secho(
            f"\nendpoint: {endpoint_info.get('host')}, python "
            f"{endpoint_info.get('python')}, parslbox "
            f"{endpoint_info.get('parslbox')}", fg=typer.colors.BLUE)

    _print_export_line(report["export_line"])


def _print_export_line(line: str) -> None:
    typer.secho(f"\n{line}", fg=typer.colors.GREEN)


def render_dir_report(found: dict, limit: int = 10) -> None:
    """Say exactly which job directories are going up, and which are not.

    The headline is the ratio: "3 of 400 jobs" is the line that catches a
    forgotten status flip, where a bare "3 directories to send" reads like
    success.
    """
    statuses = "/".join(found.get("statuses") or []) or "any status"
    total = found.get("total")
    matched = found.get("matched", 0)
    typer.secho(
        f"\U0001f4c1 Job directories — {matched} of {total} job(s) in the "
        f"database are {statuses}", fg=typer.colors.BLUE)

    present, missing, outside = found["present"], found["missing"], found["outside"]
    typer.secho(f"   {len(present):>6} to send", fg=typer.colors.GREEN if present
                else typer.colors.YELLOW)
    if missing:
        typer.secho(f"   {len(missing):>6} missing on this machine, skipped",
                    fg=typer.colors.YELLOW)
        for path in missing[:limit]:
            typer.secho(f"          {path}", fg=typer.colors.YELLOW)
        if len(missing) > limit:
            typer.secho(f"          … and {len(missing) - limit} more",
                        fg=typer.colors.YELLOW)
    if outside:
        typer.secho(f"   {len(outside):>6} outside the project, not ours to send",
                    fg=typer.colors.CYAN)
        for path in outside[:limit]:
            typer.secho(f"          {path}", fg=typer.colors.CYAN)
        if len(outside) > limit:
            typer.secho(f"          … and {len(outside) - limit} more",
                        fg=typer.colors.CYAN)

    if not present:
        if matched == 0 and total:
            typer.secho(
                f"   Nothing matched. All {total} job(s) are in some other "
                f"status — flip the ones you want to re-run back to Restart "
                f"first ('pbx update <ids> --status Restart').",
                fg=typer.colors.YELLOW)
        elif not total:
            typer.secho("   The database has no jobs yet.", fg=typer.colors.YELLOW)


@app.command("push")
def push_cmd(
    with_dirs: bool = typer.Option(
        False, "--with-dirs", help="Also send job directories over Globus Transfer."),
    apps: Optional[str] = typer.Option(
        None, "--apps", help="Only send directories for these apps (comma-separated)."),
    tags: Optional[str] = typer.Option(
        None, "--tags", help="Only send directories for these tags (comma-separated)."),
    sync_level: str = typer.Option(
        "checksum", "--sync-level",
        help="How Globus decides a file is already there: exists, size, mtime, "
             "or checksum. checksum is safest but reads every file on both ends."),
    force: bool = typer.Option(
        False, "--force", help="Overwrite the remote even if it changed."),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="Do not block until the directory transfer finishes."),
):
    """Send the local database up, and optionally the job directories."""
    try:
        project = sync_mod.require_project()
    except LocalProjectError as e:
        _fail(str(e))

    dirs = None
    if with_dirs:
        found = sync_mod.job_dirs(
            project,
            apps=[a.strip() for a in apps.split(",")] if apps else None,
            tags=[t.strip() for t in tags.split(",")] if tags else None,
        )
        dirs = found["present"]
        render_dir_report(found)

    def _progress(line: str) -> None:
        # Globus keeps retrying a task it cannot complete, so without this the
        # wait looks identical to a hang. nice_status names the real problem.
        typer.secho(f"   {line}", fg=typer.colors.CYAN)

    try:
        result = sync_mod.push(force=force, dirs=dirs, wait_for_dirs=not no_wait,
                               sync_level=sync_level, progress=_progress)
    except Exception as e:
        _fail(str(e))

    transfer = result.get("transfer") or {}
    if transfer.get("submitted"):
        plural = "s" if transfer["batches"] > 1 else ""
        typer.secho(
            f"🚚 {transfer['count']} director(ies) in {transfer['batches']} "
            f"Globus task{plural}: {', '.join(transfer['task_ids'])}",
            fg=typer.colors.BLUE)
        outcome = transfer.get("outcome")
        if outcome:
            ok = outcome["status"] == "SUCCEEDED"
            note = outcome.get("nice_status")
            typer.secho(f"   {outcome['status']}: "
                        f"{outcome.get('files_transferred')} files, "
                        f"{(outcome.get('bytes_transferred') or 0) / 1e6:.1f} MB"
                        + (f" — {note}" if note else ""),
                        fg=typer.colors.BLUE if ok else typer.colors.RED)
            if not ok and not outcome.get("done"):
                typer.secho(
                    "   Still running at https://app.globus.org/activity — "
                    "a path Globus cannot reach is retried, not failed.",
                    fg=typer.colors.YELLOW)
        else:
            typer.secho("   not waiting; watch it at https://app.globus.org/activity",
                        fg=typer.colors.CYAN)

    typer.secho(
        f"⬆️  Pushed {result['bytes'] / 1024:.0f} KB to "
        f"{result['remote_db_path']} ({result['fingerprint'].get('count')} jobs)",
        fg=typer.colors.GREEN)


@app.command("pull")
def pull_cmd(
    force: bool = typer.Option(
        False, "--force", help="Overwrite the local database even if it changed."),
):
    """Bring the remote database down over the local one."""
    try:
        result = sync_mod.pull(force=force)
    except Exception as e:
        _fail(str(e))
    typer.secho(
        f"⬇️  Pulled {result['bytes'] / 1024:.0f} KB "
        f"({result['fingerprint'].get('count')} jobs)", fg=typer.colors.GREEN)
