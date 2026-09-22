"""push, pull and status for a local project.

One side is authoritative at a time. push sends the laptop's database up and
refuses if the remote moved since the last sync; pull brings the remote's
database down and refuses if the laptop moved. Neither ever merges -- the
guard exists so that "which side is real" is never a guess.
"""

import base64
import gzip
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from parslbox.local import guard
from parslbox.local.project import (
    LOCAL_DB_NAME,
    LocalProject,
    LocalProjectError,
    find_project,
    save_project,
    utc_stamp,
)


class NotALocalProject(LocalProjectError):
    """The database in use has no .pbxlocal.yaml beside it."""


def resolve_db_path(db_path=None) -> Path:
    """Where the database is, asking PBX_DB_PATH now rather than at import.

    path_utils.DB_FILE is a module constant fixed when the process started,
    which is right for a one-shot CLI run and wrong for anything long-lived.
    Same variable, same resolution rules, read later.
    """
    if db_path is not None:
        return Path(db_path)
    from parslbox.utils import path_utils
    return Path(path_utils.get_db_path(strict=False))


def require_project(db_path=None) -> LocalProject:
    db_path = resolve_db_path(db_path)
    project = find_project(db_path)
    if project is None:
        raise NotALocalProject(
            f"Not a local project. PBX_DB_PATH resolves to {db_path}, and "
            f"there is no .pbxlocal.yaml beside it.\n"
            f"Run 'pbx local init' in a project directory, or export "
            f"PBX_DB_PATH to point at one."
        )
    return project


def require_endpoint(project: LocalProject) -> str:
    if not project.compute_endpoint:
        raise LocalProjectError(
            f"No Globus Compute endpoint is configured for this project.\n"
            f"Add compute_endpoint to {project.yaml_path} once the endpoint is "
            f"running on the remote machine."
        )
    return project.compute_endpoint


def snapshot_b64(db_path, max_bytes: int) -> Dict[str, Any]:
    """A consistent gzipped copy of a live WAL-mode database, base64'd."""
    tmpdir = tempfile.mkdtemp(prefix="pbx-snap-")
    snapshot = os.path.join(tmpdir, "snapshot.db")
    try:
        con = sqlite3.connect(str(db_path))
        try:
            con.execute("VACUUM INTO ?", (snapshot,))
        finally:
            con.close()
        raw = Path(snapshot).read_bytes()
        blob = gzip.compress(raw)
        if len(blob) > max_bytes:
            raise LocalProjectError(
                f"The database is {len(blob) / 1e6:.1f} MB compressed, over the "
                f"{max_bytes / 1e6:.0f} MB a Globus Compute payload can carry. "
                f"Move it with Globus Transfer instead."
            )
        return {"raw_bytes": len(raw), "gz_bytes": len(blob),
                "data": base64.b64encode(blob).decode()}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def write_db_from_b64(db_path, data_b64: str) -> int:
    """Replace the local database with bytes pulled from the remote."""
    db_path = Path(db_path)
    blob = gzip.decompress(base64.b64decode(data_b64))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    incoming = db_path.with_name(db_path.name + ".incoming")
    incoming.write_bytes(blob)
    for suffix in ("-wal", "-shm"):
        stale = db_path.with_name(db_path.name + suffix)
        if stale.exists():
            stale.unlink()
    os.replace(incoming, db_path)
    return len(blob)


def _record_sync(project: LocalProject, direction: str,
                 local_fp: Dict[str, Any], remote_fp: Dict[str, Any]) -> None:
    project.last_sync = {
        "at": utc_stamp(),
        "direction": direction,
        "local": {"max_timestamp": local_fp.get("max_timestamp"),
                  "count": local_fp.get("count")},
        "remote": {"max_timestamp": remote_fp.get("max_timestamp"),
                   "count": remote_fp.get("count")},
    }
    save_project(project)


def _recorded(project: LocalProject, side: str) -> Optional[Dict[str, Any]]:
    return (project.last_sync or {}).get(side)


# --------------------------------------------------------------------------

def status(db_path=None, check_remote: bool = True) -> Dict[str, Any]:
    """Everything the two sides know about each other. Changes nothing."""
    from parslbox.local.project import LocalProjectError as _LPE

    resolved = resolve_db_path(db_path)
    try:
        project = find_project(resolved)
    except _LPE as e:
        return {"is_local_project": False, "misconfigured": True,
                "db_path": str(resolved), "message": str(e)}
    if project is None:
        return {
            "is_local_project": False,
            "db_path": str(resolved),
            "message": (f"Not a local project: no .pbxlocal.yaml beside "
                        f"{resolved}."),
        }

    report: Dict[str, Any] = {
        "is_local_project": True,
        "db_path": str(project.db_path),
        "local_root": str(project.local_root),
        "remote_root": project.remote_root,
        "remote_db_path": project.remote_db_path,
        "created": project.created,
        "moved": project.moved,
        "recorded_local_root": project.recorded_local_root,
        "compute_endpoint": project.compute_endpoint,
        "transfer_local": project.transfer_local,
        "transfer_remote": project.transfer_remote,
        "transfer_remote_root": project.transfer_remote_root,
        "export_line": project.export_line,
        "last_sync": project.last_sync,
    }

    try:
        local_fp = guard.fingerprint(project.db_path)
    except FileNotFoundError as e:
        report["error"] = str(e)
        return report
    report["job_count"] = local_fp["count"]
    report["local"] = {
        "fingerprint": local_fp,
        "drift": guard.describe_drift(local_fp, _recorded(project, "local")),
    }

    if not check_remote:
        report["remote"] = {"checked": False}
        return report
    if not project.compute_endpoint:
        report["remote"] = {"checked": False,
                            "error": "no compute_endpoint configured"}
        return report

    from parslbox.local import compute
    try:
        with compute.ComputeClient(project.compute_endpoint, timeout=120) as client:
            info = client.run(compute.remote_ping)
            remote_fp = client.run(compute.remote_fingerprint,
                                   project.remote_db_path)
    except Exception as e:
        report["remote"] = {
            "checked": True, "reachable": False, "error": str(e),
            "last_known": _recorded(project, "remote"),
        }
        return report

    remote_block: Dict[str, Any] = {"checked": True, "reachable": True,
                                    "endpoint_info": info,
                                    "exists": remote_fp.get("exists", False)}
    if remote_fp.get("exists") and "error" not in remote_fp:
        remote_block["fingerprint"] = remote_fp
        remote_block["drift"] = guard.describe_drift(
            remote_fp, _recorded(project, "remote"))
        remote_block["identity_ok"] = (
            remote_fp.get("identity") in (0, None)
            or remote_fp["identity"] == guard.identity_of(project.created)
        )
    elif "error" in remote_fp:
        remote_block["error"] = remote_fp["error"]
    report["remote"] = remote_block

    local_moved = not guard.unchanged(local_fp, _recorded(project, "local"))
    remote_moved = (remote_block.get("fingerprint") is not None
                    and not guard.unchanged(remote_block["fingerprint"],
                                            _recorded(project, "remote")))
    if remote_moved and local_moved:
        report["recommendation"] = (
            "Both sides changed. Decide which one is authoritative and force "
            "that direction; there is no merge.")
    elif remote_moved:
        report["recommendation"] = "pull before you push"
    elif local_moved and remote_block.get("exists"):
        report["recommendation"] = "push to send your changes up"
    return report


def push(db_path=None, force: bool = False, dirs: Optional[List[Path]] = None,
         wait_for_dirs: bool = True, sync_level: str = "checksum",
         progress=None) -> Dict[str, Any]:
    """Send the local database up, and optionally job directories with it."""
    from parslbox.local import compute

    from parslbox.local import transfer as _transfer
    _transfer.check_sync_level(sync_level)
    project = require_project(db_path)
    require_endpoint(project)
    if dirs:
        _transfer.require_collections(project)
    local_fp = guard.fingerprint(project.db_path)
    result: Dict[str, Any] = {"direction": "push",
                              "remote_db_path": project.remote_db_path}

    with compute.ComputeClient(project.compute_endpoint) as client:
        remote_fp = client.run(compute.remote_fingerprint, project.remote_db_path)

        if remote_fp.get("exists") and not remote_fp.get("error"):
            guard.check_identity(project.created, remote_fp)
            if not force:
                guard.check_unchanged("remote", remote_fp,
                                      _recorded(project, "remote"))

        # The database goes first so that the directory transfer has something
        # to aim at: remote_put_db creates remote_root, and the file it leaves
        # there is what proves the Transfer collection is on the same
        # filesystem as the endpoint.
        snapshot = snapshot_b64(project.db_path, compute.MAX_PAYLOAD_BYTES)
        put = client.run(compute.remote_put_db,
                         project.remote_db_path, snapshot["data"])
        result["bytes"] = put["bytes"]

        if dirs:
            from parslbox.local import transfer
            try:
                tc = transfer.client(project)
                if not project.transfer_remote_root:
                    root = transfer.detect_collection_root(tc, project, put["bytes"])
                    project.transfer_remote_root = root
                    save_project(project)
                    result["transfer_remote_root"] = root
                    result["transfer_remote_root_detected"] = True
                    if progress:
                        progress(f"collection is rooted at {root} — saved as "
                                 f"transfer_remote_root, not looked up again")
                task = transfer.push_directories(
                    project, dirs, label=f"parslbox push {project.local_root.name}",
                    sync_level=sync_level, transfer_client=tc)
                result["transfer"] = task
                if task.get("submitted"):
                    if progress:
                        progress("submitted " + ", ".join(task["task_ids"]))
                    if wait_for_dirs:
                        result["transfer"]["outcome"] = transfer.wait(
                            project, task["task_ids"], transfer_client=tc,
                            progress=progress)
            except Exception:
                # The database really did land, so the sync is recorded even
                # though the transfer did not finish. Skipping it would leave
                # the remote looking changed and block the next push.
                failed = client.run(compute.remote_fingerprint,
                                    project.remote_db_path)
                _record_sync(project, "push", local_fp, failed)
                raise

        after = client.run(compute.remote_fingerprint, project.remote_db_path)

    _record_sync(project, "push", local_fp, after)
    result["fingerprint"] = after
    result["last_sync"] = project.last_sync
    return result


def pull(db_path=None, force: bool = False) -> Dict[str, Any]:
    """Bring the remote database down over the local one."""
    from parslbox.local import compute

    project = require_project(db_path)
    require_endpoint(project)
    local_fp = guard.fingerprint(project.db_path)

    with compute.ComputeClient(project.compute_endpoint) as client:
        # Whether there is anything over there at all comes first: "push before
        # you pull" is more use than a guard complaint about the local side.
        remote_fp = client.run(compute.remote_fingerprint, project.remote_db_path)
        if not remote_fp.get("exists"):
            raise LocalProjectError(
                f"There is no database at {project.remote_db_path} yet. "
                f"Push before you pull.")
        guard.check_identity(project.created, remote_fp)
        if not force:
            guard.check_unchanged("local", local_fp, _recorded(project, "local"))
        payload = client.run(compute.remote_get_db,
                             project.remote_db_path, compute.MAX_PAYLOAD_BYTES)

    if not payload.get("exists"):
        raise LocalProjectError(
            f"The database at {project.remote_db_path} disappeared mid-pull.")
    if payload.get("too_big"):
        raise LocalProjectError(
            f"The remote database is {payload['gz_bytes'] / 1e6:.1f} MB "
            f"compressed, over what a Compute payload can carry. Fetch it with "
            f"Globus Transfer instead.")

    written = write_db_from_b64(project.db_path, payload["data"])
    new_fp = guard.fingerprint(project.db_path)
    guard.check_identity(project.created, new_fp)
    _record_sync(project, "pull", new_fp, new_fp)
    return {"direction": "pull", "bytes": written, "fingerprint": new_fp,
            "last_sync": project.last_sync}


def job_dirs(project: LocalProject, statuses=("Ready", "Restart"),
             apps=None, tags=None) -> Dict[str, List[str]]:
    """Local directories behind the jobs in the database.

    Scoped to Ready/Restart by default, to match what a submission will
    actually run. Re-running an edited job means flipping it back to Restart
    first, so the same set is the right one to send up.

    The returned counts are there to be printed: `total` is every job in the
    database and `matched` is how many survived the filters, so a push that
    sends 3 directories out of 400 jobs says so instead of looking like a
    success.

    Stored paths are remote paths, so they are mapped back through the local
    root. Anything outside the remote root is a remote-only location -- a
    shared input tree, say -- and is not ours to transfer.
    """
    from parslbox.database import database
    from parslbox.local.paths import to_local
    from parslbox.utils.tag_match import expand_tag_patterns

    total = len(database.get_jobs(project.db_path))
    jobs: List[Dict[str, Any]] = []
    for status_name in statuses:
        jobs += database.get_jobs(project.db_path, status=status_name)

    app_set = set(apps) if apps else None
    # Globs are accepted on --tags everywhere else in pbx; resolving them here
    # too keeps 'pbx local push --with-dirs --tags run*' from silently sending
    # nothing.
    tag_set = None
    if tags:
        try:
            tag_set = set(expand_tag_patterns(project.db_path, list(tags)))
        except ValueError as e:
            raise LocalProjectError(str(e))

    present: List[str] = []
    missing: List[str] = []
    outside: List[str] = []
    seen = set()
    for job in jobs:
        if app_set and job.get("app") not in app_set:
            continue
        if tag_set and job.get("tag") not in tag_set:
            continue
        remote_path = job.get("path")
        if remote_path in seen:
            continue
        seen.add(remote_path)
        local = to_local(remote_path, project)
        if local is None:
            outside.append(remote_path)
        elif local.is_dir():
            present.append(str(local))
        else:
            missing.append(str(local))
    return {"present": present, "missing": missing, "outside": outside,
            "statuses": list(statuses), "total": total, "matched": len(seen)}


def remote_run_dir(project: LocalProject, run_dir=None) -> str:
    """Where the submit script and scheduler log land on the remote machine.

    Under the project root rather than the remote home: it is the filesystem
    the jobs already run on, and it keeps everything for one project together.
    """
    from datetime import datetime
    from pathlib import PurePosixPath
    from parslbox.local.paths import is_under_local_root, to_remote

    if run_dir is not None:
        if is_under_local_root(run_dir, project):
            return to_remote(run_dir, project)
        return str(run_dir)
    stamp = datetime.now().strftime("%H%M%S_%d%m%y")
    return str(PurePosixPath(project.remote_root) / "pbx_runs" / stamp)


SUBMIT_REMEDY = (
    "\nOptions, safest first:\n"
    "  pbx local status        show what changed on the remote\n"
    "  pbx local pull          bring that work down, then submit again\n"
    "  pbx local push --force  overwrite the remote with your copy, "
    "discarding it\n"
    "\nA submission has no --force of its own. If an allocation is still "
    "running there, pull: pushing over a database it is writing to discards "
    "the jobs it has already finished."
)


def submit_remote(db_path, kwargs: Dict[str, Any], run_dir=None,
                  push_dirs: bool = False) -> Dict[str, Any]:
    """push -> submit on the remote -> pull.

    The push refuses if the remote database moved since the last sync, which
    is what stops a still-running allocation's progress from being overwritten
    by a stale local copy.
    """
    from parslbox.local import compute

    project = require_project(db_path)
    if not project.compute_endpoint:
        raise LocalProjectError(
            f"This is a local project but no Compute endpoint is configured.\n"
            f"Add compute_endpoint to {project.yaml_path}, or pass --no-local "
            f"to submit against the local database as an ordinary project."
        )

    steps: Dict[str, Any] = {}
    dirs = None
    if push_dirs:
        found = job_dirs(project, apps=kwargs.get("apps"), tags=kwargs.get("tags"))
        dirs = found["present"]
        steps["dirs"] = found

    try:
        steps["push"] = push(db_path=project.db_path, dirs=dirs)
    except guard.SyncConflict as e:
        raise guard.SyncConflict(e.problem, SUBMIT_REMEDY)

    target = remote_run_dir(project, run_dir)
    payload = {
        "db_path": project.remote_db_path,
        "config_path": project.remote_config,
        "run_dir": target,
        "kwargs": kwargs,
    }
    with compute.ComputeClient(project.compute_endpoint, timeout=600) as client:
        outcome = client.run(compute.remote_submit, payload)

    if not outcome.get("ok"):
        raise LocalProjectError(
            f"The remote submission failed: {outcome.get('error')}\n"
            f"The database was pushed, so nothing needs repeating before you "
            f"try again."
        )

    result = dict(outcome["result"])
    result["remote"] = True
    result["remote_run_dir"] = target
    result["endpoint"] = project.compute_endpoint
    result["push"] = steps["push"]
    if "dirs" in steps:
        result["dirs"] = steps["dirs"]

    # Brings back whatever an earlier allocation has done since the last sync.
    # It does not carry the new scheduler job id: that is written by `pbx run`
    # once the allocation starts, not by the submission itself.
    try:
        result["pull"] = pull(db_path=project.db_path, force=True)
    except Exception as e:
        result["pull_error"] = str(e)
    return result
