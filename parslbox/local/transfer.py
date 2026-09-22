"""Moving job directories to the remote machine with Globus Transfer.

Only directories come through here. The database travels inside the Compute
call (see compute.py), which keeps push and pull working for anyone who has
an endpoint but has not set up Transfer collections on both ends.
"""

import os
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional

from parslbox.local.paths import to_remote
from parslbox.local.project import LOCAL_DB_NAME, LocalProject

TRANSFER_SCOPES = ("urn:globus:auth:scope:transfer.api.globus.org:all",)

# What Globus will accept for "is this file already there?", cheapest first.
SYNC_LEVELS = ("exists", "size", "mtime", "checksum")


class TransferError(Exception):
    """A Globus Transfer call failed, or Transfer is not configured."""


def _sdk():
    """The optional dependency, as a TransferError if it is not installed."""
    try:
        import globus_sdk
    except ImportError:
        raise TransferError(
            "Globus Transfer needs the 'remote' extra, which is not installed:\n"
            "    poetry install --extras remote\n"
            "The database syncs over the Compute endpoint without it; only job "
            "directories need Transfer."
        )
    return globus_sdk


def check_sync_level(sync_level: str) -> str:
    """Reject a bad value here, before anything triggers a Globus login."""
    if sync_level not in SYNC_LEVELS:
        raise TransferError(
            f"Unknown sync level {sync_level!r}. Choose one of: "
            f"{', '.join(SYNC_LEVELS)}."
        )
    return sync_level


def _client_id(project: LocalProject) -> str:
    client_id = os.getenv("PBX_GLOBUS_CLIENT_ID")
    if client_id:
        return client_id
    raise TransferError(
        "Globus Transfer needs a registered client id to log in with.\n"
        "Create a Native App once at https://app.globus.org/settings/developers "
        "and export its id:\n"
        '    export PBX_GLOBUS_CLIENT_ID="<uuid>"\n'
        "Transfer is only needed to move job directories; the database syncs "
        "over the Compute endpoint without it."
    )


def require_collections(project: LocalProject) -> None:
    missing = [name for name, value in
               (("transfer_local", project.transfer_local),
                ("transfer_remote", project.transfer_remote)) if not value]
    if missing:
        raise TransferError(
            f"This project has no Globus Transfer collections configured "
            f"({', '.join(missing)} missing from {project.yaml_path}).\n"
            f"Add them to move job directories automatically, or copy the "
            f"directories to {project.remote_root} yourself."
        )


# Exactly the GCS server-side mapped collection. Not a substring test: a
# Globus Connect Personal collection reports "GCP_mapped_collection", which
# also contains "mapped" but must never be sent a data_access request.
GCS_MAPPED = "GCSv5_mapped_collection"


def needs_data_access(entity_type, high_assurance: bool = False) -> bool:
    """Does this collection require a per-collection data_access consent?

    GCS mapped collections do: logging in is not enough, each one needs its own
    consent before any byte can be read or written. Personal and guest
    collections do not, and asking for it there fails the login outright with
    "Unknown Scope" -- so the two ends of one transfer often need opposite
    treatment. High-assurance collections cannot use data_access at all; those
    need a guest collection instead, which is out of scope here.
    """
    if high_assurance:
        return False
    return str(entity_type) == GCS_MAPPED


def _data_access_needed_for(tc, collection_ids) -> List[str]:
    """Ask Globus what each collection is, and pick the ones needing consent."""
    needed = []
    for collection_id in collection_ids:
        if not collection_id:
            continue
        try:
            ep = tc.get_endpoint(collection_id)
        except Exception:
            # Not fatal: if this is genuinely a mapped collection the transfer
            # fails later with ConsentRequired, which names the problem clearly.
            continue
        if needs_data_access(ep.get("entity_type"), bool(ep.get("high_assurance"))):
            needed.append(str(collection_id))
    return needed


def to_collection_path(remote_path: str, collection_root: Optional[str]) -> str:
    """Rewrite an absolute remote path as the Transfer collection sees it.

    A collection does not have to expose the whole filesystem. ALCF's
    alcf#dtn_flare, for example, is rooted at /lus/flare/projects, so the job
    living at /lus/flare/projects/ABC/you/run_a is addressed as /ABC/you/run_a.
    Everything else in parslbox -- the database, the scheduler, the job itself
    -- wants the absolute path, so the two forms have to be kept apart and only
    Transfer destinations get this treatment.

    With no collection_root configured the path is already correct: that is the
    case for a collection rooted at /, including Globus Connect Personal.
    """
    if not collection_root:
        return remote_path
    root = "/" + str(collection_root).strip("/")
    if root == "/":
        return remote_path
    if remote_path == root:
        return "/"
    if not remote_path.startswith(root + "/"):
        raise TransferError(
            f"{remote_path} is not inside the Transfer collection's root "
            f"({root}), so Globus cannot address it.\n"
            f"Check transfer_remote_root in the project file -- it must be the "
            f"filesystem path that the collection exposes as its own '/'."
        )
    return remote_path[len(root):]


def _ls_names(tc, collection_id: str, path: str):
    """Directory listing as {name: size}, or None if the path is not there."""
    try:
        listing = tc.operation_ls(collection_id, path=path)
    except Exception:
        return None
    return {item["name"]: item.get("size") for item in listing
            if item.get("type") == "file"}


def candidate_roots(remote_root: str) -> List[str]:
    """Every path the collection could be rooted at, shallowest first.

    The collection root has to be some prefix of remote_root, so there are only
    as many candidates as the path has components. Shallowest first means a
    collection that exposes the whole filesystem -- Globus Connect Personal,
    usually -- is settled on the first try.
    """
    parts = [p for p in PurePosixPath(remote_root).parts if p != "/"]
    return ["/" + "/".join(parts[:k]) if k else "/" for k in range(len(parts) + 1)]


def detect_collection_root(tc, project: LocalProject, db_bytes: int) -> str:
    """Find which prefix of remote_root the collection publishes as its '/'.

    Globus will not say: a collection reports host_path None, a default
    directory of '/{server_default}/', and an 'absolute_path' in listings that
    is collection-relative like everything else. So the root is found by trying
    each candidate and looking at what comes back.

    Finding a directory is not enough on its own. Project names repeat across
    ALCF filesystems, so /CSTEELML/you/run resolves on both flare and eagle, and
    a collection for the wrong one would answer happily and take the files.
    What settles it is the database: it was just written through the Compute
    endpoint, which is on the machine that runs the jobs by definition, so the
    listing that shows that file at that exact size is the right filesystem.
    """
    tried = []
    for root in candidate_roots(project.remote_root):
        relative = to_collection_path(project.remote_root, root)
        names = _ls_names(tc, project.transfer_remote, relative)
        if names is None:
            tried.append(f"  {relative}  (not found)")
            continue
        size = names.get(LOCAL_DB_NAME)
        if size == db_bytes:
            return root
        tried.append(
            f"  {relative}  (exists, but {LOCAL_DB_NAME} is "
            + (f"{size} bytes, not {db_bytes}" if size is not None else "not in it")
            + ")")

    raise TransferError(
        f"Could not find {project.remote_root} through Transfer collection "
        f"{project.transfer_remote}.\n"
        f"The database was just written there over the Compute endpoint, so a "
        f"collection on the same filesystem would list it. Paths tried:\n"
        + "\n".join(tried) + "\n"
        f"Either the collection is on a different filesystem than the Compute "
        f"endpoint, or it publishes a subtree that does not contain "
        f"{project.remote_root}. Set transfer_remote_root in "
        f"{project.yaml_path} by hand to skip this check."
    )


def client(project: LocalProject):
    globus_sdk = _sdk()
    require_collections(project)
    try:
        app = globus_sdk.UserApp("parslbox", client_id=_client_id(project))
        tc = globus_sdk.TransferClient(app=app)
        mapped = _data_access_needed_for(
            tc, (project.transfer_local, project.transfer_remote))
        if mapped:
            tc.add_app_data_access_scope(mapped)
    except TransferError:
        raise
    except Exception as e:
        raise TransferError(f"Could not log in to Globus Transfer: {e}")
    return tc


def push_directories(
    project: LocalProject,
    local_dirs: Iterable[Path],
    label: str = "parslbox push",
    sync_level: str = "checksum",
    batch_size: int = 500,
    transfer_client=None,
) -> Dict[str, Any]:
    """Send job directories up, in batches of `batch_size` items per task.

    Each directory is one recursive item. Neither the Transfer API docs nor the
    SDK state a cap on items per task, but a submission naming several thousand
    directories is a large request whose failure would come back as an opaque
    API error, so the work is split across tasks. Globus runs them
    concurrently; wait() blocks on all of them.

    sync_level='checksum' is the safe default and makes a repeat push cheap in
    bytes -- but it reads and checksums every file on both ends first. Drop to
    'mtime' when the directories have grown large output files that the remote
    wrote itself.
    """
    globus_sdk = _sdk()

    check_sync_level(sync_level)
    local_dirs = [Path(d) for d in local_dirs]
    if not local_dirs:
        return {"submitted": False, "reason": "nothing to transfer",
                "count": 0, "task_ids": []}

    tc = transfer_client or client(project)
    batches = [local_dirs[i:i + batch_size]
               for i in range(0, len(local_dirs), batch_size)]

    task_ids = []
    pairs = []
    for n, batch in enumerate(batches, start=1):
        suffix = f" ({n}/{len(batches)})" if len(batches) > 1 else ""
        data = globus_sdk.TransferData(
            source_endpoint=project.transfer_local,
            destination_endpoint=project.transfer_remote,
            label=(label + suffix)[:128],
            sync_level=sync_level,
            verify_checksum=True,
            notify_on_succeeded=False,
            notify_on_failed=True,
        )
        for d in batch:
            absolute = to_remote(d, project)
            destination = to_collection_path(absolute, project.transfer_remote_root)
            data.add_item(str(d), destination, recursive=True)
            pairs.append((str(d), destination))
        try:
            task = tc.submit_transfer(data)
        except globus_sdk.TransferAPIError as e:
            raise TransferError(
                f"Globus Transfer refused batch {n} of {len(batches)} "
                f"({len(batch)} directories): {e.message}"
                + (f"\nAlready submitted: {', '.join(task_ids)}" if task_ids else "")
            )
        task_ids.append(task["task_id"])

    return {"submitted": True, "task_ids": task_ids, "batches": len(batches),
            "count": len(pairs), "items": pairs}


def wait(project: LocalProject, task_ids, timeout: int = 1800,
         poll: int = 10, transfer_client=None, progress=None) -> Dict[str, Any]:
    """Block until every transfer task finishes. Returns the combined outcome.

    `progress` is called with a one-line status roughly every `poll` seconds.
    Supply it from anything with a user watching: Globus retries a task it
    cannot complete -- a wrong path is reported as PERMISSION_DENIED and tried
    again every couple of minutes -- so a silent wait is indistinguishable from
    a hang, for up to `timeout`. nice_status is what names the real problem.
    """
    import time

    if isinstance(task_ids, str):
        task_ids = [task_ids]
    tc = transfer_client or client(project)
    deadline = time.monotonic() + timeout
    try:
        return _poll(tc, task_ids, deadline, poll, progress)
    except TransferError:
        raise
    except Exception as e:
        raise TransferError(
            f"Lost track of the Globus transfer while waiting: {e}\n"
            f"The task itself is unaffected -- check it at "
            f"https://app.globus.org/activity."
        )


def _poll(tc, task_ids, deadline, poll, progress) -> Dict[str, Any]:
    import time

    outcomes = []
    for task_id in task_ids:
        done = False
        while not done:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            slice_ = max(1, int(min(poll, left)))
            done = bool(tc.task_wait(task_id, timeout=slice_,
                                     polling_interval=min(poll, slice_)))
            if not done and progress:
                t = tc.get_task(task_id)
                note = t.get("nice_status")
                progress(
                    f"{task_id[:8]}… {t.get('status')} "
                    f"{t.get('files_transferred') or 0}/{t.get('files') or 0} files"
                    + (f" — {note}" if note else "")
                )
        task = tc.get_task(task_id)
        outcomes.append({
            "task_id": task_id,
            "done": bool(done),
            "status": task["status"],
            "nice_status": task.get("nice_status"),
            "files_transferred": task.get("files_transferred"),
            "bytes_transferred": task.get("bytes_transferred"),
            "fatal_error": task.get("fatal_error"),
        })

    statuses = {o["status"] for o in outcomes}
    return {
        "done": all(o["done"] for o in outcomes),
        "status": "SUCCEEDED" if statuses == {"SUCCEEDED"} else "/".join(sorted(statuses)),
        "files_transferred": sum(o["files_transferred"] or 0 for o in outcomes),
        "bytes_transferred": sum(o["bytes_transferred"] or 0 for o in outcomes),
        "tasks": outcomes,
    }
