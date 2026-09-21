"""Running parslbox on the remote machine through a Globus Compute endpoint.

The functions prefixed with `remote_` execute on the endpoint, not here. They
are shipped as source text (see ComputeClient._serializer), so each one has to
be self-contained: every import lives inside the function body, and only
plain data crosses the wire.

The database itself travels through here rather than through Globus Transfer.
A gzipped SQLite snapshot of a few thousand jobs is well under a megabyte,
against a 10 MB cap on a Compute payload, and it means push/pull need only an
endpoint -- no Transfer collection on either end. Job *directories* are a
different matter and go through transfer.py.
"""

from typing import Any, Callable, Dict, Optional

# Globus caps task payloads and results at 10 MB. Stay clear of the edge.
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024


class ComputeError(Exception):
    """A remote call failed, or the endpoint could not be reached."""


# --------------------------------------------------------------------------
# These run on the endpoint.
# --------------------------------------------------------------------------

def remote_ping() -> dict:
    """Report what is installed on the endpoint."""
    import os
    import platform
    out = {
        "python": platform.python_version(),
        "host": platform.node(),
        "cwd": os.getcwd(),
    }
    try:
        from importlib.metadata import version
        out["parslbox"] = version("parslbox")
    except Exception as e:
        out["parslbox"] = f"not installed: {e}"
    return out


def remote_fingerprint(db_path: str) -> dict:
    """MAX(timestamp), COUNT(*) and the project id of the remote database."""
    import os
    import sqlite3
    if not os.path.isfile(db_path):
        return {"exists": False}
    con = sqlite3.connect(db_path)
    try:
        ts, count = con.execute("SELECT MAX(timestamp), COUNT(*) FROM jobs").fetchone()
        identity = con.execute("PRAGMA user_version").fetchone()[0]
    except sqlite3.DatabaseError as e:
        return {"exists": True, "error": str(e)}
    finally:
        con.close()
    return {"exists": True, "max_timestamp": ts, "count": count, "identity": identity}


def remote_get_db(db_path: str, max_bytes: int) -> dict:
    """Snapshot the remote database and hand it back gzipped and base64'd.

    VACUUM INTO rather than a plain file read: the database is in WAL mode and
    a live copy would be stale or torn, which is exactly what a pull in the
    middle of a run would produce.
    """
    import base64
    import gzip
    import os
    import shutil
    import sqlite3
    import tempfile
    if not os.path.isfile(db_path):
        return {"exists": False}
    tmpdir = tempfile.mkdtemp(prefix="pbx-snap-")
    snapshot = os.path.join(tmpdir, "snapshot.db")
    try:
        con = sqlite3.connect(db_path)
        try:
            con.execute("VACUUM INTO ?", (snapshot,))
        finally:
            con.close()
        with open(snapshot, "rb") as fh:
            raw = fh.read()
        blob = gzip.compress(raw)
        if len(blob) > max_bytes:
            return {"exists": True, "too_big": True,
                    "raw_bytes": len(raw), "gz_bytes": len(blob)}
        return {"exists": True, "too_big": False, "raw_bytes": len(raw),
                "gz_bytes": len(blob), "data": base64.b64encode(blob).decode()}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def remote_put_db(db_path: str, data_b64: str) -> dict:
    """Replace the remote database with the one sent from the laptop."""
    import base64
    import gzip
    import os
    blob = gzip.decompress(base64.b64decode(data_b64))
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    incoming = db_path + ".incoming"
    with open(incoming, "wb") as fh:
        fh.write(blob)
    # The sidecars belong to the database being replaced. Left in place,
    # SQLite would replay a stale write-ahead log over the new file.
    for suffix in ("-wal", "-shm"):
        stale = db_path + suffix
        if os.path.exists(stale):
            os.remove(stale)
    os.replace(incoming, db_path)
    return {"bytes": len(blob), "path": db_path}


def remote_submit(payload: dict) -> dict:
    """Run submit_job() on the remote machine and hand back its result dict."""
    import os
    from pathlib import Path
    os.environ["PBX_DB_PATH"] = payload["db_path"]
    if payload.get("config_path"):
        os.environ["PBX_CONFIG_PATH"] = payload["config_path"]

    kwargs = dict(payload["kwargs"])
    kwargs["db_path"] = Path(payload["db_path"])
    if payload.get("config_path"):
        kwargs["config_path"] = Path(payload["config_path"])
    if payload.get("run_dir"):
        # No chdir: submit_job already runs the scheduler with cwd=run_dir, and
        # a worker process is reused, so changing its directory would leak into
        # whatever task lands on it next.
        kwargs["run_dir"] = Path(payload["run_dir"])

    try:
        from parslbox.commands.helpers.submit_helpers import submit_job
        return {"ok": True, "result": submit_job(**kwargs)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def remote_listdir(path: str) -> dict:
    """Does this directory exist on the remote, and what is in it?"""
    import os
    if not os.path.isdir(path):
        return {"exists": False}
    try:
        entries = sorted(os.listdir(path))
    except OSError as e:
        return {"exists": True, "error": str(e)}
    return {"exists": True, "count": len(entries), "entries": entries[:50]}


# --------------------------------------------------------------------------
# This runs here.
# --------------------------------------------------------------------------

def _sdk():
    """The optional dependency, as a ComputeError if it is not installed."""
    try:
        import globus_compute_sdk
        from globus_compute_sdk import serialize  # noqa: F401
    except ImportError:
        raise ComputeError(
            "Globus Compute needs the 'remote' extra, which is not installed.\n"
            "    poetry install --extras remote\n"
            "Every 'pbx local' command reaches the remote machine through a "
            "Compute endpoint, so none of them work without it."
        )
    return globus_compute_sdk


class ComputeClient:
    """One Globus Compute endpoint, one Executor, reused across calls.

    Functions are serialized as source text rather than as dill bytecode.
    dill needs the client and the worker to agree on the Python MAJOR.MINOR
    version; source text does not, and the laptop is rarely on the same
    Python as an HPC login node.
    """

    def __init__(self, endpoint_id: str, timeout: int = 300):
        if not endpoint_id:
            raise ComputeError(
                "No Globus Compute endpoint is configured for this project. "
                "Add compute_endpoint to .pbxlocal.yaml, or re-run "
                "'pbx local init' in a fresh directory."
            )
        self.endpoint_id = str(endpoint_id)
        self.timeout = timeout
        self._executor = None

    def _serializer(self):
        serialize = _sdk().serialize
        return serialize.ComputeSerializer(
            strategy_code=serialize.PureSourceTextInspect()
        )

    @property
    def executor(self):
        if self._executor is None:
            Executor = _sdk().Executor
            self._executor = Executor(
                endpoint_id=self.endpoint_id,
                serializer=self._serializer(),
            )
        return self._executor

    def run(self, fn: Callable, *args, timeout: Optional[int] = None) -> Any:
        try:
            future = self.executor.submit(fn, *args)
            return future.result(timeout=timeout or self.timeout)
        except ComputeError:
            raise
        except Exception as e:
            raise ComputeError(f"{fn.__name__} on endpoint {self.endpoint_id[:8]}… "
                               f"failed: {type(e).__name__}: {e}")

    def close(self) -> None:
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            finally:
                self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def ping(endpoint_id: str, timeout: int = 60) -> Dict[str, Any]:
    """Reachability check. Returns {'reachable': bool, ...}."""
    try:
        with ComputeClient(endpoint_id, timeout=timeout) as client:
            info = client.run(remote_ping)
        return {"reachable": True, **info}
    except Exception as e:
        return {"reachable": False, "error": str(e)}
