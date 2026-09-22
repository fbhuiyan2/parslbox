"""push and pull, with the remote side standing in as a second directory.

ComputeClient.run is redirected so the remote_* functions execute here instead
of on an endpoint. They are unchanged -- the same VACUUM INTO, the same file
replacement -- so everything below the network is exercised for real.
"""

from pathlib import Path

import pytest

from parslbox.database import database
from parslbox.local import compute, guard, sync, transfer
from parslbox.local.guard import SyncConflict
from parslbox.local.project import LOCAL_DB_NAME, LocalProjectError, init_project


@pytest.fixture
def wired(tmp_path, monkeypatch):
    local_root = tmp_path / "local"
    remote_root = tmp_path / "remote"
    local_root.mkdir()
    remote_root.mkdir()
    outcome = init_project(local_root, remote_root=str(remote_root),
                           compute_endpoint="stand-in-endpoint")
    monkeypatch.setattr(compute.ComputeClient, "run",
                        lambda self, fn, *args, **kw: fn(*args))
    return outcome["project"]


def add(db_path, path):
    return database.add_job(db_path=db_path, path=path, app="python", num_nodes=1,
                            ngpus=0, node_occupancy=1.0, tag=None, status="Ready")


def remote_db(project):
    return project.remote_db_path


# ------------------------------------------------------------------- push

def test_push_creates_the_remote_database(wired):
    add(wired.db_path, "/remote/a")
    result = sync.push(db_path=wired.db_path)
    assert result["fingerprint"]["count"] == 1
    assert guard.fingerprint(remote_db(wired))["count"] == 1


def test_push_records_the_sync(wired):
    add(wired.db_path, "/remote/a")
    sync.push(db_path=wired.db_path)
    from parslbox.local.project import load_project
    saved = load_project(wired.local_root)
    assert saved.last_sync["direction"] == "push"
    assert saved.last_sync["local"]["count"] == 1
    assert saved.last_sync["remote"]["count"] == 1


def test_second_push_is_allowed_when_the_remote_is_untouched(wired):
    add(wired.db_path, "/remote/a")
    sync.push(db_path=wired.db_path)
    add(wired.db_path, "/remote/b")
    result = sync.push(db_path=wired.db_path)
    assert result["fingerprint"]["count"] == 2


def test_push_refuses_when_the_remote_moved(wired):
    """The case this whole guard exists for: an allocation ran while away."""
    add(wired.db_path, "/remote/a")
    sync.push(db_path=wired.db_path)
    add(remote_db(wired), "/remote/ran_over_there")
    with pytest.raises(SyncConflict) as e:
        sync.push(db_path=wired.db_path)
    assert "remote" in str(e.value)


def test_force_pushes_over_a_moved_remote(wired):
    add(wired.db_path, "/remote/a")
    sync.push(db_path=wired.db_path)
    add(remote_db(wired), "/remote/ran_over_there")
    result = sync.push(db_path=wired.db_path, force=True)
    assert result["fingerprint"]["count"] == 1


def test_push_refuses_a_foreign_remote_database(wired, tmp_path):
    """A database from another project sitting at the remote root."""
    database.initialize_database(tmp_path / "remote" / LOCAL_DB_NAME)
    guard.stamp_identity(tmp_path / "remote" / LOCAL_DB_NAME, "some-other-project")
    add(wired.db_path, "/remote/a")
    with pytest.raises(SyncConflict) as e:
        sync.push(db_path=wired.db_path)
    assert "not this project's database" in str(e.value)


def test_pushing_replaces_stale_wal_sidecars(wired):
    """A leftover -wal would be replayed over the incoming database."""
    add(wired.db_path, "/remote/a")
    sync.push(db_path=wired.db_path)
    add(remote_db(wired), "/remote/b")
    stale = wired.remote_db_path + "-wal"
    import os
    assert os.path.exists(stale)
    sync.push(db_path=wired.db_path, force=True)
    assert not os.path.exists(stale)
    assert guard.fingerprint(remote_db(wired))["count"] == 1


# ------------------------------------------------------------------- pull

def test_pull_brings_remote_work_back(wired):
    add(wired.db_path, "/remote/a")
    sync.push(db_path=wired.db_path)
    job_id = add(remote_db(wired), "/remote/finished_over_there")
    database.update_jobs(db_path=remote_db(wired), job_ids=[job_id], status="Done")
    result = sync.pull(db_path=wired.db_path)
    assert result["fingerprint"]["count"] == 2
    statuses = {j["path"]: j["status"]
                for j in database.get_jobs(wired.db_path)}
    assert statuses["/remote/finished_over_there"] == "Done"


def test_pull_refuses_when_the_local_moved(wired):
    add(wired.db_path, "/remote/a")
    sync.push(db_path=wired.db_path)
    add(wired.db_path, "/remote/authored_here")
    with pytest.raises(SyncConflict) as e:
        sync.pull(db_path=wired.db_path)
    assert "local" in str(e.value)


def test_force_pulls_over_local_changes(wired):
    add(wired.db_path, "/remote/a")
    sync.push(db_path=wired.db_path)
    add(wired.db_path, "/remote/authored_here")
    result = sync.pull(db_path=wired.db_path, force=True)
    assert result["fingerprint"]["count"] == 1


def test_pull_before_any_push_says_so(wired):
    add(wired.db_path, "/remote/a")
    with pytest.raises(LocalProjectError) as e:
        sync.pull(db_path=wired.db_path)
    assert "Push before you pull" in str(e.value)


def test_round_trip_leaves_both_sides_agreeing(wired):
    add(wired.db_path, "/remote/a")
    sync.push(db_path=wired.db_path)
    sync.pull(db_path=wired.db_path)
    assert (guard.fingerprint(wired.db_path)["count"]
            == guard.fingerprint(remote_db(wired))["count"])


def test_push_then_pull_needs_no_force(wired):
    """After a push the local side is by definition unchanged."""
    add(wired.db_path, "/remote/a")
    sync.push(db_path=wired.db_path)
    sync.pull(db_path=wired.db_path)


# -------------------------------------------------------------- job dirs

def test_job_dirs_maps_stored_paths_back_to_this_machine(wired):
    present = wired.local_root / "run_a"
    present.mkdir()
    add(wired.db_path, f"{wired.remote_root}/run_a")
    add(wired.db_path, f"{wired.remote_root}/run_missing")
    add(wired.db_path, "/somewhere/else/entirely")
    found = sync.job_dirs(wired)
    assert found["present"] == [str(present)]
    assert len(found["missing"]) == 1
    assert found["outside"] == ["/somewhere/else/entirely"]


def test_job_dirs_honours_filters(wired):
    (wired.local_root / "run_a").mkdir()
    database.add_job(db_path=wired.db_path, path=f"{wired.remote_root}/run_a",
                     app="python", num_nodes=1, ngpus=0, node_occupancy=1.0,
                     tag="keep", status="Ready")
    assert sync.job_dirs(wired, tags=["keep"])["present"]
    # globs resolve here as they do for qsub/sbatch
    assert sync.job_dirs(wired, tags=["ke*"])["present"]
    # and a tag nothing carries is a typo, not an empty push
    with pytest.raises(LocalProjectError, match="not found"):
        sync.job_dirs(wired, tags=["other"])


# ------------------------------------------------------------- remote run

def test_remote_run_dir_defaults_under_the_remote_root(wired):
    target = sync.remote_run_dir(wired)
    assert target.startswith(f"{wired.remote_root}/pbx_runs/")


def test_remote_run_dir_translates_a_local_choice(wired):
    target = sync.remote_run_dir(wired, run_dir=wired.local_root / "myrun")
    assert target == f"{wired.remote_root}/myrun"


def test_remote_run_dir_leaves_an_outside_path_alone(wired):
    target = sync.remote_run_dir(wired, run_dir="/lus/flare/elsewhere")
    assert target == "/lus/flare/elsewhere"


def test_submit_remote_pushes_submits_and_pulls(wired, monkeypatch):
    add(wired.db_path, f"{wired.remote_root}/run_a")

    calls = []

    def fake_run(self, fn, *args, **kw):
        calls.append(fn.__name__)
        if fn is compute.remote_submit:
            payload = args[0]
            assert payload["db_path"] == wired.remote_db_path
            assert payload["kwargs"]["queue"] == "debug"
            return {"ok": True, "result": {"success": True, "job_id": "12345.aurora",
                                           "run_dir": payload["run_dir"],
                                           "submit_file": payload["run_dir"] + "/submit.sh"}}
        return fn(*args)

    monkeypatch.setattr(compute.ComputeClient, "run", fake_run)
    result = sync.submit_remote(
        wired.db_path,
        {"config_name": "aurora-gpu", "job_name": "t", "queue": "debug",
         "select": "1", "walltime": 10},
    )
    assert result["success"] is True
    assert result["remote"] is True
    assert result["job_id"] == "12345.aurora"
    assert "remote_put_db" in calls and "remote_submit" in calls
    assert "remote_get_db" in calls
    assert result["pull"]["fingerprint"]["count"] == 1


def test_submit_remote_reports_a_remote_failure(wired, monkeypatch):
    add(wired.db_path, f"{wired.remote_root}/run_a")

    def fake_run(self, fn, *args, **kw):
        if fn is compute.remote_submit:
            return {"ok": False, "error": "ValidationError: no such queue"}
        return fn(*args)

    monkeypatch.setattr(compute.ComputeClient, "run", fake_run)
    with pytest.raises(LocalProjectError) as e:
        sync.submit_remote(wired.db_path,
                           {"config_name": "aurora-gpu", "job_name": "t",
                            "queue": "nope", "select": "1", "walltime": 10})
    assert "no such queue" in str(e.value)


def test_submit_remote_restates_the_refusal_for_a_submission(wired, monkeypatch):
    """qsub has no --force, so the guard's usual advice would be a dead end."""
    add(wired.db_path, f"{wired.remote_root}/run_a")
    sync.push(db_path=wired.db_path)
    add(remote_db(wired), f"{wired.remote_root}/ran_over_there")

    submitted = []
    monkeypatch.setattr(compute.ComputeClient, "run",
                        lambda self, fn, *a, **kw: submitted.append(fn.__name__)
                        or fn(*a))
    with pytest.raises(SyncConflict) as e:
        sync.submit_remote(wired.db_path,
                           {"config_name": "aurora-gpu", "job_name": "t",
                            "queue": "debug", "select": "1", "walltime": 10})
    message = str(e.value)
    assert "has changed since the last sync" in message
    assert "pbx local pull" in message
    assert "pbx local push --force" in message
    assert "remote_submit" not in submitted


def test_submit_remote_goes_through_after_a_pull(wired, monkeypatch):
    """The remedy the refusal recommends first has to actually work."""
    add(wired.db_path, f"{wired.remote_root}/run_a")
    sync.push(db_path=wired.db_path)
    add(remote_db(wired), f"{wired.remote_root}/ran_over_there")

    def fake_run(self, fn, *args, **kw):
        if fn is compute.remote_submit:
            return {"ok": True, "result": {"success": True, "job_id": "9.aurora",
                                           "run_dir": args[0]["run_dir"],
                                           "submit_file": "x"}}
        return fn(*args)

    monkeypatch.setattr(compute.ComputeClient, "run", fake_run)
    sync.pull(db_path=wired.db_path)
    result = sync.submit_remote(
        wired.db_path,
        {"config_name": "aurora-gpu", "job_name": "t", "queue": "debug",
         "select": "1", "walltime": 10})
    assert result["job_id"] == "9.aurora"


def test_submit_remote_takes_no_force_or_sync_level(wired):
    """Both were unreachable from every layer; they are gone, not defaulted."""
    import inspect
    params = inspect.signature(sync.submit_remote).parameters
    assert "force" not in params and "sync_level" not in params


def test_submit_remote_without_an_endpoint_is_refused(tmp_path, monkeypatch):
    local_root = tmp_path / "local"
    local_root.mkdir()
    outcome = init_project(local_root, remote_root=str(tmp_path / "remote"))
    with pytest.raises(LocalProjectError) as e:
        sync.submit_remote(outcome["project"].db_path, {})
    assert "--no-local" in str(e.value)


def test_pull_into_a_fresh_project_needs_no_force(tmp_path, monkeypatch):
    """Re-creating a project from its YAML and pulling should just work."""
    local_root = tmp_path / "local"
    remote_root = tmp_path / "remote"
    local_root.mkdir()
    remote_root.mkdir()
    outcome = init_project(local_root, remote_root=str(remote_root),
                           compute_endpoint="stand-in-endpoint")
    project = outcome["project"]
    monkeypatch.setattr(compute.ComputeClient, "run",
                        lambda self, fn, *args, **kw: fn(*args))
    add(project.db_path, "/remote/a")
    sync.push(db_path=project.db_path)

    # Wipe the local side back to empty, as a fresh clone of the project would be.
    project.db_path.unlink()
    database.initialize_database(project.db_path)
    guard.stamp_identity(project.db_path, project.created)
    project.last_sync = None
    from parslbox.local.project import save_project
    save_project(project)

    result = sync.pull(db_path=project.db_path)
    assert result["fingerprint"]["count"] == 1


# ------------------------------------------------------- directory batching

class FakeTransferClient:
    """Records submissions instead of talking to Globus."""

    def __init__(self):
        self.submitted = []

    def submit_transfer(self, data):
        items = list(data.iter_items())
        self.submitted.append(items)
        return {"task_id": f"task-{len(self.submitted)}"}

    def task_wait(self, task_id, timeout=None, polling_interval=None):
        return True

    def get_task(self, task_id):
        return {"status": "SUCCEEDED", "files_transferred": 3,
                "bytes_transferred": 1000, "fatal_error": None}


@pytest.fixture
def transferable(tmp_path, monkeypatch):
    local_root = tmp_path / "local"
    local_root.mkdir()
    outcome = init_project(local_root, remote_root="/lus/flare/p",
                           compute_endpoint="e", transfer_local="src",
                           transfer_remote="dst")
    return outcome["project"]


def test_directories_are_batched(transferable):
    from parslbox.local import transfer
    dirs = [transferable.local_root / f"run_{i:04d}" for i in range(1200)]
    fake = FakeTransferClient()
    result = transfer.push_directories(transferable, dirs, batch_size=500,
                                       transfer_client=fake)
    assert result["batches"] == 3
    assert [len(b) for b in fake.submitted] == [500, 500, 200]
    assert result["count"] == 1200
    assert len(result["task_ids"]) == 3


def test_a_single_batch_is_one_task(transferable):
    from parslbox.local import transfer
    dirs = [transferable.local_root / "run_a"]
    fake = FakeTransferClient()
    result = transfer.push_directories(transferable, dirs, transfer_client=fake)
    assert result["batches"] == 1
    assert result["task_ids"] == ["task-1"]


def test_items_are_recursive_and_translated(transferable):
    from parslbox.local import transfer
    fake = FakeTransferClient()
    transfer.push_directories(transferable,
                              [transferable.local_root / "run_a"],
                              transfer_client=fake)
    item = fake.submitted[0][0]
    assert item["destination_path"] == "/lus/flare/p/run_a"
    assert item["recursive"] is True


def test_sync_level_reaches_the_request(transferable):
    from parslbox.local import transfer
    fake = FakeTransferClient()

    captured = {}
    original = fake.submit_transfer

    def spy(data):
        captured["sync_level"] = data["sync_level"]
        return original(data)

    fake.submit_transfer = spy
    transfer.push_directories(transferable, [transferable.local_root / "a"],
                              sync_level="mtime", transfer_client=fake)
    assert captured["sync_level"] == 2  # globus maps mtime -> 2


def test_nothing_to_transfer_submits_nothing(transferable):
    from parslbox.local import transfer
    fake = FakeTransferClient()
    result = transfer.push_directories(transferable, [], transfer_client=fake)
    assert result["submitted"] is False
    assert result["task_ids"] == []
    assert fake.submitted == []


def test_wait_combines_every_task(transferable):
    from parslbox.local import transfer
    fake = FakeTransferClient()
    outcome = transfer.wait(transferable, ["t1", "t2", "t3"], transfer_client=fake)
    assert outcome["done"] is True
    assert outcome["status"] == "SUCCEEDED"
    assert outcome["files_transferred"] == 9
    assert outcome["bytes_transferred"] == 3000


def test_missing_collections_are_reported(tmp_path):
    from parslbox.local import transfer
    local_root = tmp_path / "nocollections"
    local_root.mkdir()
    project = init_project(local_root, remote_root="/lus/flare/p")["project"]
    with pytest.raises(transfer.TransferError) as e:
        transfer.require_collections(project)
    assert "transfer_local" in str(e.value)


def test_a_bad_sync_level_is_rejected_before_logging_in(transferable):
    from parslbox.local import transfer
    with pytest.raises(transfer.TransferError) as e:
        transfer.push_directories(transferable, [transferable.local_root / "a"],
                                  sync_level="bogus",
                                  transfer_client=FakeTransferClient())
    assert "checksum" in str(e.value)


def test_push_checks_transfer_setup_before_touching_the_network(wired):
    """dirs requested but no collections configured: fail before any call."""
    add(wired.db_path, "/remote/a")
    with pytest.raises(Exception) as e:
        sync.push(db_path=wired.db_path, dirs=["/tmp/whatever"])
    assert "transfer_local" in str(e.value)


def test_remote_functions_close_over_nothing():
    """Globus Compute ships only a function's source text.

    Anything a remote_* function reads from module scope -- a constant, an
    import at the top of compute.py, a helper -- is simply absent on the
    endpoint, and the failure is a NameError at run time rather than anything
    a local test would catch. Every name each one uses has to be a parameter,
    a local, or imported inside its own body.
    """
    import ast

    tree = ast.parse(Path(compute.__file__).read_text())

    module_names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            module_names.update(
                t.id for t in node.targets if isinstance(t, ast.Name)
            )
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            module_names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            module_names.update(
                (a.asname or a.name).split(".")[0] for a in node.names
            )

    def bound_inside(fn):
        names = {a.arg for a in fn.args.args}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    names.update(
                        n.id for n in ast.walk(target) if isinstance(n, ast.Name)
                    )
            elif isinstance(node, (ast.AugAssign, ast.For, ast.comprehension)):
                names.update(
                    n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)
                )
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(
                    (a.asname or a.name).split(".")[0] for a in node.names
                )
            elif isinstance(node, ast.ExceptHandler) and node.name:
                names.add(node.name)
            elif isinstance(node, ast.withitem) and node.optional_vars:
                names.update(
                    n.id
                    for n in ast.walk(node.optional_vars)
                    if isinstance(n, ast.Name)
                )
        return names

    offenders = {}
    checked = []
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("remote_")):
            continue
        checked.append(node.name)
        local = bound_inside(node)
        used = {
            n.id
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        leaked = sorted((used - local) & module_names)
        if leaked:
            offenders[node.name] = leaked

    assert checked, "no remote_* functions found -- did compute.py move?"
    assert not offenders, (
        "these remote functions read module-level names that will not exist "
        f"on the endpoint: {offenders}"
    )


# --- data_access consent -------------------------------------------------
#
# A GCS mapped collection needs a per-collection data_access consent before any
# byte moves. A Globus Connect Personal collection must NOT be sent one -- the
# login fails outright with "Unknown Scope". Both ends of one transfer are
# usually different kinds, so getting this wrong breaks a real transfer in a
# way no stand-in client would reveal.

def test_gcs_mapped_collection_needs_data_access():
    assert transfer.needs_data_access("GCSv5_mapped_collection") is True


def test_personal_collection_does_not_need_data_access():
    # "GCP_mapped_collection" contains "mapped": a substring test would wrongly
    # request data_access for the user's own laptop and break their login.
    assert transfer.needs_data_access("GCP_mapped_collection") is False


def test_guest_collection_does_not_need_data_access():
    assert transfer.needs_data_access("GCSv5_guest_collection") is False


def test_high_assurance_never_gets_data_access():
    # data_access is not permitted on high-assurance collections at all.
    assert transfer.needs_data_access("GCSv5_mapped_collection",
                                      high_assurance=True) is False


class _FakeEndpointClient:
    def __init__(self, table):
        self.table = table
        self.scoped = None

    def get_endpoint(self, collection_id):
        if collection_id not in self.table:
            raise RuntimeError("no such collection")
        return self.table[collection_id]

    def add_app_data_access_scope(self, ids):
        self.scoped = list(ids)
        return self


def test_only_the_mapped_end_is_given_data_access():
    tc = _FakeEndpointClient({
        "local-gcp": {"entity_type": "GCP_mapped_collection"},
        "remote-gcs": {"entity_type": "GCSv5_mapped_collection"},
    })
    assert transfer._data_access_needed_for(
        tc, ("local-gcp", "remote-gcs")) == ["remote-gcs"]


def test_unreadable_collection_is_skipped_not_fatal():
    # Losing the lookup must not block the transfer: a genuine mapped
    # collection still fails later with ConsentRequired, which says so plainly.
    tc = _FakeEndpointClient({"remote-gcs": {"entity_type": "GCSv5_mapped_collection"}})
    assert transfer._data_access_needed_for(
        tc, ("vanished", "remote-gcs")) == ["remote-gcs"]


# --- collection-relative paths -------------------------------------------
#
# A Transfer collection need not expose the whole filesystem. ALCF's
# alcf#dtn_flare is rooted at /lus/flare/projects, so the job stored in the
# database as /lus/flare/projects/ABC/run_a must be handed to Globus as
# /ABC/run_a. Send the absolute path instead and Globus does not fail fast --
# it reports PERMISSION_DENIED and retries for the whole timeout, which looks
# exactly like a hang.

def test_collection_root_is_stripped_for_transfer():
    assert transfer.to_collection_path(
        "/lus/flare/projects/ABC/me/run_a", "/lus/flare/projects") == "/ABC/me/run_a"


def test_no_collection_root_leaves_the_path_alone():
    # A collection serving the whole filesystem -- Globus Connect Personal.
    assert transfer.to_collection_path(
        "/home/me/work/run_a", None) == "/home/me/work/run_a"


def test_collection_root_of_slash_is_a_no_op():
    assert transfer.to_collection_path("/a/b", "/") == "/a/b"


def test_collection_root_tolerates_trailing_slash():
    assert transfer.to_collection_path(
        "/lus/flare/projects/ABC/run_a", "/lus/flare/projects/") == "/ABC/run_a"


def test_the_collection_root_itself_maps_to_slash():
    assert transfer.to_collection_path(
        "/lus/flare/projects", "/lus/flare/projects") == "/"


def test_a_sibling_prefix_is_not_treated_as_inside():
    # /lus/flare/projects2 must not be mistaken for a child of .../projects
    with pytest.raises(transfer.TransferError):
        transfer.to_collection_path(
            "/lus/flare/projects2/ABC/run_a", "/lus/flare/projects")


def test_a_path_outside_the_collection_is_refused_with_a_reason():
    with pytest.raises(transfer.TransferError) as e:
        transfer.to_collection_path("/home/me/run_a", "/lus/flare/projects")
    assert "transfer_remote_root" in str(e.value)


class _RecordingTransferClient:
    """Captures what would be sent to Globus."""

    def __init__(self):
        self.submitted = []

    def submit_transfer(self, data):
        self.submitted.append(data)
        return {"task_id": f"task-{len(self.submitted)}"}


def test_push_addresses_directories_relative_to_the_collection(tmp_path):
    from parslbox.local.project import LocalProject

    project = LocalProject(
        local_root=tmp_path, created="2026-01-01T00:00:00Z",
        remote_root="/lus/flare/projects/ABC/me/p1",
        transfer_local="local-id", transfer_remote="remote-id",
        transfer_remote_root="/lus/flare/projects",
    )
    tc = _RecordingTransferClient()
    result = transfer.push_directories(
        project, [tmp_path / "run_a"], transfer_client=tc)

    assert result["submitted"]
    # the stored/absolute form stays out of the Globus request
    assert result["items"] == [(str(tmp_path / "run_a"), "/ABC/me/p1/run_a")]


# --- waiting on a transfer ------------------------------------------------

class _PollingClient:
    """A task that stays ACTIVE for a couple of polls, then finishes."""

    def __init__(self, polls_before_done=2, nice_status=None, final="SUCCEEDED"):
        self.remaining = polls_before_done
        self.nice_status = nice_status
        self.final = final
        self.waits = 0

    def task_wait(self, task_id, timeout=None, polling_interval=None):
        self.waits += 1
        if self.remaining > 0:
            self.remaining -= 1
            return False
        return True

    def get_task(self, task_id):
        active = self.remaining > 0 or self.final is None
        return {
            "task_id": task_id,
            "status": "ACTIVE" if active else self.final,
            "nice_status": self.nice_status,
            "files_transferred": 0 if active else 2,
            "files": 2,
            "bytes_transferred": 0 if active else 1024,
        }


def test_wait_reports_progress_while_a_task_runs(tmp_path):
    from parslbox.local.project import LocalProject
    project = LocalProject(local_root=tmp_path, created="2026-01-01T00:00:00Z",
                           remote_root="/r")
    seen = []
    out = transfer.wait(project, ["t1"], poll=1,
                        transfer_client=_PollingClient(2), progress=seen.append)
    assert out["status"] == "SUCCEEDED"
    assert len(seen) == 2, "one progress line per unfinished poll"
    assert "0/2 files" in seen[0]


def test_wait_surfaces_nice_status_the_moment_globus_reports_it(tmp_path):
    """PERMISSION_DENIED is what a wrong path looks like, and Globus retries it.

    Without this the operation is silent for the full timeout and the user
    cannot tell a slow copy from a broken one.
    """
    from parslbox.local.project import LocalProject
    project = LocalProject(local_root=tmp_path, created="2026-01-01T00:00:00Z",
                           remote_root="/r")
    seen = []
    transfer.wait(project, ["t1"], poll=1,
                  transfer_client=_PollingClient(1, nice_status="PERMISSION_DENIED"),
                  progress=seen.append)
    assert "PERMISSION_DENIED" in seen[0]


def test_wait_returns_nice_status_for_the_caller_to_render(tmp_path):
    from parslbox.local.project import LocalProject
    project = LocalProject(local_root=tmp_path, created="2026-01-01T00:00:00Z",
                           remote_root="/r")
    out = transfer.wait(project, ["t1"], poll=1,
                        transfer_client=_PollingClient(0, nice_status="PAUSED_BY_ADMIN"))
    assert out["tasks"][0]["nice_status"] == "PAUSED_BY_ADMIN"


def test_wait_without_a_progress_callback_stays_silent(tmp_path):
    # The API and MCP layers pass nothing; it must not blow up.
    from parslbox.local.project import LocalProject
    project = LocalProject(local_root=tmp_path, created="2026-01-01T00:00:00Z",
                           remote_root="/r")
    out = transfer.wait(project, ["t1"], poll=1, transfer_client=_PollingClient(1))
    assert out["done"] is True


# --- finding the collection root -----------------------------------------
#
# Globus will not say where a collection is rooted: get_endpoint reports
# host_path None and a default_directory of '/{server_default}/', and the
# 'absolute_path' in a listing is collection-relative like everything else. So
# the root is found by trying each prefix of remote_root and seeing which one
# the collection answers to.
#
# Answering is not on its own proof. ALCF project names repeat across
# filesystems, so /ABC/me/p1 resolves on both flare and eagle and the wrong
# collection would take the files without complaint. The database settles it:
# it was just written through the Compute endpoint, which runs on the machine
# that runs the jobs, so only a collection on that same filesystem can list it
# at that exact size.

DB = "job_database_pbx-local.db"


def _rooted(tmp_path, remote_root="/lus/flare/projects/ABC/me/p1"):
    from parslbox.local.project import LocalProject
    return LocalProject(
        local_root=tmp_path, created="2026-01-01T00:00:00Z",
        remote_root=remote_root,
        transfer_local="local-id", transfer_remote="remote-id")


class _FilesystemClient:
    """A collection publishing `root`, backed by a {path: {name: size}} map."""

    def __init__(self, root, tree):
        self.root = root.rstrip("/")
        self.tree = tree
        self.asked = []

    def operation_ls(self, collection_id, path):
        self.asked.append(path)
        absolute = (self.root + path).rstrip("/") or "/"
        if absolute not in self.tree:
            raise RuntimeError("Directory not found")
        return [{"name": n, "type": "file", "size": s}
                for n, s in self.tree[absolute].items()]


def test_the_root_is_found_by_the_database_it_contains(tmp_path):
    project = _rooted(tmp_path)
    tc = _FilesystemClient("/lus/flare/projects",
                           {"/lus/flare/projects/ABC/me/p1": {DB: 4096}})
    assert transfer.detect_collection_root(tc, project, 4096) == "/lus/flare/projects"


def test_a_collection_serving_the_whole_filesystem_resolves_to_slash(tmp_path):
    project = _rooted(tmp_path)
    tc = _FilesystemClient("", {"/lus/flare/projects/ABC/me/p1": {DB: 99}})
    assert transfer.detect_collection_root(tc, project, 99) == "/"
    # shallowest candidate first, so this costs exactly one call
    assert tc.asked == ["/lus/flare/projects/ABC/me/p1"]


def test_a_collection_rooted_at_the_project_itself_is_found(tmp_path):
    project = _rooted(tmp_path)
    tc = _FilesystemClient("/lus/flare/projects/ABC/me/p1", {"/lus/flare/projects/ABC/me/p1": {DB: 7}})
    assert transfer.detect_collection_root(
        tc, project, 7) == "/lus/flare/projects/ABC/me/p1"


def test_the_wrong_filesystem_is_refused_even_when_the_path_exists(tmp_path):
    # This is the case that a path-only check cannot catch: candidate root plus
    # remainder always reassembles into remote_root, so it says yes to eagle.
    project = _rooted(tmp_path)
    tc = _FilesystemClient("/lus/eagle/projects",
                           {"/lus/eagle/projects/ABC/me/p1": {"notes.txt": 12}})
    with pytest.raises(transfer.TransferError) as e:
        transfer.detect_collection_root(tc, project, 4096)
    assert "different filesystem" in str(e.value)


def test_a_database_of_the_wrong_size_is_not_accepted(tmp_path):
    project = _rooted(tmp_path)
    tc = _FilesystemClient("/lus/eagle/projects",
                           {"/lus/eagle/projects/ABC/me/p1": {DB: 512}})
    with pytest.raises(transfer.TransferError) as e:
        transfer.detect_collection_root(tc, project, 4096)
    assert "512 bytes, not 4096" in str(e.value)


def test_the_refusal_lists_every_path_it_tried(tmp_path):
    project = _rooted(tmp_path)
    tc = _FilesystemClient("/nowhere", {})
    with pytest.raises(transfer.TransferError) as e:
        transfer.detect_collection_root(tc, project, 1)
    message = str(e.value)
    for path in ("/lus/flare/projects/ABC/me/p1", "/ABC/me/p1", "/me/p1", "/p1"):
        assert path in message
    assert "transfer_remote_root" in message


def test_candidate_roots_are_every_prefix_shallowest_first():
    assert transfer.candidate_roots("/lus/flare/p") == [
        "/", "/lus", "/lus/flare", "/lus/flare/p"]


# --- detection inside push ------------------------------------------------

class _DiskCollection:
    """A collection publishing `root` of the real filesystem."""

    def __init__(self, root, fail_transfer=False):
        self.root = str(root).rstrip("/")
        self.fail_transfer = fail_transfer
        self.asked = []
        self.submitted = []

    def operation_ls(self, collection_id, path):
        self.asked.append(path)
        d = Path(self.root + path)
        if not d.is_dir():
            raise RuntimeError("Directory not found")
        return [{"name": f.name, "type": "file" if f.is_file() else "dir",
                 "size": f.stat().st_size if f.is_file() else None}
                for f in d.iterdir()]

    def submit_transfer(self, data):
        if self.fail_transfer:
            raise RuntimeError("Globus said no")
        self.submitted.append(data)
        return {"task_id": "t1"}

    def task_wait(self, task_id, timeout=None, polling_interval=None):
        return True

    def get_task(self, task_id):
        return {"status": "SUCCEEDED", "files_transferred": 1,
                "bytes_transferred": 10, "fatal_error": None}


@pytest.fixture
def wired_transfer(wired, tmp_path, monkeypatch):
    wired.transfer_local = "src"
    wired.transfer_remote = "dst"
    from parslbox.local.project import save_project
    save_project(wired)
    return wired


def _collection(monkeypatch, tc):
    monkeypatch.setattr(transfer, "client", lambda project: tc)
    return tc


def test_the_first_directory_push_finds_and_saves_the_root(
        wired_transfer, tmp_path, monkeypatch):
    from parslbox.local.project import load_project
    (wired_transfer.local_root / "run_a").mkdir()
    add(wired_transfer.db_path, "/remote/run_a")
    tc = _collection(monkeypatch, _DiskCollection(tmp_path))

    result = sync.push(db_path=wired_transfer.db_path,
                       dirs=[wired_transfer.local_root / "run_a"])

    assert result["transfer_remote_root_detected"] is True
    assert result["transfer_remote_root"] == str(tmp_path)
    assert load_project(wired_transfer.local_root).transfer_remote_root == str(tmp_path)
    # and the directory was addressed relative to that root
    assert result["transfer"]["items"] == [
        (str(wired_transfer.local_root / "run_a"), "/remote/run_a")]


def test_a_root_already_in_the_project_file_is_not_looked_up_again(
        wired_transfer, tmp_path, monkeypatch):
    from parslbox.local.project import save_project
    wired_transfer.transfer_remote_root = str(tmp_path)
    save_project(wired_transfer)
    (wired_transfer.local_root / "run_a").mkdir()
    add(wired_transfer.db_path, "/remote/run_a")
    tc = _collection(monkeypatch, _DiskCollection(tmp_path))

    result = sync.push(db_path=wired_transfer.db_path,
                       dirs=[wired_transfer.local_root / "run_a"])

    assert "transfer_remote_root_detected" not in result
    assert tc.asked == []


def test_detection_sees_the_database_this_push_just_wrote(
        wired_transfer, tmp_path, monkeypatch):
    # Nothing is in remote_root before the first push, so the ordering matters:
    # the database has to go up over Compute before the collection is listed.
    (wired_transfer.local_root / "run_a").mkdir()
    add(wired_transfer.db_path, "/remote/run_a")
    _collection(monkeypatch, _DiskCollection(tmp_path))
    assert not (Path(wired_transfer.remote_root) / LOCAL_DB_NAME).exists()

    sync.push(db_path=wired_transfer.db_path,
              dirs=[wired_transfer.local_root / "run_a"])
    assert (Path(wired_transfer.remote_root) / LOCAL_DB_NAME).exists()


def test_a_failed_transfer_still_records_the_database_that_landed(
        wired_transfer, tmp_path, monkeypatch):
    # The database goes first now, so a transfer that dies afterwards must not
    # leave the remote looking changed -- that would block the retry.
    from parslbox.local.project import load_project
    (wired_transfer.local_root / "run_a").mkdir()
    add(wired_transfer.db_path, "/remote/run_a")
    _collection(monkeypatch, _DiskCollection(tmp_path, fail_transfer=True))

    with pytest.raises(RuntimeError):
        sync.push(db_path=wired_transfer.db_path,
                  dirs=[wired_transfer.local_root / "run_a"])

    saved = load_project(wired_transfer.local_root)
    assert saved.last_sync["remote"]["count"] == 1
    # so the next attempt is not refused as "the remote moved"
    _collection(monkeypatch, _DiskCollection(tmp_path))
    sync.push(db_path=wired_transfer.db_path,
              dirs=[wired_transfer.local_root / "run_a"])
