"""API surface for local projects, and the path translation it inherits."""

import pytest

from parslbox.api import ParslBox, ValidationError
from parslbox.local.project import LOCAL_DB_NAME, PROJECT_FILE, load_project

REMOTE = "/lus/flare/projects/CSTEELML/fbhuiyan/proj1"


@pytest.fixture
def local_pbx(tmp_path):
    """A ParslBox pointed at a freshly created local project."""
    ParslBox(db_path=tmp_path / "bootstrap.db").local_init(
        directory=tmp_path, remote_root=REMOTE)
    return ParslBox(db_path=tmp_path / LOCAL_DB_NAME)


# --------------------------------------------------------------- local_init

def test_local_init_creates_a_project(pbx, tmp_path):
    result = pbx.local_init(directory=tmp_path, remote_root=REMOTE)
    assert result["action"] == "created"
    assert result["db_path"] == str(tmp_path / LOCAL_DB_NAME)
    assert result["export_line"].startswith("export PBX_DB_PATH=")
    assert (tmp_path / PROJECT_FILE).is_file()


def test_local_init_is_idempotent(pbx, tmp_path):
    first = pbx.local_init(directory=tmp_path, remote_root=REMOTE)
    again = pbx.local_init(directory=tmp_path, remote_root=REMOTE)
    assert again["action"] == "existing"
    assert again["db_path"] == first["db_path"]


def test_local_init_records_the_endpoint(pbx, tmp_path):
    pbx.local_init(directory=tmp_path, remote_root=REMOTE,
                   compute_endpoint="abc-123", remote_config="/home/u/config.yaml")
    proj = load_project(tmp_path)
    assert proj.compute_endpoint == "abc-123"
    assert proj.remote_config == "/home/u/config.yaml"


def test_local_init_refuses_nesting_by_default(pbx, tmp_path):
    pbx.local_init(directory=tmp_path, remote_root=REMOTE)
    inner = tmp_path / "inner"
    inner.mkdir()
    with pytest.raises(ValidationError) as e:
        pbx.local_init(directory=inner, remote_root=REMOTE + "/inner")
    assert str(tmp_path) in str(e.value)


def test_local_init_allows_nesting_when_asked(pbx, tmp_path):
    pbx.local_init(directory=tmp_path, remote_root=REMOTE)
    inner = tmp_path / "inner"
    inner.mkdir()
    result = pbx.local_init(directory=inner, remote_root=REMOTE + "/inner",
                            nested_ok=True)
    assert result["action"] == "created"


def test_local_init_refuses_a_relative_remote_root(pbx, tmp_path):
    with pytest.raises(ValidationError):
        pbx.local_init(directory=tmp_path, remote_root="relative/path")


# ------------------------------------------------------------- local_status

def test_local_status_on_an_ordinary_database(pbx):
    report = pbx.local_status(check_remote=False)
    assert report["is_local_project"] is False


def test_local_status_on_a_local_project(local_pbx, tmp_path):
    report = local_pbx.local_status(check_remote=False)
    assert report["is_local_project"] is True
    assert report["remote_root"] == REMOTE
    assert report["local_root"] == str(tmp_path)
    assert report["job_count"] == 0
    assert report["local"]["drift"] == "empty, never synced"


def test_local_status_reports_a_missing_endpoint(local_pbx):
    report = local_pbx.local_status(check_remote=True)
    assert report["remote"]["checked"] is False
    assert "compute_endpoint" in report["remote"]["error"]


# ------------------------------------------------------- push / pull guards

def test_push_without_an_endpoint_is_a_validation_error(local_pbx):
    with pytest.raises(ValidationError):
        local_pbx.local_push()


def test_pull_on_an_ordinary_database_is_a_validation_error(pbx):
    with pytest.raises(ValidationError) as e:
        pbx.local_pull()
    assert "Not a local project" in str(e.value)


# ------------------------------------------------- paths stored by add_jobs

def test_add_jobs_stores_remote_paths(local_pbx, tmp_path):
    run_a = tmp_path / "run_a"
    run_a.mkdir()
    job_ids, failed, _ = local_pbx.add_jobs(
        paths=[str(run_a)], app="python", config="aurora-gpu", input_file="x.py",
        env_file="pass")
    assert not failed
    job = local_pbx.get_job(job_ids[0])
    assert job["path"] == f"{REMOTE}/run_a"


def test_add_jobs_still_requires_the_directory_to_exist(local_pbx, tmp_path):
    _, failed, _ = local_pbx.add_jobs(
        paths=[str(tmp_path / "never_created")], app="python",
        config="aurora-gpu", input_file="x.py", env_file="pass")
    assert failed
    assert "does not exist" in failed[0][1]


def test_add_jobs_passes_through_a_path_outside_the_root(local_pbx, tmp_path):
    outside = "/lus/flare/shared/precomputed"
    job_ids, failed, _ = local_pbx.add_jobs(
        paths=[outside], app="python", config="aurora-gpu", input_file="x.py",
        env_file="pass")
    assert not failed
    assert local_pbx.get_job(job_ids[0])["path"] == outside


def test_env_file_outside_the_root_is_not_checked(local_pbx, tmp_path):
    run_a = tmp_path / "run_a"
    run_a.mkdir()
    job_ids, failed, _ = local_pbx.add_jobs(
        paths=[str(run_a)], app="python", config="aurora-gpu", input_file="x.py",
        env_file="/lus/flare/shared/env.sh")
    assert not failed
    assert local_pbx.get_job(job_ids[0])["env_file"] == "/lus/flare/shared/env.sh"


def test_env_file_inside_the_root_is_translated(local_pbx, tmp_path):
    run_a = tmp_path / "run_a"
    run_a.mkdir()
    env = tmp_path / "env.sh"
    env.write_text("export X=1\n")
    job_ids, failed, _ = local_pbx.add_jobs(
        paths=[str(run_a)], app="python", config="aurora-gpu", input_file="x.py",
        env_file=str(env))
    assert not failed
    assert local_pbx.get_job(job_ids[0])["env_file"] == f"{REMOTE}/env.sh"


def test_env_file_inside_the_root_must_exist(local_pbx, tmp_path):
    run_a = tmp_path / "run_a"
    run_a.mkdir()
    with pytest.raises(ValidationError):
        local_pbx.add_jobs(
            paths=[str(run_a)], app="python", config="aurora-gpu",
            input_file="x.py", env_file=str(tmp_path / "missing.sh"))


def test_ordinary_project_paths_are_untouched(pbx, tmp_path):
    run_a = tmp_path / "run_a"
    run_a.mkdir()
    job_ids, failed, _ = pbx.add_jobs(
        paths=[str(run_a)], app="python", config="aurora-gpu", input_file="x.py",
        env_file="pass")
    assert not failed
    assert pbx.get_job(job_ids[0])["path"] == str(run_a.resolve())


def test_local_init_accepts_transfer_remote_root(pbx, tmp_path):
    """CLI, API and MCP all have to reach the same project setting."""
    d = tmp_path / "proj"
    d.mkdir()
    pbx.local_init(directory=d, remote_root="/lus/flare/projects/ABC/me/p1",
                   transfer_remote="collection-uuid",
                   transfer_remote_root="/lus/flare/projects")

    from parslbox.local.project import load_project
    assert load_project(d).transfer_remote_root == "/lus/flare/projects"


def test_a_transfer_refusal_reaches_the_api_as_a_validation_error(tmp_path, monkeypatch):
    # detect_collection_root raises TransferError when no prefix of remote_root
    # resolves to the database it just wrote. That must not escape untyped.
    from parslbox.api import ParslBox, ValidationError
    from parslbox.local import sync as local_sync, transfer
    from parslbox.local.project import init_project

    local_root = tmp_path / "proj"
    local_root.mkdir()
    outcome = init_project(local_root, remote_root=str(tmp_path / "remote"),
                           compute_endpoint="e", transfer_local="src",
                           transfer_remote="dst")
    monkeypatch.setattr(local_sync, "push", lambda **kw: (_ for _ in ()).throw(
        transfer.TransferError("Could not find it through the collection")))

    pbx = ParslBox(db_path=str(outcome["project"].db_path))
    with pytest.raises(ValidationError, match="Could not find it"):
        pbx.local_push(with_dirs=True)


def test_an_unreachable_endpoint_is_a_validation_error_on_push(tmp_path, monkeypatch):
    # ComputeError is the commonest failure of this whole feature -- a wrong
    # endpoint UUID, or one that is not running -- and it was escaping the API
    # untyped, breaking the contract that callers can catch ParslBoxError.
    from parslbox.api import ParslBox, ValidationError
    from parslbox.local import compute
    from parslbox.local.project import init_project

    local_root = tmp_path / "proj"
    local_root.mkdir()
    outcome = init_project(local_root, remote_root="/lus/flare/p",
                           compute_endpoint="not-a-real-endpoint")

    def boom(self, fn, *a, **kw):
        raise compute.ComputeError("endpoint not-a-real-endpoint did not answer")
    monkeypatch.setattr(compute.ComputeClient, "run", boom)
    monkeypatch.setattr(compute.ComputeClient, "__enter__", lambda self: self)
    monkeypatch.setattr(compute.ComputeClient, "__exit__",
                        lambda self, *a: False)

    pbx = ParslBox(db_path=str(outcome["project"].db_path))
    for call in (pbx.local_push, pbx.local_pull):
        with pytest.raises(ValidationError, match="did not answer"):
            call()


# ------------------------------------------- the wrong database in a project

def test_api_refuses_the_plain_database_inside_a_project(pbx, tmp_path):
    """ParslBox() must not create job_database_pbx.db beside a local one."""
    from parslbox.local.project import WrongDatabaseName

    pbx.local_init(directory=tmp_path, remote_root=REMOTE)
    stray = tmp_path / "job_database_pbx.db"
    with pytest.raises(WrongDatabaseName) as e:
        ParslBox(db_path=stray)
    assert str(tmp_path / LOCAL_DB_NAME) in str(e.value)
    assert not stray.exists()
