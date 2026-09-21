"""CLI for `pbx local`, plus the path translation `pbx add` inherits."""

import os

import pytest
from typer.testing import CliRunner

from parslbox.commands.local import app as local_app
from parslbox.database import database
from parslbox.local.project import LOCAL_DB_NAME, PROJECT_FILE, init_project

REMOTE = "/lus/flare/projects/CSTEELML/fbhuiyan/proj1"
WIDE = {"COLUMNS": "400"}

runner = CliRunner()


@pytest.fixture
def project_dir(tmp_path, monkeypatch):
    init_project(tmp_path, remote_root=REMOTE)
    monkeypatch.setenv("PBX_DB_PATH", str(tmp_path / LOCAL_DB_NAME))
    monkeypatch.chdir(tmp_path)
    return tmp_path


# -------------------------------------------------------------------- init

def test_init_creates_a_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(local_app, ["init", "--remote-root", REMOTE], env=WIDE)
    assert result.exit_code == 0, result.output
    assert (tmp_path / LOCAL_DB_NAME).is_file()
    assert (tmp_path / PROJECT_FILE).is_file()


def test_init_prints_the_export_line(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(local_app, ["init", "--remote-root", REMOTE], env=WIDE)
    assert f'export PBX_DB_PATH="{tmp_path / LOCAL_DB_NAME}"' in result.output


def test_init_on_an_existing_project_resumes_it(project_dir):
    result = runner.invoke(local_app, ["init"], env=WIDE)
    assert result.exit_code == 0
    assert "already exists" in result.output
    assert "export PBX_DB_PATH" in result.output


def test_init_refuses_a_populated_orphan_database(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / LOCAL_DB_NAME
    database.initialize_database(db_path)
    database.add_job(db_path=db_path, path="/lus/x/a", app="python", num_nodes=1,
                     ngpus=0, node_occupancy=1.0, tag=None, status="Ready")
    result = runner.invoke(local_app, ["init", "--remote-root", REMOTE], env=WIDE)
    assert result.exit_code == 1
    assert "1 job" in result.output
    assert not (tmp_path / PROJECT_FILE).exists()


def test_init_refuses_before_prompting(tmp_path, monkeypatch):
    """A populated database must not be asked for a remote root first."""
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / LOCAL_DB_NAME
    database.initialize_database(db_path)
    database.add_job(db_path=db_path, path="/lus/x/a", app="python", num_nodes=1,
                     ngpus=0, node_occupancy=1.0, tag=None, status="Ready")
    result = runner.invoke(local_app, ["init"], input="\n", env=WIDE)
    assert result.exit_code == 1
    assert "Remote project root" not in result.output


def test_init_errors_on_a_yaml_without_a_database(project_dir):
    (project_dir / LOCAL_DB_NAME).unlink()
    result = runner.invoke(local_app, ["init"], env=WIDE)
    assert result.exit_code == 1
    assert PROJECT_FILE in result.output


def test_init_asks_before_nesting(project_dir):
    inner = project_dir / "inner"
    inner.mkdir()
    os.chdir(inner)
    result = runner.invoke(local_app, ["init", "--remote-root", REMOTE + "/inner"],
                           input="n\n", env=WIDE)
    assert "already exists at" in result.output
    assert not (inner / PROJECT_FILE).exists()


def test_nested_ok_skips_the_prompt(project_dir):
    inner = project_dir / "inner"
    inner.mkdir()
    os.chdir(inner)
    result = runner.invoke(
        local_app, ["init", "--remote-root", REMOTE + "/inner", "--nested-ok"],
        env=WIDE)
    assert result.exit_code == 0, result.output
    assert (inner / PROJECT_FILE).is_file()


def test_init_mentions_an_ordinary_database(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    database.initialize_database(tmp_path / "job_database_pbx.db")
    result = runner.invoke(local_app, ["init", "--remote-root", REMOTE], env=WIDE)
    assert "job_database_pbx.db" in result.output
    assert result.exit_code == 0


# ------------------------------------------------------------------ status

def test_status_reports_the_project(project_dir):
    result = runner.invoke(local_app, ["status", "--local-only"], env=WIDE)
    assert result.exit_code == 0
    assert REMOTE in result.output
    assert "0 jobs" in result.output
    assert "never" in result.output


def test_status_outside_a_project_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PBX_DB_PATH", str(tmp_path / "job_database_pbx.db"))
    result = runner.invoke(local_app, ["status", "--local-only"], env=WIDE)
    assert result.exit_code == 0
    assert "Not a local project" in result.output


def test_status_runs_from_any_directory(project_dir, tmp_path):
    """status is environment-relative, not directory-relative."""
    elsewhere = tmp_path.parent / "elsewhere_for_status"
    elsewhere.mkdir(exist_ok=True)
    os.chdir(elsewhere)
    result = runner.invoke(local_app, ["status", "--local-only"], env=WIDE)
    assert result.exit_code == 0
    assert str(project_dir) in result.output


def test_status_flags_a_directory_shaped_db_path(project_dir, monkeypatch):
    monkeypatch.setenv("PBX_DB_PATH", str(project_dir))
    result = runner.invoke(local_app, ["status", "--local-only"], env=WIDE)
    assert result.exit_code == 1
    assert LOCAL_DB_NAME in result.output


def test_status_without_an_endpoint_says_so(project_dir):
    result = runner.invoke(local_app, ["status"], env=WIDE)
    assert result.exit_code == 0
    assert "not configured" in result.output


# --------------------------------------------------- push / pull refusals

def test_push_outside_a_project_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv("PBX_DB_PATH", str(tmp_path / "job_database_pbx.db"))
    result = runner.invoke(local_app, ["push"], env=WIDE)
    assert result.exit_code == 1
    assert "Not a local project" in result.output


def test_pull_outside_a_project_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv("PBX_DB_PATH", str(tmp_path / "job_database_pbx.db"))
    result = runner.invoke(local_app, ["pull"], env=WIDE)
    assert result.exit_code == 1
    assert "Not a local project" in result.output


def test_push_without_an_endpoint_fails_clearly(project_dir):
    result = runner.invoke(local_app, ["push"], env=WIDE)
    assert result.exit_code == 1
    assert "endpoint" in result.output.lower()


# ------------------------------------------------- job directory reporting

def test_job_dirs_reports_the_ratio(project_dir):
    """The headline that catches a forgotten status flip."""
    from parslbox.database import database
    from parslbox.local.project import load_project
    from parslbox.local import sync

    project = load_project(project_dir)
    (project_dir / "run_a").mkdir()
    (project_dir / "run_b").mkdir()
    for name, status in (("run_a", "Ready"), ("run_b", "Done"), ("run_c", "Done")):
        database.add_job(db_path=project.db_path,
                         path=f"{project.remote_root}/{name}", app="python",
                         num_nodes=1, ngpus=0, node_occupancy=1.0, tag=None,
                         status=status)
    found = sync.job_dirs(project)
    assert found["total"] == 3
    assert found["matched"] == 1
    assert found["statuses"] == ["Ready", "Restart"]
    assert found["present"] == [str(project_dir / "run_a")]


def test_dir_report_prints_the_ratio(capsys):
    from parslbox.commands.local import render_dir_report
    render_dir_report({"present": ["/a"], "missing": ["/b"], "outside": [],
                       "statuses": ["Ready", "Restart"], "total": 400,
                       "matched": 12})
    out = capsys.readouterr().out
    assert "12 of 400" in out
    assert "Ready/Restart" in out
    assert "1 to send" in out
    assert "missing on this machine" in out


def test_dir_report_names_the_fix_when_nothing_matches(capsys):
    from parslbox.commands.local import render_dir_report
    render_dir_report({"present": [], "missing": [], "outside": [],
                       "statuses": ["Ready", "Restart"], "total": 400,
                       "matched": 0})
    out = capsys.readouterr().out
    assert "Nothing matched" in out
    assert "--status Restart" in out


def test_dir_report_truncates_long_lists(capsys):
    from parslbox.commands.local import render_dir_report
    render_dir_report({"present": [], "missing": [f"/m{i}" for i in range(25)],
                       "outside": [], "statuses": ["Ready"], "total": 25,
                       "matched": 25}, limit=10)
    out = capsys.readouterr().out
    assert "and 15 more" in out


# ----------------------------------------------------- push flags reach sync
#
# Nothing asserted that the flags on `pbx local push` actually arrive at
# sync.push. Every one of them changes what happens on the network, and three
# of them (--sync-level, --no-wait, --force) have no visible effect in the
# output, so a dropped flag would go unnoticed.

@pytest.fixture
def captured_push(monkeypatch):
    calls = {}

    def fake_push(**kwargs):
        calls.update(kwargs)
        return {"bytes": 100, "remote_db_path": "/r/db",
                "fingerprint": {"count": 0}}

    from parslbox.commands import local as local_cmd
    monkeypatch.setattr(local_cmd.sync_mod, "push", fake_push)
    return calls


def test_push_defaults_are_what_the_shared_layer_expects(project_dir, captured_push):
    result = runner.invoke(local_app, ["push"], env=WIDE)
    assert result.exit_code == 0
    assert captured_push["force"] is False
    assert captured_push["dirs"] is None
    assert captured_push["wait_for_dirs"] is True
    assert captured_push["sync_level"] == "checksum"


def test_every_push_flag_arrives(project_dir, captured_push):
    result = runner.invoke(
        local_app, ["push", "--force", "--no-wait", "--sync-level", "mtime"],
        env=WIDE)
    assert result.exit_code == 0
    assert captured_push["force"] is True
    assert captured_push["wait_for_dirs"] is False
    assert captured_push["sync_level"] == "mtime"


def test_with_dirs_sends_the_matching_directories(project_dir, captured_push):
    (project_dir / "run_a").mkdir()
    database.add_job(db_path=project_dir / LOCAL_DB_NAME,
                     path=f"{REMOTE}/run_a", app="python", num_nodes=1, ngpus=0,
                     node_occupancy=1.0, tag="t1", status="Ready")
    result = runner.invoke(local_app, ["push", "--with-dirs"], env=WIDE)
    assert result.exit_code == 0
    assert captured_push["dirs"] == [str(project_dir / "run_a")]


def test_a_bad_sync_level_is_refused_before_anything_else(project_dir):
    # Unmocked on purpose. This project has no endpoint and no directories to
    # send, and the complaint must still be about the sync level -- it is a
    # pure string check, so it should not queue behind a network prerequisite.
    result = runner.invoke(local_app, ["push", "--sync-level", "bogus"], env=WIDE)
    assert result.exit_code != 0
    assert "bogus" in result.output
    assert "checksum" in result.output


def test_a_detected_collection_root_is_reported(project_dir, monkeypatch):
    def fake_push(**kwargs):
        kwargs["progress"]("collection is rooted at /lus/flare/projects — saved")
        return {"bytes": 1, "remote_db_path": "/r/db", "fingerprint": {"count": 0},
                "transfer_remote_root": "/lus/flare/projects",
                "transfer_remote_root_detected": True}
    from parslbox.commands import local as local_cmd
    monkeypatch.setattr(local_cmd.sync_mod, "push", fake_push)

    result = runner.invoke(local_app, ["push"], env=WIDE)
    assert result.exit_code == 0
    assert "/lus/flare/projects" in result.output


def test_status_shows_the_collection_root(project_dir, monkeypatch):
    from parslbox.local.project import load_project, save_project
    proj = load_project(project_dir)
    proj.transfer_remote = "dst-uuid"
    proj.transfer_remote_root = "/lus/flare/projects"
    save_project(proj)

    result = runner.invoke(local_app, ["status", "--local-only"], env=WIDE)
    assert result.exit_code == 0
    assert "/lus/flare/projects" in result.output


# ------------------------------------------- the guard on the submission path

def test_qsub_renders_the_refusal_as_an_error(tmp_path, monkeypatch):
    """A SyncConflict from the push is a refusal, not an 'Unexpected error'."""
    from parslbox.commands.qsub import app as qsub_app
    from parslbox.local import compute, sync
    from parslbox.utils import path_utils

    local_root = tmp_path / "local"
    remote_root = tmp_path / "remote"
    local_root.mkdir()
    remote_root.mkdir()
    project = init_project(local_root, remote_root=str(remote_root),
                           compute_endpoint="stand-in")["project"]
    monkeypatch.setattr(compute.ComputeClient, "run",
                        lambda self, fn, *a, **kw: fn(*a))
    monkeypatch.setattr(path_utils, "DB_FILE", project.db_path)

    def add_row(db_path, path):
        return database.add_job(db_path=db_path, path=path, app="python",
                                num_nodes=1, ngpus=0, node_occupancy=1.0,
                                tag=None, status="Ready")

    add_row(project.db_path, f"{project.remote_root}/a")
    sync.push(db_path=project.db_path)
    add_row(project.remote_db_path, f"{project.remote_root}/ran_over_there")

    result = runner.invoke(qsub_app, [
        "-c", "aurora-gpu", "-N", "t", "-q", "debug", "--select", "1", "-T", "30",
    ], env=WIDE)
    assert result.exit_code == 1
    assert "Unexpected error" not in result.output
    assert "has changed since the last sync" in result.output
    assert "pbx local pull" in result.output
