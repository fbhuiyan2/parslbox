"""Local project detection, the init case table, and the YAML round trip."""

import sqlite3

import pytest
import yaml

from parslbox.database import database
from parslbox.local import project as project_mod
from parslbox.local.project import (
    LOCAL_DB_NAME,
    PROJECT_FILE,
    LocalProject,
    NestedProject,
    PopulatedDatabase,
    ProjectFileError,
    WrongDatabaseName,
    check_db_name,
    find_parent_project,
    find_project,
    init_project,
    inspect_directory,
    load_project,
    save_project,
    utc_stamp,
)

REMOTE = "/lus/flare/projects/CSTEELML/fbhuiyan/proj1"


def add_row(db_path, path="/lus/flare/x/a"):
    database.add_job(db_path=db_path, path=path, app="python", num_nodes=1,
                     ngpus=0, node_occupancy=1.0, tag=None, status="Ready")


# ---------------------------------------------------------------- detection

def test_find_project_returns_none_without_yaml(tmp_path):
    assert find_project(tmp_path / LOCAL_DB_NAME) is None


def test_find_project_reads_the_yaml_beside_the_db(tmp_path):
    init_project(tmp_path, remote_root=REMOTE)
    proj = find_project(tmp_path / LOCAL_DB_NAME)
    assert proj is not None
    assert proj.local_root == tmp_path
    assert proj.remote_root == REMOTE


def test_find_project_does_not_walk_up(tmp_path):
    """A project above is somebody else's; only the DB's own directory counts."""
    init_project(tmp_path, remote_root=REMOTE)
    child = tmp_path / "sub"
    child.mkdir()
    assert find_project(child / LOCAL_DB_NAME) is None


def test_wrong_database_name_is_an_error(tmp_path):
    """Exporting the directory instead of the file is the case this catches."""
    init_project(tmp_path, remote_root=REMOTE)
    with pytest.raises(WrongDatabaseName) as e:
        find_project(tmp_path / "job_database_pbx.db")
    assert str(tmp_path / LOCAL_DB_NAME) in str(e.value)


def test_find_project_of_none_is_none():
    assert find_project(None) is None


# ---------------------------------------------------------- check_db_name

def test_check_db_name_is_false_outside_a_project(tmp_path):
    assert check_db_name(tmp_path / LOCAL_DB_NAME) is False
    assert check_db_name(tmp_path / "job_database_pbx.db") is False
    assert check_db_name(None) is False


def test_check_db_name_is_true_for_the_local_database(tmp_path):
    init_project(tmp_path, remote_root=REMOTE)
    assert check_db_name(tmp_path / LOCAL_DB_NAME) is True


def test_check_db_name_refuses_the_plain_database_in_a_project(tmp_path):
    """The one case: PBX_DB_PATH names the directory, not the file."""
    init_project(tmp_path, remote_root=REMOTE)
    with pytest.raises(WrongDatabaseName) as e:
        check_db_name(tmp_path / "job_database_pbx.db")
    message = str(e.value)
    assert f'export PBX_DB_PATH="{tmp_path / LOCAL_DB_NAME}"' in message
    assert not (tmp_path / "job_database_pbx.db").exists()


def test_check_db_name_does_not_read_the_yaml(tmp_path):
    """A broken project file must not stop an ordinary command."""
    init_project(tmp_path, remote_root=REMOTE)
    (tmp_path / PROJECT_FILE).write_text("nonsense: [")
    assert check_db_name(tmp_path / LOCAL_DB_NAME) is True
    with pytest.raises(ProjectFileError):
        find_project(tmp_path / LOCAL_DB_NAME)


# ------------------------------------------------------------------- yaml

def test_yaml_round_trip_keeps_created_a_string(tmp_path):
    proj = LocalProject(local_root=tmp_path, created=utc_stamp(), remote_root=REMOTE)
    save_project(proj)
    reloaded = load_project(tmp_path)
    assert isinstance(reloaded.created, str)
    assert reloaded.created == proj.created


def test_trailing_slash_is_stripped_from_remote_root(tmp_path):
    proj = LocalProject(local_root=tmp_path, created=utc_stamp(),
                        remote_root=REMOTE + "/")
    assert proj.remote_root == REMOTE
    assert proj.remote_db_path == f"{REMOTE}/{LOCAL_DB_NAME}"


def test_missing_keys_are_reported(tmp_path):
    (tmp_path / PROJECT_FILE).write_text(yaml.safe_dump({"created": "x"}))
    with pytest.raises(ProjectFileError) as e:
        load_project(tmp_path)
    assert "local_root" in str(e.value) and "remote_root" in str(e.value)


def test_relative_remote_root_is_rejected(tmp_path):
    (tmp_path / PROJECT_FILE).write_text(yaml.safe_dump(
        {"created": "x", "local_root": str(tmp_path), "remote_root": "relative/path"}))
    with pytest.raises(ProjectFileError):
        load_project(tmp_path)


def test_unparseable_yaml_is_reported(tmp_path):
    (tmp_path / PROJECT_FILE).write_text("{[not yaml")
    with pytest.raises(ProjectFileError):
        load_project(tmp_path)


def test_moved_project_uses_its_actual_location(tmp_path):
    init_project(tmp_path, remote_root=REMOTE)
    data = yaml.safe_load((tmp_path / PROJECT_FILE).read_text())
    data["local_root"] = "/somewhere/else"
    (tmp_path / PROJECT_FILE).write_text(yaml.safe_dump(data))
    proj = load_project(tmp_path)
    assert proj.local_root == tmp_path
    assert proj.moved is True


# ------------------------------------------------------------- init cases

def test_init_creates_db_and_yaml(tmp_path):
    out = init_project(tmp_path, remote_root=REMOTE)
    assert out["action"] == "created"
    assert (tmp_path / LOCAL_DB_NAME).is_file()
    assert (tmp_path / PROJECT_FILE).is_file()


def test_init_stamps_the_project_identity(tmp_path):
    from parslbox.local.guard import identity_of
    out = init_project(tmp_path, remote_root=REMOTE)
    con = sqlite3.connect(str(tmp_path / LOCAL_DB_NAME))
    stamped = con.execute("PRAGMA user_version").fetchone()[0]
    con.close()
    assert stamped == identity_of(out["project"].created)


def test_rerunning_init_reports_the_existing_project(tmp_path):
    first = init_project(tmp_path, remote_root=REMOTE)
    again = init_project(tmp_path)
    assert again["action"] == "existing"
    assert again["project"].created == first["project"].created


def test_yaml_without_db_is_an_error(tmp_path):
    init_project(tmp_path, remote_root=REMOTE)
    (tmp_path / LOCAL_DB_NAME).unlink()
    with pytest.raises(ProjectFileError) as e:
        init_project(tmp_path)
    assert PROJECT_FILE in str(e.value)


def test_empty_db_without_yaml_is_adopted(tmp_path):
    database.initialize_database(tmp_path / LOCAL_DB_NAME)
    out = init_project(tmp_path, remote_root=REMOTE)
    assert out["action"] == "adopted"
    assert (tmp_path / PROJECT_FILE).is_file()


def test_populated_db_without_yaml_is_refused(tmp_path):
    db_path = tmp_path / LOCAL_DB_NAME
    database.initialize_database(db_path)
    add_row(db_path)
    with pytest.raises(PopulatedDatabase) as e:
        init_project(tmp_path, remote_root=REMOTE)
    assert "1 job" in str(e.value)
    assert not (tmp_path / PROJECT_FILE).exists()


def test_ordinary_database_is_left_alone(tmp_path):
    plain = tmp_path / "job_database_pbx.db"
    database.initialize_database(plain)
    add_row(plain)
    out = init_project(tmp_path, remote_root=REMOTE)
    assert out["action"] == "created"
    assert any("job_database_pbx.db" in n for n in out["notes"])
    con = sqlite3.connect(str(plain))
    assert con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    con.close()


def test_remote_root_is_required(tmp_path):
    with pytest.raises(project_mod.LocalProjectError):
        init_project(tmp_path)


def test_relative_remote_root_is_refused(tmp_path):
    with pytest.raises(project_mod.LocalProjectError):
        init_project(tmp_path, remote_root="not/absolute")


# ---------------------------------------------------------------- nesting

def test_parent_project_is_found(tmp_path):
    init_project(tmp_path, remote_root=REMOTE)
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert find_parent_project(deep) == tmp_path


def test_find_parent_project_ignores_siblings(tmp_path):
    (tmp_path / "sibling").mkdir()
    init_project(tmp_path / "sibling", remote_root=REMOTE)
    target = tmp_path / "elsewhere"
    target.mkdir()
    assert find_parent_project(target) is None


def test_nesting_is_refused_by_default(tmp_path):
    init_project(tmp_path, remote_root=REMOTE)
    inner = tmp_path / "inner"
    inner.mkdir()
    with pytest.raises(NestedProject) as e:
        init_project(inner, remote_root=REMOTE + "/inner")
    assert e.value.parent == tmp_path


def test_nesting_is_allowed_with_nested_ok(tmp_path):
    init_project(tmp_path, remote_root=REMOTE)
    inner = tmp_path / "inner"
    inner.mkdir()
    out = init_project(inner, remote_root=REMOTE + "/inner", nested_ok=True)
    assert out["action"] == "created"
    assert any("Nested inside" in n for n in out["notes"])


# --------------------------------------------------------------- inspect

def test_inspect_reports_without_changing_anything(tmp_path):
    state = inspect_directory(tmp_path)
    assert state["yaml_exists"] is False
    assert state["db_exists"] is False
    assert state["row_count"] is None
    assert list(tmp_path.iterdir()) == []


def test_inspect_counts_rows(tmp_path):
    db_path = tmp_path / LOCAL_DB_NAME
    database.initialize_database(db_path)
    add_row(db_path, "/lus/flare/x/a")
    add_row(db_path, "/lus/flare/x/b")
    assert inspect_directory(tmp_path)["row_count"] == 2


def test_transfer_remote_root_round_trips_through_the_project_file(tmp_path):
    """The collection's root is project config, not something to re-derive."""
    from parslbox.local.project import init_project, load_project

    init_project(tmp_path, remote_root="/lus/flare/projects/ABC/me/p1",
                 transfer_remote="collection-uuid",
                 transfer_remote_root="/lus/flare/projects")
    assert load_project(tmp_path).transfer_remote_root == "/lus/flare/projects"


def test_transfer_remote_root_defaults_to_none(tmp_path):
    """Unset means the collection serves the whole filesystem -- no rewriting."""
    from parslbox.local.project import init_project, load_project

    init_project(tmp_path, remote_root="/scratch/me/p1")
    assert load_project(tmp_path).transfer_remote_root is None
