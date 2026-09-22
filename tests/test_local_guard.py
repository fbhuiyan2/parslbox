"""The change guard: what counts as the remote having moved."""

import sqlite3

import pytest

from parslbox.database import database
from parslbox.local.guard import (
    SyncConflict,
    check_identity,
    check_unchanged,
    describe_drift,
    fingerprint,
    identity_of,
    stamp_identity,
    unchanged,
)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "job_database_pbx-local.db"
    database.initialize_database(path)
    return path


def add(db_path, path):
    return database.add_job(db_path=db_path, path=path, app="python", num_nodes=1,
                            ngpus=0, node_occupancy=1.0, tag=None, status="Ready")


def test_fingerprint_of_an_empty_database(db):
    fp = fingerprint(db)
    assert fp["count"] == 0
    assert fp["max_timestamp"] is None


def test_missing_database_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        fingerprint(tmp_path / "nope.db")


def test_insert_changes_the_fingerprint(db):
    before = fingerprint(db)
    add(db, "/lus/x/a")
    assert not unchanged(fingerprint(db), before)


def test_delete_changes_the_fingerprint(db):
    """COUNT(*) is what catches this; MAX(timestamp) alone would not."""
    job_id = add(db, "/lus/x/a")
    add(db, "/lus/x/b")
    before = fingerprint(db)
    database.remove_jobs_by_id(db, [job_id])
    after = fingerprint(db)
    assert after["max_timestamp"] == before["max_timestamp"]
    assert not unchanged(after, before)


def test_update_changes_the_fingerprint(db):
    """The update_jobs_timestamp trigger is what makes this visible.

    The row is inserted with an old timestamp directly: an UPDATE to set one
    would fire the very trigger under test and stamp it with now instead.
    """
    con = sqlite3.connect(str(db))
    con.execute(
        "INSERT INTO jobs (app, path, status, timestamp) "
        "VALUES ('python', '/lus/x/a', 'Ready', '2000-01-01 00:00:00')")
    con.commit()
    job_id = con.execute("SELECT job_id FROM jobs").fetchone()[0]
    con.close()
    before = fingerprint(db)
    assert before["max_timestamp"] == "2000-01-01 00:00:00"
    database.update_jobs(db_path=db, job_ids=[job_id], status="Done")
    after = fingerprint(db)
    assert after["count"] == before["count"]
    assert after["max_timestamp"] != before["max_timestamp"]
    assert not unchanged(after, before)


def test_an_empty_database_with_no_record_has_nothing_to_lose(db):
    assert unchanged(fingerprint(db), None) is True


def test_a_populated_database_with_no_record_is_not_unchanged(db):
    add(db, "/lus/x/a")
    assert unchanged(fingerprint(db), None) is False


def test_check_unchanged_passes_when_nothing_moved(db):
    fp = fingerprint(db)
    check_unchanged("remote", fp, fp)


def test_check_unchanged_refuses_when_it_moved(db):
    before = fingerprint(db)
    add(db, "/lus/x/a")
    with pytest.raises(SyncConflict) as e:
        check_unchanged("remote", fingerprint(db), before)
    assert "remote" in str(e.value)


def test_drift_wording(db):
    before = fingerprint(db)
    assert describe_drift(before, None) == "empty, never synced"
    assert describe_drift(before, before) == "unchanged"
    add(db, "/lus/x/a")
    assert describe_drift(fingerprint(db), before) == "+1 jobs added since"
    assert describe_drift(fingerprint(db), None) == "1 jobs, never synced"


def test_identity_survives_vacuum_into(db, tmp_path):
    """VACUUM INTO is how a pull snapshots the remote; the id must survive it."""
    stamp_identity(db, "2026-09-10T00:00:00Z")
    add(db, "/lus/x/a")
    snapshot = tmp_path / "snapshot.db"
    con = sqlite3.connect(str(db))
    con.execute("VACUUM INTO ?", (str(snapshot),))
    con.close()
    assert fingerprint(snapshot)["identity"] == identity_of("2026-09-10T00:00:00Z")


def test_check_identity_accepts_a_matching_database(db):
    stamp_identity(db, "2026-09-10T00:00:00Z")
    check_identity("2026-09-10T00:00:00Z", fingerprint(db))


def test_check_identity_refuses_a_different_project(db):
    stamp_identity(db, "2026-09-10T00:00:00Z")
    with pytest.raises(SyncConflict):
        check_identity("2026-01-01T00:00:00Z", fingerprint(db))


def test_unstamped_database_is_accepted(db):
    """A database from before identities existed should not be rejected."""
    check_identity("2026-09-10T00:00:00Z", fingerprint(db))
