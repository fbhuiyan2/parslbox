"""`pbx rm` refuses a request it cannot carry out in full.

Removing five jobs when forty were named used to print "Removed 5 job(s)" —
the same sentence a complete delete prints. Both guards are exercised here
through the CLI, since that is the layer that bypasses `api.py`.
"""
import pytest
from typer.testing import CliRunner
from unittest.mock import patch

from parslbox.commands.rm import app as rm_app
from parslbox.database import database


@pytest.fixture
def db(tmp_path):
    """Jobs 1..5."""
    path = tmp_path / "jobs.db"
    database.initialize_database(path)
    for i in range(5):
        database.add_job(db_path=path, path=f"/w/j{i}", app="python", num_nodes=1,
                         ngpus=0, node_occupancy=1.0, tag="t", in_file=None,
                         ranks_per_node=1)
    return path


def run_rm(db, *args):
    with patch("parslbox.commands.rm.path_utils.DB_FILE", db):
        return CliRunner().invoke(rm_app, list(args), env={"COLUMNS": "400"})


class TestOverreachingRange:
    def test_a_range_past_the_last_job_is_refused(self, db):
        r = run_rm(db, "1-100000000")
        assert r.exit_code == 1
        assert "highest job id" in r.output
        assert len(database.get_jobs(db)) == 5

    def test_the_output_is_one_line_not_a_hundred_million_ids(self, db):
        r = run_rm(db, "1-100000000")
        assert len(r.output) < 200

    def test_a_range_within_the_database_still_works(self, db):
        r = run_rm(db, "1-3")
        assert r.exit_code == 0
        assert "Removed 3 job(s)" in r.output
        assert [j["job_id"] for j in database.get_jobs(db)] == [4, 5]


class TestMissingIdsAreNamed:
    def test_nothing_is_removed_when_an_id_is_missing(self, db):
        run_rm(db, "2-4")
        r = run_rm(db, "1-5")
        assert r.exit_code == 1
        assert "2-4" in r.output
        assert [j["job_id"] for j in database.get_jobs(db)] == [1, 5]

    def test_missing_ids_print_as_ranges_not_as_integers(self, db):
        run_rm(db, "2", "3", "4")
        r = run_rm(db, "1", "2", "3", "4", "5")
        assert "2-4" in r.output
        assert "2, 3, 4" not in r.output

    def test_a_single_unknown_id_is_refused_too(self, db):
        r = run_rm(db, "3", "99999")
        assert r.exit_code == 1
        assert "99999" in r.output
        assert len(database.get_jobs(db)) == 5

    def test_removing_every_id_that_exists_succeeds(self, db):
        r = run_rm(db, "1-5")
        assert r.exit_code == 0
        assert "Removed 5 job(s)" in r.output
        assert database.get_jobs(db) == []


class TestIdsTooLargeToStore:
    """A bare id is not bounded by the database's height, so an extra digit
    reached the sqlite3 driver and raised OverflowError -- which `rm` does
    not catch, so the user got a traceback instead of a message."""

    def test_an_oversized_bare_id_is_refused_cleanly(self, db):
        r = run_rm(db, "9" * 30)
        assert r.exit_code == 1
        assert "out of range" in r.output
        assert r.exception is None or isinstance(r.exception, SystemExit)
        assert len(database.get_jobs(db)) == 5

    def test_it_is_refused_alongside_valid_ids_too(self, db):
        r = run_rm(db, "1", "9" * 30)
        assert r.exit_code == 1
        assert len(database.get_jobs(db)) == 5
