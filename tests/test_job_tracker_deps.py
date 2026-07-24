"""
Tests for JobTracker.are_parents_done — the dependency gate.

Key behaviors:
  - a parent satisfies a child when Done OR Warning
  - Failed / Killed / non-terminal parents never satisfy
  - parent status is read FRESH from the DB (shared-DB: a parent may be run by
    another orchestrator after this tracker cached a stale copy)
"""

import tempfile
from pathlib import Path

import pytest

from parslbox.database import database
from parslbox.resource_manager.job_tracker import JobTracker


@pytest.fixture
def db():
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_path = Path(tmp.name)
    tmp.close()
    database.initialize_database(db_path)
    yield db_path
    db_path.unlink(missing_ok=True)


def _add(db_path, path, status='Ready', parents=None):
    return database.add_job(
        db_path, path, 'python', 1, 0, 1.0, 't', status=status, parents=parents
    )


def _tracker(db_path):
    jobs = database.get_jobs(db_path)
    return JobTracker(jobs, db_path)


def test_no_parents_is_satisfied(db):
    c = _add(db, '/c')
    assert _tracker(db).are_parents_done(c) is True


@pytest.mark.parametrize("parent_status,expected", [
    ('Done', True),
    ('Warning', True),
    ('Failed', False),
    ('Killed', False),
    ('Running', False),
    ('Ready', False),
    ('Submitted', False),
])
def test_single_parent_status(db, parent_status, expected):
    p = _add(db, '/p', status=parent_status)
    c = _add(db, '/c', parents=[p])
    assert _tracker(db).are_parents_done(c) is expected


def test_all_parents_must_satisfy(db):
    p1 = _add(db, '/p1', status='Done')
    p2 = _add(db, '/p2', status='Warning')
    p3 = _add(db, '/p3', status='Failed')
    c = _add(db, '/c', parents=[p1, p2, p3])
    t = _tracker(db)
    assert t.are_parents_done(c) is False  # p3 Failed
    # Fix p3 -> Done; now satisfied.
    database.update_jobs(db, [p3], status='Done')
    assert t.are_parents_done(c) is True


def test_reads_fresh_from_db_not_stale_cache(db):
    # Parent is Ready when the tracker is built (stale cache would say False).
    p = _add(db, '/p', status='Ready')
    c = _add(db, '/c', parents=[p])
    t = _tracker(db)
    assert t.are_parents_done(c) is False
    # Another orchestrator finishes the parent — only the DB reflects it.
    database.update_jobs(db, [p], status='Done')
    assert t.are_parents_done(c) is True


def test_nonexistent_parent_blocks(db):
    c = _add(db, '/c', parents=[9999])
    assert _tracker(db).are_parents_done(c) is False
