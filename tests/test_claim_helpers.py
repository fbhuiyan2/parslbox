"""
Tests for the atomic claim / revert / runnable-candidate DB helpers that back
the rewritten dispatch loop:

  - database.get_runnable_jobs
  - database.claim_jobs
  - database.revert_claims
"""

import tempfile
from pathlib import Path

import pytest

from parslbox.database import database


@pytest.fixture
def db():
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_path = Path(tmp.name)
    tmp.close()
    database.initialize_database(db_path)
    yield db_path
    db_path.unlink(missing_ok=True)


def _add(db_path, path, status='Ready', app='python', tag='t', num_nodes=1):
    return database.add_job(
        db_path, path, app, num_nodes, 0, 1.0, tag, status=status
    )


def _status(db_path, job_id):
    return database.get_jobs_by_ids(db_path, [job_id])[0]['status']


def _sched(db_path, job_id):
    return database.get_jobs_by_ids(db_path, [job_id])[0]['sched_job_id']


class TestGetRunnableJobs:
    def test_only_ready_and_restart(self, db):
        r = _add(db, '/a', status='Ready')
        rs = _add(db, '/b', status='Restart')
        _add(db, '/c', status='Running')
        _add(db, '/d', status='Done')
        ids = {j['job_id'] for j in database.get_runnable_jobs(db)}
        assert ids == {r, rs}

    def test_restart_first_then_smallest(self, db):
        big_ready = _add(db, '/a', status='Ready', num_nodes=10)
        small_ready = _add(db, '/b', status='Ready', num_nodes=1)
        restart = _add(db, '/c', status='Restart', num_nodes=5)
        order = [j['job_id'] for j in database.get_runnable_jobs(db)]
        # Restart precedes all Ready; Ready sorted smallest-first.
        assert order == [restart, small_ready, big_ready]

    def test_max_nodes_filter(self, db):
        small = _add(db, '/a', status='Ready', num_nodes=2)
        _add(db, '/b', status='Ready', num_nodes=50)
        ids = {j['job_id'] for j in database.get_runnable_jobs(db, max_nodes=10)}
        assert ids == {small}

    def test_app_and_tag_filters(self, db):
        keep = _add(db, '/a', app='vasp', tag='scale')
        _add(db, '/b', app='python', tag='scale')
        _add(db, '/c', app='vasp', tag='other')
        ids = {j['job_id'] for j in database.get_runnable_jobs(db, apps=['vasp'], tags=['scale'])}
        assert ids == {keep}

    def test_tag_glob(self, db):
        a = _add(db, '/a', tag='vasp_scale')
        b = _add(db, '/b', tag='vasp_bench')
        _add(db, '/c', tag='lammps')
        ids = {j['job_id'] for j in database.get_runnable_jobs(db, tags=['vasp_*'])}
        assert ids == {a, b}


class TestClaimJobs:
    def test_claim_flips_and_stamps_owner(self, db):
        r = _add(db, '/a', status='Ready')
        rs = _add(db, '/b', status='Restart')
        won = database.claim_jobs(db, [r], [rs], owner='BATCH_A')
        assert set(won) == {r, rs}
        assert _status(db, r) == 'Submitted'
        assert _status(db, rs) == 'Resubmitted'
        assert _sched(db, r) == 'BATCH_A'
        assert _sched(db, rs) == 'BATCH_A'

    def test_guard_skips_already_claimed(self, db):
        r = _add(db, '/a', status='Ready')
        # A wins it first.
        assert database.claim_jobs(db, [r], [], owner='A') == [r]
        # B attempts the same id — guard (status='Ready') no longer matches.
        assert database.claim_jobs(db, [r], [], owner='B') == []
        assert _sched(db, r) == 'A'
        assert _status(db, r) == 'Submitted'

    def test_disjoint_ownership_two_runs(self, db):
        ids = [_add(db, f'/j{i}', status='Ready') for i in range(6)]
        won_a = database.claim_jobs(db, ids, [], owner='A')
        won_b = database.claim_jobs(db, ids, [], owner='B')
        # A grabbed everything first; B gets none. No id is won twice.
        assert set(won_a) == set(ids)
        assert won_b == []
        assert not (set(won_a) & set(won_b))

    def test_readback_scoped_to_requested_ids(self, db):
        # A job A already owns from a prior claim must not leak into a later
        # claim's readback (which is scoped to the ids passed this call).
        old = _add(db, '/old', status='Ready')
        database.claim_jobs(db, [old], [], owner='A')  # old -> Submitted, owner A
        new = _add(db, '/new', status='Ready')
        won = database.claim_jobs(db, [new], [], owner='A')
        assert won == [new]  # not [old, new]

    def test_empty_input(self, db):
        assert database.claim_jobs(db, [], [], owner='A') == []


class TestRevertClaims:
    def test_revert_flips_back_and_clears_owner(self, db):
        r = _add(db, '/a', status='Ready')
        rs = _add(db, '/b', status='Restart')
        database.claim_jobs(db, [r], [rs], owner='A')
        n = database.revert_claims(db, [r, rs], owner='A')
        assert n == 2
        assert _status(db, r) == 'Ready'
        assert _status(db, rs) == 'Restart'
        assert _sched(db, r) is None
        assert _sched(db, rs) is None

    def test_revert_owner_scoped(self, db):
        r = _add(db, '/a', status='Ready')
        database.claim_jobs(db, [r], [], owner='A')
        # A different run must not revert A's claim.
        assert database.revert_claims(db, [r], owner='B') == 0
        assert _status(db, r) == 'Submitted'
        assert _sched(db, r) == 'A'

    def test_revert_ignores_non_claimed_states(self, db):
        running = _add(db, '/a', status='Running')
        assert database.revert_claims(db, [running]) == 0
        assert _status(db, running) == 'Running'
