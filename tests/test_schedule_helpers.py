"""
Behavioral tests for the dispatch engine (schedule_helpers): coarse_pick, and
the dynamic / static dispatch passes.

create_parsl_future is stubbed (no Parsl); assign_resources is a lightweight
fake so we can control capacity and force the rare "no fit after claim" revert.
The DB, claim/revert, and dependency checks are all real.
"""

import tempfile
import time
from concurrent.futures import Future
from pathlib import Path

import pytest

from parslbox.database import database
from parslbox.database.status_buffer import StatusBuffer
from parslbox.resource_manager.job_tracker import JobTracker
from parslbox.resource_manager.exceptions import InsufficientResources
from parslbox.commands.helpers import schedule_helpers
from parslbox.commands.helpers.schedule_helpers import (
    SchedulerContext,
    check_run_complete,
    coarse_pick,
    dispatch_dynamic,
    dispatch_static,
    drain_completions,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeApp:
    RUN_HOOKS_ON_COMPUTE = False

    def min_remaining_walltime(self, job):
        return 0  # never gate


class FakeRM:
    """Capacity-bounded fake. assign_resources succeeds until `capacity` jobs
    are held, then raises InsufficientResources. Ids in `reject` always raise
    (simulates a job that passes the node-level filter but fails fine-grained
    assignment → the coarse-pick over-claim / revert path)."""

    def __init__(self, capacity, reject=None):
        self.capacity = capacity
        self.reject = set(reject or [])
        self.assigned = set()
        self._backlogged_jobs_set = set()
        self.job_tracker = None

    def free_node_capacity(self):
        return max(0, self.capacity - len(self.assigned))

    def assign_resources(self, job):
        jid = job['job_id']
        if jid in self.reject:
            raise InsufficientResources(f"job {jid} rejected")
        if len(self.assigned) >= self.capacity:
            raise InsufficientResources("full")
        self.assigned.add(jid)

    def free_resources_with_health_check(self, job_id, **kw):
        self.assigned.discard(job_id)

    def add_to_backlog(self, job_id):
        self._backlogged_jobs_set.add(job_id)

    def get_dependency_ready_jobs_from_backlog(self):
        return self.job_tracker.get_dependency_ready_jobs(list(self._backlogged_jobs_set))


def _stub_create_future(job, ctx, is_restart):
    """Stand-in for create_parsl_future: register the live future + buffer
    Running, no Parsl."""
    fut = f"fut-{job['job_id']}"
    ctx.fut_to_item[fut] = {'job': job, 'is_restart': is_restart}
    ctx.job_tracker.update_job_status(job['job_id'], 'Running')
    ctx.status_buffer.add_status_update(job['job_id'], status='Running')
    return True


@pytest.fixture
def db():
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    db_path = Path(tmp.name)
    tmp.close()
    database.initialize_database(db_path)
    yield db_path
    db_path.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _patch_create_future(monkeypatch):
    monkeypatch.setattr(schedule_helpers, 'create_parsl_future', _stub_create_future)


def _add(db_path, path, status='Ready', num_nodes=1, parents=None, app='python'):
    return database.add_job(db_path, path, app, num_nodes, 0, 1.0, 't',
                            status=status, parents=parents)


def _mk_ctx(db_path, rm, tracker, owner='A'):
    return SchedulerContext(
        db_path=db_path, scheduler='PBS', owner=owner, config_name='test',
        resource_manager=rm, job_tracker=tracker,
        status_buffer=StatusBuffer(db_path), system_config=object(),
        app_instances={'python': FakeApp()}, app_configs={'python': {}},
        mpi_configs={'python': object()}, restarting_job_ids=set(),
        fut_to_item={}, shutdown_at=1e18,
    )


def _status(db_path, jid):
    return database.get_jobs_by_ids(db_path, [jid])[0]['status']


def _simulate_completion(ctx, job_id, status='Done'):
    """Mimic handle_completion between the as_completed wait and the next
    dispatch: pop the future, mark the tracker, BUFFER the terminal status
    (NOT flushed), and free resources. Running/Done ride the buffer, so the DB
    still shows the pre-buffer status until a flush happens."""
    ctx.fut_to_item.pop(f'fut-{job_id}', None)
    ctx.job_tracker.update_job_status(job_id, status)
    ctx.status_buffer.add_status_update(job_id, status=status)
    ctx.resource_manager.free_resources_with_health_check(job_id)


# --------------------------------------------------------------------------- #
# coarse_pick
# --------------------------------------------------------------------------- #
class TestCoarsePick:
    def test_bounded_by_budget(self):
        cands = [{'job_id': i, 'num_nodes': 1} for i in range(5)]
        assert [j['job_id'] for j in coarse_pick(cands, 2)] == [0, 1]

    def test_multinode_accumulates(self):
        cands = [{'job_id': 1, 'num_nodes': 3}, {'job_id': 2, 'num_nodes': 2}]
        assert [j['job_id'] for j in coarse_pick(cands, 3)] == [1]  # 3 fills budget
        assert [j['job_id'] for j in coarse_pick(cands, 5)] == [1, 2]

    def test_skips_too_big_keeps_smaller(self):
        cands = [{'job_id': 1, 'num_nodes': 5}, {'job_id': 2, 'num_nodes': 1}]
        # 5 doesn't fit budget 2, but 1 does — keep scanning.
        assert [j['job_id'] for j in coarse_pick(cands, 2)] == [2]

    def test_zero_budget(self):
        assert coarse_pick([{'job_id': 1, 'num_nodes': 1}], 0) == []


# --------------------------------------------------------------------------- #
# dispatch_dynamic
# --------------------------------------------------------------------------- #
class TestDispatchDynamic:
    def test_dispatches_up_to_capacity(self, db):
        ids = [_add(db, f'/j{i}') for i in range(5)]
        rm = FakeRM(capacity=2)
        tracker = JobTracker([], db)
        rm.job_tracker = tracker
        ctx = _mk_ctx(db, rm, tracker)

        n = dispatch_dynamic(ctx, apps=None, tags=None)
        assert n == 2
        running = [i for i in ids if _status(db, i) == 'Submitted']
        # 2 dispatched (claimed Submitted in DB; Running only after flush)
        assert len(ctx.fut_to_item) == 2
        ctx.status_buffer.flush_all()
        assert sum(_status(db, i) == 'Running' for i in ids) == 2
        assert sum(_status(db, i) == 'Ready' for i in ids) == 3

    def test_all_fit(self, db):
        ids = [_add(db, f'/j{i}') for i in range(3)]
        rm = FakeRM(capacity=10)
        tracker = JobTracker([], db)
        rm.job_tracker = tracker
        ctx = _mk_ctx(db, rm, tracker)
        assert dispatch_dynamic(ctx, None, None) == 3
        assert len(ctx.fut_to_item) == 3

    def test_dep_unready_not_dispatched(self, db):
        parent = _add(db, '/p', status='Ready')
        child = _add(db, '/c', parents=[parent])
        rm = FakeRM(capacity=10)
        tracker = JobTracker([], db)
        rm.job_tracker = tracker
        ctx = _mk_ctx(db, rm, tracker)

        dispatch_dynamic(ctx, None, None)
        assert _status(db, parent) == 'Submitted'   # dispatched (claimed)
        assert _status(db, child) == 'Ready'        # blocked, untouched
        assert child not in [it['job']['job_id'] for it in ctx.fut_to_item.values()]

    def test_child_dispatched_after_parent_done(self, db):
        parent = _add(db, '/p', status='Done')
        child = _add(db, '/c', parents=[parent])
        rm = FakeRM(capacity=10)
        tracker = JobTracker([], db)
        rm.job_tracker = tracker
        ctx = _mk_ctx(db, rm, tracker)
        dispatch_dynamic(ctx, None, None)
        assert _status(db, child) == 'Submitted'

    def test_restart_becomes_resubmitted(self, db):
        j = _add(db, '/r', status='Restart')
        rm = FakeRM(capacity=10)
        tracker = JobTracker([], db)
        rm.job_tracker = tracker
        ctx = _mk_ctx(db, rm, tracker)
        dispatch_dynamic(ctx, None, None)
        assert _status(db, j) == 'Resubmitted'
        assert ctx.fut_to_item[f'fut-{j}']['is_restart'] is True

    def test_ignores_other_runs_claimed_jobs(self, db):
        mine = _add(db, '/mine')
        theirs = _add(db, '/theirs')
        database.claim_jobs(db, [theirs], [], owner='B')  # owned by B → Submitted
        rm = FakeRM(capacity=10)
        tracker = JobTracker([], db)
        rm.job_tracker = tracker
        ctx = _mk_ctx(db, rm, tracker, owner='A')
        dispatch_dynamic(ctx, None, None)
        assert _status(db, mine) == 'Submitted'
        # B's job untouched, still owned by B.
        row = database.get_jobs_by_ids(db, [theirs])[0]
        assert row['sched_job_id'] == 'B'

    def test_assign_miss_reverts_claim(self, db):
        good = _add(db, '/good')
        bad = _add(db, '/bad')
        rm = FakeRM(capacity=10, reject={bad})
        tracker = JobTracker([], db)
        rm.job_tracker = tracker
        ctx = _mk_ctx(db, rm, tracker, owner='A')
        dispatch_dynamic(ctx, None, None)
        assert _status(db, good) == 'Submitted'
        # bad was claimed then reverted back to the pool, owner cleared.
        row = database.get_jobs_by_ids(db, [bad])[0]
        assert row['status'] == 'Ready'
        assert row['sched_job_id'] is None

    def test_app_filter(self, db):
        vasp = _add(db, '/v', app='vasp')
        _add(db, '/p', app='python')
        rm = FakeRM(capacity=10)
        tracker = JobTracker([], db)
        rm.job_tracker = tracker
        ctx = _mk_ctx(db, rm, tracker)
        ctx.app_instances['vasp'] = FakeApp()
        ctx.app_configs['vasp'] = {}
        ctx.mpi_configs['vasp'] = object()
        dispatch_dynamic(ctx, apps=['vasp'], tags=None)
        assert _status(db, vasp) == 'Submitted'
        assert len(ctx.fut_to_item) == 1


# --------------------------------------------------------------------------- #
# dispatch_static
# --------------------------------------------------------------------------- #
class TestDispatchStatic:
    def _setup(self, db, jobs_status, capacity, reject=None, parents_map=None):
        parents_map = parents_map or {}
        ids = {}
        for name, st in jobs_status.items():
            ids[name] = _add(db, f'/{name}', status=st, parents=parents_map.get(name))
        # claim all up front (static)
        ready = [i for n, i in ids.items() if jobs_status[n] == 'Ready']
        restart = [i for n, i in ids.items() if jobs_status[n] == 'Restart']
        database.claim_jobs(db, ready, restart, owner='A')
        tracker = JobTracker(database.get_jobs(db), db)
        rm = FakeRM(capacity=capacity, reject=reject)
        rm.job_tracker = tracker
        for i in ids.values():
            rm.add_to_backlog(i)
        ctx = _mk_ctx(db, rm, tracker, owner='A')
        return ids, rm, ctx

    def test_drains_backlog_to_capacity(self, db):
        ids, rm, ctx = self._setup(db, {'a': 'Ready', 'b': 'Ready', 'c': 'Ready'}, capacity=2)
        n = dispatch_static(ctx)
        assert n == 2
        assert len(rm._backlogged_jobs_set) == 1  # one left, didn't fit

    def test_all_dispatched_when_capacity_ample(self, db):
        ids, rm, ctx = self._setup(db, {'a': 'Ready', 'b': 'Ready'}, capacity=10)
        n = dispatch_static(ctx)
        assert n == 2
        assert rm._backlogged_jobs_set == set()

    def test_dep_blocked_stays_in_backlog(self, db):
        # Parent Ready → claimed to Submitted (not Done), so the child is not
        # dep-ready and must stay parked; only the parent dispatches.
        parent = _add(db, '/p', status='Ready')
        child = _add(db, '/c', status='Ready', parents=[parent])
        database.claim_jobs(db, [parent, child], [], owner='A')
        tracker = JobTracker(database.get_jobs(db), db)
        rm = FakeRM(capacity=10)
        rm.job_tracker = tracker
        rm.add_to_backlog(parent)
        rm.add_to_backlog(child)
        ctx = _mk_ctx(db, rm, tracker, owner='A')

        n = dispatch_static(ctx)
        assert n == 1
        assert rm._backlogged_jobs_set == {child}


# --------------------------------------------------------------------------- #
# Flush-before-dispatch ordering invariant
#
# Running/Done ride the status buffer; parents_satisfied reads parent status
# fresh from the DB. So a just-completed parent whose Done is still buffered is
# invisible to dispatch — the child looks un-ready. The main loop MUST flush the
# buffer before dispatching, or the no-idle exit skips fan-in / aggregator jobs
# whose last parent just finished. These tests pin that contract at the helper
# level for both modes.
# --------------------------------------------------------------------------- #
class TestFlushBeforeDispatchOrdering:
    def test_dynamic_child_needs_parent_done_flushed(self, db):
        parent = _add(db, '/p', status='Ready')
        child = _add(db, '/c', status='Ready', parents=[parent])
        rm = FakeRM(capacity=10)
        tracker = JobTracker([], db)
        rm.job_tracker = tracker
        ctx = _mk_ctx(db, rm, tracker)

        # Pass 1: parent dispatches; child blocked (parent not yet Done).
        dispatch_dynamic(ctx, None, None)
        assert f'fut-{parent}' in ctx.fut_to_item
        assert f'fut-{child}' not in ctx.fut_to_item

        # Parent completes — Done buffered, not flushed.
        _simulate_completion(ctx, parent, 'Done')

        # Buggy ordering (dispatch before flush): parent's Done is buffered, DB
        # still shows 'Submitted', so the child is NOT picked up.
        dispatch_dynamic(ctx, None, None)
        assert f'fut-{child}' not in ctx.fut_to_item
        assert _status(db, child) == 'Ready'

        # Correct ordering (flush THEN dispatch): child becomes dep-ready.
        ctx.status_buffer.flush_all()
        assert _status(db, parent) == 'Done'
        dispatch_dynamic(ctx, None, None)
        assert f'fut-{child}' in ctx.fut_to_item
        assert _status(db, child) == 'Submitted'

    def test_static_child_needs_parent_done_flushed(self, db):
        parent = _add(db, '/p', status='Ready')
        child = _add(db, '/c', status='Ready', parents=[parent])
        database.claim_jobs(db, [parent, child], [], owner='A')
        tracker = JobTracker(database.get_jobs(db), db)
        rm = FakeRM(capacity=10)
        rm.job_tracker = tracker
        rm.add_to_backlog(parent)
        rm.add_to_backlog(child)
        ctx = _mk_ctx(db, rm, tracker, owner='A')

        # Pass 1: only parent is dep-ready.
        assert dispatch_static(ctx) == 1
        assert f'fut-{parent}' in ctx.fut_to_item
        assert rm._backlogged_jobs_set == {child}

        _simulate_completion(ctx, parent, 'Done')

        # Buggy ordering: child stays blocked and parked in the backlog.
        assert dispatch_static(ctx) == 0
        assert child in rm._backlogged_jobs_set

        # Correct ordering: flush first, child becomes dep-ready and dispatches.
        ctx.status_buffer.flush_all()
        assert dispatch_static(ctx) == 1
        assert f'fut-{child}' in ctx.fut_to_item
        assert rm._backlogged_jobs_set == set()


# --------------------------------------------------------------------------- #
# Completion drain + exit guard
# --------------------------------------------------------------------------- #
class CompletionApp(FakeApp):
    """FakeApp plus the two hooks handle_completion calls."""

    def check_success(self, job_id, job_path, db_path, error_message=None):
        return "Failed" if error_message else "Done"

    def postprocess(self, job_id, job_path, db_path):
        return "Done"


def _mk_completion_ctx(db_path, rm, tracker, owner='A'):
    ctx = _mk_ctx(db_path, rm, tracker, owner=owner)
    ctx.app_instances['python'] = CompletionApp()
    return ctx


def _seed_future(ctx, job_id, done=True):
    """Register a live future for job_id the way create_parsl_future would."""
    fut = Future()
    if done:
        fut.set_result(None)
    ctx.fut_to_item[fut] = {
        'job': {'job_id': job_id, 'path': '/tmp', 'app': 'python', 'status': 'Ready'},
        'app_instance': ctx.app_instances['python'],
        'resource_manager': ctx.resource_manager,
        'single_rank_launcher': None,
        'env_file': None,
        'gpu_env_vars': None,
    }
    ctx.resource_manager.assigned.add(job_id)
    return fut


class TestDrainCompletions:
    def _setup(self, db, n_jobs, capacity=100):
        ids = [_add(db, f'/j{i}') for i in range(n_jobs)]
        tracker = JobTracker(database.get_jobs(db), db)
        rm = FakeRM(capacity=capacity)
        rm.job_tracker = tracker
        ctx = _mk_completion_ctx(db, rm, tracker)
        for jid in ids:
            _seed_future(ctx, jid)
        return ids, rm, ctx

    def test_drains_every_done_future_in_one_call(self, db):
        ids, rm, ctx = self._setup(db, 5)
        assert drain_completions(ctx, poll_interval=0.05) == 5
        assert ctx.fut_to_item == {}

    def test_completions_accumulate_in_buffer_before_any_flush(self, db):
        """The regression this change exists for: one drain must leave N entries
        buffered, so the caller's single flush writes them as one batched
        UPDATE. Handling one completion per pass made every flush carry 1 row."""
        ids, rm, ctx = self._setup(db, 5)
        drain_completions(ctx, poll_interval=0.05)

        assert set(ctx.status_buffer.status_updates) == set(ids)
        # Nothing written yet — the buffer is still holding all five.
        assert all(_status(db, jid) == 'Ready' for jid in ids)

        assert ctx.status_buffer.flush_all() == 5
        assert all(_status(db, jid) == 'Done' for jid in ids)

    def test_frees_resources_for_every_handled_future(self, db):
        ids, rm, ctx = self._setup(db, 4)
        drain_completions(ctx, poll_interval=0.05)
        assert rm.assigned == set()

    def test_blocking_path_when_nothing_is_done(self, db):
        """No future ready → fall back to the bounded wait and handle at most
        one, so the caller still re-checks walltime at poll_interval cadence."""
        jid = _add(db, '/pending')
        tracker = JobTracker(database.get_jobs(db), db)
        rm = FakeRM(capacity=10)
        rm.job_tracker = tracker
        ctx = _mk_completion_ctx(db, rm, tracker)
        _seed_future(ctx, jid, done=False)

        assert drain_completions(ctx, poll_interval=0.05) == 0
        assert len(ctx.fut_to_item) == 1

    def test_blocking_path_handles_one_when_future_lands(self, db):
        jid = _add(db, '/late')
        tracker = JobTracker(database.get_jobs(db), db)
        rm = FakeRM(capacity=10)
        rm.job_tracker = tracker
        ctx = _mk_completion_ctx(db, rm, tracker)
        fut = _seed_future(ctx, jid, done=False)
        fut.set_result(None)  # lands before the wait starts

        assert drain_completions(ctx, poll_interval=0.05) == 1
        assert ctx.fut_to_item == {}

    def test_walltime_bail_stops_the_drain(self, db):
        """A long drain must not sail past shutdown_at — the caller's shutdown
        check only runs between passes."""
        ids, rm, ctx = self._setup(db, 5)
        ctx.shutdown_at = time.time() - 1  # already past the deadline

        handled = drain_completions(ctx, poll_interval=0.05)
        assert handled == 1                      # one, then the guard trips
        assert len(ctx.fut_to_item) == 4         # the rest stay for reconciliation

    def test_mutation_during_iteration_is_safe(self, db):
        """handle_completion pops from fut_to_item; drain must iterate a snapshot."""
        ids, rm, ctx = self._setup(db, 50)
        assert drain_completions(ctx, poll_interval=0.05) == 50


class TestCheckRunComplete:
    def _setup(self, db, capacity=10):
        tracker = JobTracker(database.get_jobs(db), db)
        rm = FakeRM(capacity=capacity)
        rm.job_tracker = tracker
        return rm, _mk_completion_ctx(db, rm, tracker)

    def test_none_while_futures_are_active(self, db):
        jid = _add(db, '/a')
        rm, ctx = self._setup(db)
        _seed_future(ctx, jid, done=False)
        assert check_run_complete(ctx, dynamic=True) is None

    def test_continue_flushes_pending_and_asks_for_another_pass(self, db):
        jid = _add(db, '/a')
        rm, ctx = self._setup(db)
        ctx.status_buffer.add_status_update(jid, status='Done')

        assert check_run_complete(ctx, dynamic=True) == 'continue'
        assert _status(db, jid) == 'Done'          # flushed on the way through

    def test_break_when_nothing_running_and_nothing_pending(self, db):
        rm, ctx = self._setup(db)
        assert check_run_complete(ctx, dynamic=True) == 'break'

    def test_terminates_after_a_single_continue(self, db):
        """The guard must not loop: the second visit finds an empty buffer."""
        jid = _add(db, '/a')
        rm, ctx = self._setup(db)
        ctx.status_buffer.add_status_update(jid, status='Done')

        assert check_run_complete(ctx, dynamic=True) == 'continue'
        assert check_run_complete(ctx, dynamic=True) == 'break'

    def test_fan_in_child_survives_the_no_idle_exit(self, db):
        """The bug the guard exists for: the last running job is a parent, its
        Done is still buffered, so the DB-backed dependency check cannot see it.
        Without the guard the run would exit and the child would be skipped."""
        parent = _add(db, '/p')
        child = _add(db, '/c', parents=[parent])
        tracker = JobTracker(database.get_jobs(db), db)
        rm = FakeRM(capacity=10)
        rm.job_tracker = tracker
        ctx = _mk_completion_ctx(db, rm, tracker)

        # Parent is dispatched, runs, and completes as the last active future.
        database.claim_jobs(db, [parent], [], owner='A')
        _seed_future(ctx, parent)
        assert drain_completions(ctx, poll_interval=0.05) == 1
        assert ctx.fut_to_item == {}
        assert _status(db, parent) == 'Submitted'   # Done is buffered, not written

        # Child is not yet dispatchable — the DB still shows the parent unfinished.
        assert dispatch_dynamic(ctx, None, None) == 0

        # Guard flushes and buys one more pass, on which the child dispatches.
        assert check_run_complete(ctx, dynamic=True) == 'continue'
        assert _status(db, parent) == 'Done'
        assert dispatch_dynamic(ctx, None, None) == 1
        assert f'fut-{child}' in ctx.fut_to_item

    def test_static_orphans_reverted_on_the_break_path(self, db):
        parent = _add(db, '/p')
        child = _add(db, '/c', parents=[parent])
        database.claim_jobs(db, [parent, child], [], owner='A')
        tracker = JobTracker(database.get_jobs(db), db)
        rm = FakeRM(capacity=10)
        rm.job_tracker = tracker
        rm.add_to_backlog(child)
        ctx = _mk_completion_ctx(db, rm, tracker)

        assert check_run_complete(ctx, dynamic=False) == 'break'
        assert _status(db, child) == 'Ready'        # returned to the pool
