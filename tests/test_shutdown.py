"""
Tests for the graceful-shutdown machinery in run_cmd_helpers.py:
- perform_shutdown() with cleanup_parsl=True / False
- per-state, owner-scoped reconcile (Running→Killed/Restart/Failed,
  Submitted→Ready, Resubmitted→Restart)
- create_shutdown_handler() delegates to perform_shutdown(cleanup_parsl=False)
  and exits 130.
"""

import logging
import signal
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from parslbox.database import database
from parslbox.database.status_buffer import StatusBuffer
from parslbox.resource_manager.job_tracker import JobTracker
from parslbox.commands.helpers.run_cmd_helpers import (
    perform_shutdown,
    create_shutdown_handler,
)


OWNER = "BATCH_X"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def setup_db():
    """Temp sqlite db with a mix of job states, all owned by OWNER."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_path = Path(tmp.name)
    tmp.close()
    database.initialize_database(db_path)

    j1 = database.add_job(db_path, "/p1", "test_app", 1, 0, 1.0, "t", status="Running")
    j2 = database.add_job(db_path, "/p2", "test_app", 1, 0, 1.0, "t", status="Submitted")
    j3 = database.add_job(db_path, "/p3", "test_app", 1, 0, 1.0, "t", status="Done")
    j4 = database.add_job(db_path, "/p4", "test_app", 1, 0, 1.0, "t", status="Failed")
    j5 = database.add_job(db_path, "/p5", "test_app", 1, 0, 1.0, "t", status="Resubmitted")
    # Stamp the owner on this run's non-terminal jobs (claim owner token).
    database.update_jobs(db_path, [j1, j2, j5], sched_job_id=OWNER)

    initial = database.get_jobs_by_ids(db_path, [j1, j2, j3, j4, j5])
    job_tracker = JobTracker(initial, db_path)
    status_buffer = StatusBuffer(db_path)

    yield {
        "db_path": db_path,
        "job_tracker": job_tracker,
        "status_buffer": status_buffer,
        "ids": {"running": j1, "submitted": j2, "done": j3, "failed": j4, "resubmitted": j5},
    }

    db_path.unlink(missing_ok=True)


def _statuses(db_path, ids):
    rows = database.get_jobs_by_ids(db_path, list(ids.values()))
    return {row['job_id']: row['status'] for row in rows}


# ---------------------------------------------------------------------------
# perform_shutdown
# ---------------------------------------------------------------------------

class TestPerformShutdown:
    def test_signal_path_reconciles_per_state_and_skips_parsl(self, setup_db):
        logger = logging.getLogger("test_shutdown_signal")
        with patch("parslbox.commands.helpers.run_cmd_helpers.parsl") as mparsl:
            perform_shutdown(
                status_buffer=setup_db["status_buffer"],
                job_tracker=setup_db["job_tracker"],
                parsl_loaded_flag={"loaded": True},
                logger=logger,
                reason="signal SIGTERM",
                cleanup_parsl=False,
                owner=OWNER,
            )

        ids = setup_db["ids"]
        statuses = _statuses(setup_db["db_path"], ids)
        assert statuses[ids["running"]] == "Killed"        # was executing
        assert statuses[ids["submitted"]] == "Ready"       # claimed, never ran
        assert statuses[ids["resubmitted"]] == "Restart"   # claimed, never ran
        assert statuses[ids["done"]] == "Done"             # terminal untouched
        assert statuses[ids["failed"]] == "Failed"

        # Signal path explicitly skips parsl.dfk().cleanup()
        mparsl.dfk.assert_not_called()

    def test_walltime_no_respawn_running_killed(self, setup_db):
        logger = logging.getLogger("test_shutdown_walltime")
        with patch("parslbox.commands.helpers.run_cmd_helpers.parsl") as mparsl:
            perform_shutdown(
                status_buffer=setup_db["status_buffer"],
                job_tracker=setup_db["job_tracker"],
                parsl_loaded_flag={"loaded": True},
                logger=logger,
                reason="walltime",
                cleanup_parsl=True,
                owner=OWNER,
            )

        ids = setup_db["ids"]
        statuses = _statuses(setup_db["db_path"], ids)
        assert statuses[ids["running"]] == "Killed"        # no respawn_ctx
        assert statuses[ids["submitted"]] == "Ready"
        assert statuses[ids["resubmitted"]] == "Restart"

        mparsl.dfk.return_value.cleanup.assert_called_once()

    def test_walltime_path_skips_parsl_when_not_loaded(self, setup_db):
        logger = logging.getLogger("test_shutdown_unloaded")
        with patch("parslbox.commands.helpers.run_cmd_helpers.parsl") as mparsl:
            perform_shutdown(
                status_buffer=setup_db["status_buffer"],
                job_tracker=setup_db["job_tracker"],
                parsl_loaded_flag={"loaded": False},
                logger=logger,
                reason="walltime",
                cleanup_parsl=True,
                owner=OWNER,
            )
        mparsl.dfk.assert_not_called()

    def test_buffered_done_is_flushed_before_reconcile(self, setup_db):
        """A job that completed (buffered Done) must end up Done, NOT Killed —
        verifies flush-before-snapshot ordering."""
        logger = logging.getLogger("test_shutdown_ordering")
        sb = setup_db["status_buffer"]
        tracker = setup_db["job_tracker"]
        running_id = setup_db["ids"]["running"]

        # The running job just finished: tracker Done, buffer has pending Done,
        # DB still says Running.
        tracker.update_job_status(running_id, "Done")
        sb.add_status_update(running_id, status="Done")

        with patch("parslbox.commands.helpers.run_cmd_helpers.parsl"):
            perform_shutdown(
                status_buffer=sb,
                job_tracker=tracker,
                parsl_loaded_flag={"loaded": False},
                logger=logger,
                reason="walltime",
                cleanup_parsl=True,
                owner=OWNER,
            )

        statuses = _statuses(setup_db["db_path"], setup_db["ids"])
        # The just-completed job is preserved as Done (flushed before snapshot).
        assert statuses[running_id] == "Done"
        # The still-claimed job reverts to Ready.
        assert statuses[setup_db["ids"]["submitted"]] == "Ready"

    def test_no_owned_active_jobs_is_no_op(self, setup_db):
        """If none of this owner's jobs are non-terminal, no DB change."""
        logger = logging.getLogger("test_shutdown_empty")
        db_path = setup_db["db_path"]
        ids = setup_db["ids"]
        # Move this owner's non-terminal jobs to terminal states.
        database.update_jobs(db_path, [ids["running"]], status="Done")
        database.update_jobs(db_path, [ids["submitted"], ids["resubmitted"]], status="Done")

        before = _statuses(db_path, ids)
        with patch("parslbox.commands.helpers.run_cmd_helpers.parsl"):
            perform_shutdown(
                status_buffer=setup_db["status_buffer"],
                job_tracker=setup_db["job_tracker"],
                parsl_loaded_flag={"loaded": False},
                logger=logger,
                reason="walltime",
                cleanup_parsl=True,
                owner=OWNER,
            )
        after = _statuses(db_path, ids)
        assert before == after

    def test_reconcile_is_owner_scoped(self, setup_db):
        """Another run's jobs (different sched_job_id) must be left untouched."""
        logger = logging.getLogger("test_shutdown_owner_scope")
        db_path = setup_db["db_path"]
        other = database.add_job(db_path, "/other", "test_app", 1, 0, 1.0, "t", status="Running")
        database.update_jobs(db_path, [other], sched_job_id="BATCH_OTHER")

        with patch("parslbox.commands.helpers.run_cmd_helpers.parsl"):
            perform_shutdown(
                status_buffer=setup_db["status_buffer"],
                job_tracker=setup_db["job_tracker"],
                parsl_loaded_flag={"loaded": False},
                logger=logger,
                reason="walltime",
                cleanup_parsl=True,
                owner=OWNER,
            )
        # BATCH_OTHER's running job is not this run's — untouched.
        assert database.get_jobs_by_ids(db_path, [other])[0]["status"] == "Running"


# ---------------------------------------------------------------------------
# Signal handler wrapper
# ---------------------------------------------------------------------------

class TestSignalHandlerWrapper:
    def test_handler_calls_perform_shutdown_and_exits_130(self, setup_db):
        logger = logging.getLogger("test_handler_wrapper")
        handler = create_shutdown_handler(
            status_buffer=setup_db["status_buffer"],
            logger=logger,
            parsl_loaded_flag={"loaded": False},
            job_tracker=setup_db["job_tracker"],
            owner=OWNER,
        )

        with patch("parslbox.commands.helpers.run_cmd_helpers.perform_shutdown") as mshut:
            with pytest.raises(SystemExit) as excinfo:
                handler(signal.SIGTERM, None)

        assert excinfo.value.code == 130
        mshut.assert_called_once()
        kwargs = mshut.call_args.kwargs
        assert kwargs["cleanup_parsl"] is False
        assert "signal SIGTERM" in kwargs["reason"]
        assert kwargs["owner"] == OWNER
