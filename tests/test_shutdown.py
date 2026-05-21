"""
Tests for the graceful-shutdown machinery in run_cmd_helpers.py:
- perform_shutdown() with cleanup_parsl=True / False
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


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def setup_db():
    """Temp sqlite db with a mix of Running/Submitted/Done/Failed jobs."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_path = Path(tmp.name)
    tmp.close()
    database.initialize_database(db_path)

    # 4 jobs in different states
    j1 = database.add_job(db_path, "/p1", "test_app", 1, 0, 1.0, "t", status="Running")
    j2 = database.add_job(db_path, "/p2", "test_app", 1, 0, 1.0, "t", status="Submitted")
    j3 = database.add_job(db_path, "/p3", "test_app", 1, 0, 1.0, "t", status="Done")
    j4 = database.add_job(db_path, "/p4", "test_app", 1, 0, 1.0, "t", status="Failed")

    initial = database.get_jobs_by_ids(db_path, [j1, j2, j3, j4])
    job_tracker = JobTracker(initial, db_path)
    status_buffer = StatusBuffer(db_path)

    yield {
        "db_path": db_path,
        "job_tracker": job_tracker,
        "status_buffer": status_buffer,
        "ids": {"running": j1, "submitted": j2, "done": j3, "failed": j4},
    }

    db_path.unlink(missing_ok=True)


def _statuses(db_path, ids):
    rows = database.get_jobs_by_ids(db_path, list(ids.values()))
    return {row['job_id']: row['status'] for row in rows}


# ---------------------------------------------------------------------------
# perform_shutdown
# ---------------------------------------------------------------------------

class TestPerformShutdown:
    def test_signal_path_marks_active_jobs_killed_and_skips_parsl(self, setup_db):
        logger = logging.getLogger("test_shutdown_signal")
        with patch("parslbox.commands.helpers.run_cmd_helpers.parsl") as mparsl:
            perform_shutdown(
                status_buffer=setup_db["status_buffer"],
                job_tracker=setup_db["job_tracker"],
                parsl_loaded_flag={"loaded": True},
                logger=logger,
                reason="signal SIGTERM",
                cleanup_parsl=False,
            )

        # Running + Submitted -> Killed; Done + Failed untouched
        statuses = _statuses(setup_db["db_path"], setup_db["ids"])
        assert statuses[setup_db["ids"]["running"]] == "Killed"
        assert statuses[setup_db["ids"]["submitted"]] == "Killed"
        assert statuses[setup_db["ids"]["done"]] == "Done"
        assert statuses[setup_db["ids"]["failed"]] == "Failed"

        # Signal path explicitly skips parsl.dfk().cleanup()
        mparsl.dfk.assert_not_called()

    def test_walltime_path_calls_parsl_cleanup(self, setup_db):
        logger = logging.getLogger("test_shutdown_walltime")
        with patch("parslbox.commands.helpers.run_cmd_helpers.parsl") as mparsl:
            perform_shutdown(
                status_buffer=setup_db["status_buffer"],
                job_tracker=setup_db["job_tracker"],
                parsl_loaded_flag={"loaded": True},
                logger=logger,
                reason="walltime",
                cleanup_parsl=True,
            )

        # Same Killed marking
        statuses = _statuses(setup_db["db_path"], setup_db["ids"])
        assert statuses[setup_db["ids"]["running"]] == "Killed"
        assert statuses[setup_db["ids"]["submitted"]] == "Killed"

        # Walltime path runs Parsl cleanup
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
            )
        mparsl.dfk.assert_not_called()

    def test_buffered_done_is_flushed_before_killed_write(self, setup_db):
        """
        A job that completed (Done in JobTracker, buffered Done update) must
        end up Done in the DB after shutdown — NOT Killed. This verifies the
        flush-before-Killed ordering.
        """
        logger = logging.getLogger("test_shutdown_ordering")
        sb = setup_db["status_buffer"]
        tracker = setup_db["job_tracker"]
        running_id = setup_db["ids"]["running"]

        # Simulate: the running job just finished. Tracker is Done, buffer
        # has pending Done update, DB still says Running.
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
            )

        statuses = _statuses(setup_db["db_path"], setup_db["ids"])
        # The just-completed job is preserved as Done (not overwritten).
        assert statuses[running_id] == "Done"
        # The still-Submitted job is Killed.
        assert statuses[setup_db["ids"]["submitted"]] == "Killed"

    def test_no_active_jobs_is_no_op(self, setup_db):
        """If JobTracker has nothing Running/Submitted, no UPDATE happens."""
        logger = logging.getLogger("test_shutdown_empty")
        tracker = setup_db["job_tracker"]
        # Mark all Running/Submitted as Done in the tracker
        tracker.update_job_status(setup_db["ids"]["running"], "Done")
        tracker.update_job_status(setup_db["ids"]["submitted"], "Done")

        before = _statuses(setup_db["db_path"], setup_db["ids"])
        with patch("parslbox.commands.helpers.run_cmd_helpers.parsl"):
            perform_shutdown(
                status_buffer=setup_db["status_buffer"],
                job_tracker=tracker,
                parsl_loaded_flag={"loaded": False},
                logger=logger,
                reason="walltime",
                cleanup_parsl=True,
            )
        after = _statuses(setup_db["db_path"], setup_db["ids"])
        # No DB changes.
        assert before == after


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
        )

        with patch("parslbox.commands.helpers.run_cmd_helpers.perform_shutdown") as mshut:
            with pytest.raises(SystemExit) as excinfo:
                handler(signal.SIGTERM, None)

        # Exit code 130 = 128 + SIGINT-style convention used by the handler
        assert excinfo.value.code == 130
        # Called with cleanup_parsl=False (signal path)
        mshut.assert_called_once()
        kwargs = mshut.call_args.kwargs
        assert kwargs["cleanup_parsl"] is False
        assert "signal SIGTERM" in kwargs["reason"]
