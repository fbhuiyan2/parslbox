"""
Tests for `pbx qdel` and `pbx scancel` commands and their underlying
cancel_helpers + API methods.

Covers:
 - cancel_helpers.cancel_pbs_job / cancel_slurm_job
   - Happy-path subprocess ordering
   - First-stage (qsig / scancel-signal) failure short-circuits and does NOT reconcile
   - Second-stage (qdel / scancel) failure still triggers reconciliation
 - DB reconciliation
   - Flips stuck Running/Submitted with matching sched_job_id to Killed
   - Leaves other batch jobs (different sched_job_id) untouched
   - Leaves terminal-status jobs (Done/Failed/Killed/Warning) untouched
   - No-op when db_path is None (backwards compat)
 - CLI plumbing (qdel.py, scancel.py): passes db_path, surfaces reconciled_count,
   correct exit codes per stage
 - API plumbing (ParslBox.qdel / scancel): passes self.db_path
 - MCP response strings: surface reconciled_count when > 0, omit when 0
"""

import sqlite3
import subprocess
from pathlib import Path
from unittest.mock import patch, call, MagicMock

import pytest
from typer.testing import CliRunner

from parslbox.commands.helpers.cancel_helpers import cancel_pbs_job, cancel_slurm_job
from parslbox.commands.qdel import app as qdel_app
from parslbox.commands.scancel import app as scancel_app
from parslbox.database import database


# ---------------------------------------------------------------------------
# DB fixture for reconciliation tests
# ---------------------------------------------------------------------------

@pytest.fixture
def reconciliation_db(tmp_path):
    """Seed a temporary jobs DB with a mix of statuses + sched_job_ids.

    Layout:
      - Jobs 1, 2: status=Submitted, sched_job_id="BATCH_A"  ← should reconcile
      - Job  3:    status=Running,   sched_job_id="BATCH_A"  ← should reconcile
      - Job  4:    status=Done,      sched_job_id="BATCH_A"  ← terminal, leave alone
      - Job  5:    status=Killed,    sched_job_id="BATCH_A"  ← terminal, leave alone
      - Job  6:    status=Failed,    sched_job_id="BATCH_A"  ← terminal, leave alone
      - Job  7:    status=Warning,   sched_job_id="BATCH_A"  ← terminal, leave alone
      - Job  8:    status=Running,   sched_job_id="BATCH_B"  ← other batch, untouched
      - Job  9:    status=Submitted, sched_job_id="BATCH_B"  ← other batch, untouched
      - Job 10:    status=Running,   sched_job_id=NULL       ← never claimed
    """
    db_path = tmp_path / "jobs.db"
    database.initialize_database(db_path)

    seed = [
        (1, "Submitted", "BATCH_A"),
        (2, "Submitted", "BATCH_A"),
        (3, "Running",   "BATCH_A"),
        (4, "Done",      "BATCH_A"),
        (5, "Killed",    "BATCH_A"),
        (6, "Failed",    "BATCH_A"),
        (7, "Warning",   "BATCH_A"),
        (8, "Running",   "BATCH_B"),
        (9, "Submitted", "BATCH_B"),
        (10, "Running",  None),
    ]
    with sqlite3.connect(db_path) as con:
        for jid, status, sched in seed:
            # Unique (path, in_file) per row — the table has UNIQUE constraint
            # on the pair.
            con.execute(
                "INSERT INTO jobs (job_id, status, sched_job_id, app, path, in_file) "
                "VALUES (?, ?, ?, 'app', ?, 'in')",
                (jid, status, sched, f"/tmp/job{jid}"),
            )
        con.commit()
    return db_path


def _status_of(db_path: Path, job_id: int) -> str:
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# database.get_jobs_by_sched_id
# ---------------------------------------------------------------------------

class TestGetJobsBySchedId:
    def test_returns_only_matching_sched_id(self, reconciliation_db):
        rows = database.get_jobs_by_sched_id(reconciliation_db, "BATCH_A")
        ids = {r["job_id"] for r in rows}
        # BATCH_A has jobs 1-7
        assert ids == {1, 2, 3, 4, 5, 6, 7}

    def test_filters_by_status_list(self, reconciliation_db):
        rows = database.get_jobs_by_sched_id(
            reconciliation_db, "BATCH_A", statuses=["Running", "Submitted"]
        )
        ids = {r["job_id"] for r in rows}
        # Only the 3 non-terminal jobs in BATCH_A
        assert ids == {1, 2, 3}

    def test_does_not_match_other_batch(self, reconciliation_db):
        rows = database.get_jobs_by_sched_id(
            reconciliation_db, "BATCH_A", statuses=["Running", "Submitted"]
        )
        ids = {r["job_id"] for r in rows}
        assert 8 not in ids and 9 not in ids

    def test_empty_sched_id_returns_empty(self, reconciliation_db):
        assert database.get_jobs_by_sched_id(reconciliation_db, "") == []

    def test_no_match_returns_empty(self, reconciliation_db):
        assert database.get_jobs_by_sched_id(
            reconciliation_db, "DOES_NOT_EXIST", statuses=["Running"]
        ) == []


# ---------------------------------------------------------------------------
# cancel_helpers.cancel_pbs_job
# ---------------------------------------------------------------------------

class TestCancelPbsJob:
    def test_calls_qsig_sleep_qdel_in_order(self):
        """qsig SIGTERM → sleep(grace) → qdel"""
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run") as mrun, \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep") as msleep:
            mrun.return_value = MagicMock(returncode=0)
            result = cancel_pbs_job("12345", grace=7)

        assert result["success"] is True
        assert result["jobid"] == "12345"
        assert result["grace"] == 7
        assert result["reconciled_count"] == 0

        assert mrun.call_args_list == [
            call(["qsig", "-s", "SIGTERM", "12345"], check=True,
                 capture_output=True, text=True),
            call(["qdel", "12345"], check=True,
                 capture_output=True, text=True),
        ]
        msleep.assert_called_once_with(7)

    def test_default_grace_is_30(self):
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run") as mrun, \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep") as msleep:
            mrun.return_value = MagicMock(returncode=0)
            result = cancel_pbs_job("99")
        assert result["grace"] == 30
        msleep.assert_called_once_with(30)

    def test_qsig_failure_short_circuits_and_skips_reconciliation(self, reconciliation_db):
        """If qsig fails, qdel is NOT invoked, sleep NOT called, DB NOT touched.

        Critical: signal was never delivered → batch may still be running →
        flipping its claimed jobs to Killed would corrupt state.
        """
        err = subprocess.CalledProcessError(1, ["qsig"], stderr="invalid job")
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run",
                   side_effect=err) as mrun, \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep") as msleep:
            result = cancel_pbs_job("BATCH_A", grace=5, db_path=reconciliation_db)

        assert result["success"] is False
        assert result["stage"] == "qsig"
        assert "invalid job" in result["error"]
        assert result["reconciled_count"] == 0
        msleep.assert_not_called()
        assert mrun.call_count == 1
        # DB untouched: stuck jobs should still be Running/Submitted
        assert _status_of(reconciliation_db, 1) == "Submitted"
        assert _status_of(reconciliation_db, 3) == "Running"

    def test_qsig_missing_binary(self):
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run",
                   side_effect=FileNotFoundError), \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep"):
            result = cancel_pbs_job("777")
        assert result["success"] is False
        assert result["stage"] == "qsig"
        assert "not found" in result["error"]
        assert result["reconciled_count"] == 0

    def test_qdel_failure_still_triggers_reconciliation(self, reconciliation_db):
        """The user's actual scenario: qsig OK, qdel returns "Request invalid for
        state of job" (batch already gone), but the orchestrator's signal
        handler didn't finish STEP 2 — so the DB has stuck jobs. Reconciliation
        cleans them up regardless of qdel's outcome."""
        ok = MagicMock(returncode=0)
        bad = subprocess.CalledProcessError(
            1, ["qdel"], stderr="qdel: Request invalid for state of job"
        )
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run",
                   side_effect=[ok, bad]), \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep"):
            result = cancel_pbs_job("BATCH_A", grace=1, db_path=reconciliation_db)

        assert result["success"] is False
        assert result["stage"] == "qdel"
        assert "Request invalid" in result["error"]
        # 3 stuck jobs in BATCH_A (1, 2 Submitted; 3 Running) should be cleaned up
        assert result["reconciled_count"] == 3

    def test_no_db_path_skips_reconciliation(self):
        """Backwards compatibility: callers that don't pass db_path see no DB activity."""
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run") as mrun, \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep"), \
             patch("parslbox.commands.helpers.cancel_helpers.database") as mdb:
            mrun.return_value = MagicMock(returncode=0)
            result = cancel_pbs_job("anything", grace=1)  # db_path defaults to None
        assert result["reconciled_count"] == 0
        # DB module should never have been called
        mdb.get_jobs_by_sched_id.assert_not_called()
        mdb.update_jobs.assert_not_called()


# ---------------------------------------------------------------------------
# Reconciliation correctness across batches and statuses
# ---------------------------------------------------------------------------

class TestReconciliationScope:
    def test_reconciles_per_state_in_target_batch(self, reconciliation_db):
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run",
                   return_value=MagicMock(returncode=0)), \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep"):
            result = cancel_pbs_job("BATCH_A", grace=1, db_path=reconciliation_db)

        assert result["reconciled_count"] == 3
        # Submitted (claimed, never ran) → back to the pool as Ready
        assert _status_of(reconciliation_db, 1) == "Ready"
        assert _status_of(reconciliation_db, 2) == "Ready"
        # Running (was executing) → Killed
        assert _status_of(reconciliation_db, 3) == "Killed"

    def test_does_not_touch_terminal_status_jobs(self, reconciliation_db):
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run",
                   return_value=MagicMock(returncode=0)), \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep"):
            cancel_pbs_job("BATCH_A", grace=1, db_path=reconciliation_db)
        # Done/Killed/Failed/Warning in BATCH_A all preserved
        assert _status_of(reconciliation_db, 4) == "Done"
        assert _status_of(reconciliation_db, 5) == "Killed"
        assert _status_of(reconciliation_db, 6) == "Failed"
        assert _status_of(reconciliation_db, 7) == "Warning"

    def test_does_not_touch_other_batches(self, reconciliation_db):
        """Critical multi-batch safety test — concurrent pbx run B must NOT
        have its jobs flipped by a qdel against batch A."""
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run",
                   return_value=MagicMock(returncode=0)), \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep"):
            cancel_pbs_job("BATCH_A", grace=1, db_path=reconciliation_db)
        # BATCH_B's jobs intact
        assert _status_of(reconciliation_db, 8) == "Running"
        assert _status_of(reconciliation_db, 9) == "Submitted"
        # NULL sched_job_id also untouched
        assert _status_of(reconciliation_db, 10) == "Running"

    def test_no_stuck_jobs_is_silent_noop(self, tmp_path):
        """If the signal handler completed cleanly (no stuck jobs), reconciliation
        is a silent no-op — reconciled_count == 0, no errors."""
        db_path = tmp_path / "jobs.db"
        database.initialize_database(db_path)
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run",
                   return_value=MagicMock(returncode=0)), \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep"):
            result = cancel_pbs_job("BATCH_X", grace=1, db_path=db_path)
        assert result["success"] is True
        assert result["reconciled_count"] == 0


# ---------------------------------------------------------------------------
# cancel_helpers.cancel_slurm_job — mirrors PBS shape
# ---------------------------------------------------------------------------

class TestCancelSlurmJob:
    def test_calls_scancel_signal_sleep_scancel_in_order(self):
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run") as mrun, \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep") as msleep:
            mrun.return_value = MagicMock(returncode=0)
            result = cancel_slurm_job("999", grace=12)

        assert result["success"] is True
        assert result["jobid"] == "999"
        assert result["grace"] == 12
        assert result["reconciled_count"] == 0

        assert mrun.call_args_list == [
            call(["scancel", "--signal=TERM", "--batch", "999"], check=True,
                 capture_output=True, text=True),
            call(["scancel", "999"], check=True,
                 capture_output=True, text=True),
        ]
        msleep.assert_called_once_with(12)

    def test_default_grace_is_30(self):
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run") as mrun, \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep") as msleep:
            mrun.return_value = MagicMock(returncode=0)
            cancel_slurm_job("1")
        msleep.assert_called_once_with(30)

    def test_signal_stage_failure_short_circuits_and_skips_reconciliation(self, reconciliation_db):
        err = subprocess.CalledProcessError(1, ["scancel"], stderr="nope")
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run",
                   side_effect=err), \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep") as msleep:
            result = cancel_slurm_job("BATCH_A", db_path=reconciliation_db)
        assert result["success"] is False
        assert result["stage"] == "scancel-signal"
        assert result["reconciled_count"] == 0
        msleep.assert_not_called()
        assert _status_of(reconciliation_db, 1) == "Submitted"  # untouched

    def test_hard_cancel_failure_still_triggers_reconciliation(self, reconciliation_db):
        ok = MagicMock(returncode=0)
        bad = subprocess.CalledProcessError(1, ["scancel"], stderr="job already gone")
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run",
                   side_effect=[ok, bad]), \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep"):
            result = cancel_slurm_job("BATCH_A", grace=1, db_path=reconciliation_db)
        assert result["success"] is False
        assert result["stage"] == "scancel"
        assert result["reconciled_count"] == 3


# ---------------------------------------------------------------------------
# CLI: `pbx qdel`
# ---------------------------------------------------------------------------

class TestQdelCli:
    def setup_method(self):
        self.runner = CliRunner()

    def test_cli_passes_db_path_to_helper(self):
        """The CLI must thread path_utils.DB_FILE through to enable reconciliation."""
        from parslbox.utils import path_utils
        with patch("parslbox.commands.qdel.cancel_pbs_job",
                   return_value={"success": True, "jobid": "42", "grace": 30,
                                 "reconciled_count": 0}) as mcancel:
            result = self.runner.invoke(qdel_app, ["42"])
        assert result.exit_code == 0
        mcancel.assert_called_once_with("42", grace=30, db_path=path_utils.DB_FILE)
        assert "cancelled cleanly" in result.stdout

    def test_cli_grace_flag_passed_through(self):
        from parslbox.utils import path_utils
        with patch("parslbox.commands.qdel.cancel_pbs_job",
                   return_value={"success": True, "jobid": "42", "grace": 5,
                                 "reconciled_count": 0}) as mcancel:
            self.runner.invoke(qdel_app, ["42", "--grace", "5"])
        mcancel.assert_called_once_with("42", grace=5, db_path=path_utils.DB_FILE)

    def test_cli_qsig_failure_exits_nonzero(self):
        """qsig-stage failure → signal never delivered → exit 1."""
        with patch("parslbox.commands.qdel.cancel_pbs_job",
                   return_value={"success": False, "jobid": "42",
                                 "stage": "qsig", "error": "no such job",
                                 "reconciled_count": 0}):
            result = self.runner.invoke(qdel_app, ["42"])
        assert result.exit_code == 1
        assert "Cancel failed" in result.stdout
        assert "qsig" in result.stdout

    def test_cli_qdel_failure_with_reconciliation_exits_zero(self):
        """qdel-stage failure but signal was delivered and DB cleaned up →
        user's intent satisfied → exit 0 with warning."""
        with patch("parslbox.commands.qdel.cancel_pbs_job",
                   return_value={"success": False, "jobid": "42",
                                 "stage": "qdel",
                                 "error": "Request invalid for state of job",
                                 "reconciled_count": 4}):
            result = self.runner.invoke(qdel_app, ["42"])
        assert result.exit_code == 0
        assert "Hard-kill stage 'qdel'" in result.stdout
        assert "Reconciled 4 job(s)" in result.stdout

    def test_cli_surfaces_reconciliation_on_clean_success(self):
        """Even on full success, if reconciliation flipped some jobs, the CLI
        must tell the user — that's a signal the signal handler had problems."""
        with patch("parslbox.commands.qdel.cancel_pbs_job",
                   return_value={"success": True, "jobid": "42", "grace": 30,
                                 "reconciled_count": 2}):
            result = self.runner.invoke(qdel_app, ["42"])
        assert result.exit_code == 0
        assert "cancelled cleanly" in result.stdout
        assert "Reconciled 2 job(s)" in result.stdout


# ---------------------------------------------------------------------------
# CLI: `pbx scancel`
# ---------------------------------------------------------------------------

class TestScancelCli:
    def setup_method(self):
        self.runner = CliRunner()

    def test_cli_passes_db_path_to_helper(self):
        from parslbox.utils import path_utils
        with patch("parslbox.commands.scancel.cancel_slurm_job",
                   return_value={"success": True, "jobid": "777", "grace": 30,
                                 "reconciled_count": 0}) as mcancel:
            result = self.runner.invoke(scancel_app, ["777"])
        assert result.exit_code == 0
        mcancel.assert_called_once_with("777", grace=30, db_path=path_utils.DB_FILE)
        assert "cancelled cleanly" in result.stdout

    def test_cli_grace_flag_passed_through(self):
        from parslbox.utils import path_utils
        with patch("parslbox.commands.scancel.cancel_slurm_job",
                   return_value={"success": True, "jobid": "777", "grace": 10,
                                 "reconciled_count": 0}) as mcancel:
            self.runner.invoke(scancel_app, ["777", "-g", "10"])
        mcancel.assert_called_once_with("777", grace=10, db_path=path_utils.DB_FILE)

    def test_cli_signal_stage_failure_exits_nonzero(self):
        with patch("parslbox.commands.scancel.cancel_slurm_job",
                   return_value={"success": False, "jobid": "777",
                                 "stage": "scancel-signal",
                                 "error": "perm denied",
                                 "reconciled_count": 0}):
            result = self.runner.invoke(scancel_app, ["777"])
        assert result.exit_code == 1
        assert "perm denied" in result.stdout

    def test_cli_scancel_failure_with_reconciliation_exits_zero(self):
        with patch("parslbox.commands.scancel.cancel_slurm_job",
                   return_value={"success": False, "jobid": "777",
                                 "stage": "scancel", "error": "already gone",
                                 "reconciled_count": 3}):
            result = self.runner.invoke(scancel_app, ["777"])
        assert result.exit_code == 0
        assert "Hard-cancel stage 'scancel'" in result.stdout
        assert "Reconciled 3 job(s)" in result.stdout


# ---------------------------------------------------------------------------
# API: ParslBox.qdel / ParslBox.scancel
# ---------------------------------------------------------------------------

class TestApiCancelMethods:
    """The API must thread self.db_path through to enable reconciliation."""

    def test_api_qdel_passes_db_path(self):
        from parslbox.api import ParslBox
        with patch("parslbox.api.cancel_pbs_job",
                   return_value={"success": True, "jobid": "1", "grace": 30,
                                 "reconciled_count": 0}) as mcancel:
            pbx = ParslBox.__new__(ParslBox)
            pbx.db_path = Path("/tmp/fake_pbx.db")
            result = pbx.qdel("1", grace=15)
        mcancel.assert_called_once_with("1", grace=15, db_path=Path("/tmp/fake_pbx.db"))
        assert result["success"] is True

    def test_api_scancel_passes_db_path(self):
        from parslbox.api import ParslBox
        with patch("parslbox.api.cancel_slurm_job",
                   return_value={"success": True, "jobid": "2", "grace": 30,
                                 "reconciled_count": 0}) as mcancel:
            pbx = ParslBox.__new__(ParslBox)
            pbx.db_path = Path("/tmp/fake_pbx.db")
            result = pbx.scancel("2", grace=20)
        mcancel.assert_called_once_with("2", grace=20, db_path=Path("/tmp/fake_pbx.db"))
        assert result["success"] is True


# ---------------------------------------------------------------------------
# MCP: cancel_pbs_job / cancel_slurm_job response strings
# ---------------------------------------------------------------------------

class TestMcpCancelResponses:
    def test_mcp_pbs_clean_success_no_recon_no_suffix(self):
        from parslbox.mcp import mcp_server as srv
        with patch.object(srv.pbx, "qdel",
                          return_value={"success": True, "jobid": "1",
                                        "grace": 30, "reconciled_count": 0}):
            schema = srv.CancelJobSchema(jobid="1", grace=30)
            msg = srv.cancel_pbs_job(schema)
        assert "cancelled cleanly" in msg
        assert "Reconciled" not in msg

    def test_mcp_pbs_success_with_reconciliation_includes_suffix(self):
        from parslbox.mcp import mcp_server as srv
        with patch.object(srv.pbx, "qdel",
                          return_value={"success": True, "jobid": "1",
                                        "grace": 30, "reconciled_count": 5}):
            schema = srv.CancelJobSchema(jobid="1", grace=30)
            msg = srv.cancel_pbs_job(schema)
        assert "cancelled cleanly" in msg
        assert "Reconciled 5 job(s)" in msg

    def test_mcp_pbs_failure_with_reconciliation_includes_suffix(self):
        from parslbox.mcp import mcp_server as srv
        with patch.object(srv.pbx, "qdel",
                          return_value={"success": False, "jobid": "1",
                                        "stage": "qdel", "error": "gone",
                                        "reconciled_count": 2}):
            schema = srv.CancelJobSchema(jobid="1", grace=30)
            msg = srv.cancel_pbs_job(schema)
        assert "Failed to cancel" in msg
        assert "Reconciled 2 job(s)" in msg

    def test_mcp_slurm_success_with_reconciliation_includes_suffix(self):
        from parslbox.mcp import mcp_server as srv
        with patch.object(srv.pbx, "scancel",
                          return_value={"success": True, "jobid": "9",
                                        "grace": 30, "reconciled_count": 1}):
            schema = srv.CancelJobSchema(jobid="9", grace=30)
            msg = srv.cancel_slurm_job(schema)
        assert "cancelled cleanly" in msg
        assert "Reconciled 1 job(s)" in msg
