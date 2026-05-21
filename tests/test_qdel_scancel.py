"""
Tests for `pbx qdel` and `pbx scancel` commands and their underlying
cancel_helpers + API methods.
"""

import subprocess
from unittest.mock import patch, call, MagicMock

import pytest
from typer.testing import CliRunner

from parslbox.commands.helpers.cancel_helpers import cancel_pbs_job, cancel_slurm_job
from parslbox.commands.qdel import app as qdel_app
from parslbox.commands.scancel import app as scancel_app


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

        # Two subprocess calls in order: qsig, then qdel
        assert mrun.call_args_list == [
            call(["qsig", "-s", "SIGTERM", "12345"], check=True,
                 capture_output=True, text=True),
            call(["qdel", "12345"], check=True,
                 capture_output=True, text=True),
        ]
        # Sleep happens once with the requested grace
        msleep.assert_called_once_with(7)

    def test_default_grace_is_30(self):
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run") as mrun, \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep") as msleep:
            mrun.return_value = MagicMock(returncode=0)
            result = cancel_pbs_job("99")
        assert result["grace"] == 30
        msleep.assert_called_once_with(30)

    def test_qsig_failure_short_circuits(self):
        """If qsig fails, qdel is NOT invoked and sleep is NOT called."""
        err = subprocess.CalledProcessError(1, ["qsig"], stderr="invalid job")
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run",
                   side_effect=err) as mrun, \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep") as msleep:
            result = cancel_pbs_job("777", grace=5)

        assert result["success"] is False
        assert result["stage"] == "qsig"
        assert "invalid job" in result["error"]
        msleep.assert_not_called()
        # Only qsig was attempted
        assert mrun.call_count == 1

    def test_qsig_missing_binary(self):
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run",
                   side_effect=FileNotFoundError), \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep"):
            result = cancel_pbs_job("777")
        assert result["success"] is False
        assert result["stage"] == "qsig"
        assert "not found" in result["error"]

    def test_qdel_failure_after_qsig_succeeds(self):
        ok = MagicMock(returncode=0)
        bad = subprocess.CalledProcessError(1, ["qdel"], stderr="boom")
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run",
                   side_effect=[ok, bad]), \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep"):
            result = cancel_pbs_job("777", grace=1)
        assert result["success"] is False
        assert result["stage"] == "qdel"
        assert "boom" in result["error"]


# ---------------------------------------------------------------------------
# cancel_helpers.cancel_slurm_job
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

    def test_signal_stage_failure_short_circuits(self):
        err = subprocess.CalledProcessError(1, ["scancel"], stderr="nope")
        with patch("parslbox.commands.helpers.cancel_helpers.subprocess.run",
                   side_effect=err), \
             patch("parslbox.commands.helpers.cancel_helpers.time.sleep") as msleep:
            result = cancel_slurm_job("777")
        assert result["success"] is False
        assert result["stage"] == "scancel-signal"
        msleep.assert_not_called()


# ---------------------------------------------------------------------------
# CLI: `pbx qdel`
# ---------------------------------------------------------------------------

class TestQdelCli:
    def setup_method(self):
        self.runner = CliRunner()

    def test_cli_invokes_cancel_pbs_job(self):
        with patch("parslbox.commands.qdel.cancel_pbs_job",
                   return_value={"success": True, "jobid": "42", "grace": 30}) as mcancel:
            result = self.runner.invoke(qdel_app, ["42"])
        assert result.exit_code == 0
        mcancel.assert_called_once_with("42", grace=30)
        assert "cancelled cleanly" in result.stdout

    def test_cli_grace_flag_passed_through(self):
        with patch("parslbox.commands.qdel.cancel_pbs_job",
                   return_value={"success": True, "jobid": "42", "grace": 5}) as mcancel:
            self.runner.invoke(qdel_app, ["42", "--grace", "5"])
        mcancel.assert_called_once_with("42", grace=5)

    def test_cli_reports_failure_with_nonzero_exit(self):
        with patch("parslbox.commands.qdel.cancel_pbs_job",
                   return_value={"success": False, "jobid": "42",
                                 "stage": "qsig", "error": "no such job"}):
            result = self.runner.invoke(qdel_app, ["42"])
        assert result.exit_code == 1
        assert "Cancel failed" in result.stdout
        assert "qsig" in result.stdout


# ---------------------------------------------------------------------------
# CLI: `pbx scancel`
# ---------------------------------------------------------------------------

class TestScancelCli:
    def setup_method(self):
        self.runner = CliRunner()

    def test_cli_invokes_cancel_slurm_job(self):
        with patch("parslbox.commands.scancel.cancel_slurm_job",
                   return_value={"success": True, "jobid": "777", "grace": 30}) as mcancel:
            result = self.runner.invoke(scancel_app, ["777"])
        assert result.exit_code == 0
        mcancel.assert_called_once_with("777", grace=30)
        assert "cancelled cleanly" in result.stdout

    def test_cli_grace_flag_passed_through(self):
        with patch("parslbox.commands.scancel.cancel_slurm_job",
                   return_value={"success": True, "jobid": "777", "grace": 10}) as mcancel:
            self.runner.invoke(scancel_app, ["777", "-g", "10"])
        mcancel.assert_called_once_with("777", grace=10)

    def test_cli_reports_failure_with_nonzero_exit(self):
        with patch("parslbox.commands.scancel.cancel_slurm_job",
                   return_value={"success": False, "jobid": "777",
                                 "stage": "scancel-signal", "error": "perm denied"}):
            result = self.runner.invoke(scancel_app, ["777"])
        assert result.exit_code == 1
        assert "perm denied" in result.stdout


# ---------------------------------------------------------------------------
# API: ParslBox.qdel / ParslBox.scancel
# ---------------------------------------------------------------------------

class TestApiCancelMethods:
    """Thin smoke tests — verify the API methods delegate to the helpers."""

    def test_api_qdel_delegates(self):
        from parslbox.api import ParslBox
        with patch("parslbox.api.cancel_pbs_job",
                   return_value={"success": True, "jobid": "1", "grace": 30}) as mcancel:
            # Bypass __init__ (no config required for this unit test)
            pbx = ParslBox.__new__(ParslBox)
            result = pbx.qdel("1", grace=15)
        mcancel.assert_called_once_with("1", grace=15)
        assert result["success"] is True

    def test_api_scancel_delegates(self):
        from parslbox.api import ParslBox
        with patch("parslbox.api.cancel_slurm_job",
                   return_value={"success": True, "jobid": "2", "grace": 30}) as mcancel:
            pbx = ParslBox.__new__(ParslBox)
            result = pbx.scancel("2", grace=20)
        mcancel.assert_called_once_with("2", grace=20)
        assert result["success"] is True
