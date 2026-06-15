"""
Tests for `should_gate_dispatch` — the helper that decides whether a job
should be skipped at dispatch time because the batch's remaining walltime
is below the app's declared floor.

Universal gate: applies at all three dispatch sites in `pbx run`
(initial dispatch, mid-run dynamic-discovery dispatch, backlog reschedule).
Skipped jobs stay in DB at their current Ready/Restart status and are
picked up by the next `pbx run`.

Restart-continuations (jobs already in `restarting_job_ids`) are exempt —
they have checkpoint state and even brief runtime advances the simulation.
"""

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from parslbox.commands.helpers.run_cmd_helpers import should_gate_dispatch


def _mk_app(floor: int):
    """Build a mock app whose min_remaining_walltime returns `floor`."""
    app = MagicMock()
    app.min_remaining_walltime.return_value = floor
    return app


def _mk_job(job_id: int = 42, app: str = "lammps-kk-restart"):
    return {"job_id": job_id, "app": app, "status": "Ready"}


# ---------------------------------------------------------------------------
# Floor=0 means the gate is inert
# ---------------------------------------------------------------------------


class TestFloorZeroNeverGates:
    @pytest.mark.parametrize("remaining_s", [0, 60, 600, 3600, 36000, 1_000_000])
    def test_zero_floor_never_gates(self, remaining_s):
        assert should_gate_dispatch(_mk_app(0), _mk_job(), remaining_s) is False

    @pytest.mark.parametrize("floor", [-1, -100, -3600])
    def test_negative_floor_treated_as_no_gate(self, floor):
        assert should_gate_dispatch(_mk_app(floor), _mk_job(), 60) is False


# ---------------------------------------------------------------------------
# Positive floor: gate iff remaining < floor
# ---------------------------------------------------------------------------


class TestPositiveFloorGatesByRemaining:
    def test_remaining_above_floor_proceeds(self):
        assert should_gate_dispatch(_mk_app(3600), _mk_job(), 5000) is False

    def test_remaining_below_floor_gates(self):
        assert should_gate_dispatch(_mk_app(3600), _mk_job(), 1000) is True

    def test_remaining_exactly_equal_to_floor_proceeds(self):
        # Strict `<` — equal means proceed.
        assert should_gate_dispatch(_mk_app(3600), _mk_job(), 3600) is False

    def test_remaining_one_below_floor_gates(self):
        assert should_gate_dispatch(_mk_app(3600), _mk_job(), 3599) is True

    @pytest.mark.parametrize("remaining_s", [0, 1, 100, 599])
    def test_very_low_remaining_always_gates_when_floor_positive(self, remaining_s):
        assert should_gate_dispatch(_mk_app(600), _mk_job(), remaining_s) is True


# ---------------------------------------------------------------------------
# Universal gate: Restart-status jobs are gated the same as Ready jobs
# (restart() is now called lazily AFTER this gate inside create_parsl_future,
# so there's no restart-continuation exemption at this layer).
# ---------------------------------------------------------------------------


class TestRestartStatusJobsAreGatedSameAsReady:
    def test_restart_status_job_below_floor_is_gated(self):
        # The gate doesn't even inspect job status — only the app floor +
        # remaining walltime matter. Restart-status jobs gate identically to
        # Ready jobs.
        job = _mk_job(job_id=42)
        job['status'] = 'Restart'
        assert should_gate_dispatch(_mk_app(3600), job, 60) is True

    def test_restart_status_job_above_floor_proceeds(self):
        job = _mk_job(job_id=42)
        job['status'] = 'Restart'
        assert should_gate_dispatch(_mk_app(3600), job, 5000) is False


# ---------------------------------------------------------------------------
# AppBase default + LammpsKkRestart override
# ---------------------------------------------------------------------------


class TestAppBaseDefault:
    def test_appbase_default_min_remaining_walltime_is_zero(self):
        """Apps that don't override the hook get the zero default (no gate)."""
        from parslbox.apps.appbase import AppBase

        # Build a throwaway concrete subclass — AppBase is ABC with one
        # required abstract (get_command_template).
        class _Dummy(AppBase):
            INPUT_REQUIRED = False
            DFLT_INPUT = None

            def get_command_template(self, **kwargs):
                return ""

        assert _Dummy().min_remaining_walltime({"job_id": 1}) == 0


class TestLammpsKkRestartOverride:
    def test_lammps_kk_restart_overrides_to_3600(self):
        from parslbox.apps.custom_apps.lammps_kk_restart import LammpsKkRestart

        # Bypass __init__ — we only need an instance to call the method on.
        inst = LammpsKkRestart.__new__(LammpsKkRestart)
        assert inst.min_remaining_walltime({"job_id": 1}) == 3600


# ---------------------------------------------------------------------------
# Source-pattern guards — make sure run.py actually calls the helper at
# each of the three dispatch sites. A refactor that drops one of these
# calls would silently regress the gate.
# ---------------------------------------------------------------------------


def _run_py_source() -> str:
    import parslbox.commands.run as run_mod
    return Path(run_mod.__file__).read_text()


def _block_containing(src: str, marker_re: str, search_re: str) -> bool:
    """Find a block anchored on `marker_re` (a regex matching some
    distinctive line in the block) and check `search_re` appears within
    ~40 lines after it.
    """
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if re.search(marker_re, line):
            window = "\n".join(lines[i: i + 40])
            if re.search(search_re, window):
                return True
    return False


class TestDispatchSitesCallGateHelper:
    def test_should_gate_dispatch_is_imported(self):
        src = _run_py_source()
        # Multi-line `from ... import (\n  ...,\n  should_gate_dispatch,\n)`
        # — match the import block via DOTALL.
        assert re.search(
            r"from\s+parslbox\.commands\.helpers\.run_cmd_helpers\s+import\s*\("
            r".*?should_gate_dispatch.*?\)",
            src,
            re.DOTALL,
        ), (
            "run.py does not import should_gate_dispatch — the gate will not "
            "fire at any dispatch site."
        )

    def test_initial_dispatch_site_calls_helper(self):
        """Site A: the `for job in filtered_jobs:` loop must gate."""
        src = _run_py_source()
        assert _block_containing(
            src,
            marker_re=r"for\s+job\s+in\s+filtered_jobs\s*:",
            search_re=r"\bshould_gate_dispatch\s*\(",
        ), (
            "Initial dispatch (for job in filtered_jobs) does not call "
            "should_gate_dispatch — jobs would dispatch regardless of "
            "remaining walltime at pbx run startup."
        )

    def test_dynamic_discovery_dispatch_site_calls_helper(self):
        """Site B: the `for job in new_jobs:` loop inside discover_new_jobs
        must gate."""
        src = _run_py_source()
        assert _block_containing(
            src,
            marker_re=r"for\s+job\s+in\s+new_jobs\s*:",
            search_re=r"\bshould_gate_dispatch\s*\(",
        ), (
            "Dynamic-discovery dispatch (for job in new_jobs) does not call "
            "should_gate_dispatch — newly discovered jobs would dispatch "
            "regardless of remaining walltime."
        )

    def test_backlog_reschedule_site_calls_helper(self):
        """Site C: backlog reschedule must filter dependency_ready_jobs
        through the gate BEFORE passing to schedule_backlog."""
        src = _run_py_source()
        assert _block_containing(
            src,
            marker_re=r"dependency_ready_jobs\s*=\s*resource_manager\."
                      r"get_dependency_ready_jobs_from_backlog\s*\(",
            search_re=r"\bshould_gate_dispatch\s*\(",
        ), (
            "Backlog reschedule branch does not call should_gate_dispatch — "
            "backlog jobs would dispatch regardless of remaining walltime."
        )
