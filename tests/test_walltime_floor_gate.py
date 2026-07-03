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
# Source-pattern guards — the dispatch engine (schedule_helpers) must call the
# gate in both dispatch paths. A refactor that drops one would silently regress
# the gate.
# ---------------------------------------------------------------------------


def _sched_helpers_source() -> str:
    import parslbox.commands.helpers.schedule_helpers as sched_mod
    return Path(sched_mod.__file__).read_text()


def _function_source(func_name: str) -> str:
    """Return the source of a single top-level function in schedule_helpers."""
    src = _sched_helpers_source()
    lines = src.splitlines()
    out, capturing = [], False
    for line in lines:
        if re.match(rf"def\s+{func_name}\s*\(", line):
            capturing = True
            out.append(line)
            continue
        if capturing:
            if line and not line[0].isspace() and not line.startswith(")"):
                break
            out.append(line)
    return "\n".join(out)


class TestDispatchSitesCallGateHelper:
    def test_should_gate_dispatch_is_imported(self):
        src = _sched_helpers_source()
        assert re.search(
            r"from\s+parslbox\.commands\.helpers\.run_cmd_helpers\s+import\s*\("
            r".*?should_gate_dispatch.*?\)",
            src,
            re.DOTALL,
        ), (
            "schedule_helpers does not import should_gate_dispatch — the gate "
            "will not fire at any dispatch site."
        )

    def test_dynamic_dispatch_calls_helper(self):
        """dispatch_dynamic must gate every candidate before claiming."""
        assert re.search(r"\bshould_gate_dispatch\s*\(", _function_source("dispatch_dynamic")), (
            "dispatch_dynamic does not call should_gate_dispatch — jobs would "
            "dispatch regardless of remaining walltime."
        )

    def test_static_dispatch_calls_helper(self):
        """dispatch_static must gate backlog jobs before dispatching."""
        assert re.search(r"\bshould_gate_dispatch\s*\(", _function_source("dispatch_static")), (
            "dispatch_static does not call should_gate_dispatch — backlog jobs "
            "would dispatch regardless of remaining walltime."
        )
