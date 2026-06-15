"""
Tests for `should_re_dispatch_known_job` — the helper that decides whether
a job already in dispatch's `known_job_ids` should be re-routed into
`new_jobs` when dynamic discovery sees it back in a runnable status.

The previous implementation only handled the single combo
(tracker=Failed AND db=Ready). Every other valid re-dispatch intent
(e.g., user flips Ready→Restart mid-run, or resumes a Killed job via
status=Restart) was silently dropped — the `continue` fired before the
job could enter the pipeline.

The full decision table is exhaustively tested here.
"""

import re
from pathlib import Path

import pytest

from parslbox.commands.helpers.run_cmd_helpers import should_re_dispatch_known_job


# ---------------------------------------------------------------------------
# All meaningful (tracker_status × db_status) combinations
# ---------------------------------------------------------------------------

SETTLED_TRACKER_STATES = ("Done", "Failed", "Warning", "Killed", "Ready")
IN_FLIGHT_TRACKER_STATES = ("Submitted", "Running")
RUNNABLE_DB_STATES = ("Ready", "Restart")
NON_RUNNABLE_DB_STATES = ("Done", "Failed", "Killed", "Warning", "Submitted", "Running")


class TestSettledTrackerWithRunnableDb:
    """The core fix: every (settled tracker × runnable DB) combo must re-dispatch."""

    @pytest.mark.parametrize("tracker_status", SETTLED_TRACKER_STATES)
    @pytest.mark.parametrize("db_status", RUNNABLE_DB_STATES)
    def test_re_dispatches(self, tracker_status, db_status):
        assert should_re_dispatch_known_job(tracker_status, db_status) is True, (
            f"Expected re-dispatch for tracker={tracker_status}, db={db_status} "
            f"— this is exactly the case the previous bug dropped silently."
        )

    def test_failed_to_ready_still_works(self):
        """The one case the OLD code handled — must still work after the rewrite."""
        assert should_re_dispatch_known_job("Failed", "Ready") is True

    def test_ready_to_restart_now_works(self):
        """The user-reported case (jobs 36/38 scenario): user flips a job they
        were already running from Ready to Restart mid-pbx-run to trigger
        restart-continuation file mode."""
        assert should_re_dispatch_known_job("Ready", "Restart") is True

    def test_killed_to_restart_now_works(self):
        """Common scenario after a no-respawn walltime kill: user wants to
        resume the killed jobs in the next pbx run, flips them to Restart."""
        assert should_re_dispatch_known_job("Killed", "Restart") is True

    def test_done_to_restart_now_works(self):
        """User wants to re-run a completed job with restart-continuation
        semantics — e.g., to extend a converged simulation."""
        assert should_re_dispatch_known_job("Done", "Restart") is True


class TestInFlightTrackerNeverReDispatches:
    """Tracker shows live future. DB flip is a no-op for this link — let the
    future finish on its own path."""

    @pytest.mark.parametrize("tracker_status", IN_FLIGHT_TRACKER_STATES)
    @pytest.mark.parametrize("db_status", RUNNABLE_DB_STATES + NON_RUNNABLE_DB_STATES)
    def test_never_re_dispatches(self, tracker_status, db_status):
        assert should_re_dispatch_known_job(tracker_status, db_status) is False, (
            f"In-flight tracker={tracker_status} must NOT re-dispatch "
            f"(future is live; DB flip to {db_status} is a no-op this link)."
        )


class TestSettledTrackerWithNonRunnableDb:
    """If DB doesn't show the job as Ready/Restart, no re-dispatch — the
    user didn't ask for it. Covers status combinations that could come from
    races / weird intermediate states."""

    @pytest.mark.parametrize("tracker_status", SETTLED_TRACKER_STATES)
    @pytest.mark.parametrize("db_status", NON_RUNNABLE_DB_STATES)
    def test_never_re_dispatches(self, tracker_status, db_status):
        assert should_re_dispatch_known_job(tracker_status, db_status) is False


class TestUnknownStatusesAreSafe:
    """Future-proofing: an unrecognized status string must not accidentally
    trigger re-dispatch (default-deny)."""

    def test_unknown_tracker_status_does_not_re_dispatch(self):
        assert should_re_dispatch_known_job("Unknown", "Ready") is False

    def test_unknown_db_status_does_not_re_dispatch(self):
        assert should_re_dispatch_known_job("Failed", "Pending") is False

    def test_empty_strings_do_not_re_dispatch(self):
        assert should_re_dispatch_known_job("", "") is False


# ---------------------------------------------------------------------------
# Source-pattern guard — make sure discover_new_jobs actually calls the helper
# ---------------------------------------------------------------------------


class TestDiscoverNewJobsUsesHelper:
    def test_discover_new_jobs_calls_should_re_dispatch_known_job(self):
        """If a refactor removes the helper call from discover_new_jobs, the
        re-discovery branch silently goes back to the old buggy single-case
        behavior. This guard catches that regression at the source-pattern
        level."""
        import parslbox.commands.run as run_mod

        src = Path(run_mod.__file__).read_text()
        lines = src.splitlines()
        def_idx = None
        for i, line in enumerate(lines):
            if re.match(r"\s*def\s+discover_new_jobs\s*\(", line):
                def_idx = i
                break
        assert def_idx is not None, (
            "discover_new_jobs definition not found in run.py — test stale"
        )
        def_col = len(lines[def_idx]) - len(lines[def_idx].lstrip())
        found = False
        for j in range(def_idx + 1, len(lines)):
            ln = lines[j]
            if ln.strip() and (len(ln) - len(ln.lstrip())) <= def_col:
                break
            if re.search(r"\bshould_re_dispatch_known_job\s*\(", ln):
                found = True
                break
        assert found, (
            "discover_new_jobs does not invoke should_re_dispatch_known_job(). "
            "Re-discovery would silently drop most tracker × DB status combos "
            "(e.g., user flipping Ready→Restart mid-run). See test rationale."
        )
