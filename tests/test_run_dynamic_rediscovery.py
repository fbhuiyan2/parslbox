"""
Tests for `should_re_dispatch_known_job` — the helper that decides whether
a job already in dispatch's `known_job_ids` should be re-routed into
`new_jobs` when dynamic discovery sees it back in a runnable status.

History:
- v1 (initial): only handled (tracker=Failed AND db=Ready). Silently
  dropped every other valid re-dispatch intent — e.g. user flipping
  Ready→Restart mid-run, or resuming a Killed job via status=Restart.
- v2 (over-broad): expanded `tracker_settled` to include Ready. This
  unintentionally treated `tracker=Ready, db=Ready` as a re-dispatch
  trigger, which produced the production dynamic-discovery tight-loop:
  jobs that sat in the backlog from startup (tracker never moved past
  Ready, DB still Ready) were "re-discovered" on every pass, falling
  straight back into the backlog, ad infinitum.
- v3 (current): re-dispatch only on a real state transition.

The decision table:
  tracker terminal  + db runnable → re-dispatch (replay)
  tracker == Ready  + db == Restart → re-dispatch (force checkpoint resume)
  tracker == Ready  + db == Ready → NO (no transition; the v2 bug)
  tracker in_flight + any → NO (live future, DB flip is no-op this link)
  any + db non-runnable → NO (user didn't ask for it)
"""

import re
from pathlib import Path

import pytest

from parslbox.commands.helpers.run_cmd_helpers import should_re_dispatch_known_job


# ---------------------------------------------------------------------------
# Status partitions
# ---------------------------------------------------------------------------

TERMINAL_TRACKER_STATES = ("Done", "Failed", "Warning", "Killed")
IN_FLIGHT_TRACKER_STATES = ("Submitted", "Running")
RUNNABLE_DB_STATES = ("Ready", "Restart")
NON_RUNNABLE_DB_STATES = ("Done", "Failed", "Killed", "Warning", "Submitted", "Running")


class TestTerminalTrackerWithRunnableDb:
    """Every (terminal tracker × runnable DB) combo must re-dispatch — the
    core re-discovery contract."""

    @pytest.mark.parametrize("tracker_status", TERMINAL_TRACKER_STATES)
    @pytest.mark.parametrize("db_status", RUNNABLE_DB_STATES)
    def test_re_dispatches(self, tracker_status, db_status):
        assert should_re_dispatch_known_job(tracker_status, db_status) is True, (
            f"Expected re-dispatch for tracker={tracker_status}, db={db_status}"
        )

    def test_failed_to_ready_still_works(self):
        """The combo the very first implementation handled — must still work."""
        assert should_re_dispatch_known_job("Failed", "Ready") is True

    def test_killed_to_restart_still_works(self):
        """Common scenario after a no-respawn walltime kill: user wants to
        resume the killed jobs in the next pbx run, flips them to Restart."""
        assert should_re_dispatch_known_job("Killed", "Restart") is True

    def test_done_to_restart_still_works(self):
        """User wants to re-run a completed job with restart-continuation
        semantics — e.g., to extend a converged simulation."""
        assert should_re_dispatch_known_job("Done", "Restart") is True


class TestReadyTrackerOnlyRestartTransition:
    """Tracker == Ready is a special case: it can mean either 'job is fresh
    and never dispatched' (the dominant case, must NOT re-dispatch) or 'user
    flipped a fresh job to Restart to force checkpoint-resume semantics'
    (must re-dispatch). The discriminator is the DB status."""

    def test_ready_to_ready_does_not_re_dispatch(self):
        """The v2 bug fix: re-dispatching here busy-loops the orchestrator
        because tracker never moved past Ready (the job has been sitting in
        the backlog the whole time)."""
        assert should_re_dispatch_known_job("Ready", "Ready") is False

    def test_ready_to_restart_re_dispatches(self):
        """User-driven transition — must route through new_jobs so
        apply_restart_for_job adds the job to restarting_job_ids."""
        assert should_re_dispatch_known_job("Ready", "Restart") is True

    @pytest.mark.parametrize("db_status", NON_RUNNABLE_DB_STATES)
    def test_ready_to_non_runnable_does_not_re_dispatch(self, db_status):
        assert should_re_dispatch_known_job("Ready", db_status) is False


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


class TestTerminalTrackerWithNonRunnableDb:
    """Terminal tracker + non-runnable DB → no re-dispatch (user didn't ask
    for it). Covers status combinations from races / weird intermediate
    states."""

    @pytest.mark.parametrize("tracker_status", TERMINAL_TRACKER_STATES)
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
        re-discovery branch silently goes back to ad-hoc behavior. This
        guard catches that regression at the source-pattern level."""
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
