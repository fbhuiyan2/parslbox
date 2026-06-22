"""
Source-pattern guards for the empty-`fut_to_item` branch of the main
orchestrator loop in `parslbox/commands/run.py`.

History:
The previous implementation called `discover_new_jobs()` unconditionally
on every iteration when there were no active futures. Combined with a
spurious-rediscovery bug in `should_re_dispatch_known_job` (now fixed),
this spun the outer `while True:` thousands of times per second — the
60s `discovery_interval` only gated the *periodic* discovery call near
the top of the loop, not the exit-check one.

These tests don't drive the loop itself (that would require heavy
mocking). They check at the source-pattern level that:

  1. The empty-futures branch's `discover_new_jobs()` call is wrapped in
     a `time.time() - last_discovery_time >= discovery_interval` guard.

  2. The dynamic-mode pending-state path in that branch contains a
     `time.sleep(...)` before `continue`, so even with discovery gated
     the loop cannot busy-spin between cadence ticks.

Pattern-based guards mirror the style of TestDiscoverNewJobsUsesHelper
in test_run_dynamic_rediscovery.py.
"""

import re
from pathlib import Path


def _extract_empty_futures_branch(src: str) -> str:
    """Return the text of the `if not fut_to_item:` block inside
    run_orchestrator, with `#` comments stripped so guard regexes don't
    match identifier names that only appear in narrative comments."""
    lines = src.splitlines()
    start_idx = None
    indent_col = None
    for i, line in enumerate(lines):
        m = re.match(r"(\s*)if\s+not\s+fut_to_item\s*:\s*$", line)
        if m:
            start_idx = i
            indent_col = len(m.group(1))
            break
    assert start_idx is not None, (
        "Could not locate `if not fut_to_item:` block in run.py — test stale"
    )
    body = []
    for j in range(start_idx + 1, len(lines)):
        ln = lines[j]
        if not ln.strip():
            body.append(ln)
            continue
        col = len(ln) - len(ln.lstrip())
        if col <= indent_col:
            break
        # Strip line comments (no strings in this branch contain `#`).
        if "#" in ln:
            ln = ln[: ln.index("#")].rstrip()
        body.append(ln)
    return "\n".join(body)


class TestEmptyFuturesBranchGatesDiscovery:
    def test_discover_new_jobs_is_interval_gated(self):
        """The empty-futures branch must call `discover_new_jobs()` only
        inside a `last_discovery_time` interval guard. Otherwise the
        loop spins as fast as Python can iterate whenever futures drain
        and there's still pending backlog work."""
        import parslbox.commands.run as run_mod

        branch = _extract_empty_futures_branch(Path(run_mod.__file__).read_text())

        # Every discover_new_jobs() call site in this branch must be
        # preceded (within the branch text) by an `if ... last_discovery_time
        # ... >= discovery_interval` guard.
        call_positions = [m.start() for m in re.finditer(r"\bdiscover_new_jobs\s*\(", branch)]
        assert call_positions, (
            "Empty-futures branch contains no discover_new_jobs() call — "
            "test stale, or dynamic discovery was removed from this branch."
        )

        guard_pattern = re.compile(
            r"if\s+.*last_discovery_time.*>=\s*discovery_interval", re.DOTALL
        )
        for pos in call_positions:
            preceding = branch[:pos]
            assert guard_pattern.search(preceding), (
                "discover_new_jobs() in the empty-futures branch is not "
                "preceded by an `if ... last_discovery_time ... >= "
                "discovery_interval` guard. Without this gate the orchestrator "
                "loop busy-spins whenever futures drain with pending backlog."
            )

    def test_last_discovery_time_is_updated_after_call(self):
        """The interval guard relies on `last_discovery_time` being
        refreshed after each discover_new_jobs() call in this branch —
        otherwise the gate latches open forever."""
        import parslbox.commands.run as run_mod

        branch = _extract_empty_futures_branch(Path(run_mod.__file__).read_text())
        # Find each discover_new_jobs() call and check that within the
        # next few lines there's a `last_discovery_time = time.time()`
        # assignment.
        for m in re.finditer(r"\bdiscover_new_jobs\s*\([^)]*\)", branch):
            after = branch[m.end():m.end() + 400]
            assert re.search(r"last_discovery_time\s*=\s*time\.time\s*\(\s*\)", after), (
                "discover_new_jobs() call in empty-futures branch is not "
                "followed by a `last_discovery_time = time.time()` update. "
                "The interval gate would latch open forever."
            )


class TestEmptyFuturesBranchSleepsInDynamic:
    def test_dynamic_pending_state_sleeps_before_continue(self):
        """The empty-futures branch must contain a `time.sleep(...)`
        somewhere before its final `continue` so the loop cannot
        busy-spin when dynamic mode has pending work but no live futures
        (e.g., backlog jobs waiting on a new parent from a future
        discovery pass)."""
        import parslbox.commands.run as run_mod

        branch = _extract_empty_futures_branch(Path(run_mod.__file__).read_text())
        # At least one time.sleep(...) must appear in the branch text.
        assert re.search(r"\btime\.sleep\s*\(", branch), (
            "Empty-futures branch contains no time.sleep() — orchestrator "
            "loop will busy-spin whenever it has pending work but no "
            "active futures (e.g., dynamic mode between discovery ticks)."
        )
