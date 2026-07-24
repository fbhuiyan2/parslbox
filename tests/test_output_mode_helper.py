"""
Unit tests for `choose_output_mode` — the helper that decides whether a job's
stdout/stderr open in fresh 'w' mode or restart-continuation 'a' mode (with
a banner appended to any existing non-empty file).

The signal:
  is_restart_continuation = job_id in restarting_job_ids
                            or current_status == 'Restart'

Cases mirror the table laid out during design (see prev-conv-history) — 15
total scenarios across set-membership, DB status, and file pre-existence.
"""

import pytest

from parslbox.commands.helpers.run_cmd_helpers import choose_output_mode


@pytest.fixture
def paths(tmp_path):
    """Return (stdout_path, stderr_path) under tmp_path. Neither exists yet."""
    return tmp_path / "pbx_job_42.out", tmp_path / "pbx_job_42.err"


def _write(p, content):
    p.write_text(content)


def _read(p):
    return p.read_text() if p.exists() else ""


# ============================================================
# Fresh-write cases (mode 'w', no banner)
# ============================================================


class TestFreshWriteMode:
    def test_ready_status_not_in_set_returns_w(self, paths):
        out, err = paths
        mode = choose_output_mode(42, 'Ready', set(), 1800, out, err)
        assert mode == 'w'

    def test_ready_status_with_other_ids_in_set_returns_w(self, paths):
        """Set membership of OTHER job IDs must not affect this one."""
        out, err = paths
        mode = choose_output_mode(42, 'Ready', {1, 2, 99}, 1800, out, err)
        assert mode == 'w'

    def test_done_status_not_in_set_returns_w(self, paths):
        """Status values we don't recognize as 'Restart' fall through to 'w'."""
        out, err = paths
        mode = choose_output_mode(42, 'Done', set(), 1800, out, err)
        assert mode == 'w'

    def test_fresh_mode_does_not_touch_existing_file(self, paths):
        """In fresh mode we don't write a banner — Parsl will truncate on open."""
        out, err = paths
        _write(out, "stale content\n")
        choose_output_mode(42, 'Ready', set(), 1800, out, err)
        assert _read(out) == "stale content\n"  # unchanged; banner not added


# ============================================================
# Restart-continuation via set membership (post-hook flip to Ready)
# ============================================================


class TestSetMembershipTriggersAppend:
    def test_in_set_with_ready_status_returns_a(self, paths):
        out, err = paths
        mode = choose_output_mode(42, 'Ready', {42}, 1800, out, err)
        assert mode == 'a'

    def test_banner_appended_to_existing_non_empty_file(self, paths):
        out, err = paths
        _write(out, "prior run output\n")
        choose_output_mode(42, 'Ready', {42}, 1800, out, err)
        content = _read(out)
        assert content.startswith("prior run output\n")
        assert "=== Restart " in content
        assert content.count("=" * 70) == 2  # opening + closing bar

    def test_banner_skipped_on_missing_file(self, paths):
        """Missing file → no banner needed (Parsl will create it on first open)."""
        out, err = paths
        choose_output_mode(42, 'Ready', {42}, 1800, out, err)
        assert not out.exists()  # helper didn't touch it
        assert not err.exists()

    def test_banner_skipped_on_empty_file(self, paths):
        """Empty file → no banner (no prior content to separate from)."""
        out, err = paths
        out.touch()  # exists but size 0
        choose_output_mode(42, 'Ready', {42}, 1800, out, err)
        assert _read(out) == ""  # still empty

    def test_banner_on_stderr_too(self, paths):
        """Both files get the banner if both have content."""
        out, err = paths
        _write(out, "stdout data\n")
        _write(err, "stderr data\n")
        choose_output_mode(42, 'Ready', {42}, 1800, out, err)
        for p in (out, err):
            assert "=== Restart " in _read(p)

    def test_one_with_content_one_empty_only_banners_the_one_with_content(self, paths):
        """Per-file decision — empty file stays empty, non-empty gets banner."""
        out, err = paths
        _write(out, "stdout data\n")
        # err deliberately absent
        choose_output_mode(42, 'Ready', {42}, 1800, out, err)
        assert "=== Restart " in _read(out)
        assert not err.exists()


# ============================================================
# Restart-continuation via current_status == 'Restart'
# (dynamic-discovery arrival OR the tiny race between hook and dispatch)
# ============================================================


class TestStatusRestartTriggersAppend:
    def test_status_restart_not_in_set_returns_a(self, paths):
        out, err = paths
        mode = choose_output_mode(42, 'Restart', set(), 1800, out, err)
        assert mode == 'a'

    def test_status_restart_appends_banner_to_existing_file(self, paths):
        out, err = paths
        _write(out, "previous attempt\n")
        choose_output_mode(42, 'Restart', set(), 1800, out, err)
        assert "=== Restart " in _read(out)


# ============================================================
# Both signals true — same outcome (no double-banner)
# ============================================================


class TestBothSignalsTrue:
    def test_in_set_and_status_restart_still_one_banner(self, paths):
        out, err = paths
        _write(out, "data\n")
        choose_output_mode(42, 'Restart', {42}, 1800, out, err)
        # Exactly one banner — not two from "both reasons true"
        assert _read(out).count("=== Restart ") == 1


# ============================================================
# Remaining-walltime line in banner
# ============================================================


class TestBannerRemainingWalltimeLine:
    def test_banner_includes_remaining_walltime_line(self, paths):
        out, err = paths
        _write(out, "prior\n")
        choose_output_mode(42, 'Restart', set(), 5025, out, err)
        # 5025 s = 01:23:45
        assert "=== Remaining Walltime: 01:23:45" in _read(out)

    def test_banner_walltime_zero_clamped(self, paths):
        out, err = paths
        _write(out, "prior\n")
        choose_output_mode(42, 'Restart', set(), -10, out, err)
        assert "=== Remaining Walltime: 00:00:00" in _read(out)

    def test_banner_walltime_under_a_minute(self, paths):
        out, err = paths
        _write(out, "prior\n")
        choose_output_mode(42, 'Restart', set(), 42, out, err)
        assert "=== Remaining Walltime: 00:00:42" in _read(out)

    def test_banner_walltime_multi_hour(self, paths):
        out, err = paths
        _write(out, "prior\n")
        choose_output_mode(42, 'Restart', set(), 36_000, out, err)
        assert "=== Remaining Walltime: 10:00:00" in _read(out)

    def test_walltime_line_appears_between_ts_and_closing_bar(self, paths):
        """Ordering inside the banner: ts line, then walltime line, then bar."""
        out, err = paths
        _write(out, "prior\n")
        choose_output_mode(42, 'Restart', set(), 1800, out, err)
        content = _read(out)
        ts_idx = content.index("=== Restart ")
        wt_idx = content.index("=== Remaining Walltime: ")
        bar_idx = content.rindex("=" * 70)
        assert ts_idx < wt_idx < bar_idx
