"""An app's check_success must survive undecodable bytes in the files it reads.

LAMMPS echoes a non-ASCII character from an input script's `print` into
log.lammps as raw bytes, which a strict decode refuses. The decode happened
before the success marker was looked for, so a run that finished cleanly was
recorded as Failed.
"""
import pytest

from parslbox.apps.custom_apps.lammps_cpu import LammpsBasicApp
from parslbox.apps.lammps_kk import LammpsKokkosApp
from parslbox.apps.orca import OrcaApp

# What LAMMPS writes to log.lammps for an em-dash: not valid UTF-8.
UNDECODABLE = b"\x80\x94"

LAMMPS_APPS = [LammpsKokkosApp, LammpsBasicApp]


def _write(path, *chunks):
    path.write_bytes(b"".join(
        c.encode() if isinstance(c, str) else c for c in chunks))


def _check(app_cls, job_path):
    return app_cls().check_success(job_id=1, job_path=job_path, db_path=None)


class TestLammpsLogWithUndecodableBytes:
    @pytest.mark.parametrize("app_cls", LAMMPS_APPS)
    def test_a_finished_run_is_done_despite_bad_bytes(self, app_cls, tmp_path):
        _write(tmp_path / "log.lammps",
               "print ", UNDECODABLE, "\nTotal wall time: 0:01:23\n")
        assert _check(app_cls, tmp_path) == "Done"

    @pytest.mark.parametrize("app_cls", LAMMPS_APPS)
    def test_bad_bytes_do_not_invent_a_success_marker(self, app_cls, tmp_path):
        _write(tmp_path / "log.lammps", "print ", UNDECODABLE, "\nERROR: bad\n")
        assert _check(app_cls, tmp_path) == "Failed"

    @pytest.mark.parametrize("app_cls", LAMMPS_APPS)
    def test_the_marker_is_still_found_in_a_clean_log(self, app_cls, tmp_path):
        _write(tmp_path / "log.lammps", "Total wall time: 0:01:23\n")
        assert _check(app_cls, tmp_path) == "Done"

    @pytest.mark.parametrize("app_cls", LAMMPS_APPS)
    def test_a_missing_log_is_still_failed(self, app_cls, tmp_path):
        assert _check(app_cls, tmp_path) == "Failed"


class TestOrcaOutputWithUndecodableBytes:
    def test_a_finished_run_is_done_despite_bad_bytes(self, tmp_path):
        _write(tmp_path / "job.out",
               "geometry ", UNDECODABLE, "\n****ORCA TERMINATED NORMALLY****\n")
        assert _check(OrcaApp, tmp_path) == "Done"

    def test_bad_bytes_do_not_invent_a_success_marker(self, tmp_path):
        _write(tmp_path / "job.out", "geometry ", UNDECODABLE, "\nABORTING\n")
        assert _check(OrcaApp, tmp_path) == "Failed"

    def test_an_unreadable_out_file_is_reported_not_swallowed(self, tmp_path, caplog):
        """The read used to fail into a bare `continue`, so nothing reached the
        log and there was no way to tell this case from a genuine failure."""
        bad = tmp_path / "job.out"
        bad.mkdir()
        assert _check(OrcaApp, tmp_path) == "Failed"
        assert "job.out" in caplog.text

    def test_a_later_out_file_still_wins_after_an_unreadable_one(self, tmp_path):
        (tmp_path / "a.out").mkdir()
        _write(tmp_path / "b.out", "****ORCA TERMINATED NORMALLY****\n")
        assert _check(OrcaApp, tmp_path) == "Done"
