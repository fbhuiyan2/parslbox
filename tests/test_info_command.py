"""Unit tests for the `pbx info` command.

Covers field selectors, path truncation rules, ID-range parsing, missing-job
warnings, and the renamed/removed short flags.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
from typer.testing import CliRunner

from parslbox.commands.info import app as info_app
from parslbox.database import database


# Long enough path that truncate_path actually shortens it (>5 segments).
LONG_PATH = "/lus/flare/projects/CSTEELML/fbhuiyan/work/wf_kappa_al-sic/sic-3c-nomix/run1"


@pytest.fixture
def temp_db():
    d = tempfile.mkdtemp()
    yield Path(d) / "test.db"
    shutil.rmtree(d)


@pytest.fixture
def db_with_jobs(temp_db):
    database.initialize_database(temp_db)
    # Job 1: GPU multi-node
    database.add_job(
        db_path=temp_db, path=LONG_PATH, app="lammps-kk",
        num_nodes=5, ngpus=60, node_occupancy=1.0, tag="film-bulk",
        in_file="in.lammps", ranks_per_node=12, status="Ready",
    )
    # Job 2: CPU fractional
    database.add_job(
        db_path=temp_db, path=LONG_PATH + "/job2", app="python",
        num_nodes=1, ngpus=0, node_occupancy=0.25, tag="prod",
        in_file=None, ranks_per_node=1, status="Ready",
    )
    # Job 3: GPU single-node with parents
    database.add_job(
        db_path=temp_db, path=LONG_PATH + "/job3", app="vasp",
        num_nodes=1, ngpus=4, node_occupancy=1.0, tag="prod",
        in_file=None, ranks_per_node=4, parents=[1, 2], status="Ready",
    )
    return temp_db


def run_info(db_path, *args):
    """Invoke `pbx info` with DB_FILE patched. Force a wide terminal so Rich
    doesn't wrap cells and break substring assertions."""
    runner = CliRunner()
    with patch("parslbox.commands.info.path_utils.DB_FILE", db_path):
        return runner.invoke(info_app, list(args), env={"COLUMNS": "400"})


class TestDefaultView:
    def test_default_view_shows_all_fields(self, db_with_jobs):
        r = run_info(db_with_jobs, "1")
        assert r.exit_code == 0, r.output
        for header in ("ID", "App", "Status", "NGPUs", "Sched Job ID",
                       "Tag", "Input", "Env File", "Timestamp", "Path"):
            assert header in r.output

    def test_default_view_summary_line(self, db_with_jobs):
        r = run_info(db_with_jobs, "1")
        assert "1 job" in r.output

    def test_default_view_multiple_jobs(self, db_with_jobs):
        r = run_info(db_with_jobs, "1-3")
        assert r.exit_code == 0
        assert "3 jobs" in r.output
        assert "lammps-kk" in r.output
        assert "vasp" in r.output


class TestFieldSelectors:
    def test_path_flag_includes_path_column(self, db_with_jobs):
        r = run_info(db_with_jobs, "1", "--path")
        assert r.exit_code == 0
        assert "Path" in r.output
        assert "App" not in r.output  # other fields excluded

    def test_short_p_includes_path_column(self, db_with_jobs):
        r = run_info(db_with_jobs, "1", "-p")
        assert r.exit_code == 0
        assert "Path" in r.output

    def test_ngpus_flag_g(self, db_with_jobs):
        r = run_info(db_with_jobs, "1", "-g")
        assert r.exit_code == 0
        assert "NGPUs" in r.output
        assert "60" in r.output

    def test_ngpus_long_flag(self, db_with_jobs):
        r = run_info(db_with_jobs, "1", "--ngpus")
        assert r.exit_code == 0
        assert "NGPUs" in r.output

    def test_nodes_flag(self, db_with_jobs):
        r = run_info(db_with_jobs, "1", "-n")
        assert r.exit_code == 0
        assert "NNodes" in r.output
        assert "5" in r.output

    def test_ranks_flag(self, db_with_jobs):
        r = run_info(db_with_jobs, "1", "--ranks")
        assert r.exit_code == 0
        assert "Ranks/Node" in r.output
        assert "12" in r.output

    def test_nocc_flag(self, db_with_jobs):
        r = run_info(db_with_jobs, "2", "-o")
        assert r.exit_code == 0
        assert "NOcc" in r.output
        assert "0.25" in r.output

    def test_resrc_flag(self, db_with_jobs):
        r = run_info(db_with_jobs, "1", "-r")
        assert r.exit_code == 0
        assert "Resources" in r.output
        # Multi-node GPU: n:5-r:60-g:60-nocc:NA
        assert "n:5" in r.output
        assert "g:60" in r.output

    def test_resrc_for_cpu_fractional(self, db_with_jobs):
        r = run_info(db_with_jobs, "2", "--resrc")
        assert r.exit_code == 0
        assert "n:1" in r.output
        assert "nocc:0.25" in r.output

    def test_status_flag(self, db_with_jobs):
        r = run_info(db_with_jobs, "1", "-s")
        assert r.exit_code == 0
        assert "Status" in r.output
        assert "Ready" in r.output

    def test_tag_flag(self, db_with_jobs):
        r = run_info(db_with_jobs, "1", "-t")
        assert r.exit_code == 0
        assert "Tag" in r.output
        assert "film-bulk" in r.output

    def test_parents_flag(self, db_with_jobs):
        r = run_info(db_with_jobs, "3", "-P")
        assert r.exit_code == 0
        assert "Parents" in r.output
        assert "1,2" in r.output

    def test_multiple_selectors_include_id_first(self, db_with_jobs):
        r = run_info(db_with_jobs, "1", "-g", "-s", "-t")
        assert r.exit_code == 0
        # ID column auto-added when selectors are used
        assert "ID" in r.output
        assert "NGPUs" in r.output
        assert "Status" in r.output
        assert "Tag" in r.output


class TestPathTruncation:
    """Rules:
       - --path always shows full path
       - -p alone (or with ≤3 fields incl. ID) shows full
       - -p combined with enough flags so total fields > 3 → truncated
    """

    def test_path_long_flag_shows_full(self, db_with_jobs):
        r = run_info(db_with_jobs, "1", "--path", "-s", "-t", "-g", "-a")
        assert r.exit_code == 0
        # Full path must appear verbatim
        assert LONG_PATH in r.output

    def test_path_short_p_truncates_when_many_fields(self, db_with_jobs):
        # -p + ID + status + tag + ngpus + app = 5 fields → truncate
        r = run_info(db_with_jobs, "1", "-p", "-s", "-t", "-g", "-a")
        assert r.exit_code == 0
        assert "..." in r.output
        # Full long path should not appear because it's truncated
        assert LONG_PATH not in r.output

    def test_path_short_p_alone_shows_full(self, db_with_jobs):
        # -p alone → 2 fields total (ID + Path), not >3 → no truncation
        r = run_info(db_with_jobs, "1", "-p")
        assert r.exit_code == 0
        assert LONG_PATH in r.output


class TestIDParsing:
    def test_range_expansion(self, db_with_jobs):
        r = run_info(db_with_jobs, "1-3")
        assert r.exit_code == 0
        assert "3 jobs" in r.output

    def test_mixed_ids(self, db_with_jobs):
        r = run_info(db_with_jobs, "1", "3")
        assert r.exit_code == 0
        assert "2 jobs" in r.output

    def test_missing_id_warns_but_continues(self, db_with_jobs):
        r = run_info(db_with_jobs, "1", "99")
        assert r.exit_code == 0
        assert "99" in r.output
        assert "not found" in r.output.lower() or "Warning" in r.output

    def test_all_missing_exits_nonzero(self, db_with_jobs):
        r = run_info(db_with_jobs, "99", "100")
        assert r.exit_code != 0


class TestRemovedShortFlags:
    """--req lost its -r short (now means --resrc); --cmdline lost -c."""

    def test_dash_r_is_resrc_not_req(self, db_with_jobs):
        # If -r still meant --req, it would expect a system-name value and
        # invocation would fail or interpret '1' as the value.
        r = run_info(db_with_jobs, "1", "-r")
        assert r.exit_code == 0
        assert "Resources" in r.output

    def test_dash_c_no_longer_works_for_cmdline(self, db_with_jobs):
        # -c is now unknown; typer should error.
        r = run_info(db_with_jobs, "1", "-c", "aurora-tile")
        assert r.exit_code != 0


class TestReqAndCmdline:
    def test_req_long_flag_runs(self, db_with_jobs):
        r = run_info(db_with_jobs, "1", "--req", "aurora-tile")
        assert r.exit_code == 0
        assert "Resource Requirements" in r.output

    def test_req_unknown_system(self, db_with_jobs):
        r = run_info(db_with_jobs, "1", "--req", "no-such-system")
        # Doesn't raise — prints an error message in-line.
        assert "Unknown system" in r.output

    def test_cmdline_does_not_raise_nameerror(self, db_with_jobs):
        # Regression: _create_synthetic_assignment used to reference undefined
        # `system_config`. Now uses the `gpus_per_node` param.
        r = run_info(db_with_jobs, "1", "--cmdline", "aurora-tile")
        assert r.exit_code == 0
        assert "NameError" not in r.output
        assert "Command line preview" in r.output
