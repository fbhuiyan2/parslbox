"""Tests for `pbx ls`: the positional COUNT argument, the -n/--nnodes node
filter, and the parse_ls_count helper.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
from typer.testing import CliRunner

from parslbox.commands.ls import app as ls_app
from parslbox.commands.helpers.ls_cmd_helpers import (
    parse_ls_count,
    select_jobs_to_display,
)
from parslbox.database import database


@pytest.fixture
def temp_db():
    d = tempfile.mkdtemp()
    yield Path(d) / "test.db"
    shutil.rmtree(d)


@pytest.fixture
def db_with_jobs(temp_db):
    """30 jobs so the default truncated view (first 10 + last 10) kicks in."""
    database.initialize_database(temp_db)
    for i in range(1, 31):
        database.add_job(
            db_path=temp_db, path=f"/p{i:02d}", app="lammps-kk",
            num_nodes=1 + (i % 3), ngpus=0, node_occupancy=1.0,
            tag="prod", in_file=None,
            status="Ready" if i % 2 else "Done",
        )
    return temp_db


def run_ls(db_path, *args):
    runner = CliRunner()
    with patch("parslbox.commands.ls.path_utils.DB_FILE", db_path):
        return runner.invoke(ls_app, list(args), env={"COLUMNS": "400"})


# ----------------- parse_ls_count ------------------------------------------ #


class TestParseLsCount:
    def test_none_is_default_view(self):
        assert parse_ls_count(None) == (False, None)

    @pytest.mark.parametrize("token", ["all", "ALL", " All "])
    def test_all_case_insensitive(self, token):
        assert parse_ls_count(token) == (True, None)

    def test_positive(self):
        assert parse_ls_count("10") == (False, 10)

    def test_negative(self):
        assert parse_ls_count("-20") == (False, -20)

    def test_zero_rejected_points_at_all(self):
        with pytest.raises(ValueError, match="'all'"):
            parse_ls_count("0")

    @pytest.mark.parametrize("token", ["abc", "--bogus", "1.5", ""])
    def test_garbage_rejected(self, token):
        with pytest.raises(ValueError, match="Invalid count"):
            parse_ls_count(token)


# ----------------- select_jobs_to_display ---------------------------------- #


class TestSelectJobsToDisplay:
    def _jobs(self, n):
        return [{"job_id": i} for i in range(1, n + 1)]

    def test_show_all_returns_everything(self):
        jobs = self._jobs(100)
        out, msgs = select_jobs_to_display(jobs, True, None)
        assert out == jobs
        assert msgs == []

    def test_first_n(self):
        out, msgs = select_jobs_to_display(self._jobs(30), False, 5)
        assert [j["job_id"] for j in out] == [1, 2, 3, 4, 5]
        assert "first 5" in msgs[0]

    def test_last_n(self):
        out, msgs = select_jobs_to_display(self._jobs(30), False, -3)
        assert [j["job_id"] for j in out] == [28, 29, 30]
        assert "last 3" in msgs[0]

    def test_count_larger_than_total_returns_all(self):
        out, msgs = select_jobs_to_display(self._jobs(4), False, 99)
        assert len(out) == 4
        assert "Found 4 jobs" in msgs[0]

    def test_default_small_table_is_untruncated(self):
        out, _ = select_jobs_to_display(self._jobs(25), False, None)
        assert len(out) == 25
        assert "SEPARATOR" not in out

    def test_default_large_table_truncates_with_separator(self):
        out, _ = select_jobs_to_display(self._jobs(30), False, None)
        assert out.count("SEPARATOR") == 1
        assert len(out) == 21  # 10 + separator + 10


# ----------------- CLI ------------------------------------------------------ #


class TestLsCli:
    def test_default_truncates(self, db_with_jobs):
        res = run_ls(db_with_jobs)
        assert res.exit_code == 0
        assert "# of jobs in the database: 30" in res.output
        assert "..." in res.output

    def test_positional_all(self, db_with_jobs):
        res = run_ls(db_with_jobs, "all")
        assert res.exit_code == 0
        for i in range(1, 31):
            assert f"/p{i:02d}" in res.output

    def test_positional_first_n(self, db_with_jobs):
        res = run_ls(db_with_jobs, "5")
        assert res.exit_code == 0
        assert "Showing the first 5 jobs" in res.output
        assert "/p05" in res.output
        assert "/p06" not in res.output

    def test_positional_negative_last_n(self, db_with_jobs):
        """The `-20` form only parses because the command sets
        ignore_unknown_options; guard against that being dropped."""
        res = run_ls(db_with_jobs, "-3")
        assert res.exit_code == 0
        assert "Showing the last 3 jobs" in res.output
        assert "/p30" in res.output
        assert "/p27" not in res.output

    def test_negative_count_composes_with_flags(self, db_with_jobs):
        res = run_ls(db_with_jobs, "-3", "-s", "Ready")
        assert res.exit_code == 0
        assert "Showing the last 3 jobs" in res.output

    def test_flag_before_negative_count(self, db_with_jobs):
        res = run_ls(db_with_jobs, "-s", "Ready", "-3")
        assert res.exit_code == 0
        assert "Showing the last 3 jobs" in res.output

    def test_invalid_count_errors(self, db_with_jobs):
        res = run_ls(db_with_jobs, "banana")
        assert res.exit_code == 1
        assert "Invalid count" in res.output

    def test_mistyped_flag_surfaces_as_error(self, db_with_jobs):
        """ignore_unknown_options swallows unknown flags into COUNT; the count
        parser is what turns them back into a visible error."""
        res = run_ls(db_with_jobs, "--bogus")
        assert res.exit_code == 1
        assert "Invalid count" in res.output

    def test_all_flag_is_gone(self, db_with_jobs):
        res = run_ls(db_with_jobs, "--all")
        assert res.exit_code == 1

    def test_nnodes_filters_by_node_count(self, db_with_jobs):
        # num_nodes = 1 + (i % 3) -> 10 jobs each at 1, 2 and 3 nodes
        res = run_ls(db_with_jobs, "all", "-n", "3")
        assert res.exit_code == 0
        assert "# of jobs in the database: 10" in res.output

    def test_nnodes_long_form(self, db_with_jobs):
        res = run_ls(db_with_jobs, "all", "--nnodes", "2")
        assert res.exit_code == 0
        assert "# of jobs in the database: 10" in res.output

    def test_nnodes_combines_with_status(self, db_with_jobs):
        res = run_ls(db_with_jobs, "all", "-n", "2", "-s", "Ready")
        assert res.exit_code == 0
        assert "No jobs found" not in res.output

    def test_nnodes_no_match_is_graceful(self, db_with_jobs):
        res = run_ls(db_with_jobs, "-n", "99")
        assert res.exit_code == 0
        assert "No jobs found" in res.output


# ----------------- shared layer / API --------------------------------------- #


class TestNumNodesSharedLayer:
    def test_get_jobs_num_nodes(self, db_with_jobs):
        jobs = database.get_jobs(db_with_jobs, num_nodes=2)
        assert len(jobs) == 10
        assert all(j["num_nodes"] == 2 for j in jobs)

    def test_get_jobs_num_nodes_combines(self, db_with_jobs):
        jobs = database.get_jobs(db_with_jobs, num_nodes=2, status="Ready")
        assert all(j["num_nodes"] == 2 and j["status"] == "Ready" for j in jobs)

    def test_api_list_jobs_num_nodes(self, db_with_jobs):
        from parslbox.api import ParslBox
        pbx = ParslBox(db_path=db_with_jobs)
        assert len(pbx.list_jobs(num_nodes=1)) == 10

    def test_api_filter_jobs_num_nodes(self, db_with_jobs):
        from parslbox.api import ParslBox
        pbx = ParslBox(db_path=db_with_jobs)
        ids = pbx.filter_jobs(num_nodes=3)
        assert len(ids) == 10

    def test_mcp_schemas_expose_num_nodes(self):
        from parslbox.mcp.schemas import ListJobsSchema, FilterJobsSchema
        for schema in (ListJobsSchema, FilterJobsSchema):
            assert "num_nodes" in schema.model_fields
            assert "nodes" in schema.model_fields["num_nodes"].description.lower()
