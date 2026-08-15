"""Tests for job ID range compression and failure grouping, plus the bulk
`add`/`update` output that uses them.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
from typer.testing import CliRunner

from parslbox.commands.helpers.job_id_parser import (
    format_job_ids,
    group_failures,
    parse_job_ids,
)
from parslbox.database import database


class TestFormatJobIds:
    @pytest.mark.parametrize("ids,expected", [
        ([], ""),
        ([7], "7"),
        ([1, 2, 3, 4, 5], "1-5"),
        ([1, 2, 3, 8, 14, 15, 16], "1-3 8 14-16"),
        ([1, 3, 5], "1 3 5"),
        ([2, 1], "1-2"),                 # sorted
        ([1, 1, 2, 2, 3], "1-3"),        # deduped
        ([10, 11], "10-11"),             # a 2-run is still a range
    ])
    def test_compression(self, ids, expected):
        assert format_job_ids(ids) == expected

    def test_bulk_add_collapses_to_one_token(self):
        assert format_job_ids(range(1, 5001)) == "1-5000"

    @pytest.mark.parametrize("ids", [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 8, 14, 15, 16],
        [42],
        [1, 3, 5, 7, 9],
    ])
    def test_round_trips_through_parse_job_ids(self, ids):
        """Output must be re-pasteable into pbx update/rm/info."""
        assert parse_job_ids(format_job_ids(ids).split()) == sorted(set(ids))


class TestGroupFailures:
    def test_identical_messages_collapse(self):
        failed = [(i, "already exists") for i in range(1, 6)]
        grouped = group_failures(failed)
        assert list(grouped) == ["already exists"]
        assert grouped["already exists"] == [1, 2, 3, 4, 5]

    def test_distinct_messages_stay_separate(self):
        failed = [(1, "err A"), (2, "err B"), (3, "err A")]
        grouped = group_failures(failed)
        assert grouped == {"err A": [1, 3], "err B": [2]}

    def test_preserves_first_seen_order(self):
        failed = [(1, "second"), (2, "first"), (3, "second")]
        assert list(group_failures(failed)) == ["second", "first"]

    def test_empty(self):
        assert group_failures([]) == {}

    def test_works_with_path_keys(self):
        failed = [("/a", "dup"), ("/b", "dup")]
        assert group_failures(failed) == {"dup": ["/a", "/b"]}


# ----------------- CLI output ----------------------------------------------- #


@pytest.fixture
def temp_db():
    d = tempfile.mkdtemp()
    yield Path(d) / "test.db"
    shutil.rmtree(d)


@pytest.fixture
def db_with_gpu_jobs(temp_db):
    database.initialize_database(temp_db)
    for i in range(1, 11):
        database.add_job(
            db_path=temp_db, path=f"/p{i:02d}", app="lammps-kk",
            num_nodes=1, ngpus=4, node_occupancy=1.0,
            tag="prod", in_file="in.lammps", status="Ready",
        )
    return temp_db


def run_update(db_path, *args):
    from parslbox.commands.update import app as update_app
    runner = CliRunner()
    with patch("parslbox.commands.update.path_utils.DB_FILE", db_path):
        return runner.invoke(update_app, list(args), env={"COLUMNS": "400"})


class TestUpdateOutput:
    def test_success_uses_compressed_ranges(self, db_with_gpu_jobs):
        res = run_update(db_with_gpu_jobs, "1-10", "--status", "Restart")
        assert res.exit_code == 0
        assert "Successfully updated 10 job(s): 1-10" in res.output
        assert "1, 2, 3" not in res.output

    def test_non_contiguous_success(self, db_with_gpu_jobs):
        res = run_update(db_with_gpu_jobs, "1-3", "5", "8-10", "--status", "Restart")
        assert res.exit_code == 0
        assert "1-3 5 8-10" in res.output

    def test_identical_failures_are_grouped(self, db_with_gpu_jobs):
        """All 10 are single-node GPU jobs, so all 10 fail auto-scaling
        with the same message — one grouped line, not ten."""
        res = run_update(db_with_gpu_jobs, "1-10", "--nnodes", "2")
        assert "Failed to update 10 job(s):" in res.output
        assert "[10 job(s)] 1-10:" in res.output
        assert res.output.count("Cannot automatically scale") == 1

    def test_args_only_update_is_reported_as_success(self, db_with_gpu_jobs):
        """Regression: --args writes in_file directly, and app_args was missing
        from the non_dependency_updates check, so it reported 'No jobs were
        updated' despite having written to the DB."""
        res = run_update(db_with_gpu_jobs, "1-5", "--args", "-var T 300")
        assert res.exit_code == 0
        assert "Successfully updated 5 job(s): 1-5" in res.output
        assert "No jobs were updated" not in res.output

        rows = database.get_jobs_by_ids(db_with_gpu_jobs, [1, 2, 3, 4, 5])
        assert all(r["in_file"] == "in.lammps -var T 300" for r in rows)

    def test_args_only_update_returns_ids_from_core(self, db_with_gpu_jobs):
        """The shared core must report the IDs it wrote. Note app_args is
        CLI-only — it is not exposed on ParslBox.update_jobs or the MCP schema."""
        from parslbox.commands.update import update_jobs as core_update_jobs
        updated, failed, _ = core_update_jobs(
            job_ids=[1, 2, 3], db_path=db_with_gpu_jobs, app_args="-var T 300"
        )
        assert sorted(updated) == [1, 2, 3]
        assert failed == []

    def test_failure_message_has_no_embedded_job_id(self, db_with_gpu_jobs):
        """Grouping only works if the message text is job-independent."""
        res = run_update(db_with_gpu_jobs, "1-10", "--nnodes", "2")
        assert "GPU job from 1 to 2 nodes" in res.output
        assert "GPU job 1 from" not in res.output


class TestMcpAddJobsResponse:
    def test_mcp_uses_compressed_ids(self):
        from parslbox.mcp import mcp_server
        assert hasattr(mcp_server, "format_job_ids")
        assert hasattr(mcp_server, "group_failures")
