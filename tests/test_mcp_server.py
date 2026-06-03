"""MCP server: verifies tool returns are sensible and token-efficient, and
that recent CLI/API changes (tag glob expansion, matched_jobs / resolved_tags
in submit results) reach the MCP layer.
"""
import pytest
from unittest.mock import patch, MagicMock

from parslbox.mcp.mcp_server import _fmt_resources, _job_one_line


# ----------------- Formatting helpers -------------------------------------- #


class TestFmtResources:
    def test_gpu_multi_node(self):
        j = {"num_nodes": 5, "ngpus": 60, "node_occupancy": 1.0, "ranks_per_node": 12}
        assert _fmt_resources(j) == "nodes:5, ranks:60, gpus:60, nocc:1.0"

    def test_cpu_fractional(self):
        j = {"num_nodes": 1, "ngpus": 0, "node_occupancy": 0.25, "ranks_per_node": 1}
        assert _fmt_resources(j) == "nodes:1, ranks:1, gpus:0, nocc:0.25"

    def test_defaults_when_missing(self):
        assert _fmt_resources({}) == "nodes:1, ranks:1, gpus:0, nocc:1.0"


class TestJobOneLine:
    def test_one_line_format(self):
        job = {
            "job_id": 1, "app": "lammps-kk", "status": "Ready",
            "num_nodes": 5, "ngpus": 60, "node_occupancy": 1.0, "ranks_per_node": 12,
            "tag": "film-bulk", "sched_job_id": None, "in_file": "in.lammps",
            "env_file": None, "parents": None, "timestamp": "2026-05-30 21:53:42",
            "path": "/lus/flare/run1",
        }
        line = _job_one_line(job)
        # Exactly one line — no embedded newlines
        assert "\n" not in line
        # Contains all the key fields with full names
        assert "#1" in line
        assert "lammps-kk" in line
        assert "Ready" in line
        assert "nodes:5" in line
        assert "ranks:60" in line
        assert "gpus:60" in line
        assert "tag:film-bulk" in line
        assert "/lus/flare/run1" in line

    def test_compact_vs_old_verbose(self):
        """50 jobs should be ~50 lines, not 550+."""
        jobs = [{
            "job_id": i, "app": "x", "status": "Ready",
            "num_nodes": 1, "ngpus": 0, "node_occupancy": 1.0, "ranks_per_node": 1,
            "tag": "t", "sched_job_id": None, "in_file": None,
            "env_file": None, "parents": None, "timestamp": "t", "path": f"/p{i}",
        } for i in range(50)]
        all_lines = "\n".join(_job_one_line(j) for j in jobs).count("\n")
        # 50 jobs → 49 newline separators
        assert all_lines == 49


# ----------------- Tag glob support visible in MCP schemas ----------------- #


class TestSchemaGlobDocs:
    def test_filter_schema_documents_glob(self):
        from parslbox.mcp.schemas import FilterJobsSchema
        desc = FilterJobsSchema.model_fields["tag"].description
        assert "*" in desc and "glob" in desc.lower()

    def test_list_schema_documents_glob(self):
        from parslbox.mcp.schemas import ListJobsSchema
        desc = ListJobsSchema.model_fields["tag"].description
        assert "*" in desc and "glob" in desc.lower()

    def test_qsub_schema_documents_glob(self):
        from parslbox.mcp.schemas import QSubSchema
        desc = QSubSchema.model_fields["tags"].description
        assert "*" in desc and "glob" in desc.lower()

    def test_sbatch_schema_documents_glob(self):
        from parslbox.mcp.schemas import SBatchSchema
        desc = SBatchSchema.model_fields["tags"].description
        assert "*" in desc and "glob" in desc.lower()


# ----------------- Submit MCP tools surface new result fields -------------- #


def _success_status(matched=None, resolved=None):
    s = {
        "success": True,
        "job_id": "12345",
        "pbs_job_id": "12345",
        "slurm_job_id": "12345",
        "run_dir": "/tmp/run",
    }
    if matched is not None:
        s["matched_jobs"] = matched
    if resolved is not None:
        s["resolved_tags"] = resolved
    return s


class TestSubmitMcpSurfacesNewFields:
    def test_pbs_includes_matched_and_resolved(self):
        from parslbox.mcp import mcp_server as m
        with patch.object(m.pbx, "qsub",
                          return_value=_success_status(matched=3, resolved=["a", "b"])):
            out = m.submit_pbs_job(MagicMock(
                model_dump=lambda: {"config": "x", "job_name": "j", "queue": "q",
                                    "select": "1", "walltime": 10}))
        assert "Matched 3 runnable job(s)" in out
        assert "Resolved tags: a, b" in out
        assert "12345" in out

    def test_pbs_omits_extras_when_absent(self):
        from parslbox.mcp import mcp_server as m
        with patch.object(m.pbx, "qsub", return_value=_success_status()):
            out = m.submit_pbs_job(MagicMock(
                model_dump=lambda: {"config": "x", "job_name": "j", "queue": "q",
                                    "select": "1", "walltime": 10}))
        # No spurious "Matched" or "Resolved tags" lines if API didn't return them
        assert "Matched" not in out
        assert "Resolved tags" not in out

    def test_slurm_includes_matched_and_resolved(self):
        from parslbox.mcp import mcp_server as m
        with patch.object(m.pbx, "sbatch",
                          return_value=_success_status(matched=2, resolved=["foo"])):
            out = m.submit_slurm_job(MagicMock(
                model_dump=lambda: {"config": "x", "job_name": "j", "queue": "q",
                                    "select": "1", "walltime": 10}))
        assert "Matched 2 runnable job(s)" in out
        assert "Resolved tags: foo" in out
