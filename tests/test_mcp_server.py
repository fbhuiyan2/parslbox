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


def _success_status(matched=None, resolved=None, respawn_template=None):
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
    if respawn_template is not None:
        s["respawn_template_file"] = respawn_template
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

    def test_pbs_surfaces_respawn_template_when_present(self):
        from parslbox.mcp import mcp_server as m
        path = "/tmp/run/respawn_template.sh"
        with patch.object(m.pbx, "qsub",
                          return_value=_success_status(matched=5, respawn_template=path)):
            out = m.submit_pbs_job(MagicMock(
                model_dump=lambda: {"config": "x", "job_name": "j", "queue": "q",
                                    "select": "1", "walltime": 10,
                                    "respawn": 3}))
        assert "Respawn template" in out
        assert path in out
        assert "auto-resubmits at walltime" in out

    def test_slurm_surfaces_respawn_template_when_present(self):
        from parslbox.mcp import mcp_server as m
        path = "/tmp/run/respawn_template.sh"
        with patch.object(m.pbx, "sbatch",
                          return_value=_success_status(respawn_template=path)):
            out = m.submit_slurm_job(MagicMock(
                model_dump=lambda: {"config": "x", "job_name": "j", "queue": "q",
                                    "select": "1", "walltime": 10,
                                    "respawn": 2}))
        assert "Respawn template" in out
        assert path in out

    def test_pbs_omits_respawn_template_when_absent(self):
        from parslbox.mcp import mcp_server as m
        with patch.object(m.pbx, "qsub", return_value=_success_status()):
            out = m.submit_pbs_job(MagicMock(
                model_dump=lambda: {"config": "x", "job_name": "j", "queue": "q",
                                    "select": "1", "walltime": 10}))
        assert "Respawn template" not in out


# ----------------- CLI / API / MCP parameter alignment --------------------- #


class TestLayerAlignment:
    """Every user-facing field the CLI accepts must also be reachable through
    the API and the MCP schema. `app_args` was CLI-only for several releases."""

    def _cli_option_names(self, command_module, func_name):
        import inspect
        return set(inspect.signature(
            getattr(command_module, func_name)).parameters)

    def test_add_app_args_reaches_api_and_mcp(self):
        import inspect
        from parslbox.api import ParslBox
        from parslbox.mcp.schemas import AddJobSchema
        from parslbox.commands import add as add_cmd

        assert "app_args" in inspect.signature(add_cmd.add).parameters
        assert "app_args" in inspect.signature(ParslBox.add_jobs).parameters
        assert "app_args" in AddJobSchema.model_fields

    def test_update_app_args_reaches_api_and_mcp(self):
        import inspect
        from parslbox.api import ParslBox
        from parslbox.mcp.schemas import UpdateJobSchema
        from parslbox.commands import update as update_cmd

        assert "app_args" in inspect.signature(update_cmd.update).parameters
        assert "app_args" in inspect.signature(ParslBox.update_jobs).parameters
        assert "app_args" in UpdateJobSchema.model_fields

    def test_update_schema_fields_all_exist_on_api(self):
        """Guard against a schema field the API cannot accept."""
        import inspect
        from parslbox.api import ParslBox
        from parslbox.mcp.schemas import UpdateJobSchema

        api_params = set(inspect.signature(ParslBox.update_jobs).parameters)
        schema_fields = set(UpdateJobSchema.model_fields) - {"job_id"}
        assert schema_fields <= api_params, schema_fields - api_params

    def test_add_schema_fields_all_exist_on_api(self):
        import inspect
        from parslbox.api import ParslBox
        from parslbox.mcp.schemas import AddJobSchema

        api_params = set(inspect.signature(ParslBox.add_jobs).parameters)
        schema_fields = set(AddJobSchema.model_fields)
        assert schema_fields <= api_params, schema_fields - api_params

    def test_mcp_add_passes_app_args_through(self):
        from unittest.mock import patch, MagicMock
        import parslbox.mcp.mcp_server as m

        with patch.object(m.pbx, "add_jobs", return_value=([1], [], {})) as mock_add:
            m.add_jobs(MagicMock(model_dump=lambda: {
                "paths": ["/p"], "app": "lammps-kk", "config": "polaris",
                "input_file": "in.lammps", "app_args": "-var T 300",
            }))
        assert mock_add.call_args.kwargs["app_args"] == "-var T 300"

    def test_mcp_update_passes_app_args_through(self):
        from unittest.mock import patch, MagicMock
        import parslbox.mcp.mcp_server as m

        with patch.object(m.pbx, "update_jobs", return_value=([1], [], {})) as mock_upd:
            m.update_job(MagicMock(model_dump=lambda: {
                "job_id": 1, "app_args": "-var T 300",
            }))
        assert mock_upd.call_args.kwargs["app_args"] == "-var T 300"
