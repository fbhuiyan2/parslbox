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


# ----------------- Local projects: schema reaches the API ------------------ #


class TestLocalInitSchema:
    """The MCP caller sees only the schema, so a setting missing from it is
    a setting that layer cannot use, even though CLI and API can."""

    def test_transfer_remote_root_is_offered(self):
        from parslbox.mcp.schemas import LocalInitSchema
        assert "transfer_remote_root" in LocalInitSchema.model_fields

    def test_its_description_explains_the_rewrite(self):
        # The LLM caller has nothing but this text to work out that a facility
        # collection publishes a subtree rather than the whole filesystem.
        from parslbox.mcp.schemas import LocalInitSchema
        desc = LocalInitSchema.model_fields["transfer_remote_root"].description
        assert "root" in desc.lower()
        assert "/lus/flare/projects" in desc

    def test_it_is_passed_through_to_the_api(self):
        from parslbox.mcp.schemas import LocalInitSchema
        dumped = LocalInitSchema(
            directory="/home/me/p1", remote_root="/lus/flare/projects/ABC/me/p1",
            transfer_remote_root="/lus/flare/projects",
        ).model_dump()
        assert dumped["transfer_remote_root"] == "/lus/flare/projects"

    def test_the_api_accepts_every_schema_field(self):
        # local_init is called as pbx.local_init(**params.model_dump()), so any
        # schema field the API does not take is an immediate TypeError.
        import inspect
        from parslbox.api import ParslBox
        from parslbox.mcp.schemas import LocalInitSchema
        accepted = set(inspect.signature(ParslBox.local_init).parameters)
        assert set(LocalInitSchema.model_fields) <= accepted


class TestLocalToolRendering:
    """The MCP renderers for pbx local.

    These had no tests, which is how a KeyError on a perfectly ordinary report
    shape survived: sync.status() returns early with an 'error' key and no
    'local' block when the project's database has gone missing, and the
    renderer read report['local']['drift'] unconditionally.
    """

    def _status(self, report, monkeypatch):
        from parslbox.mcp import mcp_server
        monkeypatch.setattr(mcp_server.pbx, "local_status", lambda **kw: report)
        from parslbox.mcp.schemas import LocalStatusSchema
        return mcp_server.local_status(LocalStatusSchema())

    def test_a_missing_database_is_reported_not_crashed(self, monkeypatch):
        out = self._status({"is_local_project": True, "local_root": "/home/me/p1",
                            "error": "no such file: job_database_pbx-local.db"}, monkeypatch)
        assert "could not be read" in out
        assert "no such file" in out

    def test_a_misconfigured_project_says_why(self, monkeypatch):
        out = self._status({"is_local_project": False, "misconfigured": True,
                            "message": "PBX_DB_PATH names the directory"}, monkeypatch)
        assert "PBX_DB_PATH names the directory" in out

    def test_a_healthy_project_reports_the_collection_root(self, monkeypatch):
        out = self._status({
            "is_local_project": True, "local_root": "/home/me/p1",
            "remote_root": "/lus/flare/projects/ABC/me/p1",
            "db_path": "/home/me/p1/job_database_pbx-local.db", "job_count": 3,
            "compute_endpoint": "abc", "last_sync": None, "export_line": "export X=1",
            "local": {"drift": "unchanged"}, "remote": {"checked": False},
            "transfer_remote": "dst-uuid",
            "transfer_remote_root": "/lus/flare/projects",
        }, monkeypatch)
        assert "/lus/flare/projects" in out
        assert "3 jobs" in out

    def test_an_undetected_collection_root_says_so(self, monkeypatch):
        out = self._status({
            "is_local_project": True, "local_root": "/home/me/p1",
            "remote_root": "/r", "db_path": "/d", "job_count": 0,
            "compute_endpoint": "abc", "last_sync": None, "export_line": "export X=1",
            "local": {"drift": "unchanged"}, "remote": {"checked": False},
            "transfer_remote": "dst-uuid", "transfer_remote_root": None,
        }, monkeypatch)
        assert "not detected yet" in out

    def _push(self, result, monkeypatch):
        from parslbox.mcp import mcp_server
        monkeypatch.setattr(mcp_server.pbx, "local_push", lambda **kw: result)
        from parslbox.mcp.schemas import LocalPushSchema
        return mcp_server.local_push(LocalPushSchema())

    def test_push_reports_a_detected_collection_root(self, monkeypatch):
        out = self._push({
            "bytes": 4096, "remote_db_path": "/r/db", "fingerprint": {"count": 2},
            "transfer_remote_root": "/lus/flare/projects",
            "transfer_remote_root_detected": True,
        }, monkeypatch)
        assert "/lus/flare/projects" in out

    def test_push_surfaces_nice_status(self, monkeypatch):
        # The field that names why Globus is stuck; an agent has nothing else.
        out = self._push({
            "bytes": 1, "remote_db_path": "/r/db", "fingerprint": {"count": 1},
            "transfer": {"submitted": True, "count": 2, "batches": 1,
                         "task_ids": ["t1"],
                         "outcome": {"status": "ACTIVE", "files_transferred": 0,
                                     "bytes_transferred": 0,
                                     "nice_status": "PERMISSION_DENIED"}},
        }, monkeypatch)
        assert "PERMISSION_DENIED" in out

    def test_push_says_when_there_was_nothing_to_transfer(self, monkeypatch):
        out = self._push({
            "bytes": 1, "remote_db_path": "/r/db", "fingerprint": {"count": 1},
            "transfer": {"submitted": False, "reason": "nothing to transfer"},
        }, monkeypatch)
        assert "nothing to transfer" in out

    def test_a_validation_error_becomes_a_message_not_a_traceback(self, monkeypatch):
        from parslbox.mcp import mcp_server
        from parslbox.api import ValidationError
        from parslbox.mcp.schemas import LocalPushSchema

        def boom(**kw):
            raise ValidationError("the remote database has changed")
        monkeypatch.setattr(mcp_server.pbx, "local_push", boom)
        out = mcp_server.local_push(LocalPushSchema())
        assert "the remote database has changed" in out


class TestRemoteSubmissionIsVisibleToTheAgent:
    """A submission inside a local project runs on another machine.

    Rendered like a local one it is actively misleading: the run directory
    does not exist here, and a failed post-submit pull -- which leaves the
    local database behind the remote -- would be silent.
    """

    def _submit(self, status, monkeypatch):
        from parslbox.mcp import mcp_server
        from parslbox.mcp.schemas import QSubSchema
        monkeypatch.setattr(mcp_server.pbx, "qsub", lambda **kw: status)
        return mcp_server.submit_pbs_job(
            QSubSchema(config="aurora", job_name="j", queue="debug", select="1",
                       walltime=30))

    BASE = {"success": True, "job_id": "8819271.aurora", "run_dir": "/lus/x/run",
            "remote": True, "endpoint": "ep-uuid",
            "remote_run_dir": "/lus/flare/projects/ABC/me/p1/pbx_runs/1",
            "push": {"bytes": 4096}}

    def test_it_names_the_endpoint_and_the_remote_run_dir(self, monkeypatch):
        out = self._submit(dict(self.BASE), monkeypatch)
        assert "ep-uuid" in out
        assert "/lus/flare/projects/ABC/me/p1/pbx_runs/1" in out

    def test_a_failed_pull_back_is_a_warning(self, monkeypatch):
        out = self._submit(dict(self.BASE, pull_error="endpoint went away"), monkeypatch)
        assert "WARNING" in out
        assert "endpoint went away" in out

    def test_a_local_submission_gains_none_of_this(self, monkeypatch):
        out = self._submit({"success": True, "job_id": "1.local",
                            "run_dir": "/home/me/run"}, monkeypatch)
        assert "Compute endpoint" not in out
        assert "WARNING" not in out


class TestEveryLocalSchemaMatchesItsApiMethod:
    """Each tool calls pbx.<method>(**params.model_dump()); a field the API
    does not accept is an immediate TypeError at call time."""

    @pytest.mark.parametrize("schema_name,method", [
        ("LocalInitSchema", "local_init"),
        ("LocalStatusSchema", "local_status"),
        ("LocalPushSchema", "local_push"),
        ("LocalPullSchema", "local_pull"),
    ])
    def test_the_api_accepts_every_field(self, schema_name, method):
        import inspect
        from parslbox.api import ParslBox
        from parslbox.mcp import schemas
        schema = getattr(schemas, schema_name)
        accepted = set(inspect.signature(getattr(ParslBox, method)).parameters)
        assert set(schema.model_fields) <= accepted, schema_name

    @pytest.mark.parametrize("schema_name,method", [
        ("LocalPushSchema", "local_push"),
        ("LocalPullSchema", "local_pull"),
    ])
    def test_defaults_agree_with_the_api(self, schema_name, method):
        import inspect
        from parslbox.api import ParslBox
        from parslbox.mcp import schemas
        schema = getattr(schemas, schema_name)
        params = inspect.signature(getattr(ParslBox, method)).parameters
        for name, field in schema.model_fields.items():
            if params[name].default is not inspect.Parameter.empty:
                assert field.default == params[name].default, f"{schema_name}.{name}"


class TestStartupFailureIsReadable:
    """The MCP client shows stderr and nothing else, so it has to say enough."""

    def _start_with(self, monkeypatch, capsys, error):
        from parslbox.mcp import mcp_server

        def boom(*a, **kw):
            raise error

        monkeypatch.setattr(mcp_server, "ParslBox", boom)
        with pytest.raises(SystemExit) as exit_info:
            mcp_server._start()
        assert exit_info.value.code == 1
        return capsys.readouterr().err

    def test_wrong_database_name_is_reported_without_a_traceback(
            self, monkeypatch, capsys, tmp_path):
        from parslbox.local.project import LOCAL_DB_NAME, check_db_name, init_project
        from parslbox.local.project import WrongDatabaseName

        init_project(tmp_path, remote_root="/lus/flare/projects/x/proj1")
        try:
            check_db_name(tmp_path / "job_database_pbx.db")
        except WrongDatabaseName as e:
            err = self._start_with(monkeypatch, capsys, e)
        assert "ParslBox MCP server did not start" in err
        assert str(tmp_path / LOCAL_DB_NAME) in err

    def test_a_missing_config_is_reported_too(self, monkeypatch, capsys):
        err = self._start_with(
            monkeypatch, capsys, FileNotFoundError("Config file not found at: /x"))
        assert "Config file not found" in err

    def test_an_unexpected_error_still_raises(self, monkeypatch):
        from parslbox.mcp import mcp_server

        def boom(*a, **kw):
            raise RuntimeError("a real bug")

        monkeypatch.setattr(mcp_server, "ParslBox", boom)
        with pytest.raises(RuntimeError, match="a real bug"):
            mcp_server._start()
