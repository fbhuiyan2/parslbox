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
            m.update_jobs(MagicMock(model_dump=lambda: {
                "job_ids": [1], "app_args": "-var T 300",
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
            "remote_rundir": "/lus/flare/projects/ABC/me/p1/pbx_runs/1",
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


# ----------------- use_database: switching + visibility -------------------- #


class TestUseDatabase:
    """The server's binding is fixed until use_database changes it, so the
    tool must switch cleanly and every job tool must say where it answered
    from. A silent wrong database is the failure this exists to prevent.
    """

    @pytest.fixture
    def projects(self, tmp_path):
        """Two project dirs with different job counts."""
        from parslbox.database import database
        made = {}
        for name, n in (("projA", 2), ("projB", 5)):
            db = tmp_path / name / "job_database_pbx.db"
            database.initialize_database(db)
            for i in range(n):
                database.add_job(db_path=db, path=f"/{name}/j{i}", app="lammps-kk",
                                 num_nodes=1, ngpus=0, node_occupancy=1.0,
                                 tag=name, in_file=None, status="Ready")
            made[name] = tmp_path / name
        return made

    @pytest.fixture
    def server(self, projects):
        """Bound to projA, restored afterwards so tests stay independent."""
        from parslbox.mcp import mcp_server as m
        from parslbox.mcp.schemas import UseDatabaseSchema
        original = m.pbx
        m.use_database(UseDatabaseSchema(path=str(projects["projA"])))
        yield m, projects
        m.pbx = original

    def _use(self, m, path, **kw):
        from parslbox.mcp.schemas import UseDatabaseSchema
        return m.use_database(UseDatabaseSchema(path=str(path), **kw))

    def _list(self, m):
        from parslbox.mcp.schemas import ListJobsSchema
        return m.list_jobs(ListJobsSchema())

    # --- switching ---

    def test_switch_changes_the_binding(self, server):
        m, projects = server
        out = self._use(m, projects["projB"])
        assert str(projects["projB"] / "job_database_pbx.db") in out
        assert m.pbx.db_path == projects["projB"] / "job_database_pbx.db"

    def test_switch_reports_job_count(self, server):
        m, projects = server
        assert "jobs: 5" in self._use(m, projects["projB"])

    def test_switch_names_the_previous_database(self, server):
        m, projects = server
        assert "projA" in self._use(m, projects["projB"])

    def test_subsequent_calls_read_the_new_database(self, server):
        m, projects = server
        assert "Found 2 job(s)" in self._list(m)
        self._use(m, projects["projB"])
        assert "Found 5 job(s)" in self._list(m)

    def test_accepts_a_db_file_directly(self, server):
        m, projects = server
        db = projects["projB"] / "job_database_pbx.db"
        self._use(m, db)
        assert m.pbx.db_path == db

    # --- refusals leave the binding untouched ---

    def test_nonexistent_path_is_refused(self, server, tmp_path):
        m, _ = server
        before = m.pbx.db_path
        out = self._use(m, tmp_path / "typo-dir")
        assert "No database at" in out
        assert m.pbx.db_path == before

    def test_refusal_says_what_is_still_in_use(self, server, tmp_path):
        m, _ = server
        assert "projA" in self._use(m, tmp_path / "typo-dir")

    def test_relative_path_is_refused(self, server):
        m, _ = server
        before = m.pbx.db_path
        out = self._use(m, "some/relative/dir")
        assert "absolute" in out
        assert m.pbx.db_path == before

    def test_create_flag_allows_a_new_database(self, server, tmp_path):
        m, _ = server
        fresh = tmp_path / "brand-new"
        fresh.mkdir()
        out = self._use(m, fresh, create=True)
        assert "Now using" in out
        assert m.pbx.db_path == fresh / "job_database_pbx.db"

    def test_missing_config_is_refused(self, server, projects, tmp_path):
        m, _ = server
        before = m.pbx.config_path
        out = self._use(m, projects["projB"], config_path=str(tmp_path / "nope.yaml"))
        assert "No config file at" in out
        assert m.pbx.config_path == before

    # --- visibility ---

    def test_job_tools_report_the_database(self, server):
        m, projects = server
        first = self._list(m).splitlines()[0]
        assert first.startswith("db: projA (")
        assert str(projects["projA"] / "job_database_pbx.db") in first

    def test_db_line_follows_a_switch(self, server, projects):
        m, _ = server
        self._use(m, projects["projB"])
        assert self._list(m).splitlines()[0].startswith("db: projB (")

    def test_db_line_leads_with_the_project_directory_name(self, server):
        """A wrong project is easier to spot as a name than as a long path."""
        m, _ = server
        assert self._list(m).splitlines()[0].startswith("db: projA")

    def test_local_tools_are_not_double_prefixed(self, server):
        """local_status already reports the path itself."""
        from parslbox.mcp.schemas import LocalStatusSchema
        m, _ = server
        assert not m.local_status(LocalStatusSchema(check_remote=False)).startswith("db:")

    def test_instructions_name_the_binding_and_the_rule(self):
        from parslbox.mcp import mcp_server as m
        text = m.mcp.instructions
        assert "ACTIVE DATABASE:" in text
        assert "use_database" in text
        assert "does NOT follow" in text


class TestSchemaArgumentHygiene:
    """Unknown arguments must be rejected, and the error must name the valid ones.

    Pydantic's default silently drops an unrecognised key, so `nodes=4` instead
    of `nnodes=4` stored the default and the tool still answered "Successfully
    added". Rejecting without naming the alternatives is not enough either --
    the caller just guesses again.
    """

    def test_unknown_argument_is_rejected(self):
        from pydantic import ValidationError
        from parslbox.mcp.schemas import AddJobSchema

        with pytest.raises(ValidationError):
            AddJobSchema(paths=["/j"], app="lammps-kk", config="polaris", nodes=4)

    def test_error_names_the_bad_key_and_the_valid_ones(self):
        from pydantic import ValidationError
        from parslbox.mcp.schemas import AddJobSchema

        with pytest.raises(ValidationError) as exc:
            AddJobSchema(paths=["/j"], app="lammps-kk", config="polaris",
                         nodes=4, tags=["t"])
        msg = str(exc.value)
        assert "nodes" in msg and "tags" in msg
        assert "nnodes" in msg and "tag" in msg

    def test_correct_arguments_still_accepted(self):
        from parslbox.mcp.schemas import AddJobSchema

        s = AddJobSchema(paths=["/j"], app="lammps-kk", config="polaris",
                         nnodes=2, ngpus=4, tag="run-a")
        assert (s.nnodes, s.ngpus, s.tag) == (2, 4, "run-a")

    def test_every_mcp_schema_rejects_extras(self):
        from parslbox.mcp import schemas as S

        for name in dir(S):
            obj = getattr(S, name)
            if (isinstance(obj, type) and issubclass(obj, S.BaseModel)
                    and obj not in (S.BaseModel, S.McpSchema)):
                assert obj.model_config.get("extra") == "forbid", name


class TestSchemaTextBlock:
    """A tool declared `def tool(params: XSchema)` reaches the client as one
    property behind a $ref; not every model dereferences it. The rendered
    block restates the fields where there is no reference to follow."""

    def test_block_lists_fields_with_requiredness(self):
        from parslbox.mcp.schemas import AddJobSchema, schema_text_block

        block = schema_text_block(AddJobSchema)
        assert "paths (array, required)" in block
        assert "nnodes (integer, optional, default 1)" in block
        assert "app_args" in block

    def test_block_is_generated_not_hardcoded(self):
        """Adding a field to a schema must show up without further edits."""
        from pydantic import Field
        from parslbox.mcp.schemas import McpSchema, schema_text_block

        class Tmp(McpSchema):
            alpha: str = Field(description="x")
            beta: int = Field(default=7, description="y")

        block = schema_text_block(Tmp)
        assert "alpha (string, required)" in block
        assert "beta (integer, optional, default 7)" in block

    def test_every_tool_description_carries_its_arguments(self):
        import asyncio
        from parslbox.mcp.mcp_server import mcp

        tools = asyncio.run(mcp.list_tools())
        assert tools
        # Only a tool declared `def tool(params: XSchema)` puts its fields
        # behind a $ref. One that takes no arguments has nothing to restate.
        checked = 0
        for t in tools:
            if "params" not in (t.inputSchema.get("properties") or {}):
                continue
            checked += 1
            assert "Arguments, all inside the single `params` object:" in t.description, t.name
        assert checked

    def test_add_jobs_description_names_the_resource_fields(self):
        """These are the ones models most often guess wrong (nodes/gpus)."""
        import asyncio
        from parslbox.mcp.mcp_server import mcp

        tools = asyncio.run(mcp.list_tools())
        t = [x for x in tools if x.name == "add_jobs"][0]
        for field in ("paths", "app", "config", "nnodes", "ngpus", "tag"):
            assert field in t.description


# ------------- Bulk ids: ranges, plural update, tool naming ---------------- #


def _db_with(n_jobs, tmp_path):
    """A database holding job ids 1..n_jobs, for bounding ranges against."""
    from parslbox.database import database
    db = tmp_path / "mcp.db"
    database.initialize_database(db)
    for i in range(n_jobs):
        database.add_job(db_path=db, path=f"/w/j{i}", app="python", num_nodes=1,
                         ngpus=0, node_occupancy=1.0, tag="t", in_file=None,
                         ranks_per_node=1)
    return db


class TestJobIdRanges:
    """An agent doing a bulk delete used to transcribe every id from a prior
    tool's output. One corrupted element removes the wrong job, so the MCP
    layer takes the same range syntax the CLI has always accepted.

    Expansion happens in the tool body, not the schema validator: the
    validator runs before any database is in reach, so it could not refuse
    '1-100000000' without first building it.
    """

    def _ids_reaching(self, tokens, tmp_path, n_jobs=20):
        import parslbox.mcp.mcp_server as m
        db = _db_with(n_jobs, tmp_path)
        with patch.object(m.pbx, "db_path", db), \
             patch.object(m.pbx, "remove_jobs", return_value=0) as mock_rm:
            m.remove_jobs(MagicMock(model_dump=lambda: {"job_ids": tokens}))
        return mock_rm.call_args.kwargs["job_ids"]

    def test_remove_expands_a_range(self, tmp_path):
        assert self._ids_reaching(["5-8"], tmp_path) == [5, 6, 7, 8]

    def test_ids_and_ranges_mix(self, tmp_path):
        assert self._ids_reaching([1, 2, "5-7"], tmp_path) == [1, 2, 5, 6, 7]

    def test_plain_ints_still_work_and_are_sorted_unique(self, tmp_path):
        assert self._ids_reaching([3, 1, 2, 3], tmp_path) == [1, 2, 3]

    def test_update_takes_ranges_too(self, tmp_path):
        import parslbox.mcp.mcp_server as m
        db = _db_with(20, tmp_path)
        with patch.object(m.pbx, "db_path", db), \
             patch.object(m.pbx, "update_jobs",
                          return_value=([1, 2, 3], [], {})) as mock_upd:
            m.update_jobs(MagicMock(model_dump=lambda: {
                "job_ids": ["1-3"], "ngpus": 6}))
        assert mock_upd.call_args.kwargs["job_ids"] == [1, 2, 3]

    def test_a_backwards_range_is_refused(self):
        import pytest
        from pydantic import ValidationError
        from parslbox.mcp.schemas import RemoveJobsSchema
        with pytest.raises(ValidationError, match="start must be <= end"):
            RemoveJobsSchema(job_ids=["5-1"])

    def test_the_cli_parser_is_reused_not_reimplemented(self, tmp_path):
        # If these diverge, the same argument means different things in two
        # layers, which is worse than the feature being missing.
        from parslbox.commands.helpers.job_id_parser import parse_job_ids
        tokens = ["1-3", "7", "10-12"]
        assert self._ids_reaching(tokens, tmp_path) == parse_job_ids(tokens)

    def test_the_schema_hands_on_tokens_rather_than_expanding_them(self):
        """The whole point: a hundred million integers are never built."""
        from parslbox.mcp.schemas import RemoveJobsSchema
        assert RemoveJobsSchema(job_ids=["1-100000000"]).job_ids == ["1-100000000"]

    def test_a_range_past_the_last_job_is_refused_by_the_tool(self, tmp_path):
        import parslbox.mcp.mcp_server as m
        db = _db_with(5, tmp_path)
        with patch.object(m.pbx, "db_path", db), \
             patch.object(m.pbx, "remove_jobs") as mock_rm:
            out = m.remove_jobs(MagicMock(model_dump=lambda: {
                "job_ids": ["1-100000000"]}))
        assert mock_rm.call_count == 0
        assert "1-100000000" in out and "5" in out

    def test_a_bare_id_past_the_last_job_is_not_refused(self, tmp_path):
        """Only ranges are bounded. One id costs nothing to carry, and each
        command already reports the ids it could not find."""
        assert self._ids_reaching([99999], tmp_path, n_jobs=5) == [99999]


class TestUpdateJobsIsPlural:
    """update_job took one id while the CLI took ranges and api.update_jobs
    took a list, so retuning five jobs cost five round trips -- and models
    measurably stopped partway through."""

    def test_schema_field_is_plural(self):
        from parslbox.mcp.schemas import UpdateJobSchema
        assert "job_ids" in UpdateJobSchema.model_fields
        assert "job_id" not in UpdateJobSchema.model_fields

    def test_the_old_singular_name_is_refused_and_the_new_one_named(self):
        import pytest
        from pydantic import ValidationError
        from parslbox.mcp.schemas import UpdateJobSchema
        with pytest.raises(ValidationError) as exc:
            UpdateJobSchema(job_id=1, ngpus=6)
        assert "job_ids" in str(exc.value)

    def test_every_id_reaches_the_api_in_one_call(self):
        from unittest.mock import patch, MagicMock
        import parslbox.mcp.mcp_server as m
        with patch.object(m.pbx, "update_jobs",
                          return_value=([1, 2, 3], [], {})) as mock_upd:
            out = m.update_jobs(MagicMock(model_dump=lambda: {
                "job_ids": [1, 2, 3], "ngpus": 6,
            }))
        assert mock_upd.call_count == 1
        assert mock_upd.call_args.kwargs["job_ids"] == [1, 2, 3]
        assert "3 job(s)" in out

    def test_partial_failure_names_both_sides(self):
        from unittest.mock import patch, MagicMock
        import parslbox.mcp.mcp_server as m
        with patch.object(m.pbx, "update_jobs",
                          return_value=([1], [(2, "bad status")], {})):
            out = m.update_jobs(MagicMock(model_dump=lambda: {
                "job_ids": [1, 2, 3], "ngpus": 6,
            }))
        assert "Successfully updated 1 job(s)" in out
        assert "Failed to update job 2: bad status" in out
        assert "3" in out.split("Not found")[-1]   # 3 was neither updated nor failed


class TestToolNamesArePlural:
    def test_registered_names(self):
        import asyncio
        from parslbox.mcp.mcp_server import mcp
        names = {t.name for t in asyncio.run(mcp.list_tools())}
        assert {"update_jobs", "remove_jobs"} <= names
        assert "update_job" not in names and "remove_job" not in names

    def test_docs_list_the_real_names(self):
        from pathlib import Path
        doc = Path(__file__).resolve().parents[1] / "docs" / "agentic_usage.md"
        text = doc.read_text()
        assert "`update_jobs`" in text and "`remove_jobs`" in text


class TestSubmitLoglevelReachesMcp:
    """api.qsub accepted loglevel but no schema field declared it, so every
    MCP submission was pinned to the default with no way to say otherwise."""

    def test_both_submit_schemas_offer_it(self):
        from parslbox.mcp.schemas import QSubSchema, SBatchSchema
        assert "loglevel" in QSubSchema.model_fields
        assert "loglevel" in SBatchSchema.model_fields

    def test_it_defaults_to_info(self):
        from parslbox.mcp.schemas import QSubSchema
        s = QSubSchema(config="polaris", job_name="j", queue="q",
                       select="1", walltime=10)
        assert s.loglevel == "info"

    def test_it_reaches_the_api(self):
        from unittest.mock import patch, MagicMock
        import parslbox.mcp.mcp_server as m
        with patch.object(m.pbx, "qsub",
                          return_value={"success": True}) as mock_qsub:
            m.submit_pbs_job(MagicMock(model_dump=lambda: {
                "config": "polaris", "job_name": "j", "queue": "q",
                "select": "1", "walltime": 10, "loglevel": "debug",
            }))
        assert mock_qsub.call_args.kwargs["loglevel"] == "debug"


class TestInstructionsMatchRegisteredTools:
    """The server's `instructions` block lists the tools in prose, by hand.
    It drifted twice unnoticed -- it advertised `remove_jobs` while the tool
    was `remove_job`, and kept naming `update_job` after the rename. The
    instructions are the first thing a model reads, so a wrong name there
    sends it to a tool that does not exist."""

    @staticmethod
    def _advertised():
        import re
        from parslbox.mcp.mcp_server import mcp
        body = mcp.instructions.split("Available capabilities:")[1]
        return {m.group(1) for m in re.finditer(r"^- (\w+):", body, re.M)}

    @staticmethod
    def _registered():
        import asyncio
        from parslbox.mcp.mcp_server import mcp
        return {t.name for t in asyncio.run(mcp.list_tools())}

    def test_nothing_advertised_that_does_not_exist(self):
        assert not (self._advertised() - self._registered())

    def test_nothing_registered_goes_unadvertised(self):
        assert not (self._registered() - self._advertised())


class TestDynamicReachesEveryLayer:
    """submit_job has always had `dynamic`, and qsub.py forwarded it, but
    api.qsub never declared it -- so a static submission was reachable only
    from the CLI, and silently defaulted to dynamic everywhere else."""

    def test_api_declares_it(self):
        import inspect
        from parslbox.api import ParslBox
        for name in ("qsub", "sbatch"):
            sig = inspect.signature(getattr(ParslBox, name))
            assert "dynamic" in sig.parameters, name
            assert sig.parameters["dynamic"].default is True, name

    def test_both_schemas_offer_it(self):
        from parslbox.mcp.schemas import QSubSchema, SBatchSchema
        assert QSubSchema.model_fields["dynamic"].default is True
        assert SBatchSchema.model_fields["dynamic"].default is True

    def test_false_reaches_the_submit_helper(self):
        from unittest.mock import patch, MagicMock
        import parslbox.mcp.mcp_server as m
        with patch.object(m.pbx, "qsub", return_value={"success": True}) as mock:
            m.submit_pbs_job(MagicMock(model_dump=lambda: {
                "config": "polaris", "job_name": "j", "queue": "q",
                "select": "1", "walltime": 10, "dynamic": False,
            }))
        assert mock.call_args.kwargs["dynamic"] is False

    def test_api_forwards_it_to_the_scheduler_wrapper(self):
        from unittest.mock import patch
        from parslbox.api import ParslBox
        with patch("parslbox.api.submit_to_scheduler",
                   return_value={"success": True}) as mock:
            with patch.object(ParslBox, "__init__", lambda self: None):
                pbx = ParslBox()
                pbx.config_path = None
                pbx.db_path = None
                pbx.qsub(config="polaris", job_name="j", queue="q",
                         select="1", walltime=10, dynamic=False)
        assert mock.call_args.kwargs["dynamic"] is False


class TestLocalInitVerify:
    """`--verify` pinged the endpoint after init, but only from the CLI."""

    def test_schema_and_api_offer_it(self):
        import inspect
        from parslbox.api import ParslBox
        from parslbox.mcp.schemas import LocalInitSchema
        assert inspect.signature(ParslBox.local_init).parameters["verify"].default is False
        assert LocalInitSchema.model_fields["verify"].default is False

    def test_no_ping_unless_asked(self):
        from unittest.mock import patch, MagicMock
        from parslbox.api import ParslBox
        proj = MagicMock(local_root="/p", remote_root="/r", db_path="/p/db",
                         compute_endpoint="uuid", export_line="export X=1")
        with patch("parslbox.local.project.init_project",
                   return_value={"action": "created", "project": proj, "notes": []}), \
             patch("parslbox.local.compute.ping") as ping, \
             patch.object(ParslBox, "__init__", lambda self: None):
            pbx = ParslBox()
            out = pbx.local_init("/p", remote_root="/r")
        ping.assert_not_called()
        assert "endpoint_check" not in out

    def test_ping_result_is_returned_when_asked(self):
        from unittest.mock import patch, MagicMock
        from parslbox.api import ParslBox
        proj = MagicMock(local_root="/p", remote_root="/r", db_path="/p/db",
                         compute_endpoint="uuid", export_line="export X=1")
        with patch("parslbox.local.project.init_project",
                   return_value={"action": "created", "project": proj, "notes": []}), \
             patch("parslbox.local.compute.ping",
                   return_value={"reachable": True, "host": "aurora"}) as ping, \
             patch.object(ParslBox, "__init__", lambda self: None):
            pbx = ParslBox()
            out = pbx.local_init("/p", remote_root="/r", verify=True)
        ping.assert_called_once_with("uuid")
        assert out["endpoint_check"]["host"] == "aurora"

    def test_skipped_when_there_is_no_endpoint_to_ping(self):
        from unittest.mock import patch, MagicMock
        from parslbox.api import ParslBox
        proj = MagicMock(local_root="/p", remote_root="/r", db_path="/p/db",
                         compute_endpoint=None, export_line="export X=1")
        with patch("parslbox.local.project.init_project",
                   return_value={"action": "created", "project": proj, "notes": []}), \
             patch("parslbox.local.compute.ping") as ping, \
             patch.object(ParslBox, "__init__", lambda self: None):
            pbx = ParslBox()
            out = pbx.local_init("/p", remote_root="/r", verify=True)
        ping.assert_not_called()
        assert "endpoint_check" not in out


class TestMcpSchemasTrackTheApi:
    """Every MCP tool is `pbx.<method>(**params.model_dump())`, so the schema's
    field set -- not the API signature -- decides what a caller can reach. An
    undeclared parameter is not an error, it is silently pinned to the API's
    default. That is how `loglevel` and `dynamic` went missing unnoticed, so
    the two lists get compared here rather than by hand."""

    PAIRS = {
        "AddJobSchema": "add_jobs", "FilterJobsSchema": "filter_jobs",
        "ListJobsSchema": "list_jobs", "GetJobSchema": "get_job",
        "GetJobsByIdsSchema": "get_jobs_by_ids", "QSubSchema": "qsub",
        "SBatchSchema": "sbatch", "RemoveJobsSchema": "remove_jobs",
        "UpdateJobSchema": "update_jobs", "LocalInitSchema": "local_init",
        "LocalStatusSchema": "local_status", "LocalPushSchema": "local_push",
        "LocalPullSchema": "local_pull",
        "ResourceRequirementsSchema": "resource_requirements",
        "CommandLinesSchema": "command_lines",
        "CancelJobSchema": "qdel",
    }

    # Schemas with no single API method behind them, and why.
    UNPAIRED = {"UseDatabaseSchema"}   # rebinds the server's ParslBox instance

    # Deliberate omissions only. Anything else is drift.
    ALLOWED_GAPS = {}

    def test_every_schema_is_checked_by_this_guard(self):
        # A guard with a hand-written pair list has the same drift hole as the
        # lists it guards: a new schema left out of PAIRS is simply not checked.
        import inspect
        from parslbox.mcp import schemas as S
        from parslbox.mcp.schemas import McpSchema
        defined = {n for n, o in vars(S).items()
                   if inspect.isclass(o) and issubclass(o, McpSchema) and o is not McpSchema}
        assert defined - set(self.PAIRS) - self.UNPAIRED == set()

    def test_qdel_and_scancel_stay_interchangeable(self):
        # One schema serves both, so a parameter added to only one would be
        # reachable for that scheduler and silently defaulted for the other.
        import inspect
        from parslbox.api import ParslBox
        assert (inspect.signature(ParslBox.qdel).parameters.keys()
                == inspect.signature(ParslBox.scancel).parameters.keys())

    def test_no_api_parameter_is_unreachable_over_mcp(self):
        import inspect
        from parslbox.api import ParslBox
        from parslbox.mcp import schemas as S
        missing = {}
        for schema_name, api_name in self.PAIRS.items():
            fields = set(getattr(S, schema_name).model_fields)
            params = {p for p in inspect.signature(getattr(ParslBox, api_name)).parameters
                      if p != "self"}
            gap = params - fields - set(self.ALLOWED_GAPS.get(schema_name, ()))
            if gap:
                missing[schema_name] = sorted(gap)
        assert not missing, f"API parameters with no MCP field: {missing}"

    def test_no_mcp_field_is_rejected_by_the_api(self):
        import inspect
        from parslbox.api import ParslBox
        from parslbox.mcp import schemas as S
        extra = {}
        for schema_name, api_name in self.PAIRS.items():
            fields = set(getattr(S, schema_name).model_fields)
            params = {p for p in inspect.signature(getattr(ParslBox, api_name)).parameters
                      if p != "self"}
            if fields - params:
                extra[schema_name] = sorted(fields - params)
        assert not extra, f"MCP fields the API will not accept: {extra}"


class TestListJobsLimit:
    """`pbx ls 20` has always been able to cap its output; the API and MCP
    returned every matching row with no way to ask for fewer, which fights
    the rule that MCP answers stay token-efficient."""

    @staticmethod
    def _pbx(rows):
        from unittest.mock import patch
        from parslbox.api import ParslBox
        patcher = patch("parslbox.api.database.get_jobs", return_value=rows)
        return patcher, ParslBox.list_jobs

    def test_positive_keeps_the_first_n(self):
        from unittest.mock import patch
        from parslbox.api import ParslBox
        rows = [{"job_id": i} for i in range(1, 11)]
        with patch("parslbox.api.database.get_jobs", return_value=rows), \
             patch.object(ParslBox, "__init__", lambda self: None):
            pbx = ParslBox()
            pbx.db_path = None
            assert [j["job_id"] for j in pbx.list_jobs(limit=3)] == [1, 2, 3]

    def test_negative_keeps_the_last_n(self):
        from unittest.mock import patch
        from parslbox.api import ParslBox
        rows = [{"job_id": i} for i in range(1, 11)]
        with patch("parslbox.api.database.get_jobs", return_value=rows), \
             patch.object(ParslBox, "__init__", lambda self: None):
            pbx = ParslBox()
            pbx.db_path = None
            assert [j["job_id"] for j in pbx.list_jobs(limit=-3)] == [8, 9, 10]

    def test_omitted_returns_everything(self):
        from unittest.mock import patch
        from parslbox.api import ParslBox
        rows = [{"job_id": i} for i in range(1, 11)]
        with patch("parslbox.api.database.get_jobs", return_value=rows), \
             patch.object(ParslBox, "__init__", lambda self: None):
            pbx = ParslBox()
            pbx.db_path = None
            assert len(pbx.list_jobs()) == 10

    def test_a_limit_past_the_end_is_not_an_error(self):
        from unittest.mock import patch
        from parslbox.api import ParslBox
        rows = [{"job_id": i} for i in range(1, 4)]
        with patch("parslbox.api.database.get_jobs", return_value=rows), \
             patch.object(ParslBox, "__init__", lambda self: None):
            pbx = ParslBox()
            pbx.db_path = None
            assert len(pbx.list_jobs(limit=99)) == 3

    def test_zero_is_refused_the_way_the_cli_refuses_it(self):
        import pytest
        from unittest.mock import patch
        from parslbox.api import ParslBox, ValidationError
        with patch.object(ParslBox, "__init__", lambda self: None):
            pbx = ParslBox()
            pbx.db_path = None
            with pytest.raises(ValidationError, match="ambiguous"):
                pbx.list_jobs(limit=0)

    def test_the_cli_refuses_zero_too(self):
        # Both layers must mean the same thing by 0, or the same argument
        # behaves differently depending on where it is typed.
        import pytest
        from parslbox.commands.helpers.ls_cmd_helpers import parse_ls_count
        with pytest.raises(ValueError, match="Use 'all'"):
            parse_ls_count("0")

    def test_the_schema_offers_it(self):
        from parslbox.mcp.schemas import ListJobsSchema
        assert ListJobsSchema.model_fields["limit"].default is None


# ---------------- Resource sizing and command-line preview ----------------- #


class TestResourceAndCommandLineTools:
    """`--req` and `--cmdline` were CLI-only. An agent could not even work the
    numbers out by hand, because GPUS_PER_NODE and CORES_PER_NODE are not
    reachable through any other tool."""

    REQ = {
        "system": "aurora-tile", "gpus_per_node": 12, "cores_per_node": 208,
        "schedulable": 3, "skipped": {"Done": 2},
        "gpu": {"jobs": 2, "total_gpus": 16, "gpus_per_node": 12,
                "system_has_gpus": True, "oversized": [],
                "simultaneous": 2, "optimal": 1},
        "cpu": {"jobs": 1, "total_occupancy": 0.5, "simultaneous": 1, "optimal": 1},
        "total": {"simultaneous": 3, "optimal": 2},
    }

    def test_both_tools_are_registered(self):
        import asyncio
        from parslbox.mcp.mcp_server import mcp
        names = {t.name for t in asyncio.run(mcp.list_tools())}
        assert {"get_resource_requirements", "get_command_lines"} <= names

    def test_sizing_reports_the_numbers_and_what_it_skipped(self):
        from unittest.mock import patch, MagicMock
        import parslbox.mcp.mcp_server as m
        with patch.object(m.pbx, "resource_requirements", return_value=self.REQ):
            out = m.get_resource_requirements(MagicMock(model_dump=lambda: {
                "system": "aurora-tile", "job_ids": None}))
        assert "12 GPUs/node" in out and "208 cores/node" in out
        assert "schedulable: 3" in out
        assert "2 Done" in out
        assert "2 nodes simultaneous / 1 optimal" in out
        assert "total: 3 nodes simultaneous / 2 optimal" in out

    def test_sizing_says_so_when_there_is_nothing_to_size(self):
        from unittest.mock import patch, MagicMock
        import parslbox.mcp.mcp_server as m
        empty = {**self.REQ, "schedulable": 0, "gpu": None, "cpu": None,
                 "total": None, "skipped": {"Done": 2}}
        with patch.object(m.pbx, "resource_requirements", return_value=empty):
            out = m.get_resource_requirements(MagicMock(model_dump=lambda: {
                "system": "aurora-tile", "job_ids": None}))
        assert "No Ready/Restart jobs" in out

    def test_a_gpu_job_on_a_cpu_system_is_called_out(self):
        from unittest.mock import patch, MagicMock
        import parslbox.mcp.mcp_server as m
        req = {**self.REQ, "system": "crux", "gpus_per_node": 0, "cpu": None,
               "total": None,
               "gpu": {"jobs": 1, "total_gpus": 4, "gpus_per_node": 0,
                       "system_has_gpus": False, "oversized": []}}
        with patch.object(m.pbx, "resource_requirements", return_value=req):
            out = m.get_resource_requirements(MagicMock(model_dump=lambda: {
                "system": "crux", "job_ids": None}))
        assert "cannot run here" in out

    def test_an_oversized_job_names_itself(self):
        from unittest.mock import patch, MagicMock
        import parslbox.mcp.mcp_server as m
        req = {**self.REQ, "gpu": {**self.REQ["gpu"],
                                   "oversized": [{"job_id": 7, "ngpus": 99}]}}
        with patch.object(m.pbx, "resource_requirements", return_value=req):
            out = m.get_resource_requirements(MagicMock(model_dump=lambda: {
                "system": "aurora-tile", "job_ids": None}))
        assert "job 7 wants 99 GPUs" in out

    def test_command_lines_render_one_line_per_job(self):
        from unittest.mock import patch, MagicMock
        import parslbox.mcp.mcp_server as m
        preview = {"system": "aurora-tile", "jobs": [
            {"job_id": 1, "app": "vasp", "resources": "1N/4G",
             "command": "mpiexec -n 4 ...", "error": None},
            {"job_id": 2, "app": "python", "resources": None,
             "command": None, "error": "bad mpi stanza"},
        ]}
        with patch.object(m.pbx, "command_lines", return_value=preview):
            out = m.get_command_lines(MagicMock(model_dump=lambda: {
                "system": "aurora-tile", "job_ids": [1, 2]}))
        assert "1 [1N/4G] mpiexec -n 4 ..." in out
        assert "2: error: bad mpi stanza" in out

    def test_an_unknown_system_comes_back_as_a_message_not_a_crash(self):
        from unittest.mock import patch, MagicMock
        import parslbox.mcp.mcp_server as m
        from parslbox.api import ValidationError
        with patch.object(m.pbx, "resource_requirements",
                          side_effect=ValidationError("Unknown system 'x'")):
            out = m.get_resource_requirements(MagicMock(model_dump=lambda: {
                "system": "x", "job_ids": None}))
        assert "Unknown system" in out

    def test_both_tools_take_ranges(self, tmp_path):
        import parslbox.mcp.mcp_server as m
        db = _db_with(20, tmp_path)
        req = {"system": "polaris", "gpus_per_node": 4, "cores_per_node": 64,
               "schedulable": 0, "skipped": {}, "gpu": None, "cpu": None,
               "total": None}
        with patch.object(m.pbx, "db_path", db), \
             patch.object(m.pbx, "resource_requirements",
                          return_value=req) as mock_req:
            m.get_resource_requirements(MagicMock(model_dump=lambda: {
                "system": "polaris", "job_ids": ["1-3"]}))
        assert mock_req.call_args.kwargs["job_ids"] == [1, 2, 3]

        with patch.object(m.pbx, "db_path", db), \
             patch.object(m.pbx, "command_lines",
                          return_value={"system": "polaris", "jobs": []}) as mock_cmd:
            m.get_command_lines(MagicMock(model_dump=lambda: {
                "system": "polaris", "job_ids": [1, "5-6"]}))
        assert mock_cmd.call_args.kwargs["job_ids"] == [1, 5, 6]

    def test_omitting_ids_means_every_job_not_none_of_them(self):
        from parslbox.mcp.schemas import ResourceRequirementsSchema
        assert ResourceRequirementsSchema(system="polaris").job_ids is None

    def test_the_system_argument_is_required(self):
        import pytest
        from pydantic import ValidationError as PydanticError
        from parslbox.mcp.schemas import ResourceRequirementsSchema
        with pytest.raises(PydanticError):
            ResourceRequirementsSchema(job_ids=[1])

    def test_command_lines_reports_a_failure_as_a_message(self):
        # get_resource_requirements had this covered and get_command_lines
        # did not, so its whole error branch was free to say anything.
        from unittest.mock import patch, MagicMock
        import parslbox.mcp.mcp_server as m
        from parslbox.api import ValidationError
        with patch.object(m.pbx, "command_lines",
                          side_effect=ValidationError("Unknown system 'x'")):
            out = m.get_command_lines(MagicMock(model_dump=lambda: {
                "system": "x", "job_ids": None}))
        assert "Unknown system" in out

    def test_command_lines_says_so_when_there_is_nothing_to_preview(self):
        from unittest.mock import patch, MagicMock
        import parslbox.mcp.mcp_server as m
        with patch.object(m.pbx, "command_lines",
                          return_value={"system": "polaris", "jobs": []}):
            out = m.get_command_lines(MagicMock(model_dump=lambda: {
                "system": "polaris", "job_ids": None}))
        assert "no jobs to preview" in out


class TestSubmitToolsAdvertiseTheEnvSetupWarning:
    """The LLM caller never sees the server source -- only the tool
    description. A warning it has no cue to expect is a warning it skips."""

    def _descriptions(self):
        """Read what the server actually advertises, not the module globals --
        the decorator keeps the text on the registered tool, and the bare
        function carries none of it."""
        import asyncio
        from parslbox.mcp import mcp_server
        tools = asyncio.run(mcp_server.mcp.list_tools())
        return {
            t.name: (t.description or "")
            for t in tools
            if t.name in ("submit_pbs_job", "submit_slurm_job")
        }

    def test_both_submit_tools_are_registered(self):
        assert set(self._descriptions()) == {"submit_pbs_job", "submit_slurm_job"}

    def test_both_submit_tools_mention_the_line(self):
        for name, desc in self._descriptions().items():
            assert "env_setup" in desc, f"{name} never mentions the line"

    def test_they_say_what_it_means(self):
        for name, desc in self._descriptions().items():
            assert "pbx_python_env_setup" in desc, name
            assert "command not found" in desc, name
