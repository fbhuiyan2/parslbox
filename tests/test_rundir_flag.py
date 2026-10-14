"""The qsub/sbatch run-directory knob is spelled `rundir` on all three layers.

`pbx run` keeps `--run-dir`: it is a different command, and the submit.sh
template in pbx_config_template.py writes that flag into the generated script.
"""
import inspect

import pytest
from typer.testing import CliRunner

from parslbox.api import ParslBox
from parslbox.main import app
from parslbox.mcp.schemas import QSubSchema, SBatchSchema

runner = CliRunner()

REQUIRED = dict(config="polaris", job_name="j", queue="debug", select="1", walltime=10)
SUBMITTERS = [("qsub", "submit_to_scheduler"), ("sbatch", "submit_to_slurm")]


def _help(command):
    result = runner.invoke(app, [command, "--help"], env={"COLUMNS": "400"})
    assert result.exit_code == 0, result.output
    return result.output


class TestTheCliFlag:
    @pytest.mark.parametrize("command", ["qsub", "sbatch"])
    def test_it_is_rundir(self, command):
        out = _help(command)
        assert "--rundir" in out
        assert "--run-dir" not in out

    @pytest.mark.parametrize("command", ["qsub", "sbatch"])
    def test_the_old_spelling_is_gone(self, command):
        result = runner.invoke(app, [command, "--run-dir", "/tmp/x"],
                               env={"COLUMNS": "400"})
        assert result.exit_code != 0
        assert "No such option" in result.output

    def test_pbx_run_keeps_run_dir(self):
        """The generated submit.sh calls `pbx run --run-dir`, so renaming this
        one too would break every template already written to disk."""
        out = _help("run")
        assert "--run-dir" in out
        assert "--rundir" not in out


class TestTheApiKeyword:
    @pytest.mark.parametrize("method,_", SUBMITTERS)
    def test_it_is_rundir(self, method, _):
        params = inspect.signature(getattr(ParslBox, method)).parameters
        assert "rundir" in params
        assert "run_dir" not in params

    @pytest.mark.parametrize("method,submitter", SUBMITTERS)
    def test_it_reaches_the_submitter_as_run_dir(self, method, submitter, monkeypatch, tmp_path):
        """Internals and Parsl's own Config(run_dir=...) keep the old name, so
        the public keyword is translated at this boundary."""
        import parslbox.api as api_mod
        seen = {}
        monkeypatch.setattr(api_mod, submitter,
                            lambda **kw: seen.update(kw) or {"success": True})
        pbx = ParslBox.__new__(ParslBox)
        pbx.config_path = None
        pbx.db_path = tmp_path / "db.sqlite"
        getattr(pbx, method)(rundir=tmp_path / "custom", **REQUIRED)
        assert seen["run_dir"] == tmp_path / "custom"


class TestTheMcpSchemas:
    @pytest.mark.parametrize("schema", [QSubSchema, SBatchSchema])
    def test_the_field_is_rundir(self, schema):
        assert "rundir" in schema.model_fields
        assert "run_dir" not in schema.model_fields
        assert schema(**REQUIRED, rundir="/tmp/x").model_dump()["rundir"] is not None

    @pytest.mark.parametrize("schema,method", [(QSubSchema, "qsub"), (SBatchSchema, "sbatch")])
    def test_the_api_accepts_every_schema_field(self, schema, method):
        """Both tools run pbx.<method>(**params.model_dump()), so a field the
        API does not take is a TypeError the moment an agent submits. Only
        LocalInitSchema had this guard, which is how a rename here could have
        gone out green."""
        accepted = set(inspect.signature(getattr(ParslBox, method)).parameters)
        assert set(schema.model_fields) <= accepted
