"""`pbx config` prompts; every other caller of config_setup must not.

Which of the two happens used to be held in a module-level `_is_interactive`
flag that the CLI callback set to True and nothing ever reset. The CLI runs
one-shot processes, so that was harmless in a shell -- but CliRunner invokes
the callback in-process, so a single CLI test left the flag True for the rest
of the pytest session. A later non-interactive call then took the prompting
branch: instead of raising on an existing config it printed a menu and blocked
on stdin, failing somewhere unrelated and only when the tests ran in that
order. The mode is a parameter now; these tests pin that.
"""
import pytest
from typer.testing import CliRunner

from parslbox.commands import config as config_cmd
from parslbox.commands.config import app as config_app, config_setup

runner = CliRunner()


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """A config/DB home of our own, so nothing touches the real ~/.parslbox."""
    monkeypatch.setenv("PBX_CONFIG_PATH", str(tmp_path))
    monkeypatch.setenv("PBX_DB_PATH", str(tmp_path))
    return tmp_path


class TestTheModeIsNotProcessWide:
    def test_no_module_level_flag_survives(self):
        # The bug was the existence of the shared flag, not any one read of it.
        assert not hasattr(config_cmd, "_is_interactive")

    def test_nothing_sets_the_flag_back(self):
        """main.py's no-config branch used to assign it on the module object,
        which would re-create the attribute the moment that branch ran."""
        import pathlib
        root = pathlib.Path(config_cmd.__file__).parent.parent
        offenders = [
            f for f in root.rglob("*.py")
            if "_is_interactive" in f.read_text()
        ]
        assert offenders == []

    def test_a_cli_run_does_not_make_the_next_api_call_prompt(self, isolated):
        """The exact ordering that used to break: CLI first, API second."""
        result = runner.invoke(config_app, [str(isolated)], input="1\n\n")
        assert result.exit_code == 0, result.output
        assert (isolated / "config.yaml").exists()

        # Previously this took the interactive branch and blocked on a prompt.
        with pytest.raises(FileExistsError):
            config_setup(path=str(isolated), systems=["polaris"], apps=["python"])

    def test_api_then_cli_then_api_all_behave(self, isolated):
        config_setup(path=str(isolated), systems=["polaris"], apps=["python"])
        (isolated / "config.yaml").unlink()

        result = runner.invoke(config_app, [str(isolated)], input="1\n\n")
        assert result.exit_code == 0, result.output
        (isolated / "config.yaml").unlink()

        out = config_setup(path=str(isolated), systems=["crux"], apps=["vasp"])
        assert out["systems_selected"] == ["crux"]


class TestNonInteractiveCallers:
    def test_missing_systems_raises_instead_of_prompting(self, isolated):
        with pytest.raises(RuntimeError, match="systems"):
            config_setup(path=str(isolated), apps=["python"])

    def test_missing_apps_defaults_rather_than_prompting(self, isolated):
        out = config_setup(path=str(isolated), systems=["polaris"])
        assert out["apps_selected"] == ["python"]

    def test_an_unknown_app_is_rejected(self, isolated):
        with pytest.raises(ValueError, match="nosuchapp"):
            config_setup(path=str(isolated), systems=["polaris"], apps=["nosuchapp"])

    def test_an_unknown_system_is_rejected(self, isolated):
        with pytest.raises(ValueError, match="nosuchsystem"):
            config_setup(path=str(isolated), systems=["nosuchsystem"], apps=["python"])

    def test_an_existing_config_raises_rather_than_asking(self, isolated):
        config_setup(path=str(isolated), systems=["polaris"], apps=["python"])
        with pytest.raises(FileExistsError):
            config_setup(path=str(isolated), systems=["polaris"], apps=["python"])


class TestTheCliStillPrompts:
    """Step 0 touched every helper in config.py; the CLI must be unchanged."""

    def test_it_asks_for_systems_and_apps(self, isolated):
        result = runner.invoke(config_app, [str(isolated)], input="1\n\n")
        assert result.exit_code == 0, result.output
        assert "Available systems:" in result.output
        assert "Available applications:" in result.output

    def test_selections_reach_the_generated_config(self, isolated):
        result = runner.invoke(config_app, [str(isolated)], input="all\n1\n")
        assert result.exit_code == 0, result.output
        text = (isolated / "config.yaml").read_text()
        assert "polaris" in text and "lammps-kk" in text

    def test_an_empty_app_selection_defaults_to_python(self, isolated):
        result = runner.invoke(config_app, [str(isolated)], input="1\n\n")
        assert result.exit_code == 0, result.output
        assert "Defaulting to 'python' app" in result.output

    def test_an_existing_config_offers_rename_or_delete(self, isolated):
        runner.invoke(config_app, [str(isolated)], input="1\n\n")
        result = runner.invoke(config_app, [str(isolated)], input="2\n1\n\n")
        assert result.exit_code == 0, result.output
        assert "Config file already exists" in result.output
        assert "Keep old config (rename it)" in result.output
        assert "Delete old config" in result.output
