"""Absolute-path enforcement for PBX_DB_PATH / PBX_CONFIG_PATH.

A relative value resolves against the current working directory, so it would
create a fresh database in every directory pbx runs from, and — once exported
into submit.sh — point `pbx run` at an empty database inside the allocation's
working directory.
"""

import pytest
from pathlib import Path

from parslbox.utils import path_utils
from parslbox.utils.path_utils import PbxPathError


class TestGetDbPath:
    def test_relative_file_rejected(self, monkeypatch):
        monkeypatch.setenv("PBX_DB_PATH", "./somedb.db")
        with pytest.raises(PbxPathError, match="PBX_DB_PATH must be an absolute path"):
            path_utils.get_db_path()

    def test_relative_dir_rejected(self, monkeypatch):
        monkeypatch.setenv("PBX_DB_PATH", ".")
        with pytest.raises(PbxPathError):
            path_utils.get_db_path()

    def test_error_message_suggests_pwd_form(self, monkeypatch):
        monkeypatch.setenv("PBX_DB_PATH", "./somedb.db")
        with pytest.raises(PbxPathError) as exc:
            path_utils.get_db_path()
        assert 'export PBX_DB_PATH="$PWD/somedb.db"' in str(exc.value)

    def test_non_strict_keeps_relative(self, monkeypatch):
        # Module import must never raise; validation happens at the CLI/API entry.
        monkeypatch.setenv("PBX_DB_PATH", "./somedb.db")
        assert path_utils.get_db_path(strict=False) == Path("somedb.db")

    def test_absolute_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PBX_DB_PATH", str(tmp_path / "mydb.db"))
        assert path_utils.get_db_path() == tmp_path / "mydb.db"

    def test_absolute_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PBX_DB_PATH", str(tmp_path))
        assert path_utils.get_db_path() == tmp_path / "job_database_pbx.db"

    def test_tilde_expanded_not_rejected(self, monkeypatch):
        monkeypatch.setenv("PBX_DB_PATH", "~/dbs/mydb.db")
        assert path_utils.get_db_path() == Path.home() / "dbs" / "mydb.db"

    def test_unset_uses_home_default(self, monkeypatch):
        monkeypatch.delenv("PBX_DB_PATH", raising=False)
        assert path_utils.get_db_path() == Path.home() / ".parslbox" / "job_database_pbx.db"


class TestGetConfigPath:
    def test_relative_file_rejected(self, monkeypatch):
        monkeypatch.setenv("PBX_CONFIG_PATH", "cfg.yaml")
        with pytest.raises(PbxPathError, match="PBX_CONFIG_PATH must be an absolute path"):
            path_utils.get_config_path()

    def test_relative_dir_rejected(self, monkeypatch):
        monkeypatch.setenv("PBX_CONFIG_PATH", "conf")
        with pytest.raises(PbxPathError):
            path_utils.get_config_path()

    def test_non_strict_keeps_relative(self, monkeypatch):
        monkeypatch.setenv("PBX_CONFIG_PATH", "cfg.yaml")
        assert path_utils.get_config_path(strict=False) == Path("cfg.yaml")

    def test_absolute_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PBX_CONFIG_PATH", str(tmp_path / "my.yml"))
        assert path_utils.get_config_path() == tmp_path / "my.yml"

    def test_absolute_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PBX_CONFIG_PATH", str(tmp_path))
        assert path_utils.get_config_path() == tmp_path / "config.yaml"

    def test_tilde_expanded_not_rejected(self, monkeypatch):
        monkeypatch.setenv("PBX_CONFIG_PATH", "~/conf")
        assert path_utils.get_config_path() == Path.home() / "conf" / "config.yaml"


class TestValidateEnvPaths:
    def test_passes_when_unset(self, monkeypatch):
        monkeypatch.delenv("PBX_DB_PATH", raising=False)
        monkeypatch.delenv("PBX_CONFIG_PATH", raising=False)
        path_utils.validate_env_paths()

    def test_raises_on_relative_db(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PBX_DB_PATH", "somedb.db")
        monkeypatch.setenv("PBX_CONFIG_PATH", str(tmp_path))
        with pytest.raises(PbxPathError, match="PBX_DB_PATH"):
            path_utils.validate_env_paths()

    def test_raises_on_relative_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PBX_DB_PATH", str(tmp_path / "db.db"))
        monkeypatch.setenv("PBX_CONFIG_PATH", "cfg.yaml")
        with pytest.raises(PbxPathError, match="PBX_CONFIG_PATH"):
            path_utils.validate_env_paths()


class TestRequireAbsolute:
    def test_returns_expanded_path(self):
        assert path_utils.require_absolute("db_path", "~/x.db") == Path.home() / "x.db"

    def test_rejects_relative(self):
        with pytest.raises(PbxPathError, match="db_path must be an absolute path"):
            path_utils.require_absolute("db_path", Path("x.db"))

    def test_no_export_hint_for_non_env_label(self):
        with pytest.raises(PbxPathError) as exc:
            path_utils.require_absolute("db_path", "x.db")
        assert "export" not in str(exc.value)


class TestCliCallback:
    """`pbx <cmd>` must fail before the database is created in the cwd."""

    def _run(self, monkeypatch, tmp_path, var, value):
        from typer.testing import CliRunner
        from parslbox.main import app

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(var, value)
        return CliRunner().invoke(app, ["ls"], env={"COLUMNS": "400"})

    def test_relative_db_path_aborts(self, monkeypatch, tmp_path):
        result = self._run(monkeypatch, tmp_path, "PBX_DB_PATH", "./somedb.db")
        assert result.exit_code == 1
        assert "PBX_DB_PATH must be an absolute path" in result.output
        assert list(tmp_path.iterdir()) == []

    def test_relative_config_path_aborts(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PBX_DB_PATH", str(tmp_path / "db.db"))
        result = self._run(monkeypatch, tmp_path, "PBX_CONFIG_PATH", "cfg.yaml")
        assert result.exit_code == 1
        assert "PBX_CONFIG_PATH must be an absolute path" in result.output
        assert list(tmp_path.iterdir()) == []
