"""Relative db/config paths must be rejected at the API surface too.

The env-var check in path_utils does not cover callers that pass db_path /
config_path directly, and both values are exported into submit.sh.
"""

import pytest
from pathlib import Path

from parslbox.api import ParslBox
from parslbox.utils.path_utils import PbxPathError
from parslbox.commands.helpers.submit_helpers import submit_job, ValidationError


class TestParslBoxInit:
    def test_relative_db_path_rejected(self):
        with pytest.raises(PbxPathError, match="db_path must be an absolute path"):
            ParslBox(db_path=Path("mydb.db"))

    def test_relative_config_path_rejected(self, temp_db):
        with pytest.raises(PbxPathError, match="config_path must be an absolute path"):
            ParslBox(db_path=temp_db, config_path=Path("config.yaml"))

    def test_relative_env_db_path_rejected(self, monkeypatch, temp_db):
        monkeypatch.setenv("PBX_DB_PATH", "./somedb.db")
        with pytest.raises(PbxPathError, match="PBX_DB_PATH"):
            ParslBox(db_path=temp_db)

    def test_relative_env_config_path_rejected(self, monkeypatch, temp_db):
        monkeypatch.setenv("PBX_CONFIG_PATH", "cfg.yaml")
        with pytest.raises(PbxPathError, match="PBX_CONFIG_PATH"):
            ParslBox(db_path=temp_db)

    def test_tilde_db_path_expanded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        (tmp_path / "config.yaml").write_text("{}\n")
        pbx = ParslBox(db_path="~/mydb.db", config_path=tmp_path / "config.yaml")
        assert pbx.db_path == tmp_path / "mydb.db"


class TestSubmitJob:
    _kwargs = dict(
        config_name="aurora-tile", job_name="t", queue="debug",
        select="1", walltime=10, validate_runnable=False,
    )

    def test_relative_db_path_rejected(self, tmp_path):
        with pytest.raises(ValidationError, match="db_path must be an absolute path"):
            submit_job(db_path=Path("mydb.db"),
                       config_path=tmp_path / "config.yaml", **self._kwargs)

    def test_relative_config_path_rejected(self, temp_db):
        with pytest.raises(ValidationError, match="config_path must be an absolute path"):
            submit_job(db_path=temp_db, config_path=Path("config.yaml"),
                       **self._kwargs)
