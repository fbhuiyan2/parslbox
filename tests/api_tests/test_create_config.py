"""ParslBox.create_config: `pbx config` without the prompts.

The CLI protects itself by only ever offering real choices -- the menus are
built from the registries and selection is by index. An API caller passes
strings, so this is the first entry point where a typo'd system or app name
can reach ConfigGenerator, which validates nothing and would happily write
the bogus name into config.yaml.
"""
import inspect

import pytest
import yaml

from parslbox.api import ParslBox, ValidationError
from parslbox.database import database


@pytest.fixture(autouse=True)
def own_db(tmp_path, monkeypatch):
    """create_config initializes a database; keep it out of the real home."""
    monkeypatch.setenv("PBX_DB_PATH", str(tmp_path / "dbdir"))
    return tmp_path / "dbdir" / "job_database_pbx.db"


@pytest.fixture
def target(tmp_path):
    return tmp_path / "proj"


def _written(result):
    from pathlib import Path
    return yaml.safe_load(Path(result["config_path"]).read_text())


class TestItWritesAConfig:
    def test_the_file_lands_where_asked(self, target):
        result = ParslBox.create_config(target, ["polaris"])
        assert result["config_path"] == str(target / "config.yaml")
        assert (target / "config.yaml").exists()

    def test_the_result_reports_the_selection(self, target):
        result = ParslBox.create_config(target, ["polaris", "crux"], apps=["vasp"])
        assert result["systems_selected"] == ["polaris", "crux"]
        assert result["apps_selected"] == ["vasp"]
        assert result["backed_up"] is None

    def test_it_is_valid_yaml_naming_the_choices(self, target):
        result = ParslBox.create_config(target, ["polaris"], apps=["vasp"])
        data = _written(result)
        assert "vasp" in data
        assert "polaris" in data["vasp"]

    def test_a_non_default_location_comes_with_its_env_var(self, target):
        result = ParslBox.create_config(target, ["polaris"])
        assert result["config_env_var"] == {"PBX_CONFIG_PATH": str(target)}

    def test_the_config_it_writes_is_usable(self, target, tmp_path):
        ParslBox.create_config(target, ["polaris"])
        # The point of the whole exercise: ParslBox() stops raising.
        pbx = ParslBox(db_path=tmp_path / "x.db", config_path=target / "config.yaml")
        assert "polaris" in [s["name"] for s in pbx.available_systems()]

    def test_it_needs_no_instance(self, target):
        # ParslBox() raises FileNotFoundError without a config -- which is
        # exactly when a caller needs to create one.
        assert isinstance(
            inspect.getattr_static(ParslBox, "create_config"), staticmethod
        )


class TestAppsDefault:
    def test_none_means_python(self, target):
        assert ParslBox.create_config(target, ["polaris"])["apps_selected"] == ["python"]

    def test_empty_means_python_too(self, target):
        result = ParslBox.create_config(target, ["polaris"], apps=[])
        assert result["apps_selected"] == ["python"]


class TestNamesAreValidated:
    def test_unknown_system_is_refused(self, target):
        with pytest.raises(ValidationError, match="Invalid system"):
            ParslBox.create_config(target, ["polarix"])

    def test_the_message_lists_the_real_systems(self, target):
        with pytest.raises(ValidationError, match="polaris"):
            ParslBox.create_config(target, ["polarix"])

    def test_unknown_app_is_refused(self, target):
        """ConfigGenerator(['polaris'], ['lamps-kk']).generate() succeeds and
        writes the typo straight into config.yaml; the failure only surfaces
        later at `pbx add` as "Unknown application". Caught here instead."""
        with pytest.raises(ValidationError, match="Invalid app"):
            ParslBox.create_config(target, ["polaris"], apps=["lamps-kk"])

    def test_a_missing_path_is_refused(self, target):
        """str(None) is the directory name "None"; without a guard the config
        is written into ./None/ instead of failing."""
        with pytest.raises(ValidationError, match="'path' is required"):
            ParslBox.create_config(None, ["polaris"])

    def test_a_missing_path_creates_nothing(self, target, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValidationError):
            ParslBox.create_config(None, ["polaris"])
        assert not (tmp_path / "None").exists()

    def test_missing_systems_is_a_validation_error(self, target):
        """config_setup raises a bare RuntimeError here; the API promises a
        typed exception, so it must not leak through."""
        with pytest.raises(ValidationError, match="'systems' is required"):
            ParslBox.create_config(target, None)

    def test_an_empty_system_list_is_refused(self, target):
        with pytest.raises(ValidationError, match="At least one system"):
            ParslBox.create_config(target, [])

    def test_nothing_is_written_when_a_name_is_bad(self, target):
        with pytest.raises(ValidationError):
            ParslBox.create_config(target, ["polaris"], apps=["lamps-kk"])
        assert not (target / "config.yaml").exists()


class TestAnExistingConfigIsNotClobbered:
    def test_it_refuses_by_default(self, target):
        ParslBox.create_config(target, ["polaris"])
        with pytest.raises(ValidationError, match="already exists"):
            ParslBox.create_config(target, ["crux"])

    def test_the_original_survives_the_refusal(self, target):
        first = ParslBox.create_config(target, ["polaris"])
        before = (target / "config.yaml").read_text()
        with pytest.raises(ValidationError):
            ParslBox.create_config(target, ["crux"])
        assert (target / "config.yaml").read_text() == before

    def test_force_renames_rather_than_deletes(self, target):
        ParslBox.create_config(target, ["polaris"])
        original = (target / "config.yaml").read_text()

        result = ParslBox.create_config(target, ["crux"], force=True)

        assert result["backed_up"] is not None
        backup = target / result["backed_up"]
        assert backup.read_text() == original

    def test_force_writes_the_new_selection(self, target):
        ParslBox.create_config(target, ["polaris"])
        result = ParslBox.create_config(target, ["crux"], apps=["vasp"], force=True)
        assert "crux" in _written(result)["vasp"]

    def test_a_bad_name_does_not_move_the_old_config_aside(self, target):
        """force renames before the config is generated, so validation has to
        happen first -- otherwise a typo leaves the caller with no config."""
        ParslBox.create_config(target, ["polaris"])
        with pytest.raises(ValidationError):
            ParslBox.create_config(target, ["crux"], apps=["lamps-kk"], force=True)
        assert (target / "config.yaml").exists()
        assert not list(target.glob("*.backup.*"))


class TestTheDatabase:
    def test_it_gets_created(self, target, own_db):
        result = ParslBox.create_config(target, ["polaris"])
        assert result["db_path"] == str(own_db)
        assert own_db.exists()

    def test_it_lives_at_pbx_db_path_not_beside_the_config(self, target, own_db):
        ParslBox.create_config(target, ["polaris"])
        assert not (target / "job_database_pbx.db").exists()
        assert own_db.exists()

    def test_an_existing_database_is_not_replaced(self, target, own_db):
        """Setting up a second config must never cost the user their jobs."""
        own_db.parent.mkdir(parents=True, exist_ok=True)
        database.initialize_database(own_db)
        job_id = database.add_job(own_db, "/some/path", "python", 1, 0, 1.0, "t1")

        ParslBox.create_config(target, ["polaris"])

        assert database.get_jobs_by_ids(own_db, [job_id])

    def test_force_leaves_the_database_alone_too(self, target, own_db):
        ParslBox.create_config(target, ["polaris"])
        job_id = database.add_job(own_db, "/some/path", "python", 1, 0, 1.0, "t1")

        ParslBox.create_config(target, ["crux"], force=True)

        assert database.get_jobs_by_ids(own_db, [job_id])
