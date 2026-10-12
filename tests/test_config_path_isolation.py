"""An explicit config_path must govern custom-app resolution.

`ParslBox(config_path=X)` used to set self.config_path correctly while the app
registry went on reading the process-wide PBX_CONFIG_FILE, so a custom app
declared in X came back as "Unknown application". These tests pin the fixed
behaviour, including two instances on different configs in one process -- the
shape the MCP server's use_database relies on.
"""
import textwrap

import pytest
import yaml

from parslbox.api import ParslBox
from parslbox.apps import app_registry


APP_SRC = textwrap.dedent('''
    from parslbox.apps.appbase import AppBase

    class {cls}(AppBase):
        INPUT_REQUIRED = False
        DFLT_INPUT = None
        MARKER = "{marker}"
        def get_command_template(self, **kwargs):
            return "echo {marker}"
''')


def _project(root, name, app_name, cls, marker, module_ref=None):
    """Write a project dir holding an app module and a config declaring it."""
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}_app.py").write_text(APP_SRC.format(cls=cls, marker=marker))
    config = {
        "custom_apps": {
            app_name: {
                "module": module_ref or str(root / f"{name}_app.py"),
                "class": cls,
            }
        },
        app_name: {"polaris": {"executable_path": f"/{name}/bin/exe"}},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


@pytest.fixture(autouse=True)
def clear_registry_cache():
    """The registry caches per config path; keep tests independent."""
    app_registry._CUSTOM_APPS = {}
    yield
    app_registry._CUSTOM_APPS = {}


@pytest.fixture
def ambient_config(tmp_path, monkeypatch):
    """A process-wide config that declares no custom apps at all."""
    path = tmp_path / "ambient" / "config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({"lammps-kk": {"polaris": {"executable_path": "/g/lmp"}}}))
    monkeypatch.setattr("parslbox.utils.path_utils.PBX_CONFIG_FILE", path)
    return path


class TestExplicitConfigPathWins:
    def test_custom_app_from_passed_config_is_registered(self, tmp_path, ambient_config):
        cfg = _project(tmp_path / "projB", "b", "my-app", "MyApp", "B")
        ParslBox(db_path=tmp_path / "b.db", config_path=cfg)
        assert app_registry.is_app_registered("my-app", cfg)

    def test_ambient_config_still_does_not_see_it(self, tmp_path, ambient_config):
        cfg = _project(tmp_path / "projB", "b", "my-app", "MyApp", "B")
        ParslBox(db_path=tmp_path / "b.db", config_path=cfg)
        assert not app_registry.is_app_registered("my-app")

    def test_add_jobs_accepts_the_apps_its_config_declares(self, tmp_path, ambient_config):
        """The original repro: this raised Unknown application 'my-app'."""
        cfg = _project(tmp_path / "projB", "b", "my-app", "MyApp", "B")
        job = tmp_path / "projB" / "job1"
        job.mkdir()

        pbx = ParslBox(db_path=tmp_path / "b.db", config_path=cfg)
        ids, failures, _ = pbx.add_jobs(paths=[str(job)], app="my-app", config="polaris")
        assert failures == []
        assert len(ids) == 1

    def test_registered_apps_lists_the_passed_configs_app(self, tmp_path, ambient_config):
        cfg = _project(tmp_path / "projB", "b", "my-app", "MyApp", "B")
        assert "my-app" in app_registry.get_registered_apps(cfg)
        assert "my-app" not in app_registry.get_registered_apps()


class TestTwoConfigsOneProcess:
    """Two live instances on different configs must not contaminate each other.

    This is the pytest-suite shape, and the shape use_database produces when a
    session moves between projects.
    """

    def test_each_config_resolves_its_own_class(self, tmp_path, ambient_config):
        cfg_a = _project(tmp_path / "projA", "a", "shared-name", "AppA", "A")
        cfg_b = _project(tmp_path / "projB", "b", "shared-name", "AppB", "B")

        cls_a = app_registry.get_app_class("shared-name", cfg_a)
        cls_b = app_registry.get_app_class("shared-name", cfg_b)

        assert cls_a.MARKER == "A"
        assert cls_b.MARKER == "B"

    def test_order_does_not_decide_the_winner(self, tmp_path, ambient_config):
        """First-constructed used to win for the whole process."""
        cfg_a = _project(tmp_path / "projA", "a", "shared-name", "AppA", "A")
        cfg_b = _project(tmp_path / "projB", "b", "shared-name", "AppB", "B")

        app_registry.get_app_class("shared-name", cfg_b)      # warm B first
        assert app_registry.get_app_class("shared-name", cfg_a).MARKER == "A"

    def test_app_declared_in_only_one_config(self, tmp_path, ambient_config):
        cfg_a = _project(tmp_path / "projA", "a", "only-a", "AppA", "A")
        cfg_b = _project(tmp_path / "projB", "b", "only-b", "AppB", "B")

        assert app_registry.is_app_registered("only-a", cfg_a)
        assert not app_registry.is_app_registered("only-a", cfg_b)
        assert app_registry.is_app_registered("only-b", cfg_b)


class TestRelativeModulePaths:
    def test_relative_module_resolves_against_its_own_config(self, tmp_path, ambient_config):
        """A relative `module:` is resolved next to the config declaring it,
        not next to whatever config happens to be process-wide."""
        cfg = _project(tmp_path / "projB", "b", "rel-app", "RelApp", "REL",
                       module_ref="b_app.py")
        assert app_registry.get_app_class("rel-app", cfg).MARKER == "REL"


class TestDefaultsUnchanged:
    def test_no_config_path_uses_the_process_global(self, tmp_path, ambient_config):
        """CLI and pbx run pass nothing and must keep the old behaviour."""
        assert app_registry.get_registered_apps() == [
            "lammps-kk", "vasp", "python", "julia", "orca",
        ]

    def test_builtins_resolve_without_touching_config(self, tmp_path):
        assert app_registry.get_app_class("lammps-kk").__name__ == "LammpsKokkosApp"


# ---- pbx_config_utils: the shared helper that kept reading the global ---- #


class TestConfigUtilsHonourAnExplicitPath:
    """`70ff47d` threaded config_path through the app registry but stopped
    there, leaving pbx_config_utils reading the process-wide default. That
    module is a shared helper -- schedule_helpers calls into it -- so a
    helper calling a helper had no way to pass the caller's project down.
    Only CLI paths reached it, where a fresh process makes the default
    correct, which is why it went unnoticed.
    """

    @staticmethod
    def _cfg(tmp_path, name, value):
        p = tmp_path / name
        p.write_text(yaml.safe_dump({"lammps-kk": {"polaris": {"marker": value}}}))
        return p

    def test_load_full_config_reads_the_path_it_is_given(self, tmp_path):
        from parslbox.utils.pbx_config_utils import load_full_config
        cfg = self._cfg(tmp_path, "a.yaml", "A")
        assert load_full_config(cfg)["lammps-kk"]["polaris"]["marker"] == "A"

    def test_two_configs_in_one_process_do_not_bleed(self, tmp_path):
        # The shape use_database relies on: one process, two projects.
        from parslbox.utils.pbx_config_utils import load_full_config
        a = self._cfg(tmp_path, "a.yaml", "A")
        b = self._cfg(tmp_path, "b.yaml", "B")
        assert load_full_config(a)["lammps-kk"]["polaris"]["marker"] == "A"
        assert load_full_config(b)["lammps-kk"]["polaris"]["marker"] == "B"
        assert load_full_config(a)["lammps-kk"]["polaris"]["marker"] == "A"

    def test_an_explicit_path_beats_the_global(self, tmp_path, monkeypatch):
        from parslbox.utils import path_utils, pbx_config_utils
        theirs = self._cfg(tmp_path, "global.yaml", "GLOBAL")
        mine = self._cfg(tmp_path, "mine.yaml", "MINE")
        monkeypatch.setattr(path_utils, "PBX_CONFIG_FILE", theirs)
        got = pbx_config_utils.load_full_config(mine)
        assert got["lammps-kk"]["polaris"]["marker"] == "MINE"

    def test_omitting_it_still_uses_the_global(self, tmp_path, monkeypatch):
        from parslbox.utils import path_utils, pbx_config_utils
        theirs = self._cfg(tmp_path, "global.yaml", "GLOBAL")
        monkeypatch.setattr(path_utils, "PBX_CONFIG_FILE", theirs)
        got = pbx_config_utils.load_full_config()
        assert got["lammps-kk"]["polaris"]["marker"] == "GLOBAL"

    def test_load_app_config_takes_one_too(self, tmp_path):
        from parslbox.utils.pbx_config_utils import load_app_config
        cfg = self._cfg(tmp_path, "a.yaml", "A")
        assert load_app_config("lammps-kk", "polaris", cfg) == {"marker": "A"}
        assert load_app_config("lammps-kk", "crux", cfg) == {}

    def test_is_app_configured_takes_one_too(self, tmp_path):
        from parslbox.utils.pbx_config_utils import is_app_configured
        cfg = self._cfg(tmp_path, "a.yaml", "A")
        assert is_app_configured("lammps-kk", "polaris", cfg) is True
        assert is_app_configured("vasp", "polaris", cfg) is False

    def test_a_missing_file_still_raises_and_is_still_swallowed(self, tmp_path):
        from parslbox.utils.pbx_config_utils import load_full_config, is_app_configured
        gone = tmp_path / "nope.yaml"
        with pytest.raises(FileNotFoundError):
            load_full_config(gone)
        # is_app_configured has always answered False rather than raising.
        assert is_app_configured("lammps-kk", "polaris", gone) is False

    def test_the_global_is_read_in_exactly_one_place(self):
        # The fallback belongs in one helper. Any other reader in this module
        # is a function that cannot be pointed at another project.
        import inspect
        from parslbox.utils import pbx_config_utils
        src = inspect.getsource(pbx_config_utils)
        assert src.count("path_utils.PBX_CONFIG_FILE") == 1

    def test_the_dead_entry_points_are_gone(self):
        from parslbox.utils import pbx_config_utils
        assert not hasattr(pbx_config_utils, "get_configured_systems_for_app")
        assert not hasattr(pbx_config_utils, "initialize_config_file")
