"""Discovery of systems and apps through the Python API.

`ParslBox()` refuses to construct without a config.yaml, so available_systems
is a staticmethod: a caller with nothing set up yet still has to be able to
ask which systems exist.
"""
import inspect

from parslbox.api import ParslBox
from parslbox.apps.app_registry import APP_FACTORY
from parslbox.system_configs.loader import CONFIG_FACTORIES

SPEC_KEYS = {"name", "scheduler", "cores_per_node", "gpus_per_node",
             "gpu_type", "mpi_backend"}


class TestAvailableSystems:
    def test_every_configured_system_is_listed(self):
        names = [s["name"] for s in ParslBox.available_systems()]
        assert names == list(CONFIG_FACTORIES)

    def test_each_entry_carries_the_sizing_hardware(self):
        for s in ParslBox.available_systems():
            assert set(s) == SPEC_KEYS, s["name"]

    def test_it_needs_no_instance(self):
        # ParslBox() raises FileNotFoundError when no config.yaml exists --
        # which is exactly when a caller most needs the list of systems.
        assert isinstance(inspect.getattr_static(ParslBox, "available_systems"),
                          staticmethod)
        assert ParslBox.available_systems()

    def test_the_specs_match_the_config_classes(self):
        by_name = {s["name"]: s for s in ParslBox.available_systems()}
        for name, cls in CONFIG_FACTORIES.items():
            assert by_name[name]["scheduler"] == cls.SCHEDULER
            assert by_name[name]["gpus_per_node"] == cls.GPUS_PER_NODE
            assert by_name[name]["cores_per_node"] == cls.CORES_PER_NODE

    def test_cpu_only_systems_report_no_gpu_type(self):
        cpu_only = [s for s in ParslBox.available_systems() if s["gpus_per_node"] == 0]
        assert cpu_only, "fixture assumption: at least one CPU-only system"
        assert all(s["gpu_type"] is None for s in cpu_only)


class TestAvailableApps:
    def test_builtins_are_the_registry(self, pbx):
        assert pbx.available_apps()["builtin"] == list(APP_FACTORY)

    def test_custom_is_empty_without_any_declared(self, pbx):
        assert pbx.available_apps()["custom"] == []

    def test_builtin_and_custom_do_not_overlap(self, pbx):
        apps = pbx.available_apps()
        assert not set(apps["builtin"]) & set(apps["custom"])
