"""
Unit tests for MPI config merging and loading logic.

Tests that app-level mpi: config properly overrides/merges with
system-level mpi: config in merge_mpi_config_dicts and load_mpi_config.
"""

import pytest
from unittest.mock import Mock
from parslbox.resource_manager.mpi_config import (
    merge_mpi_config_dicts,
    load_mpi_config,
    MPIBackend,
    MPIConfig,
)


class TestMergeMpiConfigDicts:
    """Tests for merge_mpi_config_dicts."""

    def test_override_scalar_replaces_base(self):
        """App-level scalar values override system-level values."""
        base = {"backend": "openmpi", "use_gpu_wrapper": True, "cpu_bind_method": "depth"}
        override = {"use_gpu_wrapper": False}
        result = merge_mpi_config_dicts(base, override)
        assert result["use_gpu_wrapper"] is False
        assert result["backend"] == "openmpi"
        assert result["cpu_bind_method"] == "depth"

    def test_override_list_replaces_base(self):
        """App-level lists completely replace system-level lists (no concatenation)."""
        base = {"disable": ["-H", "--map-by"], "add": ["--verbose"]}
        override = {"disable": ["rankfile"]}
        result = merge_mpi_config_dicts(base, override)
        assert result["disable"] == ["rankfile"]
        assert result["add"] == ["--verbose"]

    def test_base_preserved_when_no_override(self):
        """System-level keys are preserved when app doesn't override them."""
        base = {"backend": "mpich", "use_gpu_wrapper": True, "cpu_bind_method": "depth"}
        override = {"add": ["--mca btl ^openib"]}
        result = merge_mpi_config_dicts(base, override)
        assert result["backend"] == "mpich"
        assert result["use_gpu_wrapper"] is True
        assert result["cpu_bind_method"] == "depth"
        assert result["add"] == ["--mca btl ^openib"]

    def test_empty_base_returns_override(self):
        """Empty base dict returns a copy of override."""
        override = {"backend": "mpich", "use_gpu_wrapper": True}
        result = merge_mpi_config_dicts({}, override)
        assert result == override
        assert result is not override  # should be a copy

    def test_empty_override_returns_base(self):
        """Empty override dict returns a copy of base."""
        base = {"backend": "openmpi", "add": ["--verbose"]}
        result = merge_mpi_config_dicts(base, {})
        assert result == base
        assert result is not base  # should be a copy

    def test_none_base_returns_override(self):
        """None base returns a copy of override."""
        override = {"backend": "srun"}
        result = merge_mpi_config_dicts(None, override)
        assert result == override

    def test_none_override_returns_base(self):
        """None override returns a copy of base."""
        base = {"backend": "openmpi"}
        result = merge_mpi_config_dicts(base, None)
        assert result == base

    def test_both_none_returns_empty(self):
        """Both None returns empty dict."""
        result = merge_mpi_config_dicts(None, None)
        assert result == {}

    def test_explicit_false_overrides_true(self):
        """Explicit False in override wins over True in base."""
        base = {"use_gpu_wrapper": True, "use_hostlist": True}
        override = {"use_gpu_wrapper": False, "use_hostlist": False}
        result = merge_mpi_config_dicts(base, override)
        assert result["use_gpu_wrapper"] is False
        assert result["use_hostlist"] is False

    def test_override_adds_new_keys(self):
        """Override can introduce keys not present in base."""
        base = {"backend": "openmpi"}
        override = {"mpi_cmd": "/opt/mpirun", "use_short_hostnames": True}
        result = merge_mpi_config_dicts(base, override)
        assert result["backend"] == "openmpi"
        assert result["mpi_cmd"] == "/opt/mpirun"
        assert result["use_short_hostnames"] is True

    def test_override_backend_replaces_base(self):
        """App can override the backend itself."""
        base = {"backend": "openmpi", "use_gpu_wrapper": True}
        override = {"backend": "mpich"}
        result = merge_mpi_config_dicts(base, override)
        assert result["backend"] == "mpich"
        assert result["use_gpu_wrapper"] is True


class TestLoadMpiConfig:
    """Tests for load_mpi_config end-to-end merging."""

    def _make_system_config(self, mpi_backend=None, mpi_cmd=None):
        """Create a mock system config class."""
        mock = Mock()
        if mpi_backend is not None:
            mock.MPI_BACKEND = mpi_backend
        else:
            del mock.MPI_BACKEND
        if mpi_cmd is not None:
            mock.MPI_CMD_TO_USE = mpi_cmd
        else:
            del mock.MPI_CMD_TO_USE
        return mock

    def test_backend_from_system_config_class(self):
        """Backend defaults to system_config_class.MPI_BACKEND when no YAML override."""
        system_config = self._make_system_config(mpi_backend="mpich")
        yaml_config = {}
        result = load_mpi_config("polaris", "lammps-kk", system_config, yaml_config)
        assert result.backend == MPIBackend.MPICH

    def test_backend_fallback_from_legacy_mpiexec(self):
        """Legacy MPI_CMD_TO_USE='mpiexec' infers mpich backend."""
        system_config = self._make_system_config(mpi_cmd="mpiexec")
        yaml_config = {}
        result = load_mpi_config("polaris", "lammps-kk", system_config, yaml_config)
        assert result.backend == MPIBackend.MPICH

    def test_backend_fallback_from_legacy_srun(self):
        """Legacy MPI_CMD_TO_USE='srun' infers srun backend."""
        system_config = self._make_system_config(mpi_cmd="srun")
        yaml_config = {}
        result = load_mpi_config("polaris", "lammps-kk", system_config, yaml_config)
        assert result.backend == MPIBackend.SRUN

    def test_backend_fallback_default_openmpi(self):
        """No MPI_BACKEND or MPI_CMD_TO_USE defaults to openmpi."""
        system_config = self._make_system_config()
        yaml_config = {}
        result = load_mpi_config("polaris", "lammps-kk", system_config, yaml_config)
        assert result.backend == MPIBackend.OPENMPI

    def test_system_yaml_overrides_class_defaults(self):
        """System-level YAML mpi: overrides class defaults."""
        system_config = self._make_system_config(mpi_backend="openmpi")
        yaml_config = {
            "polaris": {
                "mpi": {
                    "use_gpu_wrapper": True,
                    "cpu_bind_method": "depth",
                }
            }
        }
        result = load_mpi_config("polaris", "lammps-kk", system_config, yaml_config)
        assert result.backend == MPIBackend.OPENMPI
        assert result.use_gpu_wrapper is True
        assert result.cpu_bind_method == "depth"

    def test_app_yaml_overrides_system_yaml(self):
        """App-level YAML mpi: overrides system-level YAML mpi:."""
        system_config = self._make_system_config(mpi_backend="mpich")
        yaml_config = {
            "polaris": {
                "mpi": {
                    "use_gpu_wrapper": True,
                    "cpu_bind_method": "depth",
                    "disable": ["-H"],
                }
            },
            "lammps-kk": {
                "polaris": {
                    "mpi": {
                        "use_gpu_wrapper": False,
                        "disable": ["rankfile"],
                        "add": ["--verbose"],
                    }
                }
            },
        }
        result = load_mpi_config("polaris", "lammps-kk", system_config, yaml_config)
        # App override wins for scalars
        assert result.use_gpu_wrapper is False
        # System value preserved when not overridden by app
        assert result.cpu_bind_method == "depth"
        # App list replaces system list
        assert result.disable == ["rankfile"]
        assert result.add == ["--verbose"]
        # Backend from system YAML (not overridden by app)
        assert result.backend == MPIBackend.MPICH

    def test_app_overrides_backend(self):
        """App-level can override the backend itself."""
        system_config = self._make_system_config(mpi_backend="openmpi")
        yaml_config = {
            "sophia": {
                "mpi": {"backend": "openmpi"}
            },
            "vasp": {
                "sophia": {
                    "mpi": {"backend": "mpich"}
                }
            },
        }
        result = load_mpi_config("sophia", "vasp", system_config, yaml_config)
        assert result.backend == MPIBackend.MPICH

    def test_no_mpi_in_yaml_uses_class_defaults(self):
        """When YAML has no mpi: sections, result uses class defaults only."""
        system_config = self._make_system_config(mpi_backend="mpich")
        yaml_config = {
            "polaris": {"pbx_python_env_setup": "module load conda"},
            "lammps-kk": {
                "polaris": {
                    "executable_path": "/path/to/lmp",
                }
            },
        }
        result = load_mpi_config("polaris", "lammps-kk", system_config, yaml_config)
        assert result.backend == MPIBackend.MPICH
        assert result.use_gpu_wrapper is False
        assert result.cpu_bind_method == "none"
        assert result.disable == []
        assert result.add == []

    def test_system_yaml_only_no_app_override(self):
        """System YAML mpi: is used when app has no mpi: override."""
        system_config = self._make_system_config(mpi_backend="openmpi")
        yaml_config = {
            "sophia": {
                "mpi": {
                    "use_short_hostnames": True,
                    "add": ["--oversubscribe"],
                }
            },
            "lammps-kk": {
                "sophia": {
                    "executable_path": "/path/to/lmp",
                }
            },
        }
        result = load_mpi_config("sophia", "lammps-kk", system_config, yaml_config)
        assert result.use_short_hostnames is True
        assert result.add == ["--oversubscribe"]

    def test_full_three_layer_merge(self):
        """Full 3-layer merge: class default -> system YAML -> app YAML."""
        system_config = self._make_system_config(mpi_backend="mpich")
        yaml_config = {
            "polaris": {
                "mpi": {
                    "backend": "mpich",
                    "use_gpu_wrapper": True,
                    "cpu_bind_method": "depth",
                    "add": ["--depth 8"],
                }
            },
            "lammps-kk": {
                "polaris": {
                    "mpi": {
                        "cpu_bind_method": "rankfile",
                        "add": ["--mca btl ^openib"],
                    }
                }
            },
        }
        result = load_mpi_config("polaris", "lammps-kk", system_config, yaml_config)
        # From system YAML (not overridden by app)
        assert result.backend == MPIBackend.MPICH
        assert result.use_gpu_wrapper is True
        # Overridden by app YAML
        assert result.cpu_bind_method == "rankfile"
        # App list replaces system list entirely
        assert result.add == ["--mca btl ^openib"]

    def test_missing_system_in_yaml(self):
        """Gracefully handles system not present in YAML."""
        system_config = self._make_system_config(mpi_backend="openmpi")
        yaml_config = {
            "lammps-kk": {
                "polaris": {
                    "mpi": {"use_gpu_wrapper": True}
                }
            }
        }
        result = load_mpi_config("polaris", "lammps-kk", system_config, yaml_config)
        assert result.backend == MPIBackend.OPENMPI
        assert result.use_gpu_wrapper is True

    def test_missing_app_in_yaml(self):
        """Gracefully handles app not present in YAML."""
        system_config = self._make_system_config(mpi_backend="mpich")
        yaml_config = {
            "polaris": {
                "mpi": {"cpu_bind_method": "depth"}
            }
        }
        result = load_mpi_config("polaris", "lammps-kk", system_config, yaml_config)
        assert result.backend == MPIBackend.MPICH
        assert result.cpu_bind_method == "depth"

    def test_app_mpi_cmd_override(self):
        """App can specify a custom mpi_cmd path."""
        system_config = self._make_system_config(mpi_backend="openmpi")
        yaml_config = {
            "sophia": {
                "mpi": {"backend": "openmpi"}
            },
            "lammps-kk": {
                "sophia": {
                    "mpi": {"mpi_cmd": "/opt/openmpi-4.1.6/bin/mpirun"}
                }
            },
        }
        result = load_mpi_config("sophia", "lammps-kk", system_config, yaml_config)
        assert result.mpi_cmd == "/opt/openmpi-4.1.6/bin/mpirun"
        assert result.get_mpi_command() == "/opt/openmpi-4.1.6/bin/mpirun"
