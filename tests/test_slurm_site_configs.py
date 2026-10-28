"""
Tests for the SLURM system configs written from site documentation
(Kestrel, Delta, Anvil, Expanse, Midway3, Bridges-2).

None of these have been run on their systems, so the hardware table below is
the reviewable record of what the docs said, and the behavior tests pin the
parts that are shared with `perlmutter-*`.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from parslbox.system_configs.loader import (
    CONFIG_FACTORIES,
    get_system_config,
    get_system_specs,
)
from parslbox.utils.config_generator import ConfigGenerator

# name -> (cores per node, GPUs per node)
SITE_CONFIGS = {
    "kestrel-cpu": (104, 0),
    "kestrel-gpu": (128, 4),
    "delta-cpu": (128, 0),
    "delta-gpu": (64, 4),
    "delta-h200": (96, 8),
    "anvil-cpu": (128, 0),
    "anvil-gpu": (128, 4),
    "anvil-ai": (96, 4),
    "expanse-cpu": (128, 0),
    "expanse-gpu": (40, 4),
    "midway3-cpu": (48, 0),
    "midway3-gpu": (48, 4),
    "bridges2-cpu": (128, 0),
    "bridges2-v100": (40, 8),
    "bridges2-h100": (104, 8),
}
GPU_SITES = [n for n, (_, g) in SITE_CONFIGS.items() if g]
CPU_SITES = [n for n, (_, g) in SITE_CONFIGS.items() if not g]


def _nvidia_smi(count):
    out = "".join(f"GPU {i}: NVIDIA Test GPU (UUID: GPU-{i})\n" for i in range(count))
    return subprocess.CompletedProcess(["nvidia-smi", "-L"], 0, stdout=out, stderr="")


class TestRegistration:
    def test_every_site_config_is_registered(self):
        assert set(SITE_CONFIGS) <= set(CONFIG_FACTORIES)

    @pytest.mark.parametrize("name", SITE_CONFIGS)
    def test_registry_key_matches_system_name(self, name):
        assert get_system_config(name).SYSTEM_NAME == name

    @pytest.mark.parametrize("name,spec", SITE_CONFIGS.items())
    def test_hardware_matches_the_site_docs(self, name, spec):
        cfg = get_system_config(name)
        assert (cfg.CORES_PER_NODE, cfg.GPUS_PER_NODE) == spec

    def test_listed_by_get_system_specs(self):
        listed = {s["name"]: s for s in get_system_specs()}
        for name, (cores, gpus) in SITE_CONFIGS.items():
            assert listed[name]["cores_per_node"] == cores
            assert listed[name]["gpus_per_node"] == gpus
            assert listed[name]["scheduler"] == "SLURM"


class TestSharedShape:
    """Same shape as perlmutter-*: SLURM, srun, cores binding, no affinity."""

    @pytest.mark.parametrize("name", SITE_CONFIGS)
    def test_launcher_and_scheduler(self, name):
        cfg = get_system_config(name)
        assert cfg.SCHEDULER == "SLURM"
        assert cfg.MPI_BACKEND == "srun"
        assert cfg.WORKER_CPU_AFFINITY is None

    @pytest.mark.parametrize("name", SITE_CONFIGS)
    def test_default_mpi_yaml(self, name):
        mpi = get_system_config(name).get_default_mpi_config_yaml()
        assert mpi["backend"] == "srun"
        assert mpi["cpu_bind_method"] == "cores"
        assert mpi["use_gpu_wrapper"] is False

    @pytest.mark.parametrize("name", GPU_SITES)
    def test_gpu_systems_are_cuda(self, name):
        assert get_system_config(name).GPU_TYPE == "cuda"

    @pytest.mark.parametrize("name", CPU_SITES)
    def test_cpu_systems_have_no_gpu_type(self, name):
        assert get_system_config(name).GPU_TYPE is None

    @pytest.mark.parametrize("name", SITE_CONFIGS)
    def test_default_sched_opts_do_not_hardcode_a_gpu_request(self, name):
        # These sites share nodes, so a baked-in count (Perlmutter's
        # --gpus-per-node=4) would be wrong for a sub-node request. Users pass
        # the count and type through --sched-opts.
        opts = get_system_config(name).get_default_sched_opts() or ""
        assert "gpus" not in opts and "gres" not in opts

    @pytest.mark.parametrize("name", SITE_CONFIGS)
    def test_workers_default_to_one_per_gpu_or_core(self, name):
        cfg = get_system_config(name)
        assert cfg.MAX_WORKERS_PER_NODE == (cfg.GPUS_PER_NODE or cfg.CORES_PER_NODE)


class TestCpuConfigs:
    @pytest.mark.parametrize("name", CPU_SITES)
    def test_detect_reads_slurm_node_count(self, name):
        with patch.dict("os.environ", {"SLURM_JOB_NUM_NODES": "3"}):
            assert get_system_config(name).detect_resources() == (3, 0)

    @pytest.mark.parametrize("name", CPU_SITES)
    def test_detect_defaults_to_one_node(self, name):
        with patch.dict("os.environ", {}, clear=True):
            assert get_system_config(name).detect_resources() == (1, 0)

    @pytest.mark.parametrize("name", CPU_SITES)
    def test_one_worker_per_core_per_node(self, name, tmp_path):
        cfg = get_system_config(name)
        with patch.dict("os.environ", {"SLURM_JOB_NUM_NODES": "2"}):
            executor = cfg.get_config(tmp_path).executors[0]
        assert executor.max_workers_per_node == cfg.CORES_PER_NODE * 2

    @pytest.mark.parametrize("name", CPU_SITES)
    def test_max_workers_is_capped(self, name, tmp_path):
        cfg = get_system_config(name)
        with patch.dict("os.environ", {"SLURM_JOB_NUM_NODES": "1"}):
            executor = cfg.get_config(tmp_path, max_workers=10_000).executors[0]
            assert executor.max_workers_per_node == cfg.CORES_PER_NODE
            executor = cfg.get_config(tmp_path, max_workers=5).executors[0]
            assert executor.max_workers_per_node == 5


class TestGpuConfigs:
    @pytest.mark.parametrize("name", GPU_SITES)
    def test_multi_node_uses_the_known_per_node_count(self, name):
        cfg = get_system_config(name)
        with patch.dict("os.environ", {"SLURM_JOB_NUM_NODES": "4"}):
            assert cfg.detect_resources() == (4, 4 * cfg.GPUS_PER_NODE)

    @pytest.mark.parametrize("name", GPU_SITES)
    def test_single_node_counts_the_gpus_the_job_holds(self, name):
        # A shared-node allocation of 2 GPUs: nvidia-smi lists only those.
        cfg = get_system_config(name)
        with patch.dict("os.environ", {"SLURM_JOB_NUM_NODES": "1"}), \
                patch("subprocess.run", return_value=_nvidia_smi(2)):
            assert cfg.detect_resources() == (1, 2)

    @pytest.mark.parametrize("name", GPU_SITES)
    def test_falls_back_to_the_full_node_without_nvidia_smi(self, name):
        cfg = get_system_config(name)
        with patch.dict("os.environ", {"SLURM_JOB_NUM_NODES": "1"}), \
                patch("subprocess.run", side_effect=FileNotFoundError):
            assert cfg.detect_resources() == (1, cfg.GPUS_PER_NODE)

    @pytest.mark.parametrize("name", GPU_SITES)
    def test_sub_node_gets_workers_only_for_its_gpus(self, name, tmp_path):
        cfg = get_system_config(name)
        with patch.dict("os.environ", {"SLURM_JOB_NUM_NODES": "1"}), \
                patch("subprocess.run", return_value=_nvidia_smi(2)):
            executor = cfg.get_config(tmp_path).executors[0]
        assert executor.max_workers_per_node == 2
        assert executor.cores_per_worker == cfg.CORES_PER_NODE / 2

    @pytest.mark.parametrize("name", GPU_SITES)
    def test_full_node_workers_are_gpus_per_node_times_nodes(self, name, tmp_path):
        cfg = get_system_config(name)
        with patch.dict("os.environ", {"SLURM_JOB_NUM_NODES": "2"}):
            executor = cfg.get_config(tmp_path).executors[0]
        assert executor.max_workers_per_node == 2 * cfg.GPUS_PER_NODE

    @pytest.mark.parametrize("name", [n for n in GPU_SITES if SITE_CONFIGS[n][1] == 4])
    @pytest.mark.parametrize("nodes", ["1", "2", "3"])
    @pytest.mark.parametrize("max_workers", [None, 3, 100])
    def test_whole_nodes_match_what_perlmutter_gpu_computes(self, name, nodes, max_workers, tmp_path):
        # The real cross-check: on whole nodes of a 4-GPU system the worker
        # and core split must equal perlmutter-gpu's, which is proven in use.
        env = {"SLURM_JOB_NUM_NODES": nodes}
        with patch.dict("os.environ", env), patch("subprocess.run", return_value=_nvidia_smi(4)):
            ours = get_system_config(name).get_config(tmp_path, max_workers=max_workers).executors[0]
            ref = get_system_config("perlmutter-gpu").get_config(tmp_path, max_workers=max_workers).executors[0]
        assert ours.max_workers_per_node == ref.max_workers_per_node
        # perlmutter-gpu spreads 128 cores over its workers; ours spreads our own
        cfg = get_system_config(name)
        assert ours.cores_per_worker == cfg.CORES_PER_NODE / ours.max_workers_per_node

    @pytest.mark.parametrize("name", GPU_SITES)
    def test_more_devices_than_the_node_has_are_capped(self, name, tmp_path):
        # e.g. MIG slices or a larger node variant: never more workers than
        # GPUS_PER_NODE * nodes.
        cfg = get_system_config(name)
        with patch.dict("os.environ", {"SLURM_JOB_NUM_NODES": "1"}), \
                patch("subprocess.run", return_value=_nvidia_smi(cfg.GPUS_PER_NODE * 4)):
            executor = cfg.get_config(tmp_path).executors[0]
        assert executor.max_workers_per_node == cfg.GPUS_PER_NODE

    @pytest.mark.parametrize("name", GPU_SITES)
    @pytest.mark.parametrize("failure", ["error", "empty"])
    def test_other_nvidia_smi_failures_fall_back_to_the_full_node(self, name, failure):
        cfg = get_system_config(name)
        if failure == "error":
            patcher = patch("subprocess.run", side_effect=subprocess.CalledProcessError(9, "nvidia-smi"))
        else:
            patcher = patch("subprocess.run", return_value=_nvidia_smi(0))
        with patch.dict("os.environ", {"SLURM_JOB_NUM_NODES": "1"}), patcher:
            assert cfg.detect_resources() == (1, cfg.GPUS_PER_NODE)

    @pytest.mark.parametrize("name", GPU_SITES)
    def test_max_workers_is_capped_at_the_allocated_gpus(self, name, tmp_path):
        cfg = get_system_config(name)
        with patch.dict("os.environ", {"SLURM_JOB_NUM_NODES": "1"}), \
                patch("subprocess.run", return_value=_nvidia_smi(2)):
            assert cfg.get_config(tmp_path, max_workers=99).executors[0].max_workers_per_node == 2
            assert cfg.get_config(tmp_path, max_workers=1).executors[0].max_workers_per_node == 1


class TestExecutorLabels:
    def test_every_config_has_its_own_executor_label(self, tmp_path):
        labels = []
        for name in SITE_CONFIGS:
            with patch.dict("os.environ", {"SLURM_JOB_NUM_NODES": "1"}), \
                    patch("subprocess.run", return_value=_nvidia_smi(2)):
                labels.append(get_system_config(name).get_config(tmp_path).executors[0].label)
        assert len(set(labels)) == len(labels)
        assert all(label.startswith("htex_") for label in labels)


class TestConfigGeneration:
    def test_pbx_config_generates_valid_yaml_for_every_site(self):
        generated = ConfigGenerator(list(SITE_CONFIGS), ["python"]).generate()
        parsed = yaml.safe_load(generated)
        for name in SITE_CONFIGS:
            assert parsed[name]["mpi"]["backend"] == "srun"
        assert "SLURM" in generated.upper()
