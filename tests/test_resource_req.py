"""Unit tests for commands/helpers/resource_req.py.

The arithmetic here used to live inside `info.py`'s console.print calls, where
the only coverage was "the command exits 0". Nothing pinned the numbers, so a
refactor could have changed them in silence. These tests pin them.
"""
import pytest

from parslbox.commands.helpers.resource_req import (
    UnknownSystemError,
    build_command_lines,
    compute_requirements,
)


def job(job_id, num_nodes, ngpus, occupancy=1.0, status="Ready", app="vasp"):
    return dict(job_id=job_id, num_nodes=num_nodes, ngpus=ngpus,
                node_occupancy=occupancy, status=status, app=app)


class TestSystemValidation:
    def test_unknown_system_names_the_alternatives(self):
        with pytest.raises(UnknownSystemError) as e:
            compute_requirements([], "aurora-tyle")
        assert "aurora-tile" in str(e.value)
        assert e.value.system_name == "aurora-tyle"

    def test_it_is_a_valueerror(self):
        # api.py catches it and re-raises as ValidationError; CLI prints it.
        assert issubclass(UnknownSystemError, ValueError)


class TestOnlySchedulableJobsCount:
    """Sizing a project must not count work that has already finished, but a
    total that ignores jobs without saying so is worse than no total."""

    def test_finished_and_running_jobs_are_reported_not_counted(self):
        req = compute_requirements(
            [job(1, 1, 4, status="Done"), job(2, 1, 4, status="Running"),
             job(3, 1, 4)], "aurora-tile")
        assert req["schedulable"] == 1
        assert req["skipped"] == {"Done": 1, "Running": 1}

    def test_restart_counts_as_schedulable(self):
        req = compute_requirements([job(1, 1, 4, status="Restart")], "aurora-tile")
        assert req["schedulable"] == 1
        assert req["skipped"] == {}

    def test_nothing_schedulable_leaves_every_block_empty(self):
        req = compute_requirements([job(1, 1, 4, status="Done")], "aurora-tile")
        assert (req["gpu"], req["cpu"], req["total"]) == (None, None, None)
        assert req["skipped"] == {"Done": 1}


class TestGpuPacking:
    """aurora-tile has 12 GPUs/node. 'simultaneous' runs everything at once;
    'optimal' is the tighter number pbx reaches by queueing jobs behind each
    other, so optimal <= simultaneous always."""

    def test_four_four_gpu_jobs_pack_into_one_node_when_queued(self):
        gpu = compute_requirements([job(i, 1, 4) for i in range(1, 5)],
                                   "aurora-tile")["gpu"]
        assert (gpu["total_gpus"], gpu["simultaneous"], gpu["optimal"]) == (16, 2, 1)
        assert gpu["idle_slots"] == 8
        assert gpu["queued_gpus"] == 4

    def test_a_multinode_job_takes_its_nodes_whole(self):
        gpu = compute_requirements([job(1, 5, 60)], "aurora-tile")["gpu"]
        assert (gpu["simultaneous"], gpu["optimal"]) == (5, 5)
        assert gpu["multinode_nodes"] == 5
        assert gpu["min_viable_nodes"] == 5

    def test_an_exact_fit_wastes_nothing(self):
        gpu = compute_requirements([job(1, 1, 12)], "aurora-tile")["gpu"]
        assert (gpu["simultaneous"], gpu["optimal"], gpu["idle_slots"]) == (1, 1, 0)
        # The boundary itself: a job wanting exactly a node's worth fits, so
        # the oversized test has to be `>` and not `>=`.
        assert gpu["oversized"] == []

    def test_nothing_is_queued_when_packing_buys_no_nodes(self):
        # opt == sim here, so there is no second wave to queue. The guard on
        # that branch has to be `opt < sim`; `<=` would report a negative
        # count of queued GPUs.
        gpu = compute_requirements([job(1, 1, 4)], "aurora-tile")["gpu"]
        assert gpu["optimal"] == gpu["simultaneous"]
        assert gpu["queued_gpus"] == 0

    def test_min_viable_nodes_is_the_widest_job_not_the_narrowest(self):
        # An allocation smaller than the widest job cannot run it at all, so
        # this is a max. With one job in the set, max and min agree and the
        # distinction goes untested -- hence two jobs of different widths.
        gpu = compute_requirements([job(1, 5, 60), job(2, 2, 24)],
                                   "aurora-tile")["gpu"]
        assert gpu["min_viable_nodes"] == 5

    def test_a_job_too_wide_for_one_node_is_flagged(self):
        gpu = compute_requirements([job(1, 1, 99)], "aurora-tile")["gpu"]
        assert gpu["oversized"] == [{"job_id": 1, "ngpus": 99}]
        # Pre-existing behaviour, pinned deliberately: an unschedulable job
        # contributes no nodes, so the total reads 0 and the flag is the only
        # thing that says why.
        assert (gpu["simultaneous"], gpu["optimal"]) == (0, 0)

    def test_a_gpu_job_on_a_cpu_only_system_reports_no_node_counts(self):
        gpu = compute_requirements([job(1, 1, 4)], "crux")["gpu"]
        assert gpu["system_has_gpus"] is False
        assert "simultaneous" not in gpu


class TestCpuPacking:
    def test_fractional_occupancy_shares_a_node(self):
        cpu = compute_requirements(
            [job(1, 1, 0, 0.25, app="python"), job(2, 1, 0, 0.5, app="python")],
            "crux")["cpu"]
        assert cpu["total_occupancy"] == 0.75
        assert (cpu["simultaneous"], cpu["optimal"]) == (1, 1)
        assert (cpu["full_nodes"], cpu["fraction"]) == (0, 0.75)

    def test_occupancy_above_one_node_rounds_up(self):
        cpu = compute_requirements(
            [job(i, 1, 0, 0.5, app="python") for i in range(1, 6)], "crux")["cpu"]
        assert cpu["total_occupancy"] == 2.5
        assert cpu["simultaneous"] == 3

    def test_cpu_packing_is_already_optimal(self):
        # Occupancy is the tightest packing there is, so queueing buys nothing.
        cpu = compute_requirements(
            [job(1, 1, 0, 0.3, app="python"), job(2, 3, 0, app="python")],
            "crux")["cpu"]
        assert cpu["simultaneous"] == cpu["optimal"]

    def test_cpu_jobs_on_a_gpu_system_are_told_the_gpus_idle(self):
        cpu = compute_requirements([job(1, 1, 0, 0.5, app="python")],
                                   "aurora-tile")["cpu"]
        assert cpu["gpus_would_idle"] is True


class TestMixedWorkloads:
    def test_gpu_and_cpu_totals_add_up(self):
        req = compute_requirements(
            [job(1, 1, 4), job(2, 1, 0, 0.5, app="python"),
             job(3, 2, 24, app="lammps-kk")], "aurora-tile")
        assert req["total"]["simultaneous"] == (req["gpu"]["simultaneous"]
                                                + req["cpu"]["simultaneous"])
        assert req["total"] == {"simultaneous": 4, "optimal": 4}

    def test_a_single_class_workload_has_no_combined_total(self):
        # The combined line would just repeat the gpu line.
        assert compute_requirements([job(1, 1, 4)], "aurora-tile")["total"] is None

    def test_system_constants_are_reported_so_a_caller_can_check_the_maths(self):
        req = compute_requirements([job(1, 1, 4)], "aurora-tile")
        assert (req["gpus_per_node"], req["cores_per_node"]) == (12, 208)


class TestCommandLinePreview:
    def test_it_previews_one_row_per_job(self):
        out = build_command_lines([job(1, 1, 4, app="vasp"),
                                   job(2, 1, 0, 0.5, app="python")], "aurora-tile")
        assert out["system"] == "aurora-tile"
        assert [r["job_id"] for r in out["jobs"]] == [1, 2]

    def test_the_command_is_a_real_launch_line(self):
        row = build_command_lines([job(1, 1, 4, app="vasp")], "aurora-tile")["jobs"][0]
        assert row["error"] is None
        assert row["command"]
        assert row["resources"] == "1N/4G"

    def test_resources_read_differently_for_gpu_cpu_and_fractional_jobs(self):
        rows = build_command_lines(
            [job(1, 2, 24, app="lammps-kk"), job(2, 1, 0, 0.25, app="python"),
             job(3, 2, 0, app="python")], "aurora-tile")["jobs"]
        assert [r["resources"] for r in rows] == ["2N/24G", "1N/0.25occ", "2N/CPU"]

    def test_an_unknown_system_is_refused_before_any_work(self):
        with pytest.raises(UnknownSystemError):
            build_command_lines([job(1, 1, 4)], "not-a-machine")

    def test_the_preview_leaves_no_temp_directory_behind(self):
        # The command names rankfiles and wrapper scripts; generating them is
        # the only reason this is not a pure function, so the directory has to
        # go before the call returns.
        import tempfile
        from pathlib import Path
        before = set(Path(tempfile.gettempdir()).glob("pbx_cmdline_preview_*"))
        build_command_lines([job(1, 2, 24, app="lammps-kk")], "aurora-tile")
        assert set(Path(tempfile.gettempdir()).glob("pbx_cmdline_preview_*")) == before

    def test_an_unconfigured_app_still_previews(self):
        # load_mpi_config falls back through system config to backend defaults,
        # so an app absent from config.yaml previews rather than erroring.
        row = build_command_lines([job(1, 1, 4, app="no-such-app")],
                                  "aurora-tile")["jobs"][0]
        assert row["error"] is None and row["command"]

    def test_one_bad_app_does_not_sink_the_other_rows(self, monkeypatch):
        import parslbox.resource_manager.mpi_config as mc

        real = mc.load_mpi_config

        def explode(*, app_name, **kw):
            if app_name == "python":
                raise RuntimeError("bad mpi stanza")
            return real(app_name=app_name, **kw)

        monkeypatch.setattr(mc, "load_mpi_config", explode)
        ok, bad = build_command_lines([job(1, 1, 4, app="vasp"),
                                       job(2, 1, 0, 0.5, app="python")],
                                      "aurora-tile")["jobs"]
        assert ok["command"] and ok["error"] is None
        assert bad["command"] is None
        assert "bad mpi stanza" in bad["error"]

    def test_it_reads_the_config_it_is_handed(self, tmp_path, monkeypatch):
        # One server process can be rebound to another project by use_database,
        # so a preview that read the process-start config would show another
        # project's MPI settings with nothing to say it had.
        from parslbox.utils import path_utils
        import parslbox.commands.helpers.resource_req as rr

        seen = {}

        def spy(config_path=None):
            seen["path"] = config_path
            return {}

        monkeypatch.setattr("parslbox.utils.pbx_config_utils.load_full_config", spy)
        monkeypatch.setattr(path_utils, "PBX_CONFIG_FILE", tmp_path / "theirs.yaml")
        rr.build_command_lines([], "aurora-tile", config_path=tmp_path / "mine.yaml")
        assert seen["path"] == tmp_path / "mine.yaml"
