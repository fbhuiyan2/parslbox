"""
Phase 4 tests: orchestrator restart-mode behavior.

Covers:
- The three-scene `apply_restart_hook` contract (dict / None / raise).
- restart_helpers utilities (placeholder substitution, counter decrement,
  template validation, original-cap extraction, link script generation,
  resubmit subprocess wrapper).
- `compute_required_nodes` extracted helper (Phase 5).
- `perform_shutdown` branching: walltime + restart-mode-on (Restart vs
  Failed), walltime + restart-mode-off (Killed), signal path (always
  Killed regardless of restart_ctx).
"""

import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from parslbox.commands.helpers import restart_helpers
from parslbox.commands.helpers.resource_estimate import compute_required_nodes
from parslbox.commands.helpers.run_cmd_helpers import perform_shutdown
from parslbox.database import database


# ============================================================
# Helpers
# ============================================================


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "jobs.db"
    database.initialize_database(db_path)
    return db_path


def _insert_job(db_path, **kwargs):
    """Insert a job directly into the DB with sensible defaults."""
    defaults = dict(
        path="/x", app="lammps-kk",
        num_nodes=1, ngpus=2, node_occupancy=1.0,
        ranks_per_node=2, status="Restart", tag=None,
        in_file=None, mpi_opts=None, env_file=None,
        parents=None,
    )
    defaults.update(kwargs)
    return database.add_job(db_path, **defaults)


# ============================================================
# Scene A/B/C — apply_restart_hook
# ============================================================


class TestApplyRestartHook:
    def test_scene_a_dict_patches_and_flips_to_ready(self, db):
        jid = _insert_job(db, in_file="in.original")
        app = MagicMock()
        app.restart.return_value = {"in_file": "in.restart"}
        logger = MagicMock()

        buckets = restart_helpers.apply_restart_hook(
            restart_jobs=database.get_jobs(db, status="Restart"),
            app_instances={"lammps-kk": app},
            db_path=db,
            logger=logger,
        )
        assert buckets == {"patched": [jid], "rerun": [], "failed": []}
        job = database.get_jobs_by_ids(db, [jid])[0]
        assert job["status"] == "Ready"
        assert job["in_file"] == "in.restart"

    def test_scene_b_none_keeps_fields_flips_to_ready(self, db):
        jid = _insert_job(db, in_file="in.original")
        app = MagicMock()
        app.restart.return_value = None
        logger = MagicMock()

        buckets = restart_helpers.apply_restart_hook(
            restart_jobs=database.get_jobs(db, status="Restart"),
            app_instances={"lammps-kk": app},
            db_path=db,
            logger=logger,
        )
        assert buckets == {"patched": [], "rerun": [jid], "failed": []}
        job = database.get_jobs_by_ids(db, [jid])[0]
        assert job["status"] == "Ready"
        assert job["in_file"] == "in.original"

    def test_scene_b_empty_dict_also_rerun(self, db):
        jid = _insert_job(db)
        app = MagicMock()
        app.restart.return_value = {}
        buckets = restart_helpers.apply_restart_hook(
            restart_jobs=database.get_jobs(db, status="Restart"),
            app_instances={"lammps-kk": app},
            db_path=db,
            logger=MagicMock(),
        )
        assert buckets["rerun"] == [jid]
        assert database.get_jobs_by_ids(db, [jid])[0]["status"] == "Ready"

    def test_scene_c_notimplemented_marks_failed(self, db):
        jid = _insert_job(db)
        app = MagicMock()
        app.restart.side_effect = NotImplementedError("no restart support")
        buckets = restart_helpers.apply_restart_hook(
            restart_jobs=database.get_jobs(db, status="Restart"),
            app_instances={"lammps-kk": app},
            db_path=db,
            logger=MagicMock(),
        )
        assert buckets == {"patched": [], "rerun": [], "failed": [jid]}
        assert database.get_jobs_by_ids(db, [jid])[0]["status"] == "Failed"

    def test_missing_app_marks_failed(self, db):
        jid = _insert_job(db, app="missing_app")
        buckets = restart_helpers.apply_restart_hook(
            restart_jobs=database.get_jobs(db, status="Restart"),
            app_instances={},  # no app loaded
            db_path=db,
            logger=MagicMock(),
        )
        assert buckets["failed"] == [jid]
        assert database.get_jobs_by_ids(db, [jid])[0]["status"] == "Failed"

    def test_unknown_patch_fields_dropped_with_warning(self, db):
        jid = _insert_job(db)
        app = MagicMock()
        app.restart.return_value = {"in_file": "ok", "bogus_field": 42}
        logger = MagicMock()
        restart_helpers.apply_restart_hook(
            restart_jobs=database.get_jobs(db, status="Restart"),
            app_instances={"lammps-kk": app},
            db_path=db,
            logger=logger,
        )
        # Warning logged about the bogus field.
        warn_calls = [c for c in logger.warning.call_args_list if "bogus_field" in str(c)]
        assert warn_calls
        # in_file applied; bogus_field harmlessly ignored.
        assert database.get_jobs_by_ids(db, [jid])[0]["in_file"] == "ok"

    def test_mixed_buckets_all_three_in_one_call(self, db):
        jid_a = _insert_job(db, app="lammps-kk", path="/a")
        jid_b = _insert_job(db, app="vasp", path="/b")
        jid_c = _insert_job(db, app="python", path="/c")
        app_a = MagicMock(); app_a.restart.return_value = {"in_file": "x"}
        app_b = MagicMock(); app_b.restart.return_value = None
        app_c = MagicMock(); app_c.restart.side_effect = NotImplementedError()
        buckets = restart_helpers.apply_restart_hook(
            restart_jobs=database.get_jobs(db, status="Restart"),
            app_instances={"lammps-kk": app_a, "vasp": app_b, "python": app_c},
            db_path=db,
            logger=MagicMock(),
        )
        assert buckets == {"patched": [jid_a], "rerun": [jid_b], "failed": [jid_c]}


# ============================================================
# Resource recalc (Phase 5 helper)
# ============================================================


class TestComputeRequiredNodes:
    def _sys(self, gpus_per_node=4):
        return SimpleNamespace(GPUS_PER_NODE=gpus_per_node)

    def test_empty(self):
        assert compute_required_nodes([], self._sys()) == 0

    def test_skip_non_schedulable(self):
        jobs = [{"status": "Done", "ngpus": 4, "num_nodes": 1, "node_occupancy": 1.0}]
        assert compute_required_nodes(jobs, self._sys()) == 0

    def test_gpu_jobs_pack_optimally(self):
        # 8 single-node 2-GPU jobs on 4-GPU/node system → 4 nodes optimal.
        jobs = [
            {"status": "Ready", "ngpus": 2, "num_nodes": 1, "node_occupancy": 1.0}
        ] * 8
        assert compute_required_nodes(jobs, self._sys(gpus_per_node=4)) == 4

    def test_multinode_gpu_jobs_dedicate_their_own(self):
        jobs = [
            {"status": "Ready", "ngpus": 8, "num_nodes": 2, "node_occupancy": 1.0},
            {"status": "Ready", "ngpus": 4, "num_nodes": 1, "node_occupancy": 1.0},
        ]
        # multi-node: 2 dedicated + single-node: ceil(4/4) = 1 → 3
        assert compute_required_nodes(jobs, self._sys(gpus_per_node=4)) == 3

    def test_cpu_jobs_pack_by_occupancy(self):
        # 4 quarter-node CPU jobs → 1 node.
        jobs = [
            {"status": "Ready", "ngpus": 0, "num_nodes": 1, "node_occupancy": 0.25}
        ] * 4
        assert compute_required_nodes(jobs, self._sys()) == 1

    def test_min_one_when_anything_schedulable(self):
        jobs = [{"status": "Ready", "ngpus": 0, "num_nodes": 1, "node_occupancy": 0.01}]
        assert compute_required_nodes(jobs, self._sys()) == 1


# ============================================================
# restart_helpers utilities — placeholder, counter, validation, cap
# ============================================================


class TestPlaceholderSubstitution:
    def test_select_replaced(self):
        out = restart_helpers._substitute_resource_placeholders(
            "select=<<PBX_AUTO_SELECT>>", computed_nodes=3, original_cap=10
        )
        assert "select=3" in out
        assert "<<PBX_AUTO_SELECT>>" not in out

    def test_nodes_replaced(self):
        out = restart_helpers._substitute_resource_placeholders(
            "--nodes=<<PBX_AUTO_NODES>>", computed_nodes=2, original_cap=None
        )
        assert "--nodes=2" in out

    def test_capped_at_original(self):
        out = restart_helpers._substitute_resource_placeholders(
            "select=<<PBX_AUTO_SELECT>>", computed_nodes=100, original_cap=4
        )
        assert "select=4" in out

    def test_minimum_one_node(self):
        out = restart_helpers._substitute_resource_placeholders(
            "select=<<PBX_AUTO_SELECT>>", computed_nodes=0, original_cap=10
        )
        assert "select=1" in out


class TestDecrementMaxRestarts:
    def test_simple_replace(self):
        out = restart_helpers._decrement_max_restarts(
            "pbx run --restart-mode --max-restarts 5 --config polaris", 4
        )
        assert "--max-restarts 4" in out
        assert "--max-restarts 5" not in out

    def test_preserves_other_args(self):
        line = "pbx run --restart-mode --max-restarts 3 --config polaris --apps lammps-kk"
        out = restart_helpers._decrement_max_restarts(line, 2)
        assert "--apps lammps-kk" in out
        assert "--config polaris" in out


class TestValidatePbxRunLine:
    def _line(self, **overrides):
        defaults = {
            "restart-mode": True,
            "max-restarts": 3,
            "config": "polaris",
            "run-dir": "./",
        }
        defaults.update(overrides)
        parts = ["pbx run"]
        for k, v in defaults.items():
            if v is True:
                parts.append(f"--{k}")
            elif v is None or v is False:
                continue
            else:
                parts.append(f"--{k} {v}")
        return " ".join(parts)

    def test_all_args_present(self):
        assert restart_helpers.validate_pbx_run_line(self._line())

    def test_missing_max_restarts_raises(self):
        line = "exec pbx run --restart-mode --config polaris --run-dir ./"
        with pytest.raises(ValueError, match="max-restarts"):
            restart_helpers.validate_pbx_run_line(line)

    def test_missing_config_raises(self):
        line = "exec pbx run --restart-mode --max-restarts 3 --run-dir ./"
        with pytest.raises(ValueError, match="config"):
            restart_helpers.validate_pbx_run_line(line)

    def test_no_pbx_run_line_raises(self):
        with pytest.raises(ValueError, match="no `pbx run"):
            restart_helpers.validate_pbx_run_line("#!/bin/bash\necho hi\n")


class TestExtractOriginalAllocation:
    def test_pbs_integer(self, tmp_path):
        p = tmp_path / "submit.sh"
        p.write_text("#!/bin/bash\n#PBS -l select=4\n#PBS -l walltime=01:00:00\n")
        assert restart_helpers.extract_original_allocation(p, "pbs") == 4

    def test_pbs_complex_returns_none(self, tmp_path):
        p = tmp_path / "submit.sh"
        p.write_text("#!/bin/bash\n#PBS -l select=2:ncpus=32:ngpus=4\n")
        assert restart_helpers.extract_original_allocation(p, "pbs") is None

    def test_slurm_integer(self, tmp_path):
        p = tmp_path / "submit.sh"
        p.write_text("#!/bin/bash\n#SBATCH --nodes=8\n")
        assert restart_helpers.extract_original_allocation(p, "slurm") == 8

    def test_missing_file_returns_none(self, tmp_path):
        assert restart_helpers.extract_original_allocation(
            tmp_path / "absent.sh", "pbs"
        ) is None


class TestDetectSchedulerCommand:
    def test_pbs(self, monkeypatch):
        monkeypatch.setenv("PBS_JOBID", "12345.x")
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        assert restart_helpers.detect_scheduler_command() == "qsub"

    def test_slurm(self, monkeypatch):
        monkeypatch.delenv("PBS_JOBID", raising=False)
        monkeypatch.setenv("SLURM_JOB_ID", "99")
        assert restart_helpers.detect_scheduler_command() == "sbatch"

    def test_neither(self, monkeypatch):
        monkeypatch.delenv("PBS_JOBID", raising=False)
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        assert restart_helpers.detect_scheduler_command() is None


# ============================================================
# build_restart_link_script — full integration
# ============================================================


class TestBuildRestartLinkScript:
    def _make_template(self, run_dir):
        (run_dir / "submit.sh").write_text(
            "#!/bin/bash\n#PBS -l select=4\n"
            "exec pbx run --restart-mode --max-restarts 3 "
            "--config polaris --run-dir ./\n"
        )
        (run_dir / "restart_template.sh").write_text(
            "#!/bin/bash\n#PBS -l select=<<PBX_AUTO_SELECT>>\n"
            "exec pbx run --restart-mode --max-restarts 3 "
            "--config polaris --run-dir ./\n"
        )

    def test_generates_link_1_with_substitutions(self, tmp_path):
        self._make_template(tmp_path)
        link = restart_helpers.build_restart_link_script(
            template_path=tmp_path / "restart_template.sh",
            run_dir=tmp_path,
            current_max_restarts=3,
            scheduler_type="pbs",
            computed_nodes=2,
            submit_file_path=tmp_path / "submit.sh",
        )
        assert link == tmp_path / "restart_link_1.sh"
        text = link.read_text()
        # placeholder replaced
        assert "select=2" in text
        assert "<<PBX_AUTO_SELECT>>" not in text
        # counter decremented
        assert "--max-restarts 2" in text
        assert "--max-restarts 3" not in text

    def test_increments_link_index_across_calls(self, tmp_path):
        self._make_template(tmp_path)
        for expected_idx in (1, 2, 3):
            link = restart_helpers.build_restart_link_script(
                template_path=tmp_path / "restart_template.sh",
                run_dir=tmp_path,
                current_max_restarts=3,
                scheduler_type="pbs",
                computed_nodes=1,
                submit_file_path=tmp_path / "submit.sh",
            )
            assert link == tmp_path / f"restart_link_{expected_idx}.sh"

    def test_cap_applied(self, tmp_path):
        self._make_template(tmp_path)
        link = restart_helpers.build_restart_link_script(
            template_path=tmp_path / "restart_template.sh",
            run_dir=tmp_path,
            current_max_restarts=3,
            scheduler_type="pbs",
            computed_nodes=100,  # huge
            submit_file_path=tmp_path / "submit.sh",  # original = 4
        )
        assert "select=4" in link.read_text()

    def test_missing_template_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            restart_helpers.build_restart_link_script(
                template_path=tmp_path / "absent.sh",
                run_dir=tmp_path,
                current_max_restarts=2,
                scheduler_type="pbs",
                computed_nodes=1,
                submit_file_path=tmp_path / "submit.sh",
            )

    def test_broken_template_raises(self, tmp_path):
        (tmp_path / "restart_template.sh").write_text(
            "#!/bin/bash\necho missing run line\n"
        )
        with pytest.raises(ValueError, match="no `pbx run"):
            restart_helpers.build_restart_link_script(
                template_path=tmp_path / "restart_template.sh",
                run_dir=tmp_path,
                current_max_restarts=2,
                scheduler_type="pbs",
                computed_nodes=1,
                submit_file_path=tmp_path / "submit.sh",
            )


# ============================================================
# submit_restart_link — subprocess success / failure
# ============================================================


class TestSubmitRestartLink:
    def test_success_logs_ok(self, tmp_path):
        link = tmp_path / "restart_link_1.sh"
        link.write_text("#!/bin/bash\n")
        logger = MagicMock()
        with patch(
            "parslbox.commands.helpers.restart_helpers.subprocess.run"
        ) as m:
            m.return_value = MagicMock(returncode=0, stdout="12346.x", stderr="")
            ok = restart_helpers.submit_restart_link(link, "qsub", tmp_path, logger)
        assert ok is True
        logger.info.assert_called()

    def test_nonzero_returncode_returns_false(self, tmp_path):
        link = tmp_path / "restart_link_1.sh"
        link.write_text("#!/bin/bash\n")
        logger = MagicMock()
        with patch(
            "parslbox.commands.helpers.restart_helpers.subprocess.run"
        ) as m:
            m.return_value = MagicMock(returncode=1, stdout="", stderr="quota exceeded")
            ok = restart_helpers.submit_restart_link(link, "qsub", tmp_path, logger)
        assert ok is False
        err_text = " ".join(str(c) for c in logger.error.call_args_list)
        assert "quota exceeded" in err_text
        assert "Chain stopped" in err_text

    def test_command_not_found(self, tmp_path):
        link = tmp_path / "restart_link_1.sh"
        link.write_text("#!/bin/bash\n")
        logger = MagicMock()
        with patch(
            "parslbox.commands.helpers.restart_helpers.subprocess.run",
            side_effect=FileNotFoundError(),
        ):
            ok = restart_helpers.submit_restart_link(link, "qsub", tmp_path, logger)
        assert ok is False
        err_text = " ".join(str(c) for c in logger.error.call_args_list)
        assert "not found" in err_text


# ============================================================
# perform_shutdown branching
# ============================================================


def _shutdown_fixtures(db_path, active_ids):
    """Mock status_buffer, job_tracker, parsl_loaded_flag for perform_shutdown."""
    sb = MagicMock()
    sb.flush_all.return_value = 0
    sb.db_path = db_path
    jt = MagicMock()
    jt.get_jobs_by_status.side_effect = lambda s: (
        [{"job_id": jid} for jid in active_ids] if s in ("Running", "Submitted") else []
    )
    return sb, jt, {"loaded": False}


class TestPerformShutdownBranching:
    def test_walltime_no_restart_marks_killed(self, db):
        jid = _insert_job(db, status="Running")
        sb, jt, pfl = _shutdown_fixtures(db, [jid])
        perform_shutdown(
            status_buffer=sb, job_tracker=jt, parsl_loaded_flag=pfl,
            logger=MagicMock(), reason="walltime", cleanup_parsl=False,
            restart_ctx=None,
        )
        assert database.get_jobs_by_ids(db, [jid])[0]["status"] == "Killed"

    def test_walltime_restart_mode_n_gt_0_marks_restart_and_resubmits(self, db, tmp_path):
        jid = _insert_job(db, status="Running")
        sb, jt, pfl = _shutdown_fixtures(db, [jid])
        sb.db_path = db
        system_config = SimpleNamespace(GPUS_PER_NODE=4)
        # Need a template + submit for the resubmit step to succeed.
        (tmp_path / "submit.sh").write_text("#PBS -l select=4\n")
        (tmp_path / "restart_template.sh").write_text(
            "#!/bin/bash\n#PBS -l select=<<PBX_AUTO_SELECT>>\n"
            "exec pbx run --restart-mode --max-restarts 3 "
            "--config polaris --run-dir ./\n"
        )
        ctx = {
            "max_restarts": 3, "run_dir": tmp_path, "system_config": system_config,
            "db_path": db, "app_filter": None, "tag_filter": None,
        }
        with patch(
            "parslbox.commands.helpers.restart_helpers.detect_scheduler_command",
            return_value="qsub",
        ), patch(
            "parslbox.commands.helpers.restart_helpers.subprocess.run"
        ) as m_sub:
            m_sub.return_value = MagicMock(returncode=0, stdout="55555.x", stderr="")
            perform_shutdown(
                status_buffer=sb, job_tracker=jt, parsl_loaded_flag=pfl,
                logger=MagicMock(), reason="walltime", cleanup_parsl=False,
                restart_ctx=ctx,
            )
        assert database.get_jobs_by_ids(db, [jid])[0]["status"] == "Restart"
        # Resubmit was attempted
        m_sub.assert_called_once()
        # Link script was created with decremented counter
        link = tmp_path / "restart_link_1.sh"
        assert link.exists()
        assert "--max-restarts 2" in link.read_text()

    def test_walltime_restart_mode_n_eq_0_marks_failed_no_resubmit(self, db, tmp_path):
        jid = _insert_job(db, status="Running")
        sb, jt, pfl = _shutdown_fixtures(db, [jid])
        sb.db_path = db
        ctx = {
            "max_restarts": 0, "run_dir": tmp_path,
            "system_config": SimpleNamespace(GPUS_PER_NODE=4),
            "db_path": db, "app_filter": None, "tag_filter": None,
        }
        with patch(
            "parslbox.commands.helpers.restart_helpers.subprocess.run"
        ) as m_sub:
            perform_shutdown(
                status_buffer=sb, job_tracker=jt, parsl_loaded_flag=pfl,
                logger=MagicMock(), reason="walltime", cleanup_parsl=False,
                restart_ctx=ctx,
            )
        assert database.get_jobs_by_ids(db, [jid])[0]["status"] == "Failed"
        m_sub.assert_not_called()

    def test_signal_path_always_killed_even_with_restart_ctx(self, db, tmp_path):
        """External SIGTERM must NOT trigger restart-mode resubmit."""
        jid = _insert_job(db, status="Running")
        sb, jt, pfl = _shutdown_fixtures(db, [jid])
        sb.db_path = db
        ctx = {
            "max_restarts": 3, "run_dir": tmp_path,
            "system_config": SimpleNamespace(GPUS_PER_NODE=4),
            "db_path": db, "app_filter": None, "tag_filter": None,
        }
        with patch(
            "parslbox.commands.helpers.restart_helpers.subprocess.run"
        ) as m_sub:
            perform_shutdown(
                status_buffer=sb, job_tracker=jt, parsl_loaded_flag=pfl,
                logger=MagicMock(), reason="signal SIGTERM", cleanup_parsl=False,
                restart_ctx=ctx,
            )
        assert database.get_jobs_by_ids(db, [jid])[0]["status"] == "Killed"
        m_sub.assert_not_called()
