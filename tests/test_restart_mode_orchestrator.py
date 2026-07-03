"""
Phase 4 tests: orchestrator restart-mode behavior.

Covers:
- The three-scene `apply_restart_for_job` contract (dict / None / raise) +
  the strict no-resource-fields rule (returning ngpus/num_nodes/etc. fails
  the job with a clear error).
- restart_helpers utilities (placeholder substitution, respawn-count decrement,
  template validation, original-cap extraction, link script generation,
  resubmit subprocess wrapper).
- `compute_required_nodes` extracted helper (Phase 5).
- `perform_shutdown` branching: walltime + respawn-on (Restart vs Failed),
  walltime + respawn-off (Killed), signal path (always Killed regardless of
  respawn_ctx).
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
# Scene A/B/C — apply_restart_for_job (per-job, lazy)
#
# The lazy per-job helper returns (bucket, patch, err) and does NOT touch
# the DB — the caller (create_parsl_future) is responsible for the buffered
# Running/Failed write. This matches the new "one DB call per dispatch"
# contract: patches ride along on the Running buffer write; failures get
# a Failed buffer write.
# ============================================================


def _job(**kwargs):
    """Build a minimal job dict for apply_restart_for_job tests."""
    defaults = {"job_id": 1, "app": "lammps-kk", "path": "/x", "in_file": "in"}
    defaults.update(kwargs)
    return defaults


class TestApplyRestartForJob:
    def test_scene_a_dict_returns_patched_bucket(self):
        app = MagicMock()
        app.restart.return_value = {"in_file": "in.restart"}
        bucket, patch, err = restart_helpers.apply_restart_for_job(
            _job(in_file="in.original"), app, MagicMock()
        )
        assert bucket == "patched"
        assert patch == {"in_file": "in.restart"}
        assert err is None

    def test_scene_b_none_returns_rerun_bucket(self):
        app = MagicMock()
        app.restart.return_value = None
        bucket, patch, err = restart_helpers.apply_restart_for_job(
            _job(), app, MagicMock()
        )
        assert bucket == "rerun"
        assert patch is None
        assert err is None

    def test_scene_b_empty_dict_also_rerun(self):
        app = MagicMock()
        app.restart.return_value = {}
        bucket, _, _ = restart_helpers.apply_restart_for_job(
            _job(), app, MagicMock()
        )
        assert bucket == "rerun"

    def test_scene_c_notimplemented_returns_failed_with_reason(self):
        app = MagicMock()
        app.restart.side_effect = NotImplementedError("no restart support")
        bucket, patch, err = restart_helpers.apply_restart_for_job(
            _job(), app, MagicMock()
        )
        assert bucket == "failed"
        assert patch is None
        assert "does not support restart" in err

    def test_generic_exception_returns_failed_with_class_name(self):
        app = MagicMock()
        app.restart.side_effect = RuntimeError("disk full")
        bucket, _, err = restart_helpers.apply_restart_for_job(
            _job(), app, MagicMock()
        )
        assert bucket == "failed"
        assert "RuntimeError" in err and "disk full" in err

    def test_missing_app_returns_failed(self):
        bucket, _, err = restart_helpers.apply_restart_for_job(
            _job(app="missing"), None, MagicMock()
        )
        assert bucket == "failed"
        assert "missing" in err and "not loaded" in err

    def test_unknown_non_resource_field_warns_and_drops(self):
        app = MagicMock()
        app.restart.return_value = {"in_file": "ok", "bogus_field": 42}
        logger = MagicMock()
        bucket, patch, err = restart_helpers.apply_restart_for_job(
            _job(), app, logger
        )
        assert bucket == "patched"
        assert patch == {"in_file": "ok"}  # bogus dropped
        # Warning emitted about the unknown field.
        warn_calls = [c for c in logger.warning.call_args_list if "bogus_field" in str(c)]
        assert warn_calls, "expected a warning about the unknown field"

    def test_only_unknown_fields_collapses_to_rerun(self):
        """If every returned key is unknown (none are patchable), there's
        nothing to apply — treat as a no-op rerun rather than an empty patch."""
        app = MagicMock()
        app.restart.return_value = {"bogus_field": 42}
        bucket, patch, err = restart_helpers.apply_restart_for_job(
            _job(), app, MagicMock()
        )
        assert bucket == "rerun"
        assert patch is None


class TestApplyRestartForJobForbidsResourceFields:
    """Strict contract: restart() may not change resource fields. Resources
    are allocated BEFORE restart() runs in the lazy design, so patching
    them after would be silently ignored. Fail loud instead."""

    @pytest.mark.parametrize("forbidden", [
        "ngpus", "num_nodes", "node_occupancy", "ranks_per_node", "mpi_opts",
    ])
    def test_returning_any_resource_field_is_a_contract_violation(self, forbidden):
        app = MagicMock()
        app.restart.return_value = {"in_file": "ok", forbidden: 7}
        bucket, patch, err = restart_helpers.apply_restart_for_job(
            _job(), app, MagicMock()
        )
        assert bucket == "failed"
        assert patch is None
        assert "forbidden" in err
        assert forbidden in err

    def test_multiple_forbidden_fields_all_reported(self):
        app = MagicMock()
        app.restart.return_value = {"ngpus": 4, "num_nodes": 2}
        bucket, _, err = restart_helpers.apply_restart_for_job(
            _job(), app, MagicMock()
        )
        assert bucket == "failed"
        assert "ngpus" in err and "num_nodes" in err


# ============================================================
# Source-pattern guards: lazy restart wiring must stay intact
# ============================================================


class TestRestartingJobIdsPlumbing:
    """Source-level guards for the restart-continuation tracking that drives
    per-job stdout/stderr file mode in create_parsl_future.

    The set is populated by apply_restart_hook calls (startup + dynamic
    discovery) with patched + rerun bucket IDs, threaded into
    create_parsl_future, and cleared on terminal status. Each piece is
    asserted here so silent removal of any one of them shows up in CI.
    """

    def test_create_parsl_future_uses_restarting_job_ids(self):
        """create_parsl_future (now in schedule_helpers, taking a ctx) must
        still drive file-mode / restart() logic off restarting_job_ids (via
        ctx) and expose an `is_restart` selector."""
        import inspect
        from parslbox.commands.helpers.schedule_helpers import create_parsl_future
        sig = inspect.signature(create_parsl_future)
        assert "is_restart" in sig.parameters, (
            "create_parsl_future is missing the `is_restart` parameter — "
            "restart() handling and 'a' vs 'w' file mode need it."
        )
        src = inspect.getsource(create_parsl_future)
        assert "restarting_job_ids" in src, (
            "create_parsl_future no longer references restarting_job_ids — "
            "restart-continuation append mode would be silently broken."
        )

    def test_create_parsl_future_uses_choose_output_mode(self):
        """The body must call `choose_output_mode` to set the stdout/stderr
        mode, not hardcode 'w' as before."""
        import inspect
        from parslbox.commands.helpers.schedule_helpers import create_parsl_future
        src = inspect.getsource(create_parsl_future)
        assert "choose_output_mode(" in src, (
            "create_parsl_future no longer calls choose_output_mode — "
            "per-job restart-continuation append mode would be silently broken."
        )

    def test_discards_from_set_on_terminal_status(self):
        """The result-handling path must clear set entries on Done/Failed/
        Warning so a Failed→Ready re-discovery (same pbx run invocation)
        opens in 'w' mode instead of stale 'a'+banner."""
        import re
        from pathlib import Path
        import parslbox.commands.helpers.schedule_helpers as sched_mod
        src = Path(sched_mod.__file__).read_text()
        assert re.search(r"restarting_job_ids\.discard\s*\(", src), (
            "schedule_helpers never calls restarting_job_ids.discard — stale "
            "set entries would survive across same-invocation re-dispatches."
        )


class TestLazyRestartWiring:
    """Source-pattern guards locking in the lazy per-job restart design.

    Three invariants:
      1. The bulk `apply_restart_hook` (eager startup + dynamic-discovery
         calls) MUST be gone — any remaining call would restore the eager
         design and reintroduce the wasted-work bug it replaced.
      2. `apply_restart_for_job` must be CALLED inside `create_parsl_future`
         so per-job restart actually fires.
      3. The bulk `apply_restart_hook` function itself must be REMOVED from
         restart_helpers — keeping it around invites future re-introduction.
    """

    def _run_py_source(self) -> str:
        from pathlib import Path
        import parslbox.commands.run as run_mod
        return Path(run_mod.__file__).read_text()

    def test_run_py_does_not_call_eager_apply_restart_hook(self):
        import re
        src = self._run_py_source()
        # Match any callable form: `apply_restart_hook(`. Comments referencing
        # the old name in prose are fine — those don't have `(`.
        callsites = re.findall(r"\bapply_restart_hook\s*\(", src)
        assert not callsites, (
            f"run.py still contains {len(callsites)} eager apply_restart_hook() "
            f"call(s). The lazy design replaces these with per-job "
            f"apply_restart_for_job calls inside create_parsl_future. "
            f"Re-introducing them brings back the wasted-work bug "
            f"(restart() side effects on jobs that get backlogged/gated)."
        )

    def test_create_parsl_future_calls_apply_restart_for_job(self):
        import inspect
        from parslbox.commands.helpers.schedule_helpers import create_parsl_future
        src = inspect.getsource(create_parsl_future)
        assert "apply_restart_for_job(" in src, (
            "create_parsl_future does not call apply_restart_for_job. "
            "Restart-status jobs would dispatch without their app's "
            "restart() hook running — checkpoint-resume would silently break."
        )

    def test_restart_helpers_does_not_export_bulk_apply_restart_hook(self):
        # The bulk function must be deleted, not just unused. Keeping it
        # around invites silent re-introduction.
        assert not hasattr(restart_helpers, "apply_restart_hook"), (
            "restart_helpers.apply_restart_hook still exists. The lazy design "
            "deletes the bulk eager helper entirely — see apply_restart_for_job "
            "for the per-job replacement."
        )


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


class TestDecrementRespawn:
    def test_simple_replace(self):
        out = restart_helpers._decrement_respawn(
            "pbx run --respawn 5 --config polaris", 4
        )
        assert "--respawn 4" in out
        assert "--respawn 5" not in out

    def test_preserves_other_args(self):
        line = "pbx run --respawn 3 --config polaris --apps lammps-kk"
        out = restart_helpers._decrement_respawn(line, 2)
        assert "--apps lammps-kk" in out
        assert "--config polaris" in out


class TestValidatePbxRunLine:
    def _line(self, **overrides):
        defaults = {
            "respawn": 3,
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

    def test_missing_respawn_raises(self):
        # No --respawn at all → not even recognized as a respawn line
        line = "exec pbx run --config polaris --run-dir ./"
        with pytest.raises(ValueError, match="no `pbx run"):
            restart_helpers.validate_pbx_run_line(line)

    def test_missing_config_raises(self):
        line = "exec pbx run --respawn 3 --run-dir ./"
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
# build_respawn_link_script — full integration
# ============================================================


class TestBuildRespawnLinkScript:
    def _make_template(self, run_dir):
        (run_dir / "submit.sh").write_text(
            "#!/bin/bash\n#PBS -l select=4\n#PBS -o pbx_scheduler_link-0.out\n"
            "exec pbx run --respawn 3 --config polaris --run-dir ./\n"
        )
        (run_dir / "respawn_template.sh").write_text(
            "#!/bin/bash\n#PBS -l select=<<PBX_AUTO_SELECT>>\n"
            "#PBS -o pbx_scheduler_link-0.out\n"
            "exec pbx run --respawn 3 --config polaris --run-dir ./\n"
        )

    def test_generates_link_1_with_substitutions(self, tmp_path):
        self._make_template(tmp_path)
        link = restart_helpers.build_respawn_link_script(
            template_path=tmp_path / "respawn_template.sh",
            run_dir=tmp_path,
            current_respawn=3,
            scheduler_type="pbs",
            computed_nodes=2,
            submit_file_path=tmp_path / "submit.sh",
        )
        assert link == tmp_path / "respawn_link_1.sh"
        text = link.read_text()
        # placeholder replaced
        assert "select=2" in text
        assert "<<PBX_AUTO_SELECT>>" not in text
        # counter decremented
        assert "--respawn 2" in text
        assert "--respawn 3" not in text

    def test_increments_link_index_across_calls(self, tmp_path):
        self._make_template(tmp_path)
        for expected_idx in (1, 2, 3):
            link = restart_helpers.build_respawn_link_script(
                template_path=tmp_path / "respawn_template.sh",
                run_dir=tmp_path,
                current_respawn=3,
                scheduler_type="pbs",
                computed_nodes=1,
                submit_file_path=tmp_path / "submit.sh",
            )
            assert link == tmp_path / f"respawn_link_{expected_idx}.sh"

    def test_rolls_scheduler_log_name_per_link(self, tmp_path):
        """Each generated link script gets its own pbx_scheduler_link-<N>.out
        so the scheduler log isn't overwritten cycle to cycle."""
        self._make_template(tmp_path)
        for expected_idx in (1, 2, 3):
            link = restart_helpers.build_respawn_link_script(
                template_path=tmp_path / "respawn_template.sh",
                run_dir=tmp_path,
                current_respawn=3,
                scheduler_type="pbs",
                computed_nodes=1,
                submit_file_path=tmp_path / "submit.sh",
            )
            text = link.read_text()
            assert f"pbx_scheduler_link-{expected_idx}.out" in text
            # The template's link-0 form must have been rewritten.
            assert "pbx_scheduler_link-0.out" not in text

    def test_cap_applied(self, tmp_path):
        self._make_template(tmp_path)
        link = restart_helpers.build_respawn_link_script(
            template_path=tmp_path / "respawn_template.sh",
            run_dir=tmp_path,
            current_respawn=3,
            scheduler_type="pbs",
            computed_nodes=100,  # huge
            submit_file_path=tmp_path / "submit.sh",  # original = 4
        )
        assert "select=4" in link.read_text()

    def test_missing_template_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            restart_helpers.build_respawn_link_script(
                template_path=tmp_path / "absent.sh",
                run_dir=tmp_path,
                current_respawn=2,
                scheduler_type="pbs",
                computed_nodes=1,
                submit_file_path=tmp_path / "submit.sh",
            )

    def test_broken_template_raises(self, tmp_path):
        (tmp_path / "respawn_template.sh").write_text(
            "#!/bin/bash\necho missing run line\n"
        )
        with pytest.raises(ValueError, match="no `pbx run"):
            restart_helpers.build_respawn_link_script(
                template_path=tmp_path / "respawn_template.sh",
                run_dir=tmp_path,
                current_respawn=2,
                scheduler_type="pbs",
                computed_nodes=1,
                submit_file_path=tmp_path / "submit.sh",
            )


# ============================================================
# submit_respawn_link — subprocess success / failure
# ============================================================


class TestSubmitRespawnLink:
    def test_success_logs_ok(self, tmp_path):
        link = tmp_path / "respawn_link_1.sh"
        link.write_text("#!/bin/bash\n")
        logger = MagicMock()
        with patch(
            "parslbox.commands.helpers.restart_helpers.subprocess.run"
        ) as m:
            m.return_value = MagicMock(returncode=0, stdout="12346.x", stderr="")
            ok = restart_helpers.submit_respawn_link(link, "qsub", tmp_path, logger)
        assert ok is True
        logger.info.assert_called()

    def test_nonzero_returncode_returns_false(self, tmp_path):
        link = tmp_path / "respawn_link_1.sh"
        link.write_text("#!/bin/bash\n")
        logger = MagicMock()
        with patch(
            "parslbox.commands.helpers.restart_helpers.subprocess.run"
        ) as m:
            m.return_value = MagicMock(returncode=1, stdout="", stderr="quota exceeded")
            ok = restart_helpers.submit_respawn_link(link, "qsub", tmp_path, logger)
        assert ok is False
        err_text = " ".join(str(c) for c in logger.error.call_args_list)
        assert "quota exceeded" in err_text
        assert "Chain stopped" in err_text

    def test_command_not_found(self, tmp_path):
        link = tmp_path / "respawn_link_1.sh"
        link.write_text("#!/bin/bash\n")
        logger = MagicMock()
        with patch(
            "parslbox.commands.helpers.restart_helpers.subprocess.run",
            side_effect=FileNotFoundError(),
        ):
            ok = restart_helpers.submit_respawn_link(link, "qsub", tmp_path, logger)
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
    def test_walltime_no_respawn_marks_killed(self, db):
        jid = _insert_job(db, status="Running")
        sb, jt, pfl = _shutdown_fixtures(db, [jid])
        perform_shutdown(
            status_buffer=sb, job_tracker=jt, parsl_loaded_flag=pfl,
            logger=MagicMock(), reason="walltime", cleanup_parsl=False,
            respawn_ctx=None,
        )
        assert database.get_jobs_by_ids(db, [jid])[0]["status"] == "Killed"

    def test_walltime_respawn_gt_0_marks_restart_and_resubmits(self, db, tmp_path):
        jid = _insert_job(db, status="Running")
        sb, jt, pfl = _shutdown_fixtures(db, [jid])
        sb.db_path = db
        system_config = SimpleNamespace(GPUS_PER_NODE=4)
        # Need a template + submit for the resubmit step to succeed.
        (tmp_path / "submit.sh").write_text("#PBS -l select=4\n")
        (tmp_path / "respawn_template.sh").write_text(
            "#!/bin/bash\n#PBS -l select=<<PBX_AUTO_SELECT>>\n"
            "exec pbx run --respawn 3 --config polaris --run-dir ./\n"
        )
        ctx = {
            "respawn": 3, "run_dir": tmp_path, "system_config": system_config,
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
                respawn_ctx=ctx,
            )
        assert database.get_jobs_by_ids(db, [jid])[0]["status"] == "Restart"
        # Resubmit was attempted
        m_sub.assert_called_once()
        # Link script was created with decremented counter
        link = tmp_path / "respawn_link_1.sh"
        assert link.exists()
        assert "--respawn 2" in link.read_text()

    def test_walltime_respawn_eq_0_marks_failed_no_resubmit(self, db, tmp_path):
        jid = _insert_job(db, status="Running")
        sb, jt, pfl = _shutdown_fixtures(db, [jid])
        sb.db_path = db
        ctx = {
            "respawn": 0, "run_dir": tmp_path,
            "system_config": SimpleNamespace(GPUS_PER_NODE=4),
            "db_path": db, "app_filter": None, "tag_filter": None,
        }
        with patch(
            "parslbox.commands.helpers.restart_helpers.subprocess.run"
        ) as m_sub:
            perform_shutdown(
                status_buffer=sb, job_tracker=jt, parsl_loaded_flag=pfl,
                logger=MagicMock(), reason="walltime", cleanup_parsl=False,
                respawn_ctx=ctx,
            )
        assert database.get_jobs_by_ids(db, [jid])[0]["status"] == "Failed"
        m_sub.assert_not_called()

    def test_signal_path_always_killed_even_with_respawn_ctx(self, db, tmp_path):
        """External SIGTERM must NOT trigger respawn resubmit."""
        jid = _insert_job(db, status="Running")
        sb, jt, pfl = _shutdown_fixtures(db, [jid])
        sb.db_path = db
        ctx = {
            "respawn": 3, "run_dir": tmp_path,
            "system_config": SimpleNamespace(GPUS_PER_NODE=4),
            "db_path": db, "app_filter": None, "tag_filter": None,
        }
        with patch(
            "parslbox.commands.helpers.restart_helpers.subprocess.run"
        ) as m_sub:
            perform_shutdown(
                status_buffer=sb, job_tracker=jt, parsl_loaded_flag=pfl,
                logger=MagicMock(), reason="signal SIGTERM", cleanup_parsl=False,
                respawn_ctx=ctx,
            )
        assert database.get_jobs_by_ids(db, [jid])[0]["status"] == "Killed"
        m_sub.assert_not_called()
