"""
Phase 4 tests: orchestrator restart-mode behavior.

Covers:
- The three-scene `apply_restart_hook` contract (dict / None / raise).
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
# Regression: apply_restart_hook must NOT be gated on the respawn flag
#
# A user may manually mark a job Restart (via `pbx update --status Restart`)
# in a non-respawn run — e.g., on clusters that don't allow qsub from
# compute nodes, when debugging interactively, or when one job dies while
# the rest of the run continues. The app's restart() hook should still fire
# so checkpoint-resume logic works in that scenario. Previously gated by
# `if restart_mode:` in run.py; now must be unconditional, regardless of
# whether --respawn is set.
# ============================================================


class TestRestartingJobIdsPlumbing:
    """Source-level guards for the restart-continuation tracking that drives
    per-job stdout/stderr file mode in create_parsl_future.

    The set is populated by apply_restart_hook calls (startup + dynamic
    discovery) with patched + rerun bucket IDs, threaded into
    create_parsl_future, and cleared on terminal status. Each piece is
    asserted here so silent removal of any one of them shows up in CI.
    """

    def test_create_parsl_future_accepts_restarting_job_ids_param(self):
        """Signature must include the parameter so all 3 call sites in run.py
        can pass the set through."""
        import inspect
        from parslbox.commands.run import create_parsl_future
        sig = inspect.signature(create_parsl_future)
        assert "restarting_job_ids" in sig.parameters, (
            "create_parsl_future is missing the `restarting_job_ids` "
            "parameter — file-mode logic needs it to decide 'a' vs 'w'."
        )

    def test_create_parsl_future_uses_choose_output_mode(self):
        """The body must call `choose_output_mode` to set the stdout/stderr
        mode, not hardcode 'w' as before."""
        import inspect
        from parslbox.commands.run import create_parsl_future
        src = inspect.getsource(create_parsl_future)
        assert "choose_output_mode(" in src, (
            "create_parsl_future no longer calls choose_output_mode — "
            "per-job restart-continuation append mode would be silently broken."
        )

    def test_run_py_discards_from_set_on_terminal_status(self):
        """The result-handling loop must clear set entries on Done/Failed/
        Warning so a Failed→Ready re-discovery (same pbx run invocation)
        opens in 'w' mode instead of stale 'a'+banner."""
        import re
        from pathlib import Path
        import parslbox.commands.run as run_mod
        src = Path(run_mod.__file__).read_text()
        # Look for restarting_job_ids.discard(...) call somewhere in the file.
        assert re.search(r"restarting_job_ids\.discard\s*\(", src), (
            "run.py never calls restarting_job_ids.discard — stale set "
            "entries would survive across same-invocation re-dispatches."
        )


class TestApplyRestartHookOnDynamicDiscovery:
    def test_discover_new_jobs_invokes_apply_restart_hook(self):
        """Source-level guard: `discover_new_jobs` must call `apply_restart_hook`
        on the Restart-status subset of newly-discovered jobs.

        Rationale: a job that goes Done/Failed/Killed/Ready → Restart during a
        running pbx run invocation (e.g., via `pbx update --status Restart`
        from another terminal) gets picked up by the discovery polling. Without
        this call, those jobs would dispatch with status='Restart' but their
        app's `restart()` hook would never fire — defeating checkpoint-resume
        logic. Closes the only mid-run gap left by the startup-time hook.
        """
        import re
        from pathlib import Path
        import parslbox.commands.run as run_mod

        src = Path(run_mod.__file__).read_text()
        lines = src.splitlines()
        # Find the `def discover_new_jobs(` line, then scan its body (until
        # dedent to <= def's indent column) for an apply_restart_hook( call.
        def_idx = None
        for i, line in enumerate(lines):
            if re.match(r"\s*def\s+discover_new_jobs\s*\(", line):
                def_idx = i
                break
        assert def_idx is not None, (
            "discover_new_jobs definition not found in run.py — test stale"
        )
        def_col = len(lines[def_idx]) - len(lines[def_idx].lstrip())
        found = False
        for j in range(def_idx + 1, len(lines)):
            ln = lines[j]
            if ln.strip() and (len(ln) - len(ln.lstrip())) <= def_col:
                break  # left the function body
            if re.search(r"\bapply_restart_hook\s*\(", ln):
                found = True
                break
        assert found, (
            "discover_new_jobs does not invoke apply_restart_hook(). "
            "Dynamically-discovered Restart jobs must go through the "
            "per-job restart() hook just like startup-time Restart jobs. "
            "See test docstring for rationale."
        )


class TestRestartHookDecoupledFromRespawn:
    def test_run_py_calls_apply_restart_hook_unconditionally(self):
        """Source-level guard: `apply_restart_hook` must NOT be inside any
        `if respawn ...:` / `if restart_mode:` block in run.py. The hook
        fires for every Restart-status job, regardless of orchestrator mode.

        Pattern-based test: cheaper than spinning up the full typer command
        and locks in the contract against silent re-gating.
        """
        import re
        from pathlib import Path
        import parslbox.commands.run as run_mod

        src = Path(run_mod.__file__).read_text()
        lines = src.splitlines()
        # Match any call to apply_restart_hook( on a line — including the
        # `buckets = apply_restart_hook(...)` capture form added when the
        # restart-continuation set was introduced.
        hook_call_lines = [
            i for i, line in enumerate(lines)
            if re.search(r"\bapply_restart_hook\s*\(", line)
        ]
        assert hook_call_lines, (
            "apply_restart_hook call site not found in run.py — test stale"
        )
        # Match `if respawn ...:` (any expression involving the respawn var)
        # and also the legacy `if restart_mode:` form.
        gate_re = re.compile(r"\s*if\s+(respawn|restart_mode)\b")
        for hook_line in hook_call_lines:
            window = lines[max(0, hook_line - 40):hook_line]
            gating = [ln for ln in window if gate_re.match(ln)]
            assert not gating, (
                f"apply_restart_hook at line {hook_line + 1} is gated by "
                f"{gating[0].strip()!r}. The hook must fire for any "
                f"Restart-status job, regardless of --respawn. See test "
                f"docstring for rationale."
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
