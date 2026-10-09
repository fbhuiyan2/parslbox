"""Tests for the per-app resolution summary printed at qsub/sbatch time.

Covers the helper that resolves each matched app against the target system's
config block, the three env states it distinguishes, and how the CLI, the
result dict, and MCP surface it.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from parslbox.database import database
from parslbox.commands.helpers.submit_helpers import (
    count_matching_runnable_jobs,
    summarize_app_configs,
    submit_job,
)


def _make_pbs_config(system_name="aurora-tile", extra=None):
    cfg = {
        "schedulers": {"pbs": {"template": (
            "#!/bin/bash\n#PBS -N {job_name}\n#PBS -q {queue}\n"
            "#PBS -l select={select}\n#PBS -l walltime={walltime}\n"
            "#PBS -A {project}\n{sched_opts}\n\n"
            "{pbx_python_env_setup}\n{pbx_env_vars}\n"
            "pbx run --config {config} --run-dir {run_dir} {run_options}\n"
        )}},
        system_name: {"pbx_python_env_setup": "module load conda"},
    }
    if extra:
        cfg.update(extra)
    return cfg


APP_BLOCKS = {
    "lammps-kk": {
        "aurora-tile": {
            "executable_path": "/soft/lammps/lmp",
            "environment_setup": "module load kokkos",
        }
    },
    # registered, per-system block exists for a *different* system
    "lammps-kk-kappa": {
        "polaris": {"executable_path": "/soft/lammps/lmp-kappa"},
    },
    # block exists for the target system but names no executable
    "python": {
        "aurora-tile": {"environment_setup": "module load frameworks"},
    },
}


@pytest.fixture
def temp_db():
    d = tempfile.mkdtemp()
    yield Path(d) / "test.db"
    shutil.rmtree(d)


@pytest.fixture
def db_with_apps(temp_db):
    """Ready jobs across three apps, plus a Done job that must not count."""
    database.initialize_database(temp_db)
    seed = [
        ("lammps-kk", "Ready"),
        ("lammps-kk", "Ready"),
        ("lammps-kk-kappa", "Ready"),
        ("python", "Restart"),
        ("vasp", "Done"),
    ]
    for i, (app, status) in enumerate(seed):
        database.add_job(
            db_path=temp_db, path=f"/p{i}", app=app,
            num_nodes=1, ngpus=0, node_occupancy=1.0,
            tag="run-a", in_file=None, status=status,
        )
    return temp_db


# ============================================================
# count_matching_runnable_jobs() now also reports the apps
# ============================================================


class TestCountMatchingRunnableJobs:
    def test_returns_count_and_sorted_apps(self, db_with_apps):
        count, apps = count_matching_runnable_jobs(db_with_apps, None, None)
        assert count == 4
        assert apps == ["lammps-kk", "lammps-kk-kappa", "python"]

    def test_done_jobs_excluded_from_apps(self, db_with_apps):
        _, apps = count_matching_runnable_jobs(db_with_apps, None, None)
        assert "vasp" not in apps

    def test_apps_filter_narrows_the_list(self, db_with_apps):
        count, apps = count_matching_runnable_jobs(
            db_with_apps, ["lammps-kk"], None
        )
        assert count == 2
        assert apps == ["lammps-kk"]

    def test_no_matches_gives_empty_app_list(self, db_with_apps):
        count, apps = count_matching_runnable_jobs(db_with_apps, ["orca"], None)
        assert count == 0
        assert apps == []


# ============================================================
# summarize_app_configs() — the three env states
# ============================================================


class TestSummarizeAppConfigs:
    def _one(self, app, config_name="aurora-tile"):
        cfg = _make_pbs_config(extra=APP_BLOCKS)
        out = summarize_app_configs([app], config_name, cfg, Path("/cfg/config.yaml"))
        assert len(out) == 1
        return out[0]

    def test_configured_reports_executable(self):
        e = self._one("lammps-kk")
        assert e["env_state"] == "configured"
        assert e["executable"] == "/soft/lammps/lmp"
        assert e["detail"] == "configured for aurora-tile"

    def test_app_block_present_but_wrong_system(self):
        """The case that looks like success: right app, wrong system key."""
        e = self._one("lammps-kk-kappa")
        assert e["env_state"] == "no_system_entry"
        assert e["executable"] is None
        assert e["detail"] == "no `aurora-tile` entry under `lammps-kk-kappa`"

    def test_no_app_block_at_all(self):
        e = self._one("lammps-custom")
        assert e["env_state"] == "no_app_block"
        assert e["executable"] is None
        assert "no `lammps-custom` block" in e["detail"]
        assert "/cfg/config.yaml" in e["detail"]

    def test_configured_but_no_executable_is_still_configured(self):
        """python/julia carry their own default, so this is normal for them."""
        e = self._one("python")
        assert e["env_state"] == "configured"
        assert e["executable"] is None

    def test_preserves_input_order(self):
        cfg = _make_pbs_config(extra=APP_BLOCKS)
        out = summarize_app_configs(
            ["python", "lammps-kk"], "aurora-tile", cfg, Path("/cfg/config.yaml")
        )
        assert [e["app"] for e in out] == ["python", "lammps-kk"]

    def test_empty_app_list(self):
        assert summarize_app_configs([], "aurora-tile", {}, Path("/c.yaml")) == []

    def test_missing_config_is_not_a_crash(self):
        out = summarize_app_configs(
            ["lammps-kk"], "aurora-tile", None, Path("/c.yaml")
        )
        assert out[0]["env_state"] == "no_app_block"

    def test_non_dict_app_block_treated_as_missing(self):
        cfg = {"lammps-kk": "oops-a-string"}
        out = summarize_app_configs(
            ["lammps-kk"], "aurora-tile", cfg, Path("/c.yaml")
        )
        assert out[0]["env_state"] == "no_app_block"


# ============================================================
# submit_job() carries app_summary in its result dict
# ============================================================


class TestSubmitJobResultDict:
    def _submit(self, db_path, tmp_path):
        with patch(
            "parslbox.commands.helpers.submit_helpers.load_config",
            return_value=_make_pbs_config(extra=APP_BLOCKS),
        ), patch(
            "parslbox.commands.helpers.submit_helpers.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(stdout="12345.pbs01\n", returncode=0)
            return submit_job(
                config_name="aurora-tile", job_name="t", queue="prod",
                select="4", walltime=60, project="proj", run_dir=tmp_path,
                scheduler_type="pbs", submit_command="qsub",
                db_path=db_path, no_local=True,
            )

    def test_summary_present_for_every_matched_app(self, db_with_apps, tmp_path):
        result = self._submit(db_with_apps, tmp_path)
        assert result["success"] is True
        states = {e["app"]: e["env_state"] for e in result["app_summary"]}
        assert states == {
            "lammps-kk": "configured",
            "lammps-kk-kappa": "no_system_entry",
            "python": "configured",
        }

    def test_summary_is_empty_when_validation_skipped(self, tmp_path):
        """validate_runnable=False is the test-only path: no DB, so no apps."""
        with patch(
            "parslbox.commands.helpers.submit_helpers.load_config",
            return_value=_make_pbs_config(extra=APP_BLOCKS),
        ), patch(
            "parslbox.commands.helpers.submit_helpers.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(stdout="1.pbs01\n", returncode=0)
            result = submit_job(
                config_name="aurora-tile", job_name="t", queue="prod",
                select="1", walltime=10, project="proj", run_dir=tmp_path,
                scheduler_type="pbs", submit_command="qsub",
                validate_runnable=False,
            )
        assert result["app_summary"] == []


# ============================================================
# CLI rendering
# ============================================================


@pytest.fixture
def qsub_runner(db_with_apps, tmp_path):
    from parslbox.commands import qsub as qsub_mod
    runner = CliRunner()
    with patch("parslbox.commands.helpers.submit_helpers.load_config",
               return_value=_make_pbs_config(extra=APP_BLOCKS)), \
         patch("parslbox.commands.helpers.submit_helpers.subprocess.run") as mock_run, \
         patch("parslbox.commands.helpers.submit_helpers.get_default_run_dir",
               return_value=tmp_path / "fake_run"), \
         patch("parslbox.utils.path_utils.DB_FILE", db_with_apps):
        mock_run.return_value = MagicMock(stdout="12345.pbs01\n", returncode=0)
        yield runner, qsub_mod.app


QSUB_ARGS = [
    "--config", "aurora-tile", "--job-name", "t", "--queue", "prod",
    "--select", "4", "--walltime", "60", "--project", "proj",
    "--no-local",
]


class TestQsubCliOutput:
    def _run(self, qsub_runner):
        runner, app = qsub_runner
        r = runner.invoke(app, QSUB_ARGS, env={"COLUMNS": "400"})
        assert r.exit_code == 0, r.output
        return r.output

    def test_lists_each_app(self, qsub_runner):
        out = self._run(qsub_runner)
        assert "Apps in this submission (aurora-tile)" in out
        assert "lammps-kk" in out
        assert "lammps-kk-kappa" in out
        assert "python" in out

    def test_shows_resolved_executable(self, qsub_runner):
        assert "/soft/lammps/lmp" in self._run(qsub_runner)

    def test_flags_missing_system_entry(self, qsub_runner):
        out = self._run(qsub_runner)
        assert "no `aurora-tile` entry under `lammps-kk-kappa`" in out

    def test_unset_executable_is_called_out(self, qsub_runner):
        out = self._run(qsub_runner)
        assert "(not set)" in out
        assert "make sure this is intended" in out

    def test_cancel_hint_offers_both_commands(self, qsub_runner):
        out = self._run(qsub_runner)
        assert "qdel 12345.pbs01" in out
        assert "pbx qdel 12345.pbs01" in out
        assert "if the job already started" in out


# ============================================================
# MCP rendering
# ============================================================


class TestMcpAppSummaryLines:
    def _lines(self, status, cancel="qdel"):
        from parslbox.mcp.mcp_server import _app_summary_lines
        return _app_summary_lines(status, cancel)

    def test_compact_one_line_for_all_apps(self):
        status = {
            "job_id": "12345.pbs01",
            "app_summary": [
                {"app": "lammps-kk", "executable": "/soft/lmp",
                 "env_state": "configured"},
                {"app": "lammps-kk-kappa", "executable": None,
                 "env_state": "no_system_entry"},
            ],
        }
        lines = self._lines(status)
        assert lines[0] == (
            "apps: lammps-kk [exe:/soft/lmp, env:configured] | "
            "lammps-kk-kappa [exe:NOT SET, env:NO SYSTEM ENTRY]"
        )

    def test_no_app_block_state(self):
        status = {
            "job_id": "1",
            "app_summary": [
                {"app": "x", "executable": None, "env_state": "no_app_block"},
            ],
        }
        assert "env:NO APP BLOCK" in self._lines(status)[0]

    def test_cancel_hint_present(self):
        status = {
            "job_id": "12345.pbs01",
            "app_summary": [
                {"app": "x", "executable": "/e", "env_state": "configured"},
            ],
        }
        hint = self._lines(status)[1]
        assert "qdel 12345.pbs01" in hint
        assert "pbx qdel 12345.pbs01" in hint

    def test_scancel_variant(self):
        status = {
            "job_id": "999",
            "app_summary": [
                {"app": "x", "executable": "/e", "env_state": "configured"},
            ],
        }
        assert "pbx scancel 999" in self._lines(status, "scancel")[1]

    def test_remote_says_check_the_remote_config(self):
        status = {"job_id": "7", "remote": True}
        lines = self._lines(status)
        assert any("remote host" in ln for ln in lines)
        assert any("config.yaml" in ln for ln in lines)
        assert not any(ln.startswith("apps:") for ln in lines)

    def test_no_summary_yields_no_lines(self):
        assert self._lines({"job_id": "7"}) == []
