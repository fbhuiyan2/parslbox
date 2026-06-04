"""
Phase 2 + 3 tests for `--restart` / `--max-restarts`.

Phase 2 covers the validation contract at every layer (submit_job core,
CLI, API, MCP schemas).

Phase 3 covers script generation: the `pbx run` line in submit.sh gets
`--restart-mode --max-restarts N`, and a parallel `restart_template.sh`
file is written alongside with resource placeholders + header comment.
"""

import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from parslbox.commands.helpers.submit_helpers import submit_job, ValidationError


# ============================================================
# submit_job core validation
# ============================================================


class TestSubmitJobRestartValidation:
    """The deepest shared layer must reject inconsistent restart args."""

    def test_restart_without_max_restarts_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="requires --max-restarts"):
            submit_job(
                config_name="polaris",
                job_name="x", queue="q", select="1", walltime=10,
                run_dir=tmp_path,
                restart=True,
                max_restarts=None,
                validate_runnable=False,
            )

    def test_max_restarts_without_restart_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="only applies when --restart"):
            submit_job(
                config_name="polaris",
                job_name="x", queue="q", select="1", walltime=10,
                run_dir=tmp_path,
                restart=False,
                max_restarts=3,
                validate_runnable=False,
            )

    def test_negative_max_restarts_raises(self, tmp_path):
        with pytest.raises(ValidationError, match=">= 0"):
            submit_job(
                config_name="polaris",
                job_name="x", queue="q", select="1", walltime=10,
                run_dir=tmp_path,
                restart=True,
                max_restarts=-1,
                validate_runnable=False,
            )

    def test_zero_max_restarts_passes_validation(self, tmp_path):
        """max_restarts=0 is a valid (degenerate: just-one-run-no-resubmit) combo."""
        # Will still fail later (no config), but our restart-validation must pass.
        with pytest.raises(ValidationError) as exc:
            submit_job(
                config_name="polaris",
                job_name="x", queue="q", select="1", walltime=10,
                run_dir=tmp_path,
                restart=True,
                max_restarts=0,
                validate_runnable=False,
            )
        # The failure should not be about restart/max-restarts.
        msg = str(exc.value).lower()
        assert "restart" not in msg or "configuration" in msg or "yaml" in msg


# ============================================================
# CLI flag propagation (typer level)
# ============================================================


class TestCliRestartFlags:
    """Both qsub and sbatch CLIs should reject inconsistent flag pairs."""

    def test_pbx_qsub_restart_without_max_restarts_errors(self):
        from parslbox.commands.qsub import app as qsub_app
        runner = CliRunner()
        result = runner.invoke(qsub_app, [
            "--config", "polaris",
            "--job-name", "x",
            "--queue", "q",
            "--select", "1",
            "--walltime", "10",
            "--restart",
        ])
        assert result.exit_code != 0
        # Error message text comes from submit_job ValidationError surfaced by CLI
        assert "max-restarts" in result.output

    def test_pbx_qsub_max_restarts_without_restart_errors(self):
        from parslbox.commands.qsub import app as qsub_app
        runner = CliRunner()
        result = runner.invoke(qsub_app, [
            "--config", "polaris",
            "--job-name", "x",
            "--queue", "q",
            "--select", "1",
            "--walltime", "10",
            "--max-restarts", "3",
        ])
        assert result.exit_code != 0
        assert "restart" in result.output.lower()

    def test_pbx_sbatch_restart_without_max_restarts_errors(self):
        from parslbox.commands.sbatch import app as sbatch_app
        runner = CliRunner()
        result = runner.invoke(sbatch_app, [
            "--config", "polaris",
            "--job-name", "x",
            "--queue", "q",
            "--select", "1",
            "--walltime", "10",
            "--restart",
        ])
        assert result.exit_code != 0
        assert "max-restarts" in result.output

    def test_pbx_sbatch_max_restarts_without_restart_errors(self):
        from parslbox.commands.sbatch import app as sbatch_app
        runner = CliRunner()
        result = runner.invoke(sbatch_app, [
            "--config", "polaris",
            "--job-name", "x",
            "--queue", "q",
            "--select", "1",
            "--walltime", "10",
            "--max-restarts", "3",
        ])
        assert result.exit_code != 0
        assert "restart" in result.output.lower()


# ============================================================
# API validation
# ============================================================


class TestApiRestartValidation:
    """ParslBox.qsub/sbatch should raise ValidationError on inconsistent args."""

    def test_api_qsub_restart_without_max_restarts(self, tmp_path):
        from parslbox.api import ParslBox, ValidationError as ApiValidationError
        # Need a real-ish config file
        cfg = tmp_path / "config.yaml"
        cfg.write_text("polaris: {}\n")
        pbx = ParslBox(db_path=tmp_path / "db.sqlite", config_path=cfg)
        with pytest.raises(ApiValidationError, match="max-restarts"):
            pbx.qsub(
                config="polaris", job_name="x", queue="q",
                select="1", walltime=10,
                restart=True,
            )

    def test_api_sbatch_max_restarts_without_restart(self, tmp_path):
        from parslbox.api import ParslBox, ValidationError as ApiValidationError
        cfg = tmp_path / "config.yaml"
        cfg.write_text("polaris: {}\n")
        pbx = ParslBox(db_path=tmp_path / "db.sqlite", config_path=cfg)
        with pytest.raises(ApiValidationError, match="restart"):
            pbx.sbatch(
                config="polaris", job_name="x", queue="q",
                select="1", walltime=10,
                max_restarts=2,
            )


# ============================================================
# MCP schemas surface the new fields
# ============================================================


class TestMcpSchemasRestartFields:
    """LLM callers see `restart` and `max_restarts` in both submit schemas."""

    def test_qsub_schema_has_restart_field(self):
        from parslbox.mcp.schemas import QSubSchema
        assert "restart" in QSubSchema.model_fields
        assert "max_restarts" in QSubSchema.model_fields
        assert QSubSchema.model_fields["restart"].default is False
        assert QSubSchema.model_fields["max_restarts"].default is None

    def test_qsub_restart_description_mentions_chain(self):
        from parslbox.mcp.schemas import QSubSchema
        desc = QSubSchema.model_fields["restart"].description
        assert "restart" in desc.lower()
        assert "max_restarts" in desc.lower()

    def test_sbatch_schema_has_restart_field(self):
        from parslbox.mcp.schemas import SBatchSchema
        assert "restart" in SBatchSchema.model_fields
        assert "max_restarts" in SBatchSchema.model_fields
        assert SBatchSchema.model_fields["restart"].default is False
        assert SBatchSchema.model_fields["max_restarts"].default is None

    def test_sbatch_restart_description_mentions_chain(self):
        from parslbox.mcp.schemas import SBatchSchema
        desc = SBatchSchema.model_fields["restart"].description
        assert "restart" in desc.lower()
        assert "max_restarts" in desc.lower()


# ============================================================
# Phase 3: script generation (submit.sh + restart_template.sh)
# ============================================================


def _minimal_config(scheduler_type="pbs", system_name="polaris"):
    """Build a minimal config dict for submit_job script-generation tests."""
    if scheduler_type == "pbs":
        template = (
            "#!/bin/bash\n"
            "#PBS -N {job_name}\n"
            "#PBS -q {queue}\n"
            "#PBS -l select={select}\n"
            "#PBS -l walltime={walltime}\n"
            "#PBS -A {project}\n"
            "{sched_opts}\n"
            "\n"
            "{pbx_python_env_setup}\n"
            "{pbx_env_vars}\n"
            "exec pbx run --config {config} --run-dir {run_dir} {run_options}\n"
        )
    else:
        template = (
            "#!/bin/bash\n"
            "#SBATCH --job-name={job_name}\n"
            "#SBATCH --partition={queue}\n"
            "#SBATCH --nodes={select}\n"
            "#SBATCH --time={walltime}\n"
            "#SBATCH --account={project}\n"
            "{sched_opts}\n"
            "\n"
            "{pbx_python_env_setup}\n"
            "{pbx_env_vars}\n"
            "exec pbx run --config {config} --run-dir {run_dir} {run_options}\n"
        )
    return {
        "schedulers": {scheduler_type: {"template": template}},
        system_name: {"pbx_python_env_setup": "module load conda"},
    }


def _run_submit(tmp_path, scheduler_type, **kwargs):
    """Invoke submit_job with mocked subprocess + load_config."""
    config = _minimal_config(scheduler_type)
    sched_cmd = "qsub" if scheduler_type == "pbs" else "sbatch"
    with patch(
        "parslbox.commands.helpers.submit_helpers.load_config",
        return_value=config,
    ), patch(
        "parslbox.commands.helpers.submit_helpers.subprocess.run"
    ) as mock_run:
        mock_run.return_value = MagicMock(stdout="12345.x\n", returncode=0)
        return submit_job(
            config_name="polaris",
            job_name="testjob",
            queue="prod",
            select="4",
            walltime=60,
            project="proj",
            run_dir=tmp_path,
            scheduler_type=scheduler_type,
            submit_command=sched_cmd,
            validate_runnable=False,
            **kwargs,
        )


class TestRestartScriptGeneration:
    """submit.sh embeds --restart-mode; restart_template.sh is written with placeholders."""

    # --- non-restart path: nothing about restart should appear ---

    def test_no_restart_no_template_file(self, tmp_path):
        result = _run_submit(tmp_path, "pbs")
        assert result.get("restart_template_file") is None
        assert not (tmp_path / "restart_template.sh").exists()
        script = (tmp_path / "submit.sh").read_text()
        assert "--restart-mode" not in script
        assert "--max-restarts" not in script

    # --- PBS restart path ---

    def test_pbs_submit_sh_has_restart_flags(self, tmp_path):
        result = _run_submit(tmp_path, "pbs", restart=True, max_restarts=3)
        assert result["restart_template_file"] == str(tmp_path / "restart_template.sh")
        script = (tmp_path / "submit.sh").read_text()
        # Resource line uses the original user value
        assert "select=4" in script
        # pbx run line carries restart-mode + max-restarts
        assert "--restart-mode" in script
        assert "--max-restarts 3" in script

    def test_pbs_restart_template_has_placeholder_and_header(self, tmp_path):
        _run_submit(tmp_path, "pbs", restart=True, max_restarts=2)
        template = (tmp_path / "restart_template.sh").read_text()
        # Header comment present
        assert "PBX RESTART TEMPLATE" in template
        assert "NEVER overwritten by pbx" in template
        assert "DO NOT EDIT" in template
        # Resource placeholder replaces the literal value
        assert "<<PBX_AUTO_SELECT>>" in template
        assert "select=4" not in template
        # pbx run line still has restart-mode + literal initial max-restarts
        assert "--restart-mode" in template
        assert "--max-restarts 2" in template
        # Shebang stayed at line 1
        assert template.startswith("#!/bin/bash")

    # --- SLURM restart path ---

    def test_slurm_submit_sh_has_restart_flags(self, tmp_path):
        result = _run_submit(tmp_path, "slurm", restart=True, max_restarts=5)
        assert result["restart_template_file"] == str(tmp_path / "restart_template.sh")
        script = (tmp_path / "submit.sh").read_text()
        assert "--nodes=4" in script
        assert "--restart-mode" in script
        assert "--max-restarts 5" in script

    def test_slurm_restart_template_uses_nodes_placeholder(self, tmp_path):
        _run_submit(tmp_path, "slurm", restart=True, max_restarts=1)
        template = (tmp_path / "restart_template.sh").read_text()
        assert "<<PBX_AUTO_NODES>>" in template
        assert "--nodes=4" not in template
        assert "--restart-mode" in template
        assert "--max-restarts 1" in template

    # --- zero max-restarts is a valid (degenerate) case ---

    def test_max_restarts_zero_still_generates_template(self, tmp_path):
        _run_submit(tmp_path, "pbs", restart=True, max_restarts=0)
        assert (tmp_path / "restart_template.sh").exists()
        script = (tmp_path / "submit.sh").read_text()
        assert "--max-restarts 0" in script
