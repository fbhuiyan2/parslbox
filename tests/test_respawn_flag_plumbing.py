"""
Phase 2 + 3 tests for `--respawn N`.

Phase 2 covers the validation contract at every layer (submit_job core,
CLI, API, MCP schemas).

Phase 3 covers script generation: the `pbx run` line in submit.sh gets
`--respawn N`, and a parallel `respawn_template.sh` file is written
alongside with resource placeholders + header comment.
"""

import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from parslbox.commands.helpers.submit_helpers import submit_job, ValidationError


# ============================================================
# submit_job core validation
# ============================================================


class TestSubmitJobRespawnValidation:
    """The deepest shared layer must reject inconsistent respawn args."""

    def test_negative_respawn_raises(self, tmp_path):
        with pytest.raises(ValidationError, match=">= 0"):
            submit_job(
                config_name="polaris",
                job_name="x", queue="q", select="1", walltime=10,
                run_dir=tmp_path,
                respawn=-1,
                validate_runnable=False,
            )

    def test_zero_respawn_passes_validation(self, tmp_path):
        """respawn=0 is a valid (degenerate: just-one-run-no-resubmit) combo."""
        # Will still fail later (no config), but our respawn-validation must pass.
        with pytest.raises(ValidationError) as exc:
            submit_job(
                config_name="polaris",
                job_name="x", queue="q", select="1", walltime=10,
                run_dir=tmp_path,
                respawn=0,
                validate_runnable=False,
            )
        # The failure should not be about respawn.
        msg = str(exc.value).lower()
        assert "respawn" not in msg or "configuration" in msg or "yaml" in msg


# ============================================================
# CLI flag propagation (typer level)
# ============================================================


class TestCliRespawnFlags:
    """Both qsub and sbatch CLIs should reject negative --respawn."""

    def test_pbx_qsub_negative_respawn_errors(self):
        from parslbox.commands.qsub import app as qsub_app
        runner = CliRunner()
        result = runner.invoke(qsub_app, [
            "--config", "polaris",
            "--job-name", "x",
            "--queue", "q",
            "--select", "1",
            "--walltime", "10",
            "--respawn", "-1",
        ])
        assert result.exit_code != 0
        assert "respawn" in result.output.lower()

    def test_pbx_sbatch_negative_respawn_errors(self):
        from parslbox.commands.sbatch import app as sbatch_app
        runner = CliRunner()
        result = runner.invoke(sbatch_app, [
            "--config", "polaris",
            "--job-name", "x",
            "--queue", "q",
            "--select", "1",
            "--walltime", "10",
            "--respawn", "-1",
        ])
        assert result.exit_code != 0
        assert "respawn" in result.output.lower()


# ============================================================
# API validation
# ============================================================


class TestApiRespawnValidation:
    """ParslBox.qsub/sbatch should raise ValidationError on negative respawn."""

    def test_api_qsub_negative_respawn(self, tmp_path):
        from parslbox.api import ParslBox, ValidationError as ApiValidationError
        # Need a real-ish config file
        cfg = tmp_path / "config.yaml"
        cfg.write_text("polaris: {}\n")
        pbx = ParslBox(db_path=tmp_path / "db.sqlite", config_path=cfg)
        with pytest.raises(ApiValidationError, match="respawn"):
            pbx.qsub(
                config="polaris", job_name="x", queue="q",
                select="1", walltime=10,
                respawn=-1,
            )

    def test_api_sbatch_negative_respawn(self, tmp_path):
        from parslbox.api import ParslBox, ValidationError as ApiValidationError
        cfg = tmp_path / "config.yaml"
        cfg.write_text("polaris: {}\n")
        pbx = ParslBox(db_path=tmp_path / "db.sqlite", config_path=cfg)
        with pytest.raises(ApiValidationError, match="respawn"):
            pbx.sbatch(
                config="polaris", job_name="x", queue="q",
                select="1", walltime=10,
                respawn=-2,
            )


# ============================================================
# MCP schemas surface the new field
# ============================================================


class TestMcpSchemasRespawnFields:
    """LLM callers see `respawn` in both submit schemas."""

    def test_qsub_schema_has_respawn_field(self):
        from parslbox.mcp.schemas import QSubSchema
        assert "respawn" in QSubSchema.model_fields
        assert QSubSchema.model_fields["respawn"].default is None
        # Old fields should be gone
        assert "restart" not in QSubSchema.model_fields
        assert "max_restarts" not in QSubSchema.model_fields

    def test_qsub_respawn_description_mentions_chain(self):
        from parslbox.mcp.schemas import QSubSchema
        desc = QSubSchema.model_fields["respawn"].description
        assert "respawn" in desc.lower()
        assert "chain" in desc.lower()

    def test_sbatch_schema_has_respawn_field(self):
        from parslbox.mcp.schemas import SBatchSchema
        assert "respawn" in SBatchSchema.model_fields
        assert SBatchSchema.model_fields["respawn"].default is None
        assert "restart" not in SBatchSchema.model_fields
        assert "max_restarts" not in SBatchSchema.model_fields

    def test_sbatch_respawn_description_mentions_chain(self):
        from parslbox.mcp.schemas import SBatchSchema
        desc = SBatchSchema.model_fields["respawn"].description
        assert "respawn" in desc.lower()
        assert "chain" in desc.lower()


# ============================================================
# Phase 3: script generation (submit.sh + respawn_template.sh)
# ============================================================


def _minimal_config(scheduler_type="pbs", system_name="polaris"):
    """Build a minimal config dict for submit_job script-generation tests.

    Templates intentionally include `pbx_scheduler.out` (matching the real
    PBS_TEMPLATE / SLURM_TEMPLATE in pbx_config_template.py) so the
    respawn-mode scheduler-log rolling substitution has something to hit.
    """
    if scheduler_type == "pbs":
        template = (
            "#!/bin/bash\n"
            "#PBS -N {job_name}\n"
            "#PBS -q {queue}\n"
            "#PBS -l select={select}\n"
            "#PBS -l walltime={walltime}\n"
            "#PBS -A {project}\n"
            "#PBS -o pbx_scheduler.out\n"
            "{sched_opts}\n"
            "\n"
            "> pbx_scheduler.out\n"
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
            "#SBATCH --output=pbx_scheduler.out\n"
            "#SBATCH --error=pbx_scheduler.out\n"
            "{sched_opts}\n"
            "\n"
            "> pbx_scheduler.out\n"
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


class TestRespawnScriptGeneration:
    """submit.sh embeds --respawn; respawn_template.sh is written with placeholders."""

    # --- no respawn path: nothing about respawn should appear ---

    def test_no_respawn_no_template_file(self, tmp_path):
        result = _run_submit(tmp_path, "pbs")
        assert result.get("respawn_template_file") is None
        assert not (tmp_path / "respawn_template.sh").exists()
        script = (tmp_path / "submit.sh").read_text()
        assert "--respawn" not in script

    # --- PBS respawn path ---

    def test_pbs_submit_sh_has_respawn_flag(self, tmp_path):
        result = _run_submit(tmp_path, "pbs", respawn=3)
        assert result["respawn_template_file"] == str(tmp_path / "respawn_template.sh")
        script = (tmp_path / "submit.sh").read_text()
        # Resource line uses the original user value
        assert "select=4" in script
        # pbx run line carries --respawn N
        assert "--respawn 3" in script

    def test_pbs_respawn_template_has_placeholder_and_header(self, tmp_path):
        _run_submit(tmp_path, "pbs", respawn=2)
        template = (tmp_path / "respawn_template.sh").read_text()
        # Header comment present
        assert "PBX RESPAWN TEMPLATE" in template
        assert "NEVER overwritten by pbx" in template
        assert "DO NOT EDIT" in template
        # Resource placeholder replaces the literal value
        assert "<<PBX_AUTO_SELECT>>" in template
        assert "select=4" not in template
        # pbx run line still has --respawn with the literal initial value
        assert "--respawn 2" in template
        # Shebang stayed at line 1
        assert template.startswith("#!/bin/bash")

    # --- SLURM respawn path ---

    def test_slurm_submit_sh_has_respawn_flag(self, tmp_path):
        result = _run_submit(tmp_path, "slurm", respawn=5)
        assert result["respawn_template_file"] == str(tmp_path / "respawn_template.sh")
        script = (tmp_path / "submit.sh").read_text()
        assert "--nodes=4" in script
        assert "--respawn 5" in script

    def test_slurm_respawn_template_uses_nodes_placeholder(self, tmp_path):
        _run_submit(tmp_path, "slurm", respawn=1)
        template = (tmp_path / "respawn_template.sh").read_text()
        assert "<<PBX_AUTO_NODES>>" in template
        assert "--nodes=4" not in template
        assert "--respawn 1" in template

    # --- zero respawn is a valid (degenerate) case ---

    def test_respawn_zero_still_generates_template(self, tmp_path):
        _run_submit(tmp_path, "pbs", respawn=0)
        assert (tmp_path / "respawn_template.sh").exists()
        script = (tmp_path / "submit.sh").read_text()
        assert "--respawn 0" in script


# ============================================================
# Scheduler log rolling (pbx_scheduler.out → pbx_scheduler_link-0.out)
# Only activates when --respawn is set so each chain link writes its own log.
# ============================================================


class TestSchedulerLogRolling:
    def test_no_respawn_keeps_original_scheduler_log_name(self, tmp_path):
        """Without --respawn, the scheduler log name is the unrolled
        pbx_scheduler.out (no chain semantics → no need to roll)."""
        _run_submit(tmp_path, "pbs")
        script = (tmp_path / "submit.sh").read_text()
        assert "pbx_scheduler.out" in script
        assert "pbx_scheduler_link-" not in script

    def test_pbs_respawn_submit_sh_rolls_scheduler_log_to_link_0(self, tmp_path):
        _run_submit(tmp_path, "pbs", respawn=3)
        script = (tmp_path / "submit.sh").read_text()
        # The link-0 form must appear …
        assert "pbx_scheduler_link-0.out" in script
        # … and the unrolled form must NOT (would risk being overwritten by
        # link 1's batch job if both ended up in the same dir).
        # Match only `pbx_scheduler.out` exactly (not its `_link-N.out` cousin).
        import re
        bare_hits = re.findall(r"\bpbx_scheduler\.out\b", script)
        assert bare_hits == [], (
            f"Found unrolled pbx_scheduler.out in respawn submit.sh: {bare_hits}"
        )

    def test_pbs_respawn_template_rolls_scheduler_log_to_link_0(self, tmp_path):
        _run_submit(tmp_path, "pbs", respawn=3)
        template = (tmp_path / "respawn_template.sh").read_text()
        assert "pbx_scheduler_link-0.out" in template
        import re
        bare_hits = re.findall(r"\bpbx_scheduler\.out\b", template)
        assert bare_hits == []

    def test_slurm_respawn_submit_sh_rolls_both_output_and_error(self, tmp_path):
        """SLURM has both --output= and --error= directives — both must roll."""
        _run_submit(tmp_path, "slurm", respawn=2)
        script = (tmp_path / "submit.sh").read_text()
        assert "#SBATCH --output=pbx_scheduler_link-0.out" in script
        assert "#SBATCH --error=pbx_scheduler_link-0.out" in script

    def test_pbs_respawn_rolls_truncation_line_too(self, tmp_path):
        """The `> pbx_scheduler.out` truncation line must also roll — otherwise
        it would zero out a stale unrelated file."""
        _run_submit(tmp_path, "pbs", respawn=1)
        script = (tmp_path / "submit.sh").read_text()
        assert "> pbx_scheduler_link-0.out" in script
