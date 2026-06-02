"""
Unit tests for sched_opts helpers and submit_helpers.

Tests the directive parsing, merging, and job submission logic
for both PBS and SLURM schedulers.
"""

import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from parslbox.commands.helpers.sched_opts_helpers import (
    extract_directive_key,
    merge_sched_opts,
)
from parslbox.commands.helpers.submit_helpers import (
    submit_job,
    ValidationError,
)


# ============================================================
# Tests for extract_directive_key()
# ============================================================


class TestExtractDirectiveKey:
    """Unit tests for extract_directive_key()."""

    # --- PBS -l resources ---

    def test_pbs_walltime(self):
        assert extract_directive_key("#PBS -l walltime=01:30:00") == "walltime"

    def test_pbs_filesystems(self):
        assert extract_directive_key("#PBS -l filesystems=home:eagle") == "filesystems"

    def test_pbs_select(self):
        assert extract_directive_key("#PBS -l select=4") == "select"

    def test_pbs_place(self):
        assert extract_directive_key("#PBS -l place=scatter") == "place"

    # --- PBS single-letter flags ---

    def test_pbs_job_name(self):
        assert extract_directive_key("#PBS -N my_job") == "-N"

    def test_pbs_queue(self):
        assert extract_directive_key("#PBS -q prod") == "-q"

    def test_pbs_account(self):
        assert extract_directive_key("#PBS -A my_project") == "-A"

    def test_pbs_join(self):
        assert extract_directive_key("#PBS -j oe") == "-j"

    def test_pbs_mail(self):
        assert extract_directive_key("#PBS -m be") == "-m"

    def test_pbs_mail_user(self):
        assert extract_directive_key("#PBS -M user@example.com") == "-M"

    def test_pbs_output(self):
        assert extract_directive_key("#PBS -o output.log") == "-o"

    # --- SLURM long flags ---

    def test_slurm_job_name(self):
        assert extract_directive_key("#SBATCH --job-name=myjob") == "--job-name"

    def test_slurm_partition(self):
        assert extract_directive_key("#SBATCH --partition=gpu") == "--partition"

    def test_slurm_nodes(self):
        assert extract_directive_key("#SBATCH --nodes=4") == "--nodes"

    def test_slurm_time(self):
        assert extract_directive_key("#SBATCH --time=01:30:00") == "--time"

    def test_slurm_output(self):
        assert extract_directive_key("#SBATCH --output=job.out") == "--output"

    def test_slurm_error(self):
        assert extract_directive_key("#SBATCH --error=job.err") == "--error"

    def test_slurm_account(self):
        assert extract_directive_key("#SBATCH --account=myproj") == "--account"

    def test_slurm_mem(self):
        assert extract_directive_key("#SBATCH --mem=64G") == "--mem"

    def test_slurm_gres(self):
        assert extract_directive_key("#SBATCH --gres=gpu:4") == "--gres"

    def test_slurm_space_separated(self):
        """SLURM flag with space instead of = should still extract the flag."""
        assert extract_directive_key("#SBATCH --partition gpu") == "--partition"

    # --- Non-directives return None ---

    def test_shebang(self):
        assert extract_directive_key("#!/bin/bash") is None

    def test_comment(self):
        assert extract_directive_key("# This is a comment") is None

    def test_shell_command(self):
        assert extract_directive_key("cd $PBS_O_WORKDIR") is None

    def test_empty_line(self):
        assert extract_directive_key("") is None

    def test_blank_line(self):
        assert extract_directive_key("   ") is None

    def test_echo_command(self):
        assert extract_directive_key('echo "hello"') is None

    def test_export_command(self):
        assert extract_directive_key('export PATH="/usr/bin:$PATH"') is None

    # --- Indented directives (template indentation) ---

    def test_indented_pbs(self):
        assert extract_directive_key("      #PBS -l walltime=01:00:00") == "walltime"

    def test_indented_slurm(self):
        assert extract_directive_key("      #SBATCH --nodes=2") == "--nodes"


# ============================================================
# Tests for merge_sched_opts()
# ============================================================


class TestMergeSchedOpts:
    """Unit tests for merge_sched_opts()."""

    PBS_TEMPLATE = (
        "#!/bin/bash\n"
        "#PBS -N testjob\n"
        "#PBS -q prod\n"
        "#PBS -l select=4\n"
        "#PBS -l walltime=01:00:00\n"
        "#PBS -A myproject\n"
        "#PBS -j oe\n"
        "{sched_opts}\n"
        "\n"
        "cd $PBS_O_WORKDIR\n"
        "echo hello\n"
    )

    SLURM_TEMPLATE = (
        "#!/bin/bash\n"
        "#SBATCH --job-name=testjob\n"
        "#SBATCH --partition=gpu\n"
        "#SBATCH --nodes=4\n"
        "#SBATCH --time=01:00:00\n"
        "#SBATCH --account=myproject\n"
        "{sched_opts}\n"
        "\n"
        "echo hello\n"
    )

    def test_no_overrides_removes_placeholder(self):
        """With no overrides, {sched_opts} placeholder is removed."""
        result = merge_sched_opts(self.PBS_TEMPLATE, None, None)
        assert "{sched_opts}" not in result
        # Template directives should remain
        assert "#PBS -N testjob" in result
        assert "#PBS -l walltime=01:00:00" in result
        # Non-directive lines preserved
        assert "cd $PBS_O_WORKDIR" in result
        assert "echo hello" in result

    def test_config_overrides_template_directive(self):
        """Config sched_opts overrides a matching template directive (walltime)."""
        config_opts = "#PBS -l walltime=02:00:00"
        result = merge_sched_opts(self.PBS_TEMPLATE, config_opts, None)
        assert "#PBS -l walltime=02:00:00" in result
        assert "#PBS -l walltime=01:00:00" not in result

    def test_cli_overrides_config_directive(self):
        """CLI sched_opts overrides a matching config directive."""
        config_opts = "#PBS -l walltime=02:00:00"
        cli_opts = ["#PBS -l walltime=03:00:00"]
        result = merge_sched_opts(self.PBS_TEMPLATE, config_opts, cli_opts)
        assert "#PBS -l walltime=03:00:00" in result
        assert "#PBS -l walltime=02:00:00" not in result
        assert "#PBS -l walltime=01:00:00" not in result

    def test_new_directives_appended_at_placeholder(self):
        """New directives (not in template) are inserted at {sched_opts} location."""
        config_opts = "#PBS -l filesystems=home:eagle"
        result = merge_sched_opts(self.PBS_TEMPLATE, config_opts, None)
        assert "#PBS -l filesystems=home:eagle" in result
        assert "{sched_opts}" not in result
        # The new directive should appear after the last template directive
        lines = result.split("\n")
        fs_idx = next(i for i, l in enumerate(lines) if "filesystems" in l)
        oe_idx = next(i for i, l in enumerate(lines) if "#PBS -j oe" in l)
        assert fs_idx > oe_idx

    def test_full_three_layer_chain(self):
        """Full override chain: template → config → CLI."""
        config_opts = "#PBS -l walltime=02:00:00\n#PBS -l filesystems=home:eagle"
        cli_opts = ["#PBS -l place=scatter", "#PBS -l walltime=03:00:00"]
        result = merge_sched_opts(self.PBS_TEMPLATE, config_opts, cli_opts)

        # Template walltime overridden by config, then by CLI
        assert "#PBS -l walltime=03:00:00" in result
        assert "#PBS -l walltime=02:00:00" not in result
        assert "#PBS -l walltime=01:00:00" not in result

        # Config-added filesystems preserved
        assert "#PBS -l filesystems=home:eagle" in result

        # CLI-added place directive
        assert "#PBS -l place=scatter" in result

    def test_slurm_directive_merging(self):
        """Test merging with SLURM directives."""
        config_opts = "#SBATCH --time=02:00:00"
        cli_opts = ["#SBATCH --mem=64G"]
        result = merge_sched_opts(self.SLURM_TEMPLATE, config_opts, cli_opts)

        # Time overridden
        assert "#SBATCH --time=02:00:00" in result
        assert "#SBATCH --time=01:00:00" not in result

        # New directive added
        assert "#SBATCH --mem=64G" in result

    def test_non_directive_lines_preserved(self):
        """Non-directive lines must not be altered."""
        result = merge_sched_opts(self.PBS_TEMPLATE, None, None)
        assert "#!/bin/bash" in result
        assert "cd $PBS_O_WORKDIR" in result
        assert "echo hello" in result

    def test_cli_overrides_new_config_directive(self):
        """CLI can override a new directive added by config (not in template)."""
        config_opts = "#PBS -l filesystems=home:eagle"
        cli_opts = ["#PBS -l filesystems=home:grand"]
        result = merge_sched_opts(self.PBS_TEMPLATE, config_opts, cli_opts)
        assert "#PBS -l filesystems=home:grand" in result
        assert "#PBS -l filesystems=home:eagle" not in result

    def test_multiple_cli_opts(self):
        """Multiple CLI options are all applied."""
        cli_opts = [
            "#PBS -l filesystems=home:eagle",
            "#PBS -l place=scatter",
            "#PBS -m be",
        ]
        result = merge_sched_opts(self.PBS_TEMPLATE, None, cli_opts)
        assert "#PBS -l filesystems=home:eagle" in result
        assert "#PBS -l place=scatter" in result
        assert "#PBS -m be" in result

    def test_indentation_preserved(self):
        """Indented template directives keep their indentation after override."""
        indented = (
            "#!/bin/bash\n"
            "      #PBS -l walltime=01:00:00\n"
            "      {sched_opts}\n"
            "      echo hello\n"
        )
        result = merge_sched_opts(indented, "#PBS -l walltime=02:00:00", None)
        # The overridden line should preserve the 6-space indent
        for line in result.split("\n"):
            if "walltime=02:00:00" in line:
                assert line.startswith("      ")
                break
        else:
            pytest.fail("walltime=02:00:00 not found in result")

    def test_no_placeholder_no_crash(self):
        """Template without {sched_opts} placeholder doesn't crash."""
        no_placeholder = (
            "#!/bin/bash\n"
            "#PBS -l walltime=01:00:00\n"
            "echo done\n"
        )
        result = merge_sched_opts(no_placeholder, "#PBS -l walltime=02:00:00", None)
        assert "#PBS -l walltime=02:00:00" in result


# ============================================================
# Tests for submit_job() (integration, mocked subprocess)
# ============================================================


def _make_config(scheduler_type="pbs", system_name="polaris", sched_opts=None):
    """Build a minimal config dict for testing submit_job."""
    if scheduler_type == "pbs":
        template = (
            "#!/bin/bash\n"
            "#PBS -N {job_name}\n"
            "#PBS -q {queue}\n"
            "#PBS -l select={select}\n"
            "#PBS -l walltime={walltime}\n"
            "#PBS -A {project}\n"
            "#PBS -j oe\n"
            "{sched_opts}\n"
            "\n"
            "{pbx_python_env_setup}\n"
            "{pbx_env_vars}\n"
            "pbx run --config {config} --run-dir {run_dir} {run_options}\n"
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
            "pbx run --config {config} --run-dir {run_dir} {run_options}\n"
        )

    system_cfg = {"pbx_python_env_setup": "module load conda"}
    if sched_opts is not None:
        system_cfg["sched_opts"] = sched_opts

    return {
        "schedulers": {scheduler_type: {"template": template}},
        system_name: system_cfg,
    }


class TestSubmitJob:
    """Integration tests for submit_job() with mocked subprocess."""

    def test_pbs_submission_no_sched_opts(self, tmp_path):
        """PBS submission without sched_opts produces a valid script."""
        config = _make_config("pbs")

        with patch(
            "parslbox.commands.helpers.submit_helpers.load_config",
            return_value=config,
        ), patch(
            "parslbox.commands.helpers.submit_helpers.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(
                stdout="12345.pbs01\n", returncode=0
            )

            result = submit_job(
                config_name="polaris",
                job_name="testjob",
                queue="prod",
                select="4",
                walltime=60,
                project="proj",
                run_dir=tmp_path,
                scheduler_type="pbs",
                submit_command="qsub",
                validate_runnable=False,
            )

            assert result["success"] is True
            assert "job_id" in result
            assert "pbs_job_id" in result

            # Verify submit.sh was written
            script = (tmp_path / "submit.sh").read_text()
            assert "#PBS -N testjob" in script
            assert "{sched_opts}" not in script

    def test_pbs_submission_with_cli_sched_opts(self, tmp_path):
        """PBS submission with CLI sched_opts adds directives."""
        config = _make_config("pbs")

        with patch(
            "parslbox.commands.helpers.submit_helpers.load_config",
            return_value=config,
        ), patch(
            "parslbox.commands.helpers.submit_helpers.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(
                stdout="12345.pbs01\n", returncode=0
            )

            result = submit_job(
                config_name="polaris",
                job_name="testjob",
                queue="prod",
                select="4",
                walltime=60,
                project="proj",
                run_dir=tmp_path,
                sched_opts=["#PBS -l filesystems=home:eagle", "#PBS -l place=scatter"],
                scheduler_type="pbs",
                submit_command="qsub",
                validate_runnable=False,
            )

            assert result["success"] is True
            script = (tmp_path / "submit.sh").read_text()
            assert "#PBS -l filesystems=home:eagle" in script
            assert "#PBS -l place=scatter" in script

    def test_slurm_submission(self, tmp_path):
        """SLURM submission produces correct script and result keys."""
        config = _make_config("slurm")

        with patch(
            "parslbox.commands.helpers.submit_helpers.load_config",
            return_value=config,
        ), patch(
            "parslbox.commands.helpers.submit_helpers.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(
                stdout="98765\n", returncode=0
            )

            result = submit_job(
                config_name="polaris",
                job_name="testjob",
                queue="gpu",
                select="4",
                walltime=60,
                project="proj",
                run_dir=tmp_path,
                scheduler_type="slurm",
                submit_command="sbatch",
                validate_runnable=False,
            )

            assert result["success"] is True
            assert "job_id" in result
            assert "slurm_job_id" in result

            script = (tmp_path / "submit.sh").read_text()
            assert "#SBATCH --job-name=testjob" in script

    def test_config_level_sched_opts(self, tmp_path):
        """Config-level sched_opts from YAML are applied."""
        config = _make_config(
            "pbs",
            sched_opts="#PBS -l filesystems=home:eagle\n#PBS -m be",
        )

        with patch(
            "parslbox.commands.helpers.submit_helpers.load_config",
            return_value=config,
        ), patch(
            "parslbox.commands.helpers.submit_helpers.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(
                stdout="12345.pbs01\n", returncode=0
            )

            result = submit_job(
                config_name="polaris",
                job_name="testjob",
                queue="prod",
                select="4",
                walltime=60,
                project="proj",
                run_dir=tmp_path,
                scheduler_type="pbs",
                submit_command="qsub",
                validate_runnable=False,
            )

            assert result["success"] is True
            script = (tmp_path / "submit.sh").read_text()
            assert "#PBS -l filesystems=home:eagle" in script
            assert "#PBS -m be" in script

    def test_validation_error_missing_scheduler(self, tmp_path):
        """Missing scheduler template raises ValidationError."""
        config = {"schedulers": {}, "polaris": {"pbx_python_env_setup": ""}}

        with patch(
            "parslbox.commands.helpers.submit_helpers.load_config",
            return_value=config,
        ):
            with pytest.raises(ValidationError, match="PBS.*not found"):
                submit_job(
                    config_name="polaris",
                    job_name="testjob",
                    queue="prod",
                    select="4",
                    walltime=60,
                    project="proj",
                    run_dir=tmp_path,
                    scheduler_type="pbs",
                    submit_command="qsub",
                validate_runnable=False,
                )

    def test_validation_error_missing_system(self, tmp_path):
        """Missing system config raises ValidationError."""
        config = _make_config("pbs")

        with patch(
            "parslbox.commands.helpers.submit_helpers.load_config",
            return_value=config,
        ):
            with pytest.raises(ValidationError, match="not found"):
                submit_job(
                    config_name="nonexistent",
                    job_name="testjob",
                    queue="prod",
                    select="4",
                    walltime=60,
                    project="proj",
                    run_dir=tmp_path,
                    scheduler_type="pbs",
                    submit_command="qsub",
                validate_runnable=False,
                )

    def test_validation_error_empty_config(self, tmp_path):
        """Empty config raises ValidationError."""
        with patch(
            "parslbox.commands.helpers.submit_helpers.load_config",
            return_value=None,
        ):
            with pytest.raises(ValidationError, match="empty"):
                submit_job(
                    config_name="polaris",
                    job_name="testjob",
                    queue="prod",
                    select="4",
                    walltime=60,
                    project="proj",
                    run_dir=tmp_path,
                    scheduler_type="pbs",
                    submit_command="qsub",
                validate_runnable=False,
                )

    def test_submit_command_not_found(self, tmp_path):
        """Missing submit command raises ValidationError."""
        config = _make_config("pbs")

        with patch(
            "parslbox.commands.helpers.submit_helpers.load_config",
            return_value=config,
        ), patch(
            "parslbox.commands.helpers.submit_helpers.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            with pytest.raises(ValidationError, match="qsub command not found"):
                submit_job(
                    config_name="polaris",
                    job_name="testjob",
                    queue="prod",
                    select="4",
                    walltime=60,
                    project="proj",
                    run_dir=tmp_path,
                    scheduler_type="pbs",
                    submit_command="qsub",
                validate_runnable=False,
                )


# ============================================================
# Tests for system-level default sched_opts
# ============================================================


class TestSystemDefaultSchedOpts:
    """Tests for the system-class default sched_opts override chain."""

    def test_system_default_sched_opts_applied(self, tmp_path):
        """System defaults are applied when no config or CLI sched_opts."""
        config = _make_config("pbs")

        mock_sys = MagicMock()
        mock_sys.get_default_sched_opts.return_value = "#PBS -l filesystems=home:eagle"

        with patch(
            "parslbox.commands.helpers.submit_helpers.load_config",
            return_value=config,
        ), patch(
            "parslbox.system_configs.loader.get_system_config",
            return_value=mock_sys,
        ), patch(
            "parslbox.commands.helpers.submit_helpers.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(
                stdout="12345.pbs01\n", returncode=0
            )

            result = submit_job(
                config_name="polaris",
                job_name="testjob",
                queue="prod",
                select="4",
                walltime=60,
                project="proj",
                run_dir=tmp_path,
                scheduler_type="pbs",
                submit_command="qsub",
                validate_runnable=False,
            )

            assert result["success"] is True
            script = (tmp_path / "submit.sh").read_text()
            assert "#PBS -l filesystems=home:eagle" in script

    def test_config_overrides_system_default(self, tmp_path):
        """Config-level sched_opts override system defaults with same key."""
        config = _make_config(
            "pbs",
            sched_opts="#PBS -l filesystems=home:grand",
        )

        mock_sys = MagicMock()
        mock_sys.get_default_sched_opts.return_value = "#PBS -l filesystems=home:eagle"

        with patch(
            "parslbox.commands.helpers.submit_helpers.load_config",
            return_value=config,
        ), patch(
            "parslbox.system_configs.loader.get_system_config",
            return_value=mock_sys,
        ), patch(
            "parslbox.commands.helpers.submit_helpers.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(
                stdout="12345.pbs01\n", returncode=0
            )

            result = submit_job(
                config_name="polaris",
                job_name="testjob",
                queue="prod",
                select="4",
                walltime=60,
                project="proj",
                run_dir=tmp_path,
                scheduler_type="pbs",
                submit_command="qsub",
                validate_runnable=False,
            )

            assert result["success"] is True
            script = (tmp_path / "submit.sh").read_text()
            assert "#PBS -l filesystems=home:grand" in script
            assert "#PBS -l filesystems=home:eagle" not in script

    def test_cli_overrides_system_and_config(self, tmp_path):
        """CLI sched_opts override both system defaults and config."""
        config = _make_config(
            "pbs",
            sched_opts="#PBS -l filesystems=home:grand",
        )

        mock_sys = MagicMock()
        mock_sys.get_default_sched_opts.return_value = "#PBS -l filesystems=home:eagle"

        with patch(
            "parslbox.commands.helpers.submit_helpers.load_config",
            return_value=config,
        ), patch(
            "parslbox.system_configs.loader.get_system_config",
            return_value=mock_sys,
        ), patch(
            "parslbox.commands.helpers.submit_helpers.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(
                stdout="12345.pbs01\n", returncode=0
            )

            result = submit_job(
                config_name="polaris",
                job_name="testjob",
                queue="prod",
                select="4",
                walltime=60,
                project="proj",
                run_dir=tmp_path,
                sched_opts=["#PBS -l filesystems=home:flare"],
                scheduler_type="pbs",
                submit_command="qsub",
                validate_runnable=False,
            )

            assert result["success"] is True
            script = (tmp_path / "submit.sh").read_text()
            assert "#PBS -l filesystems=home:flare" in script
            assert "#PBS -l filesystems=home:grand" not in script
            assert "#PBS -l filesystems=home:eagle" not in script

    def test_system_default_no_system_config_class(self, tmp_path):
        """Submission works even when get_system_config raises ValueError."""
        config = _make_config("pbs")

        with patch(
            "parslbox.commands.helpers.submit_helpers.load_config",
            return_value=config,
        ), patch(
            "parslbox.system_configs.loader.get_system_config",
            side_effect=ValueError("Unknown system"),
        ), patch(
            "parslbox.commands.helpers.submit_helpers.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(
                stdout="12345.pbs01\n", returncode=0
            )

            result = submit_job(
                config_name="polaris",
                job_name="testjob",
                queue="prod",
                select="4",
                walltime=60,
                project="proj",
                run_dir=tmp_path,
                scheduler_type="pbs",
                submit_command="qsub",
                validate_runnable=False,
            )

            assert result["success"] is True
            script = (tmp_path / "submit.sh").read_text()
            assert "{sched_opts}" not in script

    def test_system_default_with_none_preserves_config(self, tmp_path):
        """When system returns None, config sched_opts still work."""
        config = _make_config(
            "pbs",
            sched_opts="#PBS -l filesystems=home:eagle",
        )

        mock_sys = MagicMock()
        mock_sys.get_default_sched_opts.return_value = None

        with patch(
            "parslbox.commands.helpers.submit_helpers.load_config",
            return_value=config,
        ), patch(
            "parslbox.system_configs.loader.get_system_config",
            return_value=mock_sys,
        ), patch(
            "parslbox.commands.helpers.submit_helpers.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(
                stdout="12345.pbs01\n", returncode=0
            )

            result = submit_job(
                config_name="polaris",
                job_name="testjob",
                queue="prod",
                select="4",
                walltime=60,
                project="proj",
                run_dir=tmp_path,
                scheduler_type="pbs",
                submit_command="qsub",
                validate_runnable=False,
            )

            assert result["success"] is True
            script = (tmp_path / "submit.sh").read_text()
            assert "#PBS -l filesystems=home:eagle" in script


# ============================================================
# Tests for parse_walltime() and minutes_to_hms()
# ============================================================

from parslbox.commands.helpers.qsub_cmd_helpers import parse_walltime, minutes_to_hms


class TestParseWalltime:
    """Unit tests for parse_walltime()."""

    def test_plain_integer_minutes(self):
        assert parse_walltime("90") == 90.0

    def test_plain_float_minutes(self):
        assert parse_walltime("90.5") == 90.5

    def test_explicit_m_suffix(self):
        assert parse_walltime("120m") == 120.0

    def test_hours_integer(self):
        assert parse_walltime("2h") == 120.0

    def test_hours_fractional(self):
        assert parse_walltime("4.25h") == 255.0

    def test_hours_half(self):
        assert parse_walltime("0.5h") == 30.0

    def test_days_integer(self):
        assert parse_walltime("1d") == 1440.0

    def test_days_fractional(self):
        assert parse_walltime("3.5d") == 5040.0

    def test_uppercase_suffix(self):
        assert parse_walltime("2H") == 120.0
        assert parse_walltime("1D") == 1440.0
        assert parse_walltime("60M") == 60.0

    def test_whitespace_stripped(self):
        assert parse_walltime("  90  ") == 90.0
        assert parse_walltime(" 2h ") == 120.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_walltime("")

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_walltime("abc")

    def test_no_number_raises(self):
        with pytest.raises(ValueError):
            parse_walltime("h")


class TestMinutesToHms:
    """Unit tests for minutes_to_hms()."""

    def test_exact_hours(self):
        assert minutes_to_hms(120) == "02:00:00"

    def test_hours_and_minutes(self):
        assert minutes_to_hms(90) == "01:30:00"

    def test_fractional_minutes_with_seconds(self):
        assert minutes_to_hms(90.5) == "01:30:30"

    def test_zero(self):
        assert minutes_to_hms(0) == "00:00:00"

    def test_large_value(self):
        assert minutes_to_hms(5040) == "84:00:00"

    def test_quarter_hour(self):
        assert minutes_to_hms(255) == "04:15:00"

    def test_float_from_hours_conversion(self):
        """4.25h -> 255 min -> 04:15:00"""
        assert minutes_to_hms(parse_walltime("4.25h")) == "04:15:00"

    def test_float_from_days_conversion(self):
        """3.5d -> 5040 min -> 84:00:00"""
        assert minutes_to_hms(parse_walltime("3.5d")) == "84:00:00"

    def test_one_third_hour(self):
        """1/3 hour = 20 min exactly"""
        result = minutes_to_hms(20)
        assert result == "00:20:00"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
