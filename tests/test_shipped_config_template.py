"""The scheduler templates `pbx config` writes into a new config.yaml.

Nothing exercised these before: every other test hand-writes its own
near-copy, so the shipped PBS_TEMPLATE/SLURM_TEMPLATE could drift out of step
with the `pbx run` flags and the respawn parser without a single failure.
"""
import pytest
import yaml

from parslbox.commands.helpers.restart_helpers import validate_pbx_run_line
from parslbox.utils.config_generator import ConfigGenerator

# The keys submit_job fills a template with (submit_helpers.py, template_vars).
TEMPLATE_VARS = dict(
    job_name="j", queue="debug", select="1", walltime="00:10:00", project="p",
    pbx_python_env_setup="module load conda", pbx_env_vars="",
    config="polaris", run_dir="./", run_options="--walltime-seconds 600",
    sched_opts="",
)

SCHEDULERS = [("pbs", "polaris"), ("slurm", "perlmutter-gpu")]


def _shipped_template(scheduler, system):
    config = yaml.safe_load(ConfigGenerator([system], ["lammps-kk"]).generate())
    return config["schedulers"][scheduler]["template"]


class TestTheShippedTemplate:
    @pytest.mark.parametrize("scheduler,system", SCHEDULERS)
    def test_it_renders_with_the_keys_submit_job_supplies(self, scheduler, system):
        """A placeholder with no matching key is a KeyError at submit time."""
        rendered = _shipped_template(scheduler, system).format(**TEMPLATE_VARS)
        assert "pbx run" in rendered

    @pytest.mark.parametrize("scheduler,system", SCHEDULERS)
    def test_it_uses_the_rundir_spelling(self, scheduler, system):
        rendered = _shipped_template(scheduler, system).format(**TEMPLATE_VARS)
        assert "--rundir ./" in rendered

    @pytest.mark.parametrize("scheduler,system", SCHEDULERS)
    def test_the_run_line_satisfies_the_respawn_parser(self, scheduler, system):
        """The chain re-reads this line at every walltime expiry. A template
        the parser rejects fails hours into an allocation, not at submit."""
        vars_with_respawn = dict(TEMPLATE_VARS,
                                 run_options="--walltime-seconds 600 --respawn 3")
        rendered = _shipped_template(scheduler, system).format(**vars_with_respawn)
        assert "--rundir" in validate_pbx_run_line(rendered)


class TestTemplatesAlreadyOnDisk:
    """`--run-dir` and `{run_dir}` must keep working: every config.yaml
    written before the rename still says both, and users edit these by hand."""

    @pytest.mark.parametrize("flag", ["--run-dir", "--rundir"])
    def test_either_flag_spelling_still_validates(self, flag):
        line = (f"exec pbx run --config polaris {flag} ./ "
                f"--walltime-seconds 600 --respawn 3")
        assert validate_pbx_run_line(line)

    def test_the_run_dir_placeholder_still_resolves(self):
        old = "exec pbx run --config {config} --run-dir {run_dir} {run_options}"
        assert "--run-dir ./" in old.format(**TEMPLATE_VARS)
