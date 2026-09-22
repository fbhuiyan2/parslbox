"""API-layer tests for ParslBox.qsub / ParslBox.sbatch.

Confirms the tag-glob expansion and zero-runnable-jobs guard (which live in
submit_job) reach API callers — and therefore reach the MCP server, which
delegates to the API.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from parslbox.api import ParslBox, ValidationError
from parslbox.database import database


def _make_pbs_config(system_name="aurora-tile"):
    return {
        "schedulers": {"pbs": {"template": (
            "#!/bin/bash\n#PBS -N {job_name}\n#PBS -q {queue}\n"
            "#PBS -l select={select}\n#PBS -l walltime={walltime}\n"
            "#PBS -A {project}\n{sched_opts}\n\n"
            "{pbx_python_env_setup}\n{pbx_env_vars}\n"
            "pbx run --config {config} --run-dir {run_dir} {run_options}\n"
        )}},
        system_name: {"pbx_python_env_setup": "module load conda"},
    }


def _make_slurm_config(system_name="aurora-tile"):
    return {
        "schedulers": {"slurm": {"template": (
            "#!/bin/bash\n#SBATCH -J {job_name}\n#SBATCH -p {queue}\n"
            "#SBATCH -N {select}\n#SBATCH -t {walltime}\n#SBATCH -A {project}\n"
            "{sched_opts}\n\n"
            "{pbx_python_env_setup}\n{pbx_env_vars}\n"
            "pbx run --config {config} --run-dir {run_dir} {run_options}\n"
        )}},
        system_name: {"pbx_python_env_setup": "module load conda"},
    }


@pytest.fixture
def db_with_tags(temp_db):
    """Seed the API's temp_db with a mix of tags for glob tests."""
    database.initialize_database(temp_db)
    tags = [
        "film-bulk_3c-nomix",
        "film-bulk_3c-mix",
        "prod-run",
        "prod-test",
        "misc",
    ]
    for i, tag in enumerate(tags):
        database.add_job(
            db_path=temp_db, path=f"/p{i}", app="lammps-kk",
            num_nodes=1, ngpus=0, node_occupancy=1.0,
            tag=tag, in_file=None, status="Ready",
        )
    return temp_db


@pytest.fixture
def api_pbs(db_with_tags, tmp_path):
    """ParslBox instance whose qsub() exercises the real submit_job (and its
    new validation). Mocks only the subprocess + config-loading layer.

    Deliberately does NOT patch path_utils.DB_FILE — submit_job must honour
    the instance's db_path on its own.
    """
    fake_config = tmp_path / "pbx_config.yaml"
    fake_config.write_text("schedulers: {}\n")
    with patch("parslbox.commands.helpers.submit_helpers.load_config",
               return_value=_make_pbs_config()), \
         patch("parslbox.commands.helpers.submit_helpers.subprocess.run") as mock_run, \
         patch("parslbox.commands.helpers.submit_helpers.get_default_run_dir",
               return_value=tmp_path / "fake_run"):
        mock_run.return_value = MagicMock(stdout="12345.pbs01\n", returncode=0)
        pbx = ParslBox(db_path=db_with_tags, config_path=fake_config)
        yield pbx, mock_run


@pytest.fixture
def api_slurm(db_with_tags, tmp_path):
    fake_config = tmp_path / "pbx_config.yaml"
    fake_config.write_text("schedulers: {}\n")
    with patch("parslbox.commands.helpers.submit_helpers.load_config",
               return_value=_make_slurm_config()), \
         patch("parslbox.commands.helpers.submit_helpers.subprocess.run") as mock_run, \
         patch("parslbox.commands.helpers.submit_helpers.get_default_run_dir",
               return_value=tmp_path / "fake_run"):
        mock_run.return_value = MagicMock(stdout="12345\n", returncode=0)
        pbx = ParslBox(db_path=db_with_tags, config_path=fake_config)
        yield pbx, mock_run


class TestApiQsubValidation:
    _kwargs = dict(
        config="aurora-tile", job_name="test", queue="debug",
        select="1", walltime=10, project="proj",
    )

    def test_glob_expands_to_literal_tags(self, api_pbs):
        pbx, mock_run = api_pbs
        result = pbx.qsub(apps=["lammps-kk"], tags=["*3c*"], **self._kwargs)
        assert result["success"] is True
        assert set(result["resolved_tags"]) == {
            "film-bulk_3c-mix", "film-bulk_3c-nomix"
        }
        assert result["matched_jobs"] == 2

    def test_mixed_glob_and_literal(self, api_pbs):
        pbx, _ = api_pbs
        result = pbx.qsub(apps=["lammps-kk"],
                          tags=["*3c*", "prod-run"], **self._kwargs)
        assert result["success"] is True
        assert set(result["resolved_tags"]) == {
            "film-bulk_3c-mix", "film-bulk_3c-nomix", "prod-run"
        }
        assert result["matched_jobs"] == 3

    def test_unmatched_tag_raises_before_submission(self, api_pbs):
        pbx, mock_run = api_pbs
        with pytest.raises(ValidationError, match="not found"):
            pbx.qsub(apps=["lammps-kk"], tags=["*nope*"], **self._kwargs)
        assert mock_run.call_count == 0

    def test_unmatched_literal_raises(self, api_pbs):
        pbx, mock_run = api_pbs
        with pytest.raises(ValidationError, match="doesnotexist"):
            pbx.qsub(apps=["lammps-kk"],
                     tags=["prod-run", "doesnotexist"], **self._kwargs)
        assert mock_run.call_count == 0

    def test_zero_runnable_jobs_raises(self, api_pbs):
        # tag exists, but no Ready jobs match app=vasp
        pbx, mock_run = api_pbs
        with pytest.raises(ValidationError, match="No Ready/Restart"):
            pbx.qsub(apps=["vasp"], tags=["prod-run"], **self._kwargs)
        assert mock_run.call_count == 0


class TestApiSbatchValidation:
    """Same guarantees as qsub — sbatch must also benefit since it shares
    submit_job."""
    _kwargs = dict(
        config="aurora-tile", job_name="test", queue="debug",
        select="1", walltime=10, project="proj",
    )

    def test_glob_expansion(self, api_slurm):
        pbx, _ = api_slurm
        result = pbx.sbatch(apps=["lammps-kk"], tags=["*3c*"], **self._kwargs)
        assert result["success"] is True
        assert set(result["resolved_tags"]) == {
            "film-bulk_3c-mix", "film-bulk_3c-nomix"
        }

    def test_zero_runnable_raises(self, api_slurm):
        pbx, mock_run = api_slurm
        with pytest.raises(ValidationError, match="No Ready/Restart"):
            pbx.sbatch(apps=["vasp"], tags=["prod-run"], **self._kwargs)
        assert mock_run.call_count == 0


class TestInstancePathsReachSubmitJob:
    """ParslBox(db_path=..., config_path=...) must govern qsub/sbatch the same
    way it governs every other method — both for validation and for the paths
    baked into the generated batch script."""

    _kwargs = dict(
        config="aurora-tile", job_name="test", queue="debug",
        select="1", walltime=10, project="proj",
    )

    def test_qsub_validates_against_instance_db(self, api_pbs, db_with_tags):
        # Tags only exist in db_with_tags, never in the process-wide default DB.
        pbx, _ = api_pbs
        result = pbx.qsub(apps=["lammps-kk"], tags=["prod-run"], **self._kwargs)
        assert result["matched_jobs"] == 1

    def test_submit_script_pins_instance_paths(self, api_pbs, db_with_tags,
                                               tmp_path, monkeypatch):
        # No PBX_* env at all: the script must still carry both exports,
        # resolved from the instance rather than the environment.
        monkeypatch.delenv("PBX_DB_PATH", raising=False)
        monkeypatch.delenv("PBX_CONFIG_PATH", raising=False)

        pbx, _ = api_pbs
        result = pbx.qsub(apps=["lammps-kk"], tags=["prod-run"], **self._kwargs)

        script = (Path(result["run_dir"]) / "submit.sh").read_text()
        assert f'export PBX_DB_PATH="{db_with_tags}"' in script
        assert f'export PBX_CONFIG_PATH="{tmp_path / "pbx_config.yaml"}"' in script

    def test_sbatch_script_pins_instance_paths(self, api_slurm, db_with_tags,
                                               tmp_path, monkeypatch):
        monkeypatch.delenv("PBX_DB_PATH", raising=False)
        monkeypatch.delenv("PBX_CONFIG_PATH", raising=False)

        pbx, _ = api_slurm
        result = pbx.sbatch(apps=["lammps-kk"], tags=["prod-run"], **self._kwargs)

        script = (Path(result["run_dir"]) / "submit.sh").read_text()
        assert f'export PBX_DB_PATH="{db_with_tags}"' in script
        assert f'export PBX_CONFIG_PATH="{tmp_path / "pbx_config.yaml"}"' in script

    def test_env_var_does_not_override_explicit_db_path(self, api_pbs,
                                                        db_with_tags,
                                                        monkeypatch):
        # An unrelated PBX_DB_PATH in the environment must not leak into the
        # script when the caller pinned a db_path explicitly.
        monkeypatch.setenv("PBX_DB_PATH", "/somewhere/else/stale.db")

        pbx, _ = api_pbs
        result = pbx.qsub(apps=["lammps-kk"], tags=["prod-run"], **self._kwargs)

        script = (Path(result["run_dir"]) / "submit.sh").read_text()
        assert f'export PBX_DB_PATH="{db_with_tags}"' in script
        assert "stale.db" not in script
