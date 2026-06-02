"""Tests for `*`-glob tag matching: helper, database integration, and the
qsub/sbatch CLI expansion that resolves globs to literal tags before submission.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from parslbox.database import database
from parslbox.utils.tag_match import has_glob, to_sql_like, expand_tag_patterns


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
def temp_db():
    d = tempfile.mkdtemp()
    yield Path(d) / "test.db"
    shutil.rmtree(d)


@pytest.fixture
def db_with_tags(temp_db):
    """DB seeded with a mix of tag values for glob tests."""
    database.initialize_database(temp_db)
    tags = [
        "film-bulk_3c-nomix",
        "film-bulk_3c-mix",
        "prod-run",
        "prod-test",
        "misc",
        None,  # untagged job
        "weird_100%_done",  # SQL LIKE wildcard in the actual tag value
        "snake_case_tag",   # underscore — SQL LIKE wildcard
    ]
    for i, tag in enumerate(tags):
        database.add_job(
            db_path=temp_db, path=f"/p{i}", app="lammps-kk",
            num_nodes=1, ngpus=0, node_occupancy=1.0,
            tag=tag, in_file=None, status="Ready",
        )
    return temp_db


class TestHasGlob:
    def test_plain_string(self):
        assert not has_glob("prod-run")

    def test_star_present(self):
        assert has_glob("*test")
        assert has_glob("test*")
        assert has_glob("*test*")
        assert has_glob("a*b")

    def test_other_glob_chars_not_active(self):
        # We only treat `*` as a glob char; `?` and `[]` are literals.
        assert not has_glob("test?")
        assert not has_glob("[abc]")


class TestToSqlLike:
    def test_star_to_percent(self):
        assert to_sql_like("*foo") == "%foo"
        assert to_sql_like("foo*") == "foo%"
        assert to_sql_like("*foo*") == "%foo%"
        assert to_sql_like("a*b*c") == "a%b%c"

    def test_escapes_existing_percent(self):
        # `%` in input must be escaped so it matches literally, not as wildcard.
        assert to_sql_like("100%") == "100\\%"
        assert to_sql_like("*100%") == "%100\\%"

    def test_escapes_existing_underscore(self):
        assert to_sql_like("a_b") == "a\\_b"
        assert to_sql_like("snake_*") == "snake\\_%"

    def test_escapes_backslash(self):
        assert to_sql_like("a\\b") == "a\\\\b"


class TestExpandTagPatterns:
    def test_pure_literals_pass_through(self, db_with_tags):
        out = expand_tag_patterns(db_with_tags, ["prod-run", "misc"])
        assert out == ["prod-run", "misc"]

    def test_glob_expands_to_multiple(self, db_with_tags):
        out = expand_tag_patterns(db_with_tags, ["*3c*"])
        assert set(out) == {"film-bulk_3c-mix", "film-bulk_3c-nomix"}

    def test_mixed_list(self, db_with_tags):
        out = expand_tag_patterns(db_with_tags, ["*3c*", "prod-run"])
        assert set(out) == {"film-bulk_3c-mix", "film-bulk_3c-nomix", "prod-run"}

    def test_dedup_overlapping_patterns(self, db_with_tags):
        # Both globs match prod-run; should appear once.
        out = expand_tag_patterns(db_with_tags, ["prod*", "*run"])
        assert sorted(out) == ["prod-run", "prod-test"]
        assert out.count("prod-run") == 1

    def test_dedup_glob_and_literal(self, db_with_tags):
        out = expand_tag_patterns(db_with_tags, ["prod-run", "prod*"])
        assert out.count("prod-run") == 1
        assert "prod-test" in out

    def test_preserves_first_seen_order(self, db_with_tags):
        out = expand_tag_patterns(db_with_tags, ["misc", "prod-run"])
        assert out == ["misc", "prod-run"]

    def test_unmatched_literal_raises(self, db_with_tags):
        with pytest.raises(ValueError, match="doesnotexist"):
            expand_tag_patterns(db_with_tags, ["prod-run", "doesnotexist"])

    def test_unmatched_glob_raises(self, db_with_tags):
        with pytest.raises(ValueError, match=r"\*nope\*"):
            expand_tag_patterns(db_with_tags, ["*nope*"])

    def test_error_lists_all_unmatched(self, db_with_tags):
        with pytest.raises(ValueError) as exc:
            expand_tag_patterns(db_with_tags, ["*nope*", "alsogone", "misc"])
        msg = str(exc.value)
        assert "*nope*" in msg
        assert "alsogone" in msg

    def test_literal_with_sql_wildcard_chars_treated_literally(self, db_with_tags):
        # Tag value "weird_100%_done" contains `_` and `%` which are SQL LIKE
        # wildcards. Without escaping, "weird_100%_done" would match many things,
        # but our helper escapes them — so the literal must match the exact tag.
        out = expand_tag_patterns(db_with_tags, ["weird_100%_done"])
        assert out == ["weird_100%_done"]

    def test_glob_with_literal_underscore_escapes_correctly(self, db_with_tags):
        # `*_tag` should match `snake_case_tag` (ends with `_tag`).
        # The `_` in the pattern must match a literal `_`, not "any char".
        out = expand_tag_patterns(db_with_tags, ["*_tag"])
        assert out == ["snake_case_tag"]

    def test_null_tags_ignored(self, db_with_tags):
        # An untagged job exists; `*` glob should not return NULL.
        out = expand_tag_patterns(db_with_tags, ["*"])
        assert all(t is not None for t in out)


class TestGetJobsTagGlob:
    """database.get_jobs(tag=...) — used by `pbx filter` and `pbx ls`."""

    def test_exact_tag_match(self, db_with_tags):
        rows = database.get_jobs(db_with_tags, tag="prod-run")
        assert len(rows) == 1
        assert rows[0]["tag"] == "prod-run"

    def test_glob_tag_match(self, db_with_tags):
        rows = database.get_jobs(db_with_tags, tag="*3c*")
        assert sorted(r["tag"] for r in rows) == ["film-bulk_3c-mix", "film-bulk_3c-nomix"]

    def test_glob_prefix(self, db_with_tags):
        rows = database.get_jobs(db_with_tags, tag="prod*")
        assert sorted(r["tag"] for r in rows) == ["prod-run", "prod-test"]

    def test_glob_suffix(self, db_with_tags):
        rows = database.get_jobs(db_with_tags, tag="*test")
        assert [r["tag"] for r in rows] == ["prod-test"]

    def test_glob_no_match_returns_empty(self, db_with_tags):
        # get_jobs is silent (read-only query) — no error, just empty result.
        assert database.get_jobs(db_with_tags, tag="*nomatch*") == []

    def test_literal_with_underscore_is_exact(self, db_with_tags):
        # Without our escaping, "snake_case_tag" used with LIKE would match too
        # many things due to `_` being LIKE wildcard. But the no-glob branch
        # uses `=`, so this is just exact match.
        rows = database.get_jobs(db_with_tags, tag="snake_case_tag")
        assert len(rows) == 1


# -------- qsub / sbatch CLI expansion -------------------------------------- #


@pytest.fixture
def qsub_runner(db_with_tags, tmp_path):
    """Real submit_job runs against db_with_tags; only load_config and
    subprocess.run are mocked. Lets us exercise the new tag-expansion and
    runnable-jobs guard that lives in submit_job.

    Yields (runner, app, captured) where captured is a dict with:
      - 'kwargs'  : the kwargs submit_job was called with (pre-expansion)
      - 'result'  : submit_job's return dict (post-expansion); absent if
                    validation raised before submit_job returned
      - 'qsub_called': bool — did subprocess.run for the qsub command run?
    """
    from parslbox.commands import qsub as qsub_mod
    runner = CliRunner()

    captured = {}
    real_submit_job = qsub_mod.submit_job

    def spy_submit_job(**kwargs):
        captured["kwargs"] = kwargs
        result = real_submit_job(**kwargs)
        captured["result"] = result
        return result

    with patch.object(qsub_mod, "submit_job", side_effect=spy_submit_job), \
         patch("parslbox.commands.helpers.submit_helpers.load_config",
               return_value=_make_pbs_config()), \
         patch("parslbox.commands.helpers.submit_helpers.subprocess.run") as mock_run, \
         patch("parslbox.commands.helpers.submit_helpers.get_default_run_dir",
               return_value=tmp_path / "fake_run"), \
         patch("parslbox.utils.path_utils.DB_FILE", db_with_tags):
        mock_run.return_value = MagicMock(stdout="12345.pbs01\n", returncode=0)
        captured["_mock_run"] = mock_run
        yield runner, qsub_mod.app, captured


class TestQsubTagExpansion:
    def _common_args(self, tags):
        return [
            "--config", "aurora-tile",
            "--job-name", "test",
            "--queue", "debug",
            "--select", "1",
            "--walltime", "10",
            "--project", "proj",
            "--apps", "lammps-kk",
            "--tags", tags,
        ]

    def test_literal_tags_pass_through(self, qsub_runner):
        runner, app, captured = qsub_runner
        r = runner.invoke(app, self._common_args("prod-run,misc"))
        assert r.exit_code == 0, r.output
        assert captured["result"]["resolved_tags"] == ["prod-run", "misc"]

    def test_glob_resolves_to_literals(self, qsub_runner):
        runner, app, captured = qsub_runner
        r = runner.invoke(app, self._common_args("*3c*"))
        assert r.exit_code == 0, r.output
        assert set(captured["result"]["resolved_tags"]) == {
            "film-bulk_3c-mix", "film-bulk_3c-nomix"
        }

    def test_mixed_glob_and_literal(self, qsub_runner):
        runner, app, captured = qsub_runner
        r = runner.invoke(app, self._common_args("*3c*,prod-run"))
        assert r.exit_code == 0, r.output
        assert set(captured["result"]["resolved_tags"]) == {
            "film-bulk_3c-mix", "film-bulk_3c-nomix", "prod-run"
        }

    def test_unmatched_glob_exits_nonzero(self, qsub_runner):
        runner, app, captured = qsub_runner
        r = runner.invoke(app, self._common_args("*nope*"))
        assert r.exit_code != 0
        assert "not found" in r.output.lower()
        # qsub subprocess must NOT have been called (validation aborts first)
        assert captured["_mock_run"].call_count == 0

    def test_unmatched_literal_exits_nonzero(self, qsub_runner):
        runner, app, captured = qsub_runner
        r = runner.invoke(app, self._common_args("prod-run,doesnotexist"))
        assert r.exit_code != 0
        assert "doesnotexist" in r.output
        assert captured["_mock_run"].call_count == 0

    def test_one_bad_token_aborts_whole_submission(self, qsub_runner):
        # Strict rule: if *any* token doesn't match, no submission.
        runner, app, captured = qsub_runner
        r = runner.invoke(app, self._common_args("*3c*,bogus,prod-run"))
        assert r.exit_code != 0
        assert "bogus" in r.output
        assert captured["_mock_run"].call_count == 0


@pytest.fixture
def sbatch_runner(db_with_tags, tmp_path):
    """Same pattern as qsub_runner — see its docstring."""
    from parslbox.commands import sbatch as sbatch_mod
    runner = CliRunner()

    captured = {}
    real_submit_job = sbatch_mod.submit_job

    def spy_submit_job(**kwargs):
        captured["kwargs"] = kwargs
        result = real_submit_job(**kwargs)
        captured["result"] = result
        return result

    with patch.object(sbatch_mod, "submit_job", side_effect=spy_submit_job), \
         patch("parslbox.commands.helpers.submit_helpers.load_config",
               return_value=_make_slurm_config()), \
         patch("parslbox.commands.helpers.submit_helpers.subprocess.run") as mock_run, \
         patch("parslbox.commands.helpers.submit_helpers.get_default_run_dir",
               return_value=tmp_path / "fake_run"), \
         patch("parslbox.utils.path_utils.DB_FILE", db_with_tags):
        mock_run.return_value = MagicMock(stdout="12345\n", returncode=0)
        captured["_mock_run"] = mock_run
        yield runner, sbatch_mod.app, captured


class TestQsubJobCountAndPanel:
    """New behavior: print matched job count, abort if zero; print submit
    script in a box with `pbx run` line in red."""

    def _common_args(self, **overrides):
        args = {
            "--config": "aurora-tile", "--job-name": "test",
            "--queue": "debug", "--select": "1", "--walltime": "10",
            "--project": "proj", "--apps": "lammps-kk",
            "--tags": "prod-run",
        }
        args.update(overrides)
        out = []
        for k, v in args.items():
            out += [k, v]
        return out

    def test_matched_count_printed(self, qsub_runner):
        runner, app, _ = qsub_runner
        r = runner.invoke(app, self._common_args())
        assert r.exit_code == 0, r.output
        assert "Matched 1 runnable job" in r.output

    def test_zero_match_aborts(self, qsub_runner):
        # `--apps vasp` has no Ready jobs in our seed DB
        # (db_with_tags only adds lammps-kk jobs)
        runner, app, captured = qsub_runner
        r = runner.invoke(app, self._common_args(**{"--apps": "vasp"}))
        assert r.exit_code != 0
        assert "No Ready/Restart jobs match" in r.output
        # qsub subprocess must NOT have been called
        assert captured["_mock_run"].call_count == 0

    def test_no_tag_no_app_counts_all(self, qsub_runner):
        # Drop the --apps and --tags entirely
        runner, app, _ = qsub_runner
        args = self._common_args()
        # Remove --apps / --tags pairs
        for flag in ("--apps", "--tags"):
            i = args.index(flag)
            del args[i:i + 2]
        r = runner.invoke(app, args)
        assert r.exit_code == 0, r.output
        # db_with_tags has 8 jobs total (one per tag entry), all lammps-kk Ready
        assert "Matched 8 runnable job" in r.output

    def test_submit_script_panel_shown(self, qsub_runner):
        runner, app, _ = qsub_runner
        r = runner.invoke(app, self._common_args(),
                          env={"COLUMNS": "300", "FORCE_COLOR": "0"})
        assert r.exit_code == 0
        # The panel renders the script contents — check key lines appear
        assert "pbx run" in r.output
        assert "#PBS -N test" in r.output
        # Panel title appears
        assert "Submit Script" in r.output


class TestSbatchTagExpansion:
    """Same strict-expansion behavior as qsub."""

    def _common_args(self, tags):
        return [
            "--config", "aurora-tile",
            "--job-name", "test",
            "--queue", "debug",
            "--select", "1",
            "--walltime", "10",
            "--project", "proj",
            "--apps", "lammps-kk",
            "--tags", tags,
        ]

    def test_glob_resolves_to_literals(self, sbatch_runner):
        runner, app, captured = sbatch_runner
        r = runner.invoke(app, self._common_args("*3c*"))
        assert r.exit_code == 0, r.output
        assert set(captured["result"]["resolved_tags"]) == {
            "film-bulk_3c-mix", "film-bulk_3c-nomix"
        }

    def test_unmatched_token_aborts(self, sbatch_runner):
        runner, app, captured = sbatch_runner
        r = runner.invoke(app, self._common_args("prod-run,bogus"))
        assert r.exit_code != 0
        assert "bogus" in r.output
        assert captured["_mock_run"].call_count == 0

    def test_zero_runnable_match_aborts(self, sbatch_runner):
        runner, app, captured = sbatch_runner
        args = self._common_args("prod-run")
        i = args.index("--apps")
        args[i + 1] = "vasp"
        r = runner.invoke(app, args)
        assert r.exit_code != 0
        assert "No Ready/Restart jobs match" in r.output
        assert captured["_mock_run"].call_count == 0

    def test_matched_count_and_panel_printed(self, sbatch_runner):
        runner, app, _ = sbatch_runner
        r = runner.invoke(app, self._common_args("prod-run"),
                          env={"COLUMNS": "300", "FORCE_COLOR": "0"})
        assert r.exit_code == 0, r.output
        assert "Matched 1 runnable job" in r.output
        assert "Submit Script" in r.output
        assert "pbx run" in r.output

# API/MCP validation lives in tests/api_tests/test_submission.py — the
# submit_job-level validation reaches every caller, so those tests verify the
# Python API path explicitly.
