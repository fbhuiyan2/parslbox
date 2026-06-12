"""
Tests for the RUN_HOOKS_ON_COMPUTE feature.

Covers:
- Default RUN_HOOKS_ON_COMPUTE=False keeps the in-process behavior unchanged.
- dispatch_hook_on_compute builds the correct subprocess command (resource
  launcher prefix, env_file sourcing, shlex-quoted paths).
- PBX_HOOK_RETURN file-as-return-channel: string → string, empty file → None.
- Subprocess failure (CalledProcessError) propagates so the orchestrator can
  mark the job Failed via its existing exception path.
- File cleanup: args JSON and PBX_HOOK_RETURN are unlinked in `finally`.
- env_file edge cases: None means no `source` prefix; quoting handles spaces.
- _hook_runner allowlist rejects non-{preprocess,postprocess} method names.
- _hook_runner.main() end-to-end with a fake app instance.
"""

import json
import shlex
import subprocess
from pathlib import Path

import pytest

from parslbox.apps import _hook_runner
from parslbox.commands.helpers import hook_dispatch
from parslbox.commands.helpers.hook_dispatch import (
    PBX_HOOK_RETURN_FILE,
    dispatch_hook_on_compute,
)


# ============================================================
# Defaults: RUN_HOOKS_ON_COMPUTE default value
# ============================================================


def test_appbase_default_run_hooks_on_compute_is_false():
    from parslbox.apps.appbase import AppBase
    assert AppBase.RUN_HOOKS_ON_COMPUTE is False


def test_builtin_apps_do_not_opt_in():
    from parslbox.apps.lammps_kk import LammpsKokkosApp
    from parslbox.apps.python import PythonApp
    from parslbox.apps.julia import JuliaApp
    from parslbox.apps.vasp import VaspApp
    assert LammpsKokkosApp.RUN_HOOKS_ON_COMPUTE is False
    assert PythonApp.RUN_HOOKS_ON_COMPUTE is False
    assert JuliaApp.RUN_HOOKS_ON_COMPUTE is False
    assert VaspApp.RUN_HOOKS_ON_COMPUTE is False


# ============================================================
# Helpers
# ============================================================


def _fake_run_writes_return(tmp_path: Path, return_value: str | None):
    """Build a subprocess.run replacement that simulates the runner writing
    PBX_HOOK_RETURN. Pass return_value=None to simulate the runner crashing
    before it could write the file."""
    def fake_run(cmd, *, capture_output, text, check, cwd=None):
        if return_value is not None:
            (tmp_path / PBX_HOOK_RETURN_FILE).write_text(return_value)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return fake_run


def _call_preprocess(tmp_path: Path, **overrides):
    """Default preprocess call. Caller can override individual kwargs."""
    kwargs = dict(
        app_name="example",
        method_name="preprocess",
        single_rank_launcher="mpiexec -n 1 --ppn 1",
        env_file=None,
        # method kwargs:
        job_id=42,
        job_path=tmp_path,
        db_path=tmp_path / "jobs.db",
        app_config={"executable_path": "x"},
        config_name="polaris",
    )
    kwargs.update(overrides)
    return dispatch_hook_on_compute(**kwargs)


def _call_postprocess(tmp_path: Path, **overrides):
    kwargs = dict(
        app_name="example",
        method_name="postprocess",
        single_rank_launcher="srun -N 1 -n 1",
        env_file=None,
        job_id=1,
        job_path=tmp_path,
        db_path=tmp_path / "jobs.db",
    )
    kwargs.update(overrides)
    return dispatch_hook_on_compute(**kwargs)


# ============================================================
# dispatch_hook_on_compute: command construction
# ============================================================


def test_dispatch_builds_bash_c_command_with_launcher_prefix(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, *, capture_output, text, check, cwd=None):
        captured['cmd'] = cmd
        (tmp_path / PBX_HOOK_RETURN_FILE).write_text("Done")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(hook_dispatch.subprocess, "run", fake_run)

    result = _call_preprocess(tmp_path)

    assert result == "Done"
    cmd = captured['cmd']
    assert cmd[0:2] == ["bash", "-c"]
    inner = cmd[2]
    assert "mpiexec -n 1 --ppn 1" in inner
    assert "python -m parslbox.apps._hook_runner" in inner
    assert "preprocess" in inner
    assert "example" in inner


def test_dispatch_sources_env_file_when_present(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, *, capture_output, text, check, cwd=None):
        captured['cmd'] = cmd
        (tmp_path / PBX_HOOK_RETURN_FILE).write_text("")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(hook_dispatch.subprocess, "run", fake_run)

    _call_preprocess(tmp_path, env_file="/etc/env.sh")

    inner = captured['cmd'][2]
    # shlex.quote keeps simple paths unquoted, so accept either form
    assert "source /etc/env.sh && " in inner or "source '/etc/env.sh' && " in inner


def test_dispatch_quotes_env_file_with_spaces(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, *, capture_output, text, check, cwd=None):
        captured['cmd'] = cmd
        (tmp_path / PBX_HOOK_RETURN_FILE).write_text("")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(hook_dispatch.subprocess, "run", fake_run)

    _call_preprocess(tmp_path, env_file="/path with spaces/env.sh")

    inner = captured['cmd'][2]
    # shlex.quote wraps strings with shell metacharacters in single quotes
    assert "'/path with spaces/env.sh'" in inner


def test_dispatch_no_source_when_env_file_is_none(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, *, capture_output, text, check, cwd=None):
        captured['cmd'] = cmd
        (tmp_path / PBX_HOOK_RETURN_FILE).write_text("")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(hook_dispatch.subprocess, "run", fake_run)

    _call_preprocess(tmp_path, env_file=None)

    inner = captured['cmd'][2]
    # No `source ... &&` prefix should be emitted. Check the start of the
    # command rather than substring (tmp paths can contain the word "source").
    assert not inner.startswith("source ")


def test_dispatch_no_source_when_env_file_is_empty_string(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, *, capture_output, text, check, cwd=None):
        captured['cmd'] = cmd
        (tmp_path / PBX_HOOK_RETURN_FILE).write_text("")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(hook_dispatch.subprocess, "run", fake_run)

    _call_preprocess(tmp_path, env_file="")

    inner = captured['cmd'][2]
    # No `source ... &&` prefix should be emitted. Check the start of the
    # command rather than substring (tmp paths can contain the word "source").
    assert not inner.startswith("source ")


# ============================================================
# gpu_env_vars: unset + export pattern
#
# The orchestrator's outer allocation may have CUDA_VISIBLE_DEVICES /
# ZE_AFFINITY_MASK set across the whole node. We MUST unset those before
# exporting whatever ResourceAssignment.get_env_vars() returned for THIS
# specific job — otherwise a CPU-only job sees the orchestrator's GPUs.
# ============================================================


def test_dispatch_always_unsets_gpu_vars_regardless_of_input(tmp_path, monkeypatch):
    """The unset line MUST appear even when gpu_env_vars is None or empty —
    that's the whole point of the defensive clear for CPU-only jobs."""
    captured = {}

    def fake_run(cmd, *, capture_output, text, check, cwd=None):
        captured.setdefault('inners', []).append(cmd[2])
        (tmp_path / PBX_HOOK_RETURN_FILE).write_text("")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(hook_dispatch.subprocess, "run", fake_run)

    # gpu_env_vars omitted (defaults to None)
    _call_preprocess(tmp_path)
    # gpu_env_vars explicitly empty dict
    _call_preprocess(tmp_path, gpu_env_vars={})

    for inner in captured['inners']:
        assert "unset CUDA_VISIBLE_DEVICES ZE_AFFINITY_MASK" in inner
        # And no stale export lines from the orchestrator
        assert "export CUDA_VISIBLE_DEVICES" not in inner
        assert "export ZE_AFFINITY_MASK" not in inner


def test_dispatch_exports_provided_gpu_env_vars(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, *, capture_output, text, check, cwd=None):
        captured['inner'] = cmd[2]
        (tmp_path / PBX_HOOK_RETURN_FILE).write_text("")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(hook_dispatch.subprocess, "run", fake_run)

    _call_preprocess(tmp_path, gpu_env_vars={
        "CUDA_VISIBLE_DEVICES": "0,2",
        "ZE_AFFINITY_MASK": "0.0,2.0",
    })

    inner = captured['inner']
    assert "export CUDA_VISIBLE_DEVICES=0,2" in inner
    assert "export ZE_AFFINITY_MASK=0.0,2.0" in inner
    # Exports come AFTER the unset (so they take effect)
    unset_pos = inner.find("unset CUDA_VISIBLE_DEVICES")
    cuda_export_pos = inner.find("export CUDA_VISIBLE_DEVICES")
    ze_export_pos = inner.find("export ZE_AFFINITY_MASK")
    assert unset_pos != -1 and unset_pos < cuda_export_pos < ze_export_pos


def test_dispatch_command_ordering_source_unset_export_launcher(tmp_path, monkeypatch):
    """Full ordering check: source env_file → unset → export → launcher."""
    captured = {}

    def fake_run(cmd, *, capture_output, text, check, cwd=None):
        captured['inner'] = cmd[2]
        (tmp_path / PBX_HOOK_RETURN_FILE).write_text("")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(hook_dispatch.subprocess, "run", fake_run)

    _call_preprocess(
        tmp_path,
        env_file="/etc/env.sh",
        gpu_env_vars={"CUDA_VISIBLE_DEVICES": "1"},
    )

    inner = captured['inner']
    pos_source = inner.find("source /etc/env.sh")
    pos_unset = inner.find("unset CUDA_VISIBLE_DEVICES")
    pos_export = inner.find("export CUDA_VISIBLE_DEVICES=1")
    pos_launcher = inner.find("mpiexec")
    # Each step is present and comes strictly before the next
    assert -1 < pos_source < pos_unset < pos_export < pos_launcher


def test_dispatch_does_not_set_subprocess_env_kwarg(tmp_path, monkeypatch):
    """GPU env vars are handled via bash `export` in the inner command, NOT
    via subprocess.run's env= kwarg. Setting env= would override the
    orchestrator's PATH/PYTHONPATH/etc. and is fragile w.r.t. mpiexec env
    propagation. This test pins down the design: env= is never passed."""
    captured = {}

    def fake_run(cmd, *, capture_output, text, check, cwd=None, env=None, **extra):
        captured['env_kwarg'] = env
        captured['extra'] = extra
        (tmp_path / PBX_HOOK_RETURN_FILE).write_text("")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(hook_dispatch.subprocess, "run", fake_run)

    _call_preprocess(tmp_path, gpu_env_vars={"CUDA_VISIBLE_DEVICES": "3"})

    assert captured['env_kwarg'] is None, (
        "dispatch_hook_on_compute must not pass env= to subprocess.run — "
        "env management belongs in the bash inner command (see hook_dispatch.py)."
    )


def test_dispatch_rejects_unknown_method(tmp_path):
    with pytest.raises(ValueError, match="method_name must be one of"):
        dispatch_hook_on_compute(
            app_name="example",
            method_name="restart",  # not in allowlist
            single_rank_launcher="x",
            env_file=None,
            job_id=1,
            job_path=tmp_path,
            db_path=tmp_path / "jobs.db",
        )


def test_dispatch_rejects_missing_job_path(tmp_path):
    with pytest.raises(ValueError, match="must include 'job_path'"):
        dispatch_hook_on_compute(
            app_name="example",
            method_name="preprocess",
            single_rank_launcher="x",
            env_file=None,
            # missing job_path
            job_id=1,
            db_path=tmp_path / "jobs.db",
        )


# ============================================================
# Return-value channel: PBX_HOOK_RETURN file
# ============================================================


def test_dispatch_returns_string_from_return_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hook_dispatch.subprocess, "run",
        _fake_run_writes_return(tmp_path, "Done"),
    )
    assert _call_postprocess(tmp_path) == "Done"


def test_dispatch_returns_none_for_empty_return_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hook_dispatch.subprocess, "run",
        _fake_run_writes_return(tmp_path, ""),  # empty file == returned None
    )
    assert _call_postprocess(tmp_path) is None


def test_dispatch_returns_none_when_return_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hook_dispatch.subprocess, "run",
        _fake_run_writes_return(tmp_path, None),  # don't write the file
    )
    assert _call_postprocess(tmp_path) is None


# ============================================================
# Failure propagation and file cleanup
# ============================================================


def test_dispatch_raises_on_subprocess_failure(tmp_path, monkeypatch):
    def fake_run(cmd, *, capture_output, text, check, cwd=None):
        # check=True semantics: simulate non-zero exit by raising
        raise subprocess.CalledProcessError(1, cmd, output="", stderr="boom")

    monkeypatch.setattr(hook_dispatch.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        _call_preprocess(tmp_path)

    # Even after the failure, the finally block must have cleaned up the
    # args JSON and any stale PBX_HOOK_RETURN file.
    assert not (tmp_path / "PBX_HOOK_ARGS_preprocess.json").exists()
    assert not (tmp_path / PBX_HOOK_RETURN_FILE).exists()


def test_dispatch_cleans_up_files_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hook_dispatch.subprocess, "run",
        _fake_run_writes_return(tmp_path, "Done"),
    )
    _call_preprocess(tmp_path)
    assert not (tmp_path / "PBX_HOOK_ARGS_preprocess.json").exists()
    assert not (tmp_path / PBX_HOOK_RETURN_FILE).exists()


def test_dispatch_clears_stale_return_file_before_run(tmp_path, monkeypatch):
    """A leftover PBX_HOOK_RETURN from a previous run must not poison the read."""
    (tmp_path / PBX_HOOK_RETURN_FILE).write_text("Stale")

    seen_during_run = {}

    def fake_run(cmd, *, capture_output, text, check, cwd=None):
        f = tmp_path / PBX_HOOK_RETURN_FILE
        seen_during_run['exists_before_runner'] = f.exists()
        f.write_text("Fresh")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(hook_dispatch.subprocess, "run", fake_run)

    result = _call_postprocess(tmp_path)
    assert seen_during_run['exists_before_runner'] is False, \
        "stale PBX_HOOK_RETURN should have been cleared before invoking the runner"
    assert result == "Fresh"


def test_dispatch_runs_subprocess_with_cwd_set_to_job_path(tmp_path, monkeypatch):
    """Relative paths in single_rank_launcher (mpiexec --rankfile ./rankfile_*.txt,
    GPU wrapper scripts) live in job_path. The bash subprocess must run from
    job_path so those resolve — same convention as the bash_app's `cd job_path`.
    Regression test for the 'mpiexec exit 127' failure mode."""
    captured = {}

    def fake_run(cmd, *, capture_output, text, check, cwd=None):
        captured['cwd'] = cwd
        (tmp_path / PBX_HOOK_RETURN_FILE).write_text("")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(hook_dispatch.subprocess, "run", fake_run)

    _call_preprocess(tmp_path)

    assert captured['cwd'] == str(tmp_path)


def test_dispatch_serializes_path_args_as_strings(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, *, capture_output, text, check, cwd=None):
        tokens = shlex.split(cmd[-1])
        args_path = next(t for t in tokens if t.endswith(".json"))
        captured['args'] = json.loads(Path(args_path).read_text())
        (tmp_path / PBX_HOOK_RETURN_FILE).write_text("")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(hook_dispatch.subprocess, "run", fake_run)

    _call_preprocess(tmp_path)

    args = captured['args']
    assert args['job_id'] == 42
    assert args['job_path'] == str(tmp_path)
    assert args['db_path'] == str(tmp_path / "jobs.db")
    assert args['config_name'] == "polaris"


# ============================================================
# _hook_runner module: direct invocation
# ============================================================


class _FakeApp:
    """Used by hook_runner tests via monkeypatching get_app_instance."""

    def __init__(self):
        self.called = []

    def preprocess(self, *, job_id, job_path, db_path, app_config, config_name):
        self.called.append(("preprocess", job_id, job_path, db_path))
        return None  # preprocess returns None

    def postprocess(self, *, job_id, job_path, db_path):
        self.called.append(("postprocess", job_id, job_path, db_path))
        return "Done"


def test_hook_runner_writes_return_file_and_cleans_args(tmp_path, monkeypatch):
    args_file = tmp_path / "args.json"
    args_file.write_text(json.dumps({
        "job_id": 7,
        "job_path": str(tmp_path),
        "db_path": str(tmp_path / "jobs.db"),
    }))

    fake = _FakeApp()
    monkeypatch.setattr(_hook_runner, "get_app_instance", lambda name: fake)

    rc = _hook_runner.main(["fakeapp", "postprocess", str(args_file)])
    assert rc == 0
    assert (tmp_path / PBX_HOOK_RETURN_FILE).read_text() == "Done"
    assert not args_file.exists(), "args JSON must be deleted by the runner's finally"
    # Method called with rehydrated Path objects
    assert fake.called[0][0] == "postprocess"
    assert isinstance(fake.called[0][2], Path)
    assert isinstance(fake.called[0][3], Path)


def test_hook_runner_preprocess_writes_empty_string_for_none(tmp_path, monkeypatch):
    args_file = tmp_path / "args.json"
    args_file.write_text(json.dumps({
        "job_id": 7,
        "job_path": str(tmp_path),
        "db_path": str(tmp_path / "jobs.db"),
        "app_config": {},
        "config_name": "polaris",
    }))

    monkeypatch.setattr(_hook_runner, "get_app_instance", lambda name: _FakeApp())

    rc = _hook_runner.main(["fakeapp", "preprocess", str(args_file)])
    assert rc == 0
    assert (tmp_path / PBX_HOOK_RETURN_FILE).read_text() == ""


def test_hook_runner_rejects_disallowed_method(tmp_path):
    args_file = tmp_path / "args.json"
    args_file.write_text(json.dumps({"job_path": str(tmp_path)}))
    rc = _hook_runner.main(["fakeapp", "restart", str(args_file)])
    assert rc == 2


def test_hook_runner_propagates_exception_as_non_zero_exit(tmp_path, monkeypatch):
    args_file = tmp_path / "args.json"
    args_file.write_text(json.dumps({
        "job_id": 1,
        "job_path": str(tmp_path),
        "db_path": str(tmp_path / "jobs.db"),
    }))

    class _BoomApp:
        def postprocess(self, **kw):
            raise RuntimeError("boom from postprocess")

    monkeypatch.setattr(_hook_runner, "get_app_instance", lambda name: _BoomApp())

    rc = _hook_runner.main(["fakeapp", "postprocess", str(args_file)])
    assert rc == 1
    assert not (tmp_path / PBX_HOOK_RETURN_FILE).exists(), \
        "no return file should be written on failure"
    assert not args_file.exists(), "args JSON cleaned in finally even on failure"


def test_hook_runner_bad_argv_returns_2(tmp_path):
    assert _hook_runner.main([]) == 2
    assert _hook_runner.main(["only-one"]) == 2


# ============================================================
# Regression: run.py must read job['app'], not job['app_name']
#
# DB schema has a column named `app`, not `app_name` (database.py).
# Twice we shipped run.py code (in the RUN_HOOKS_ON_COMPUTE = True
# branches) that read `job['app_name']`, blowing up with KeyError
# the moment any custom app opted into compute-side hooks. Existing
# tests in this file mock dispatch_hook_on_compute(...) directly and
# never exercise the caller, so the bug slipped through. These two
# tests pin down the convention on both sides.
# ============================================================


def test_db_job_row_uses_app_key_not_app_name(tmp_path):
    """A job row fetched from the DB must expose its app name under key 'app'.

    Locks in the schema contract the RUN_HOOKS_ON_COMPUTE caller depends on.
    """
    from parslbox.database import database

    db_path = tmp_path / "jobs.db"
    job_dir = tmp_path / "j1"
    job_dir.mkdir()
    database.initialize_database(db_path)
    database.add_job(
        db_path,
        path=str(job_dir),
        app="lammps-kk",
        num_nodes=1,
        ngpus=0,
        node_occupancy=1.0,
        tag=None,
    )
    rows = database.get_jobs(db_path)
    assert len(rows) == 1
    row = rows[0]
    assert "app" in row, "DB row must expose app name under 'app'"
    assert "app_name" not in row, (
        "DB row must NOT expose 'app_name' — the schema column is 'app' "
        "(see database.py). Callers that read job['app_name'] crash with KeyError."
    )
    assert row["app"] == "lammps-kk"


def test_run_py_does_not_read_job_app_name(tmp_path):
    """Source-level guard against the KeyError regression on run.py:156 / :820.

    Both dispatch_hook_on_compute call sites in run.py must read job['app'],
    matching the DB column. job['app_name'] does not exist in any row dict and
    raises KeyError the first time a RUN_HOOKS_ON_COMPUTE = True app reaches
    the dispatch path.
    """
    import re
    from pathlib import Path
    import parslbox.commands.run as run_mod

    src = Path(run_mod.__file__).read_text()
    # Match both quoting styles
    bad = re.findall(r"""job\[\s*['"]app_name['"]\s*\]""", src)
    assert not bad, (
        f"run.py contains {len(bad)} reference(s) to job['app_name'] — "
        "the DB column is 'app' (see database.py). Use job['app'] instead. "
        "This bug previously broke every custom app with "
        "RUN_HOOKS_ON_COMPUTE = True on its first dispatch."
    )
