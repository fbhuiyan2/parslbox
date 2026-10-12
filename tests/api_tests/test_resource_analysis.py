"""API tests for resource_requirements() and command_lines().

Both were CLI-only until now: `pbx info --req` could size a project but no
agent or script could, and the system constants it needs (GPUS_PER_NODE,
CORES_PER_NODE) are not reachable any other way, so the numbers could not
even be recomputed by hand.
"""
import pytest

from parslbox.api import ValidationError


@pytest.fixture
def seeded(pbx, tmp_path):
    """Three Ready jobs: one GPU single-node, one GPU multi-node, one CPU."""
    env_file = tmp_path / "env.sh"
    env_file.write_text("#!/bin/bash\n")
    specs = [
        dict(app="vasp", nnodes=1, ngpus=4, input_file="INCAR"),
        dict(app="lammps-kk", nnodes=2, ngpus=24, input_file="in.lammps"),
        dict(app="python", nnodes=1, ngpus=0, node_occupancy=0.5, input_file="run.py"),
    ]
    for i, s in enumerate(specs):
        d = tmp_path / f"job{i}"
        d.mkdir()
        kwargs = dict(paths=[str(d)], config="polaris", **s)
        if s["app"] == "python":
            kwargs["env_file"] = str(env_file)
        pbx.add_jobs(**kwargs)
    return pbx


class TestResourceRequirements:
    def test_it_sizes_every_job_when_no_ids_are_given(self, seeded):
        req = seeded.resource_requirements("aurora-tile")
        assert req["schedulable"] == 3
        assert req["gpu"]["jobs"] == 2
        assert req["cpu"]["jobs"] == 1

    def test_named_ids_size_only_those_jobs(self, seeded):
        req = seeded.resource_requirements("aurora-tile", job_ids=[1])
        assert req["schedulable"] == 1
        assert req["cpu"] is None

    def test_it_reports_the_system_constants(self, seeded):
        req = seeded.resource_requirements("aurora-tile")
        assert (req["gpus_per_node"], req["cores_per_node"]) == (12, 208)

    def test_an_unknown_system_raises_validationerror(self, pbx):
        with pytest.raises(ValidationError, match="Unknown system"):
            pbx.resource_requirements("aurora-tyle")

    def test_an_empty_database_sizes_to_nothing(self, pbx):
        req = pbx.resource_requirements("polaris")
        assert req["schedulable"] == 0
        assert req["gpu"] is None

    def test_it_changes_nothing(self, seeded):
        before = seeded.list_jobs()
        seeded.resource_requirements("aurora-tile")
        assert seeded.list_jobs() == before


class TestCommandLines:
    def test_it_previews_one_row_per_job(self, seeded):
        out = seeded.command_lines("aurora-tile")
        assert [r["job_id"] for r in out["jobs"]] == [1, 2, 3]
        assert all(r["command"] for r in out["jobs"])

    def test_named_ids_preview_only_those_jobs(self, seeded):
        out = seeded.command_lines("aurora-tile", job_ids=[2])
        assert len(out["jobs"]) == 1
        assert out["jobs"][0]["resources"] == "2N/24G"

    def test_an_unknown_system_raises_validationerror(self, pbx):
        with pytest.raises(ValidationError, match="Unknown system"):
            pbx.command_lines("not-a-machine")

    def test_it_previews_against_this_instance_s_config(self, seeded, monkeypatch):
        # The API threads config_path everywhere else; a preview reading the
        # process-wide default would show another project's MPI settings after
        # use_database rebinds the server.
        seen = {}
        monkeypatch.setattr(
            "parslbox.utils.pbx_config_utils.load_full_config",
            lambda config_path=None: seen.setdefault("path", config_path) or {},
        )
        seeded.command_lines("aurora-tile", job_ids=[1])
        assert seen["path"] == seeded.config_path

    def test_it_changes_nothing(self, seeded):
        before = seeded.list_jobs()
        seeded.command_lines("aurora-tile")
        assert seeded.list_jobs() == before
