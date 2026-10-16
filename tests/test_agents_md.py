"""AGENTS.md is the entry point an agent reads after cloning the repo.

Every path it names is an instruction the agent will act on, so a stale link
sends it somewhere that does not exist -- and unlike a human, it has no
README index to fall back to. These tests keep the pointers honest.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
AGENTS = REPO / "AGENTS.md"


@pytest.fixture(scope="module")
def text():
    return AGENTS.read_text()


def test_it_exists_and_is_tracked(text):
    """An untracked AGENTS.md is invisible in a fresh clone, which is the
    only situation it exists for."""
    import subprocess
    tracked = subprocess.run(["git", "ls-files", "AGENTS.md"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()
    assert tracked == "AGENTS.md"


@pytest.mark.parametrize("target", sorted(set(
    re.findall(r"\]\((?!https?:)([^)#]+)\)", AGENTS.read_text())
)))
def test_every_link_resolves(target):
    assert (REPO / target).exists(), target


def test_it_points_at_all_three_skills(text):
    for skill in ("pbx-setup", "parslbox-cli", "parslbox-api"):
        assert f"skills/{skill}/SKILL.md" in text, skill


def test_every_command_it_names_exists(text):
    for script in re.findall(r"`(scripts/[\w.-]+\.sh)", text):
        assert (REPO / script).is_file(), script


def test_the_readme_points_here(text):
    readme = (REPO / "README.md").read_text()
    head = readme[:readme.index("## Highlights")]
    assert "AGENTS.md" in head, "the pointer must be above the fold, not buried"


def test_the_config_keys_it_warns_about_are_real(text):
    """It tells the agent a fresh config is a scaffold; the two keys it names
    have to be the ones actually left blank."""
    from parslbox.utils.config_generator import ConfigGenerator
    generated = ConfigGenerator(["polaris"], ["vasp"]).generate()
    for key in ("pbx_python_env_setup", "executable_path"):
        assert key in text and key in generated, key
