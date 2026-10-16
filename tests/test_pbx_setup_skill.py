"""The /pbx-setup playbook is prose an agent follows literally.

Its instructions name config keys, placeholder strings and API methods. If
any of those change in the code, the playbook keeps saying the old thing and
the agent silently does the wrong work -- the worst case being the
placeholder scan, which would report a clean config while every executable
path is still unfilled. These tests pin the playbook to the code.
"""
import re
from pathlib import Path

import pytest
import yaml

from parslbox.api import ParslBox
from parslbox.utils.config_generator import ConfigGenerator

REPO = Path(__file__).resolve().parent.parent
SKILL = REPO / "skills" / "pbx-setup" / "SKILL.md"


@pytest.fixture(scope="module")
def skill_text():
    return SKILL.read_text()


@pytest.fixture(scope="module")
def generated():
    """A config covering an interpreter app and compiled apps."""
    return ConfigGenerator(["polaris", "perlmutter-cpu"],
                           ["python", "julia", "vasp", "lammps-kk", "orca"]).generate()


class TestTheSkillIsInstallable:
    def test_it_has_frontmatter_naming_the_command(self, skill_text):
        assert skill_text.startswith("---\n")
        front = yaml.safe_load(skill_text.split("---")[1])
        assert front["name"] == "pbx-setup"
        assert front["description"].strip()

    def test_every_doc_it_links_is_bundled(self, skill_text):
        for ref in set(re.findall(r"\(docs/([\w.-]+\.md)\)", skill_text)):
            assert (SKILL.parent / "docs" / ref).is_file(), ref


class TestThePlaceholderScanStillMatches:
    """The playbook greps for these literals to decide what is unfilled."""

    def test_the_executable_placeholder_is_what_the_generator_writes(self, skill_text, generated):
        assert "/path/to/your/" in skill_text
        assert "/path/to/your/" in generated

    def test_an_interpreter_app_leaves_no_executable_placeholder(self, generated):
        """The playbook tells the agent python and julia have no
        executable_path; if that changed, it would skip real blanks."""
        data = yaml.safe_load(generated)
        for app in ("python", "julia"):
            assert "executable_path" not in data[app]["polaris"]

    def test_every_other_builtin_does_have_one(self, generated):
        data = yaml.safe_load(generated)
        for app in ("vasp", "lammps-kk", "orca"):
            assert data[app]["polaris"]["executable_path"].startswith("/path/to/your/")

    ANCHORED = re.compile(r'^[ \t]*executable_path: "/path/to/your/', re.M)

    def test_the_anchored_grep_finds_one_per_compiled_app_and_system(self, generated):
        # 3 compiled apps x 2 systems; python and julia contribute none.
        assert len(self.ANCHORED.findall(generated)) == 6

    def test_an_unanchored_grep_would_over_report(self, generated):
        """The commented example-app template contains the same string, so a
        bare grep reports a leftover even when every path is filled."""
        assert generated.count("/path/to/your/") > len(self.ANCHORED.findall(generated))

    def test_the_skill_uses_the_anchored_pattern(self, skill_text):
        assert 'executable_path: "/path/to/your/' in skill_text
        assert 'grep -n "/path/to/your/"' not in skill_text


class TestTheFatalBlank:
    def test_pbx_python_env_setup_is_emitted_for_every_system(self, generated):
        data = yaml.safe_load(generated)
        for system in ("polaris", "perlmutter-cpu"):
            assert "pbx_python_env_setup" in data[system]

    def test_it_is_generated_empty(self, generated):
        """Every line a comment -- which is why the playbook must fill it."""
        block = yaml.safe_load(generated)["polaris"]["pbx_python_env_setup"]
        assert block.strip()
        assert all(ln.strip().startswith("#") for ln in block.strip().splitlines())

    def test_the_skill_names_the_key_and_the_symptom(self, skill_text):
        assert "pbx_python_env_setup" in skill_text
        assert "pbx: command not found" in skill_text

    def test_it_is_injected_before_the_run_command(self):
        """The reason a blank block is fatal: submit.sh substitutes it just
        above `exec pbx run`."""
        from parslbox.utils.pbx_config_template import PBS_TEMPLATE, SLURM_TEMPLATE
        for template in (PBS_TEMPLATE, SLURM_TEMPLATE):
            assert template.index("{pbx_python_env_setup}") < template.index("exec pbx run")


class TestThePurgeRestoreRule:
    def test_the_generator_emits_them_live_not_commented(self, generated):
        block = yaml.safe_load(generated)["vasp"]["polaris"]["environment_setup"]
        live = [ln.strip() for ln in block.splitlines() if not ln.strip().startswith("#")]
        assert live[:2] == ["module purge", "module restore"]

    def test_the_skill_says_to_keep_them(self, skill_text):
        assert "module purge" in skill_text and "module restore" in skill_text


class TestTheInstallBranch:
    """With a conda env active (base counts, and it is often auto-activated),
    Poetry ignores the uv/venv environment you just created and installs into
    its own under ~/.cache/pypoetry/virtualenvs -- so `pbx` never lands on
    PATH. Verified in a clean copy of the repo: identical steps gave
    .uvenv/bin/pbx ABSENT with CONDA_PREFIX set and PRESENT without it."""

    def test_it_matches_the_install_the_readme_documents(self, skill_text):
        """Not just "both mention poetry" -- the actual uv command lines the
        skill tells an agent to run must appear in README's uv section."""
        install = skill_text.split("## Phase 1")[1].split("## Phase 2")[0]
        readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
        uv_section = readme.split("### uv")[1].split("###")[0]
        for cmd in ("uv venv -p 3.11.9 .uvenv", "uv pip install poetry", "poetry install"):
            assert cmd in install, cmd
            assert cmd in uv_section, cmd

    def test_it_does_not_tell_an_agent_to_run_conda_deactivate(self, skill_text):
        """`conda deactivate` is a shell function that does not exist in a
        non-interactive shell -- the only kind an agent gets. It fails with
        "Run 'conda init' before 'conda deactivate'" and leaves CONDA_PREFIX
        set, so an agent told to use it proceeds believing it is cleared."""
        install = skill_text.split("## Phase 1")[1].split("## Phase 2")[0]
        assert "env -u CONDA_PREFIX" in install
        import re
        for line in install.splitlines():
            bare = line.strip().strip("`")
            assert bare != "conda deactivate", line

    def test_it_warns_about_an_active_conda_env(self, skill_text):
        install = skill_text.split("## Phase 1")[1].split("## Phase 2")[0]
        assert "CONDA_PREFIX" in install
        assert "pypoetry" in install

    def test_it_does_not_probe_a_flag_that_does_not_exist(self, skill_text):
        """`pbx --version` is not a thing -- the probe would exit 2 and look
        like a broken install rather than a missing one."""
        from parslbox.main import app
        import typer, click
        cli = typer.main.get_command(app)
        params = {o for p in cli.params for o in p.opts}
        assert "--version" not in params
        # Named only to warn against it, never run as a probe.
        install = skill_text.split("## Phase 1")[1].split("## Phase 2")[0]
        assert "command -v pbx" in install
        assert not re.search(r"^\s*pbx --version", install, re.M)

    def test_it_checks_the_import_from_outside_the_repo(self, skill_text):
        """Inside the repo, `import parslbox` resolves from the source tree
        whether or not anything is installed."""
        install = skill_text.split("## Phase 1")[1].split("## Phase 2")[0]
        assert "cd /tmp" in install


class TestEveryApiSymbolItNamesExists:
    @pytest.mark.parametrize("name", ["available_systems", "builtin_apps",
                                      "create_config", "available_apps"])
    def test_the_method_is_on_parslbox(self, name, skill_text):
        assert hasattr(ParslBox, name)
        assert name in skill_text

    def test_create_config_takes_the_arguments_it_passes(self):
        import inspect
        params = inspect.signature(ParslBox.create_config).parameters
        assert {"path", "systems", "apps", "force"} <= set(params)
