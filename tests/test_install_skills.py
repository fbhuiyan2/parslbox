"""scripts/install-skills.sh places the skills where each harness looks.

The script is the only thing standing between a cloned repo and a usable
`/pbx-setup`, and it writes into the user's home directory -- so these tests
run it with HOME redirected into tmp_path and check both that it installs and
that it refuses to damage anything it did not create.
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "install-skills.sh"
SKILLS = sorted(d.name for d in (REPO / "skills").iterdir() if (d / "SKILL.md").is_file())


def run(home, *args):
    return subprocess.run(
        ["sh", str(SCRIPT), *args],
        capture_output=True, text=True,
        env={**os.environ, "HOME": str(home)},
    )


@pytest.fixture
def home(tmp_path):
    """A home with two harnesses already present."""
    (tmp_path / ".claude" / "skills").mkdir(parents=True)
    (tmp_path / ".codex" / "skills").mkdir(parents=True)
    return tmp_path


class TestItInstalls:
    def test_every_skill_lands_in_every_detected_harness(self, home):
        assert run(home).returncode == 0
        for harness in (".claude/skills", ".codex/skills"):
            for skill in SKILLS:
                assert (home / harness / skill).is_dir(), f"{harness}/{skill}"

    def test_there_is_something_to_install(self):
        # A glob that matched nothing would make every assertion here vacuous.
        assert len(SKILLS) >= 3

    def test_they_are_symlinks_into_the_repo(self, home):
        run(home)
        link = home / ".claude" / "skills" / "pbx-setup"
        assert link.is_symlink()
        assert os.readlink(link) == str(REPO / "skills" / "pbx-setup")

    def test_the_skill_is_usable_through_the_link(self, home):
        """A harness reads SKILL.md and the bundled docs through the link."""
        run(home)
        installed = home / ".claude" / "skills" / "pbx-setup"
        assert (installed / "SKILL.md").read_text().startswith("---")
        assert (installed / "docs" / "config-file.md").is_file()

    def test_it_skips_harnesses_that_are_not_installed(self, home):
        run(home)
        assert not (home / ".gemini").exists()

    def test_a_named_harness_is_created_even_if_absent(self, home):
        assert run(home, "--harness", "gemini").returncode == 0
        assert (home / ".gemini" / "skills" / "pbx-setup").is_symlink()

    def test_running_twice_changes_nothing(self, home):
        run(home)
        assert run(home).returncode == 0
        assert run(home, "--check").returncode == 0


class TestItRefusesToDamageThings:
    def test_a_directory_it_did_not_create_is_left_alone(self, home):
        foreign = home / ".codex" / "skills" / "pbx-setup"
        foreign.mkdir(parents=True)
        (foreign / "SKILL.md").write_text("someone else's skill")

        result = run(home, "--harness", "codex")

        assert result.returncode == 1
        assert "refusing to replace" in result.stderr
        assert (foreign / "SKILL.md").read_text() == "someone else's skill"

    def test_the_other_skills_still_install_around_it(self, home):
        (home / ".codex" / "skills" / "pbx-setup").mkdir(parents=True)
        run(home, "--harness", "codex")
        assert (home / ".codex" / "skills" / "parslbox-cli").is_symlink()

    def test_uninstall_leaves_foreign_entries(self, home):
        foreign = home / ".claude" / "skills" / "someone-elses"
        foreign.mkdir(parents=True)
        run(home)
        run(home, "--uninstall")
        assert foreign.is_dir()
        assert not (home / ".claude" / "skills" / "pbx-setup").exists()

    def test_an_unknown_harness_is_an_error(self, home):
        result = run(home, "--harness", "nosuchharness")
        assert result.returncode == 2
        assert "unknown harness" in result.stderr

    def test_no_harness_at_all_explains_what_to_do(self, tmp_path):
        result = run(tmp_path)
        assert result.returncode == 1
        assert "--harness" in result.stderr


class TestAHomeWithASpaceInIt:
    """The target lists are newline-separated and split with IFS set to
    newline. Space-separated, a home like /Users/Ada Lovelace field-split
    into bogus targets: the script printed success, left the real directory
    empty, and created stray dirs relative to the current directory."""

    @pytest.fixture
    def spaced(self, tmp_path):
        home = tmp_path / "a home with spaces"
        (home / ".claude" / "skills").mkdir(parents=True)
        return home

    def test_it_installs_where_it_says_it_did(self, spaced):
        assert run(spaced).returncode == 0
        assert (spaced / ".claude" / "skills" / "pbx-setup").is_symlink()

    def test_an_explicit_harness_works_too(self, spaced):
        run(spaced, "--harness", "claude")
        assert (spaced / ".claude" / "skills" / "pbx-setup").is_symlink()

    def test_it_creates_nothing_in_the_working_directory(self, spaced, tmp_path):
        before = set(os.listdir(REPO))
        run(spaced)
        assert set(os.listdir(REPO)) == before

    def test_check_and_uninstall_survive_it(self, spaced):
        run(spaced)
        assert run(spaced, "--check").returncode == 0
        run(spaced, "--uninstall")
        assert not (spaced / ".claude" / "skills" / "pbx-setup").exists()


class TestAForeignSymlinkIsNotOurs:
    """A directory we did not create is refused; a symlink we did not create
    used to be deleted unconditionally, which is the same loss with less
    warning -- someone's own skill, or their link to it, silently replaced."""

    def test_it_is_refused(self, home, tmp_path):
        theirs = tmp_path / "their-own-skill"
        theirs.mkdir()
        link = home / ".claude" / "skills" / "pbx-setup"
        link.symlink_to(theirs)

        result = run(home, "--harness", "claude")

        assert result.returncode == 1
        assert "refusing to replace" in result.stderr
        assert os.readlink(link) == str(theirs)

    def test_the_other_skills_still_install(self, home, tmp_path):
        (home / ".claude" / "skills" / "pbx-setup").symlink_to(tmp_path / "x")
        run(home, "--harness", "claude")
        assert (home / ".claude" / "skills" / "parslbox-cli").is_symlink()

    def test_our_own_link_is_replaced_silently(self, home):
        run(home, "--harness", "claude")
        result = run(home, "--harness", "claude")
        assert result.returncode == 0
        assert "refusing" not in result.stderr


class TestCopyMode:
    def test_it_writes_real_directories(self, home):
        run(home, "--copy")
        installed = home / ".claude" / "skills" / "pbx-setup"
        assert installed.is_dir() and not installed.is_symlink()
        assert (installed / "SKILL.md").read_text() == (
            REPO / "skills" / "pbx-setup" / "SKILL.md").read_text()

    def test_a_copy_is_still_uninstallable(self, home):
        """Copies carry a marker so uninstall can tell them from a user's own."""
        run(home, "--copy")
        run(home, "--uninstall")
        assert not (home / ".claude" / "skills" / "pbx-setup").exists()


class TestCheck:
    def test_it_fails_when_a_skill_is_missing(self, home):
        run(home)
        (home / ".claude" / "skills" / "pbx-setup").unlink()
        result = run(home, "--check")
        assert result.returncode == 1
        assert "missing" in result.stderr

    def test_it_fails_on_a_stale_link(self, home):
        run(home)
        link = home / ".claude" / "skills" / "pbx-setup"
        link.unlink()
        link.symlink_to("/somewhere/else")
        assert run(home, "--check").returncode == 1


class TestTheDoubleLoadWarning:
    """OpenCode scans ~/.config/opencode/skills, ~/.claude/skills AND
    ~/.agents/skills, so installing into several of them loads each skill
    once per location."""

    def test_it_fires_when_two_of_them_are_written(self, home):
        (home / ".config" / "opencode" / "skills").mkdir(parents=True)
        out = run(home).stdout
        assert "WARNING" in out
        assert "opencode" in out
        assert "~/.claude/skills" in out
        assert "~/.config/opencode/skills" in out

    def test_it_names_the_way_out(self, home):
        (home / ".config" / "opencode" / "skills").mkdir(parents=True)
        assert "--uninstall --harness opencode" in run(home).stdout

    def test_it_stays_quiet_for_a_single_location(self, home):
        assert "WARNING" not in run(home, "--harness", "claude").stdout

    def test_a_non_overlapping_pair_is_not_warned_about(self, home):
        """codex and claude are read by different harnesses -- no duplicate."""
        assert "WARNING" not in run(home, "--harness", "claude,codex").stdout
