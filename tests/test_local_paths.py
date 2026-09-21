"""The local -> remote prefix swap and its containment rules."""

import pytest

from parslbox.local.paths import is_under_local_root, normalize, to_local, to_remote
from parslbox.local.project import LocalProject

REMOTE = "/lus/flare/projects/CSTEELML/fbhuiyan/proj1"


@pytest.fixture
def proj(tmp_path):
    return LocalProject(local_root=tmp_path, created="2026-09-10T00:00:00Z",
                        remote_root=REMOTE)


def test_path_under_root_is_swapped(proj, tmp_path):
    assert to_remote(tmp_path / "run_a", proj) == f"{REMOTE}/run_a"


def test_nested_path_keeps_its_whole_tail(proj, tmp_path):
    assert to_remote(tmp_path / "a" / "b" / "c", proj) == f"{REMOTE}/a/b/c"


def test_the_root_itself_maps_to_the_remote_root(proj, tmp_path):
    assert to_remote(tmp_path, proj) == REMOTE


def test_dotdot_is_collapsed_before_the_swap(proj, tmp_path):
    assert to_remote(tmp_path / "a" / ".." / "b", proj) == f"{REMOTE}/b"


def test_path_outside_the_root_passes_through(proj):
    assert to_remote("/scratch/shared/env.sh", proj) == "/scratch/shared/env.sh"


def test_a_sibling_with_the_same_prefix_is_not_under_the_root(tmp_path):
    """proj1-other must not be mistaken for something inside proj1."""
    root = tmp_path / "proj1"
    root.mkdir()
    sibling = tmp_path / "proj1-other"
    proj = LocalProject(local_root=root, created="x", remote_root=REMOTE)
    assert is_under_local_root(sibling, proj) is False
    assert to_remote(sibling / "run", proj) == str(sibling / "run")


def test_symlinks_are_not_followed(proj, tmp_path):
    """resolve() would follow the link out of the root and break the swap."""
    outside = tmp_path.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    link = tmp_path / "run_link"
    link.symlink_to(outside)
    assert is_under_local_root(link, proj) is True
    assert to_remote(link, proj) == f"{REMOTE}/run_link"


def test_normalize_expands_user(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert normalize("~/work") == tmp_path / "work"


def test_to_local_inverts_the_swap(proj, tmp_path):
    assert to_local(f"{REMOTE}/run_a", proj) == tmp_path / "run_a"


def test_to_local_returns_none_outside_the_remote_root(proj):
    assert to_local("/lus/flare/somebody/else", proj) is None


def test_round_trip(proj, tmp_path):
    original = tmp_path / "a" / "b"
    assert to_local(to_remote(original, proj), proj) == original
