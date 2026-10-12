"""The two agent skills bundle their own copies of docs/.

`skills/parslbox-api/docs/` and `skills/parslbox-cli/docs/` are byte copies of
the top-level `docs/`. Nothing kept them that way, so editing a doc updated
the copy an agent never reads and left the one it does read stale -- the same
hand-maintained-parallel-list failure as the MCP server's prose `instructions`
block and the schema/API parameter lists, both of which now have guards.
"""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SKILLS = REPO / "skills"


def _copies():
    for skill_doc in sorted(SKILLS.glob("*/docs/*.md")):
        yield skill_doc, REPO / "docs" / skill_doc.name


@pytest.mark.parametrize("skill_doc,canonical",
                         list(_copies()),
                         ids=lambda p: str(p.relative_to(REPO)) if isinstance(p, Path) else p)
def test_bundled_copy_matches_the_canonical_doc(skill_doc, canonical):
    assert canonical.is_file(), f"{skill_doc} has no counterpart in docs/"
    assert skill_doc.read_text() == canonical.read_text(), (
        f"{skill_doc.relative_to(REPO)} has drifted from "
        f"{canonical.relative_to(REPO)} -- copy the canonical file over it."
    )


def test_there_is_something_to_check():
    # A glob that matched nothing would make every assertion above vacuous.
    assert len(list(_copies())) >= 8


def test_every_api_method_named_in_the_skill_exists():
    """The skill's method table is prose an LLM reads as ground truth; a
    renamed method there sends it to one that does not exist."""
    import re
    from parslbox.api import ParslBox

    text = (SKILLS / "parslbox-api" / "SKILL.md").read_text()
    named = set(re.findall(r"^\| `(\w+)\(", text, re.M))
    named |= set(re.findall(r"`(\w+)\(", text))
    public = {n for n in named if not n.startswith("_")}
    # Only check names the table presents as ParslBox methods.
    table_names = set(re.findall(r"^\| `(\w+)\(", text, re.M))
    missing = {n for n in table_names if not hasattr(ParslBox, n)}
    assert not missing, f"SKILL.md names methods that do not exist: {sorted(missing)}"
    assert public  # the regex still matches something


def test_the_skill_table_covers_every_public_api_method():
    import inspect
    import re
    from parslbox.api import ParslBox

    text = (SKILLS / "parslbox-api" / "SKILL.md").read_text()
    documented = set(re.findall(r"`(\w+)\(", text))
    public = {n for n, _ in inspect.getmembers(ParslBox, inspect.isfunction)
              if not n.startswith("_")}
    assert not public - documented, (
        f"API methods absent from SKILL.md: {sorted(public - documented)}")
