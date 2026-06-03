"""Glob-style tag matching helpers.

A pattern containing `*` is treated as a glob (matches any sequence of chars);
patterns without `*` are matched exactly.
"""
from typing import List


def has_glob(pattern: str) -> bool:
    return '*' in pattern


def to_sql_like(pattern: str) -> str:
    """Translate a `*`-glob pattern to a SQL LIKE pattern.

    Escapes existing SQL LIKE wildcards (`%`, `_`) in the input so they match
    literally, then maps `*` -> `%`. Use with `ESCAPE '\\'`.
    """
    escaped = pattern.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
    return escaped.replace('*', '%')


def expand_tag_patterns(db_path, patterns: List[str]) -> List[str]:
    """Expand a mixed list of literal tags and `*`-globs into the literal tag
    values found in the database.

    Strict: every pattern (glob or literal) must match at least one tag in the
    DB; otherwise raises ValueError listing the unmatched patterns.

    Returns a de-duplicated literal-tag list, in first-seen order.
    """
    from parslbox.database.database import get_configured_connection

    expanded: List[str] = []
    seen = set()
    unmatched: List[str] = []

    def add(t: str):
        if t not in seen:
            seen.add(t)
            expanded.append(t)

    with get_configured_connection(db_path) as con:
        cur = con.cursor()
        for p in patterns:
            if has_glob(p):
                rows = cur.execute(
                    "SELECT DISTINCT tag FROM jobs WHERE tag IS NOT NULL AND tag LIKE ? ESCAPE '\\'",
                    (to_sql_like(p),),
                ).fetchall()
                if not rows:
                    unmatched.append(p)
                    continue
                for (tag,) in rows:
                    add(tag)
            else:
                row = cur.execute(
                    "SELECT 1 FROM jobs WHERE tag = ? LIMIT 1", (p,)
                ).fetchone()
                if not row:
                    unmatched.append(p)
                    continue
                add(p)

    if unmatched:
        raise ValueError(
            "Tag(s) not found in DB: " + ", ".join(f"'{p}'" for p in unmatched)
        )

    return expanded
