"""Helpers for `pbx filter` exclude logic.

Exclusions are applied as a post-filter over jobs already fetched by the
include filters. Tag exclusion supports `*` glob (same semantics as the
include side); status/app are exact match.
"""
import fnmatch
from typing import List, Dict, Any, Optional

from parslbox.utils.tag_match import has_glob


def apply_excludes(
    jobs: List[Dict[str, Any]],
    exclude_status: Optional[str] = None,
    exclude_app: Optional[str] = None,
    exclude_tag: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Drop jobs matching any of the given exclude criteria."""
    out = jobs
    if exclude_status:
        target = exclude_status.capitalize()
        out = [j for j in out if j.get('status') != target]
    if exclude_app:
        out = [j for j in out if j.get('app') != exclude_app]
    if exclude_tag:
        if has_glob(exclude_tag):
            out = [j for j in out
                   if not (j.get('tag') and fnmatch.fnmatchcase(j['tag'], exclude_tag))]
        else:
            out = [j for j in out if j.get('tag') != exclude_tag]
    return out
