"""Turning local job paths into the remote paths the database stores.

The rule has two cases (see the plan, "Path handling"):

  under the local root   must exist locally, then the prefix is swapped
  outside the local root must be absolute, stored unchanged -- it already
                         describes the remote filesystem
"""

import os
from pathlib import Path, PurePosixPath
from typing import Optional

from parslbox.local.project import LocalProject


def normalize(path) -> Path:
    """Absolute and '..'-collapsed, but deliberately not symlink-resolved.

    Path.resolve() would follow a symlink out of the local root and defeat the
    prefix swap: ~/work/proj1/run_a -> /scratch/run_a is still run_a of this
    project, and /scratch/run_a means nothing on the remote machine.
    """
    return Path(os.path.normpath(Path(path).expanduser().absolute()))


def is_under_local_root(path, project: LocalProject) -> bool:
    p = normalize(path)
    root = normalize(project.local_root)
    return p == root or root in p.parents


def to_remote(path, project: LocalProject) -> str:
    """Map a local path onto the remote filesystem.

    Paths outside the local root are returned normalized but unswapped; the
    caller is responsible for having required them to be absolute.
    """
    p = normalize(path)
    root = normalize(project.local_root)
    if not (p == root or root in p.parents):
        return str(p)
    rel = p.relative_to(root)
    remote = PurePosixPath(project.remote_root)
    return str(remote / rel) if rel.parts else str(remote)


def to_local(remote_path, project: LocalProject) -> Optional[Path]:
    """Inverse of to_remote: where a stored path lives on this machine.

    Returns None for paths outside the remote root -- shared input directories
    and the like, which exist only on the remote and are never transferred.
    """
    remote = PurePosixPath(str(remote_path))
    root = PurePosixPath(project.remote_root)
    if remote != root and root not in remote.parents:
        return None
    rel = remote.relative_to(root)
    return normalize(project.local_root) / rel if rel.parts else normalize(project.local_root)
