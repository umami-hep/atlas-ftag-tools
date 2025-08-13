"""Small utilities for interacting with a local Git repository.

This module provides helper functions to:

- detect whether a given path is inside a Git working tree,
- fail fast when there are uncommitted changes,
- verify that a local clone is a fork of an expected upstream,
- create and push an annotated tag, and
- read the current short commit hash.

All functions run Git commands via :mod:`subprocess`. If Git is not installed
or the command cannot be executed, the underlying OS error (e.g. :class:`OSError`)
will propagate.
"""

from __future__ import annotations

import subprocess
import sys
from os import PathLike
from subprocess import CalledProcessError


class GitError(Exception):
    """Raised when a Git-related precondition is not satisfied."""


def is_git_repo(path: str | PathLike[str]) -> bool:
    """Return whether ``path`` is inside a Git working tree.

    Parameters
    ----------
    path : str | PathLike[str]
        Filesystem path used as the current working directory for the Git command.

    Returns
    -------
    bool
        ``True`` if ``path`` is inside a Git working tree, ``False`` otherwise.

    Notes
    -----
    This function runs::

        git rev-parse --is-inside-work-tree HEAD

    Any non-zero exit status is treated as "not a Git repository". If Git is not
    available on the system, an :class:`OSError` may be raised by :mod:`subprocess`.
    """
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree", "HEAD"],
            cwd=path,
        )
    except CalledProcessError:
        return False
    else:
        return True


def check_for_uncommitted_changes(path: str | PathLike[str]) -> None:
    """Raise if the repository at ``path`` has uncommitted changes.

    Parameters
    ----------
    path : str | PathLike[str]
        Filesystem path to the repository root or any directory within it.

    Raises
    ------
    GitError
        If ``path`` is a Git repository and there are uncommitted changes.

    Notes
    -----
    - If ``path`` is **not** a Git repository, the function returns silently.
    - If the current process is running under ``pytest`` (detected via
      :data:`sys.modules`), the check is skipped and the function returns.
    """
    if not is_git_repo(path):
        return
    if "pytest" in sys.modules:
        return

    try:
        subprocess.check_output(
            ["git", "diff", "--quiet", "--exit-code"],
            cwd=path,
        )
    except CalledProcessError:
        raise GitError(
            "Uncommitted changes detected. Please commit them before running, or use --force."
        ) from None


def check_for_fork(path: str | PathLike[str], upstream: str) -> None:
    """Ensure the local clone's ``origin`` remote is a fork of ``upstream``.

    Parameters
    ----------
    path : str | PathLike[str]
        Filesystem path to the repository root or any directory within it.
    upstream : str
        Expected upstream repository URL substring (e.g. ``'github.com/org/repo'``).

    Raises
    ------
    GitError
        If the repository is present but its ``origin`` URL does not contain ``upstream``.

    Notes
    -----
    If ``path`` is not a Git repository, the function returns silently.
    """
    if not is_git_repo(path):
        return

    cmd = ["git", "remote", "get-url", "origin"]
    origin = subprocess.check_output(cmd, cwd=path).decode("utf-8").strip()
    if upstream not in origin:
        raise GitError(f"Your origin {origin} is not a fork of the upstream repo {upstream}")


def create_and_push_tag(
    path: str | PathLike[str],
    upstream: str,
    tagname: str,
    msg: str,
) -> None:
    """Create an annotated Git tag and push it to ``origin``.

    Parameters
    ----------
    path : str | PathLike[str]
        Filesystem path to the repository root or any directory within it.
    upstream : str
        Expected upstream repository URL substring; passed to :func:`check_for_fork`.
    tagname : str
        Name of the tag to create.
    msg : str
        Annotation message for the tag (``git tag -m``).

    Notes
    -----
    If ``path`` is not a Git repository, the function returns silently.
    """
    print(f"Pushing tag {tagname}")
    if not is_git_repo(path):
        return
    check_for_fork(path, upstream)
    subprocess.check_output(["git", "tag", tagname, "-m", msg], cwd=path)
    subprocess.check_output(["git", "push", "-q", "origin", "--tags"], cwd=path)


def get_git_hash(path: str | PathLike[str]) -> str | None:
    """Return the short commit hash for ``HEAD`` at ``path``, if available.

    Parameters
    ----------
    path : str | PathLike[str]
        Filesystem path to the repository root or any directory within it.

    Returns
    -------
    str | None
        The short (``--short``) commit hash as a string, or ``None`` if ``path`` is
        not a Git repository.
    """
    if not is_git_repo(path):
        return None

    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=path,
    )
    return git_hash.decode("ascii").strip()
