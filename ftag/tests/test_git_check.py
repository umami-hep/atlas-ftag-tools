from __future__ import annotations

import contextlib

from ftag.git_check import (
    GitError,
    check_for_fork,
    check_for_uncommitted_changes,
    get_git_hash,
    is_git_repo,
)


def test_is_git_repo():
    assert isinstance(is_git_repo("."), bool)


def test_check_for_uncommitted_changes():
    with contextlib.suppress(GitError):
        check_for_uncommitted_changes(".")


def test_check_for_fork():
    with contextlib.suppress(GitError):
        check_for_fork(".", "test")


def test_get_git_hash():
    assert isinstance(get_git_hash("."), str)
