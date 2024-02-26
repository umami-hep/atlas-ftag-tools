from __future__ import annotations

import subprocess
import sys
from subprocess import CalledProcessError


class GitError(Exception):
    pass


def is_git_repo(path) -> bool:
    try:
        subprocess.check_output(["git", "rev-parse", "--is-inside-work-tree", "HEAD"], cwd=path)
    except CalledProcessError:
        return False
    else:
        return True


def check_for_uncommitted_changes(path) -> None:
    if not is_git_repo(path):
        return
    if "pytest" in sys.modules:
        return

    try:
        subprocess.check_output(["git", "diff", "--quiet", "--exit-code"], cwd=path)
    except CalledProcessError:
        raise GitError(
            "Uncommitted changes detected. Please commit them before running, or use --force."
        ) from None


def check_for_fork(path, upstream) -> None:
    cmd = ["git", "remote", "get-url", "origin"]
    if not is_git_repo(path):
        return

    origin = subprocess.check_output(cmd, cwd=path).decode("utf-8").strip()
    if upstream not in origin:
        raise GitError(f"Your origin {origin} is not a fork of the upstream repo {upstream}")


def create_and_push_tag(path, upstream, tagname, msg) -> None:
    print(f"Pushing tag {tagname}")
    if not is_git_repo(path):
        return
    check_for_fork(path, upstream)
    subprocess.check_output(["git", "tag", tagname, "-m", msg], cwd=path)
    subprocess.check_output(["git", "push", "-q", "origin", "--tags"], cwd=path)


def get_git_hash(path):
    if not is_git_repo(path):
        return None

    git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=path)
    return git_hash.decode("ascii").strip()
