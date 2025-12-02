# ftag/tests/test_git_check.py
from __future__ import annotations

import sys
import types
import unittest
from subprocess import CalledProcessError
from unittest.mock import MagicMock, call, patch

from ftag.git_check import (
    GitError,
    check_for_fork,
    check_for_uncommitted_changes,
    create_and_push_tag,
    get_git_hash,
    is_git_repo,
)


class TestIsGitRepo(unittest.TestCase):
    @patch("ftag.git_check.subprocess.check_output")
    def test_returns_true_when_git_command_succeeds(self, m_check_output: MagicMock):
        m_check_output.return_value = b"true\n"
        self.assertTrue(is_git_repo("/some/path"))
        m_check_output.assert_called_once_with(
            ["git", "rev-parse", "--is-inside-work-tree", "HEAD"],
            cwd="/some/path",
        )

    @patch("ftag.git_check.subprocess.check_output")
    def test_returns_false_when_git_command_fails(self, m_check_output: MagicMock):
        m_check_output.side_effect = CalledProcessError(128, ["git"])
        self.assertFalse(is_git_repo("/some/path"))
        m_check_output.assert_called_once()


class TestCheckForUncommittedChanges(unittest.TestCase):
    def test_noop_when_not_a_git_repo(self):
        with (
            patch("ftag.git_check.is_git_repo", return_value=False) as m_is_git_repo,
            patch("ftag.git_check.subprocess.check_output") as m_check_output,
        ):
            check_for_uncommitted_changes("/not/repo")
            m_is_git_repo.assert_called_once_with("/not/repo")
            m_check_output.assert_not_called()

    def test_skips_when_running_under_pytest(self):
        with (
            patch("ftag.git_check.is_git_repo", return_value=True) as m_is_git_repo,
            patch("ftag.git_check.subprocess.check_output") as m_check_output,
        ):
            # Simulate pytest present using a real ModuleType (keeps mypy happy)
            had_pytest = "pytest" in sys.modules
            try:
                sys.modules["pytest"] = types.ModuleType("pytest")
                check_for_uncommitted_changes("/repo")
                m_is_git_repo.assert_called_once_with("/repo")
                m_check_output.assert_not_called()
            finally:
                if not had_pytest:
                    sys.modules.pop("pytest", None)

    @patch("ftag.git_check.is_git_repo", return_value=True)
    @patch("ftag.git_check.subprocess.check_output")
    def test_raises_giterror_when_dirty(self, m_check_output: MagicMock, _m_is_git_repo: MagicMock):
        m_check_output.side_effect = CalledProcessError(1, ["git", "diff"])
        # Temporarily hide pytest so the function doesn't early-return
        saved_pytest = sys.modules.pop("pytest", None)
        try:
            with self.assertRaises(GitError) as ctx:
                check_for_uncommitted_changes("/repo")
            self.assertIn("Uncommitted changes detected", str(ctx.exception))
            _m_is_git_repo.assert_called_once_with("/repo")
            m_check_output.assert_called_once_with(
                ["git", "diff", "--quiet", "--exit-code"], cwd="/repo"
            )
        finally:
            if saved_pytest is not None:
                sys.modules["pytest"] = saved_pytest

    @patch("ftag.git_check.is_git_repo", return_value=True)
    @patch("ftag.git_check.subprocess.check_output")
    def test_returns_silently_when_clean(
        self, m_check_output: MagicMock, _m_is_git_repo: MagicMock
    ):
        m_check_output.return_value = b""
        # Temporarily hide pytest so the function doesn't early-return
        saved_pytest = sys.modules.pop("pytest", None)
        try:
            check_for_uncommitted_changes("/repo")  # should not raise
            _m_is_git_repo.assert_called_once_with("/repo")
            m_check_output.assert_called_once_with(
                ["git", "diff", "--quiet", "--exit-code"], cwd="/repo"
            )
        finally:
            if saved_pytest is not None:
                sys.modules["pytest"] = saved_pytest


class TestCheckForFork(unittest.TestCase):
    def test_noop_when_not_a_git_repo(self):
        with (
            patch("ftag.git_check.is_git_repo", return_value=False) as m_is_git_repo,
            patch("ftag.git_check.subprocess.check_output") as m_check_output,
        ):
            check_for_fork("/not/repo", upstream="github.com/org/repo")
            m_is_git_repo.assert_called_once_with("/not/repo")
            m_check_output.assert_not_called()

    @patch("ftag.git_check.is_git_repo", return_value=True)
    @patch("ftag.git_check.subprocess.check_output")
    def test_passes_when_origin_contains_upstream(
        self, m_check_output: MagicMock, _m_is_git_repo: MagicMock
    ):
        m_check_output.return_value = b"https://github.com/org/repo.git\n"
        check_for_fork("/repo", upstream="github.com/org/repo")  # should not raise
        _m_is_git_repo.assert_called_once_with("/repo")  # use injected mock → no PT019
        m_check_output.assert_called_once_with(["git", "remote", "get-url", "origin"], cwd="/repo")

    @patch("ftag.git_check.is_git_repo", return_value=True)
    @patch("ftag.git_check.subprocess.check_output")
    def test_raises_when_origin_does_not_contain_upstream(
        self, m_check_output: MagicMock, _m_is_git_repo: MagicMock
    ):
        m_check_output.return_value = b"git@github.com:someone/else.git\n"
        with self.assertRaises(GitError):
            check_for_fork("/repo", upstream="github.com/org/repo")
        _m_is_git_repo.assert_called_once_with("/repo")  # use injected mock → no PT019
        m_check_output.assert_called_once()


class TestCreateAndPushTag(unittest.TestCase):
    def test_noop_when_not_a_git_repo(self):
        with (
            patch("ftag.git_check.is_git_repo", return_value=False) as m_is_git_repo,
            patch("ftag.git_check.subprocess.check_output") as m_check_output,
            patch("ftag.git_check.check_for_fork") as m_check_for_fork,
        ):
            create_and_push_tag("/not/repo", upstream="u", tagname="v1.0", msg="hello")
            m_is_git_repo.assert_called_once_with("/not/repo")
            m_check_for_fork.assert_not_called()
            m_check_output.assert_not_called()

    def test_creates_and_pushes_tag_when_repo_and_fork_ok(self):
        with (
            patch("ftag.git_check.is_git_repo", return_value=True),
            patch("ftag.git_check.subprocess.check_output") as m_check_output,
            patch("ftag.git_check.check_for_fork") as m_check_for_fork,
        ):
            create_and_push_tag(
                "/repo", upstream="github.com/org/repo", tagname="v1.2.3", msg="Release"
            )
            m_check_for_fork.assert_called_once_with("/repo", "github.com/org/repo")
            self.assertEqual(
                m_check_output.call_args_list,
                [
                    call(["git", "tag", "v1.2.3", "-m", "Release"], cwd="/repo"),
                    call(["git", "push", "-q", "origin", "--tags"], cwd="/repo"),
                ],
            )


class TestGetGitHash(unittest.TestCase):
    def test_returns_none_when_not_repo(self):
        with patch("ftag.git_check.is_git_repo", return_value=False):
            self.assertIsNone(get_git_hash("/not/repo"))

    def test_returns_short_hash_when_repo(self):
        with (
            patch("ftag.git_check.is_git_repo", return_value=True),
            patch("ftag.git_check.subprocess.check_output") as m_check_output,
        ):
            m_check_output.return_value = b"abc123\n"
            self.assertEqual(get_git_hash("/repo"), "abc123")
            m_check_output.assert_called_once_with(
                ["git", "rev-parse", "--short", "HEAD"], cwd="/repo"
            )
