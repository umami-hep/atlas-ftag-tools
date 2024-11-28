from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pytest

from ftag.cli_utils import valid_path


def test_valid_path_existing_file():
    # Test when the input path is an existing file
    # get a temp directory
    with TemporaryDirectory() as tmpdir, NamedTemporaryFile(dir=tmpdir) as f:
        input_path = f.name
        expected_output = Path(f.name)
        result = valid_path(input_path)
        assert result == expected_output


def test_valid_path_non_existing_file():
    # Test when the input path is a non-existing file
    input_path = "non_existing_file.txt"
    with pytest.raises(FileNotFoundError) as e:
        valid_path(input_path)
    assert str(e.value) == input_path


def test_valid_path_directory():
    # Test when the input path is a directory
    input_path = "directory/"
    with pytest.raises(FileNotFoundError) as e:
        valid_path(input_path)
    assert str(e.value) == input_path
