from __future__ import annotations

from pathlib import Path

import pytest

from ftag.cli_utils import valid_path


def test_valid_path_existing_file():
    # Test when the input path is an existing file
    input_path = "existing_file.txt"
    expected_output = Path("existing_file.txt")
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


def test_valid_path_invalid_input():
    # Test when the input path is an invalid input
    input_path = 123  # Invalid input (not a string)
    with pytest.raises(TypeError) as e:
        valid_path(input_path)
    assert str(e) == input_path
