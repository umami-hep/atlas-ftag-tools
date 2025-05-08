import numpy as np
import h5py
import pytest
from pathlib import Path

from ftag import get_mock_file
from ftag.hdf5 import h5_add_column


@pytest.fixture
def input_file(tmp_path):
    """Create a mock HDF5 file for testing."""
    dst = tmp_path / "input.h5"
    mock = get_mock_file()
    with h5py.File(dst, "w") as f:
        f.create_dataset("jets", data=mock[1]["jets"][:100])
    return dst


@pytest.fixture
def append_func():
    def add_phi(batch):
        return {
            "jets": {
                "new_phi": batch["jets"]["pt"] * 0.1  # some dummy computation
            }
        }
    return add_phi


def test_h5_add_column_creates_output(tmp_path, input_file, append_func):
    output_file = tmp_path / "output.h5"
    h5_add_column(input_file, output_file, append_func)

    assert output_file.exists()
    with h5py.File(output_file, "r") as f:
        assert "jets" in f
        assert "new_phi" in f["jets"].dtype.names
        assert np.allclose(f["jets"]["new_phi"], f["jets"]["pt"][:100] * 0.1)


def test_h5_add_column_overwrite_protection(tmp_path, input_file, append_func):
    output_file = tmp_path / "output.h5"
    h5_add_column(input_file, output_file, append_func)

    # Attempting to write again should raise unless overwrite=True
    with pytest.raises(FileExistsError):
        h5_add_column(input_file, output_file, append_func)

    # Should succeed with overwrite=True
    h5_add_column(input_file, output_file, append_func, overwrite=True)
    assert output_file.exists()


def test_h5_add_column_invalid_group(tmp_path, input_file, append_func):
    def wrong_group_func(batch):
        return {"tracks": {"phi": np.ones(len(batch["jets"]))}}

    with pytest.raises(ValueError, match="Trying to output to"):
        h5_add_column(input_file, tmp_path / "bad_output.h5", wrong_group_func)


def test_h5_add_column_variable_conflict(tmp_path, input_file):
    def conflict_func(batch):
        return {"jets": {"pt": np.ones(len(batch["jets"]))}}  # pt already exists

    with pytest.raises(ValueError, match="Trying to append pt to jets but it already exists in batch"):
        h5_add_column(input_file, tmp_path / "conflict.h5", conflict_func)