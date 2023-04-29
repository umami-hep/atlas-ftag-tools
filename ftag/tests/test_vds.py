import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from vds import create_virtual_file, create_virtual_dataset, create_fixed_size_chunks


@pytest.fixture(scope="function")
def test_h5_files():
    # create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # create temporary h5 files
        file_paths = []
        for i in range(5):
            filename = os.path.join(tmpdir, f"test_file_{i}.h5")
            data = np.zeros((5,), dtype=[("var", "i8")])
            data["var"] = i
            with h5py.File(filename, "w") as f:
                f.create_dataset("data", data=data)
            file_paths.append(filename)
        yield file_paths


def test_create_virtual_dataset(test_h5_files):
    groups = ["data"]
    layouts = create_virtual_dataset(
        test_h5_files, groups, filter_fraction=1, filtering_var_group=None, filtering_var=None
    )
    layout = layouts["data"]
    assert isinstance(layout, h5py.VirtualLayout)
    assert layout.shape == (25,)
    assert layout.dtype == np.dtype("int64")


def test_create_virtual_file(test_h5_files):
    # create temporary output file
    with tempfile.NamedTemporaryFile() as tmpfile:
        # create virtual file
        output_path = Path(tmpfile.name)
        pattern = test_h5_files[0].replace("test_file_0", "test_file_*")
        create_virtual_file(
            pattern,
            output_path,
            overwrite=True,
            filter_fraction=1,
            filtering_var_group=None,
            filtering_var=None,
        )
        # check if file exists
        assert output_path.is_file()
        # check if file has expected content
        with h5py.File(output_path) as f:
            assert "data" in f
            assert len(f["data"]) == 25


def test_create_virtual_dataset_with_filter(test_h5_files):
    groups = ["data"]
    layouts = create_virtual_dataset(
        test_h5_files, groups, filter_fraction=0.5, filtering_var_group="data", filtering_var="data"
    )
    layout = layouts["data"]
    assert isinstance(layout, h5py.VirtualLayout)
    assert layout.shape[0] <= 25
    assert layout.dtype == np.dtype("int64")


def test_create_fixed_size_chunks():
    indices = np.arange(10)
    chunk_size = 3
    expected_chunks = [
        np.array([0, 1, 2]),
        np.array([3, 4, 5]),
        np.array([6, 7, 8]),
        np.array([9]),
    ]
    chunks = create_fixed_size_chunks(indices, chunk_size)
    assert len(chunks) == len(expected_chunks)
    for chunk, expected_chunk in zip(chunks, expected_chunks):
        assert np.array_equal(chunk, expected_chunk)
