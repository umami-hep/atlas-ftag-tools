import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from ftag.vds import create_virtual_file, get_virtual_layout, main


@pytest.fixture(scope="function")
def test_h5_files():
    # create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # create temporary h5 files
        file_paths = []
        for i in range(5):
            filename = os.path.join(tmpdir, f"test_file_{i}.h5")
            with h5py.File(filename, "w") as f:
                f.create_dataset("data", data=[i] * 5)
            file_paths.append(filename)
        yield file_paths


def test_get_virtual_layout(test_h5_files):
    layout = get_virtual_layout(test_h5_files, "data")
    assert isinstance(layout, h5py.VirtualLayout)
    assert layout.shape == (25,)
    assert layout.dtype == np.dtype("int64")


def test_create_virtual_file(test_h5_files):
    # create temporary output file
    with tempfile.NamedTemporaryFile() as tmpfile:
        # create virtual file
        output_path = Path(tmpfile.name)
        pattern = test_h5_files[0].replace("test_file_0", "test_file_*")
        create_virtual_file(pattern, output_path, overwrite=True)
        # check if file exists
        assert output_path.is_file()
        # check if file has expected content
        with h5py.File(output_path) as f:
            assert "data" in f
            print(f["data"])
            assert len(f["data"]) == 25


def test_main():
    # Create temporary directory to store test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some dummy h5 files
        for i in range(3):
            fname = Path(tmpdir) / f"test_{i}.h5"
            with h5py.File(fname, "w") as f:
                dset = f.create_dataset("data", (10,), dtype="f")
                dset.attrs["key"] = f"value_{i}"

        # Run the main function
        output_fname = Path(tmpdir) / "test_output.h5"
        pattern = Path(tmpdir) / "*.h5"
        args = [str(pattern), str(output_fname)]
        main(args)

        # Check that the output file exists
        assert output_fname.is_file()

        # Check that the output file contains the expected data
        with h5py.File(output_fname, "r") as f:
            assert len(f) == 1
            key = list(f.keys())[0]
            assert key == "data"
            assert f[key].shape == (30,)
            assert f[key].attrs["key"] == "value_0"
