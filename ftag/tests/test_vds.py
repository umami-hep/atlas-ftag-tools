from __future__ import annotations

import re
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from ftag.vds import create_virtual_file, get_virtual_layout, glob_re, main, regex_files_from_dir


@pytest.fixture
def test_h5_files():
    # create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # create temporary h5 files
        file_paths = []
        for i in range(5):
            filename = Path(tmpdir) / f"test_file_{i}.h5"
            with h5py.File(filename, "w") as f:
                f.create_dataset("data", data=[i] * 5)
            file_paths.append(filename)
        yield file_paths


@pytest.fixture
def test_h5_dirs():
    # create temporary directory
    with tempfile.TemporaryDirectory(prefix="outer_tmp_", suffix="_h5dump") as top_level:
        nested_dirs = []
        file_paths = []
        for _ in range(3):
            nested = tempfile.TemporaryDirectory(dir=top_level, prefix="inner_tmp_", suffix=".h5")
            nested_dirs.append(nested)
            for j in range(5):
                filename = Path(str(nested.name)) / f"test_file_{j}.h5"
                with h5py.File(filename, "w") as f:
                    f.create_dataset("data", data=[j] * 5)
                file_paths.append(filename)
        yield file_paths


def test_glob_re_files(test_h5_files):
    pattern = "(test_file_1|test_file_3|test_file_5)" + ".h5"
    regex_path = str(Path(test_h5_files[0].parent))
    matched_reg = glob_re(pattern, regex_path)
    assert len(matched_reg) == 2
    assert matched_reg == ["test_file_3.h5", "test_file_1.h5"]


def test_glob_re_dirs(test_h5_dirs):
    pattern = "(test_file_1|test_file_3|test_file_5)" + ".h5"
    regex_path = str(Path(test_h5_dirs[0]).parent)
    matched_reg = glob_re(pattern, regex_path)
    assert len(matched_reg) == 2
    assert matched_reg == ["test_file_3.h5", "test_file_1.h5"]


def test_glob_re_dirs_nopath(test_h5_dirs):
    pattern = str(Path(test_h5_dirs[0]).parent) + "(test_file_1|test_file_3|test_file_5)" + ".h5"
    regex_path = None
    matched_reg = glob_re(pattern, regex_path)
    # Not assigning the regex_path makes this regex not match
    # because the local directory of this test is not the temporary folder
    assert len(matched_reg) == 0
    assert matched_reg == []


def test_h5dir_files(test_h5_files):
    matched_reg = ["test_file_3.h5", "test_file_1.h5"]
    regex_path = str(Path(test_h5_files[0].parent))
    fnames = regex_files_from_dir(matched_reg, regex_path)
    assert fnames == [
        str(Path(test_h5_files[0].parent)) + "/test_file_3.h5",
        str(Path(test_h5_files[0].parent)) + "/test_file_1.h5",
    ]


def test_h5dir_dirs(test_h5_dirs):
    to_match = str(test_h5_dirs[0].parent)
    test_match = re.match("(/tmp/)(outer_tmp.*)(inner_tmp.*h5)", to_match).group(3)
    matched_reg = [test_match]
    regex_path = str(Path(test_h5_dirs[0].parent.parent))
    fnames = regex_files_from_dir(matched_reg, regex_path)
    assert set(fnames) == {
        str(Path(test_h5_dirs[0].parent)) + "/test_file_0.h5",
        str(Path(test_h5_dirs[0].parent)) + "/test_file_1.h5",
        str(Path(test_h5_dirs[0].parent)) + "/test_file_2.h5",
        str(Path(test_h5_dirs[0].parent)) + "/test_file_3.h5",
        str(Path(test_h5_dirs[0].parent)) + "/test_file_4.h5",
    }


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
        pattern = Path(test_h5_files[0]).parent / "test_file_*"
        print("pattern", pattern)
        print("file create:", test_h5_files)
        create_virtual_file(pattern, output_path, overwrite=True)
        # check if file exists
        assert output_path.is_file()
        # check if file has expected content
        with h5py.File(output_path) as f:
            assert "data" in f
            print(f["data"])
            assert len(f["data"]) == 25


def test_create_virtual_file_regex(test_h5_files):
    # create temporary output file
    use_regex = True
    with tempfile.NamedTemporaryFile() as tmpfile:
        # create virtual file
        output_path = Path(tmpfile.name)
        regex_path = str(Path(test_h5_files[0].parent))
        pattern = "(test_file_0|test_file_1|test_file_2).h5"
        create_virtual_file(pattern, output_path, use_regex, regex_path, overwrite=True)
        # check if file exists
        assert output_path.is_file()
        # check if file has expected content
        with h5py.File(output_path) as f:
            assert "data" in f
            print(f["data"])
            assert len(f["data"]) == 15


def test_create_virtual_file_common_groups(test_h5_files):
    # create additional h5 files with different group
    with tempfile.TemporaryDirectory() as tmpdir:
        extra_files = []
        for i in range(2):
            filename = Path(tmpdir) / f"extra_file_{i}.h5"
            with h5py.File(filename, "w") as f:
                f.create_dataset("extra_data", data=[i] * 5)
            extra_files.append(filename)

        all_files = test_h5_files + extra_files
        pattern = Path(all_files[0]).parent / "*.h5"

        # create temporary output file
        with tempfile.NamedTemporaryFile() as tmpfile:
            output_path = Path(tmpfile.name)
            create_virtual_file(pattern, output_path, overwrite=True)

            # check the output file
            with h5py.File(output_path) as f:
                assert "data" in f
                assert "extra_data" not in f


def test_main():
    # Create temporary directory to store test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some dummy h5 files
        for i in range(3):
            fname = Path(tmpdir) / f"test_{i}.h5"
            with h5py.File(fname, "w") as f:
                dset = f.create_dataset("data", (10,), dtype="f")
                dset.attrs["key"] = "value"

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
            key = next(iter(f.keys()))
            assert key == "data"
            assert f[key].shape == (30,)
            assert f[key].attrs["key"] == "value"
