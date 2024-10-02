from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import pytest

from ftag import get_mock_file
from ftag.hdf5.h5copy import main, parse_args


@pytest.fixture
def get_fname():
    return get_mock_file()[0]


def test_parse_args():
    """Test argument parsing for the copy operation."""
    with NamedTemporaryFile() as src_file, NamedTemporaryFile() as dst_file:
        args = [
            "--src_fname",
            src_file.name,
            "--src",
            "/source/dataset",
            "--dst_fname",
            dst_file.name,
            "--dst",
            "/destination/dataset",
        ]
        parsed_args = parse_args(args)
        assert parsed_args.src_fname == Path(src_file.name)
        assert parsed_args.src == "/source/dataset"
        assert parsed_args.dst_fname == dst_file.name
        assert parsed_args.dst == "/destination/dataset"


def test_main_copy(get_fname):
    """Test copying a dataset from source to destination."""
    src_fname = get_fname
    dst_fname = NamedTemporaryFile().name

    old_data = h5py.File(src_fname, "r")["/jets"][:]

    # Run the copy operation
    main([
        "--src_fname",
        src_fname,
        "--src",
        "/jets",
        "--dst_fname",
        dst_fname,
        "--dst",
        "/jets_copy",
    ])

    # Verify the copy
    with h5py.File(dst_fname, "r") as f:
        assert "/jets_copy" in f
        assert all(f["/jets_copy"][:] == old_data)


def test_main_default_dst(get_fname):
    """Test copying when destination path is not provided (defaults to source)."""
    src_fname = get_fname
    dst_fname = NamedTemporaryFile().name

    old_data = h5py.File(src_fname, "r")["/jets"][:]

    # Run the copy operation with no destination path provided
    main(["--src_fname", src_fname, "--src", "/jets", "--dst_fname", dst_fname])

    # Verify the copy (dst should be "/jets")
    with h5py.File(dst_fname, "r") as f:
        assert "/jets" in f
        assert all(f["/jets"][:] == old_data)


def test_main_errors(get_fname):
    """Test that an error is raised when the destination dataset already exists."""
    src_fname = get_fname
    dst_fname = NamedTemporaryFile().name

    old_data = h5py.File(src_fname, "r")["/jets"][:]

    # Pre-create the destination dataset in the destination file
    with h5py.File(dst_fname, "a") as f:
        f.create_dataset("/jets_copy", data=old_data)

    # Run the copy operation and expect a FileExistsError
    with pytest.raises(FileExistsError, match="already exists"):
        main([
            "--src_fname",
            src_fname,
            "--src",
            "/jets",
            "--dst_fname",
            dst_fname,
            "--dst",
            "/jets_copy",
        ])

    with pytest.raises(KeyError, match="not found"):
        main([
            "--src_fname",
            src_fname,
            "--src",
            "/xyz",
            "--dst_fname",
            dst_fname,
            "--dst",
            "/jets_copy",
        ])
