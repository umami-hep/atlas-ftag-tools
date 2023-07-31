from pathlib import Path

import h5py
import pytest

from ftag import get_mock_file
from ftag.hdf5.h5move import main, parse_args


@pytest.fixture
def get_fname():
    return get_mock_file()[0]


def test_parse_args():
    # Test for the parse_args function
    args = ["/path/to/file.h5", "/source/dataset", "/destination/dataset"]
    parsed_args = parse_args(args)
    assert parsed_args.fname == Path("/path/to/file.h5")
    assert parsed_args.src == "/source/dataset"
    assert parsed_args.dst == "/destination/dataset"


def test_main(get_fname):
    fname = get_fname
    old_data = h5py.File(fname)["/jets"][:]

    main([fname, "/jets", "/jets_new"])

    f = h5py.File(fname)
    assert "/jets" not in f
    assert "/jets_new" in f
    assert all(f["/jets_new"][:] == old_data)
    f.close()
