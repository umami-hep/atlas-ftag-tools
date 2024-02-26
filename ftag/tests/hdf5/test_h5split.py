from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from ftag import get_mock_file
from ftag.hdf5.h5split import main, parse_args


# Define a fixture to provide mock data
@pytest.fixture
def mock_h5_file():
    file_path = get_mock_file()[0]
    return Path(file_path)


def test_parse_args():
    args = ["--src", "input.h5", "--jets_per_file", "1000"]
    parsed_args = parse_args(args)
    assert parsed_args.src == Path("input.h5")
    assert parsed_args.jets_per_file == 1000


def test_main(mock_h5_file, capsys):
    args = ["--src", str(mock_h5_file), "--jets_per_file", "100", "--batch_size", "10"]
    main(args)

    captured = capsys.readouterr()

    assert "Done!" in captured.out
    assert not captured.err
    split_dir = mock_h5_file.parent / f"split_{mock_h5_file.stem}"
    assert split_dir.exists()
    num_output_files = len(list(split_dir.glob("*.h5")))
    assert num_output_files == 10

    # check file content
    for i, f in enumerate(sorted(split_dir.glob("*.h5"))):
        start = i * 100
        stop = start + 100
        with h5py.File(f) as dst, h5py.File(mock_h5_file) as src:
            assert src.keys() == dst.keys()
            for k in src:
                assert (src[k][start:stop] == dst[k]).all()


def test_remainder(mock_h5_file, capsys):
    n_per_file = 201
    args = ["--src", str(mock_h5_file), "--jets_per_file", str(n_per_file), "--batch_size", "10"]
    main(args)

    captured = capsys.readouterr()
    assert "Done!" in captured.out
    assert not captured.err
    split_dir = mock_h5_file.parent / f"split_{mock_h5_file.stem}"
    assert split_dir.exists()
    num_output_files = len(list(split_dir.glob("*.h5")))
    assert num_output_files == 5

    # check file content
    for i, f in enumerate(sorted(split_dir.glob("*.h5"))):
        start = i * n_per_file
        stop = start + n_per_file if i <= 3 else 1000
        print(start, stop)
        with h5py.File(f) as dst, h5py.File(mock_h5_file) as src:
            assert src.keys() == dst.keys()
            for k in src:
                assert (src[k][start:stop] == dst[k]).all()


def test_attrs(mock_h5_file):
    args = ["--src", str(mock_h5_file), "--jets_per_file", "100", "--batch_size", "10"]
    main(args)

    split_dir = mock_h5_file.parent / f"split_{mock_h5_file.stem}"
    assert split_dir.exists()

    # for each file, check the h5 attrs are the same as the input file
    for f in sorted(split_dir.glob("*.h5")):
        with h5py.File(f) as dst, h5py.File(mock_h5_file) as src:
            assert set(src.attrs).issubset(set(dst.attrs))

            # check each dataset has the same attrs
            for k in src:
                print(dict(src[k].attrs), dict(dst[k].attrs))
                assert set(src[k].attrs).issubset(set(dst[k].attrs))
