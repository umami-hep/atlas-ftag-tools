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
    assert captured.err == ""
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
    args = ["--src", str(mock_h5_file), "--jets_per_file", "201", "--batch_size", "10"]
    main(args)
    
    captured = capsys.readouterr()
    assert "Done!" in captured.out
    assert captured.err == ""
    split_dir = mock_h5_file.parent / f"split_{mock_h5_file.stem}"
    assert split_dir.exists()
    num_output_files = len(list(split_dir.glob("*.h5")))
    assert num_output_files == 5

    # check file content
    for i, f in enumerate(sorted(split_dir.glob("*.h5"))):
        start = i * 201
        stop = start + 201 if i <= 3 else 1000
        with h5py.File(f) as dst, h5py.File(mock_h5_file) as src:
            assert src.keys() == dst.keys()
            for k in src:
                assert (src[k][start:stop] == dst[k]).all()
