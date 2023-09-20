from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from ftag import get_mock_file
from ftag.hdf5 import H5Writer


@pytest.fixture
def mock_data():
    f = get_mock_file()[1]
    jets = f["jets"][:100][["pt", "eta"]]
    tracks = f["tracks"][:100]
    return jets, tracks


@pytest.fixture
def mock_data_path():
    f = get_mock_file()[1]
    return f.filename


@pytest.fixture
def jet_dtype():
    return np.dtype([("pt", "f4"), ("eta", "f4")])


def test_create_ds(tmp_path, jet_dtype):
    writer = H5Writer(
        dst=Path(tmp_path) / "test.h5", dtypes={"jets": jet_dtype}, shapes={"jets": (100,)}
    )

    assert "jets" in writer.file
    assert writer.file["jets"].dtype == jet_dtype


def test_write(tmp_path, mock_data):
    writer = H5Writer(
        dst=Path(tmp_path) / "test.h5",
        dtypes={"jets": np.dtype([("pt", "f4"), ("eta", "f4")])},
        shapes={"jets": (100,)},
        shuffle=False,
    )

    data = {"jets": mock_data[0]}
    writer.write(data)
    assert writer.num_written == len(data["jets"])
    assert np.array_equal(writer.file["jets"][0 : writer.num_written], data["jets"])


def test_close(tmp_path, mock_data):
    writer = H5Writer(
        dst=Path(tmp_path) / "test.h5",
        dtypes={"jets": np.dtype([("pt", "f4"), ("eta", "f4")])},
        shapes={"jets": (100,)},
    )

    data = {"jets": mock_data[0]}
    writer.write(data)
    writer.close()

    with pytest.raises(ValueError):
        writer.file["jets"].resize(writer.num_written)


def test_add_attr(tmp_path, jet_dtype):
    writer = H5Writer(
        dst=Path(tmp_path) / "test.h5", dtypes={"jets": jet_dtype}, shapes={"jets": (100,)}
    )

    writer.add_attr("test_attr", "test_value")
    assert "test_attr" in writer.file.attrs
    assert writer.get_attr("test_attr") == "test_value"


def test_post_init(tmp_path, jet_dtype):
    writer = H5Writer(
        dst=Path(tmp_path) / "test.h5", dtypes={"jets": jet_dtype}, shapes={"jets": (100,)}
    )

    assert writer.num_jets == 100
    assert writer.dst == Path(tmp_path) / "test.h5"
    assert writer.rng is not None


def test_invalid_write(tmp_path, jet_dtype):
    writer = H5Writer(
        dst=Path(tmp_path) / "test.h5", dtypes={"jets": jet_dtype}, shapes={"jets": (100,)}
    )

    data = {"jets": np.zeros(110, dtype=writer.dtypes["jets"])}
    with pytest.raises(ValueError):
        writer.write(data)


def test_from_file(tmp_path, mock_data_path):
    f = get_mock_file()[1]
    jets = f["jets"][:]
    tracks = f["tracks"][:]

    dst_path = Path(tmp_path) / "test.h5"
    writer = H5Writer.from_file(source=mock_data_path, dst=dst_path, shuffle=False)

    writer.write({"jets": jets, "tracks": tracks})
    with h5py.File(dst_path) as f:
        assert np.array_equal(f["jets"][:], jets)
