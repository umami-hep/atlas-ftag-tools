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

    with pytest.raises(KeyError):
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
        dst=Path(tmp_path) / "test.h5",
        dtypes={"jets": jet_dtype},
        shapes={"jets": (100,)},
        num_jets=100,
    )

    assert writer.num_jets == 100
    assert writer.dst == Path(tmp_path) / "test.h5"
    assert writer.rng is not None


def test_invalid_write(tmp_path, jet_dtype):
    writer = H5Writer(
        dst=Path(tmp_path) / "test.h5",
        dtypes={"jets": jet_dtype},
        shapes={"jets": (100,)},
        num_jets=100,
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


def test_half_full_precision(tmp_path, mock_data_path):
    f_old = get_mock_file()[1]

    dst_path = Path(tmp_path) / "test.h5"
    full_precision_vars = ["pt"]
    writer = H5Writer.from_file(
        source=mock_data_path,
        dst=dst_path,
        shuffle=False,
        precision="half",
        full_precision_vars=full_precision_vars,
    )

    writer.write(f_old)
    with h5py.File(dst_path) as f:
        for key in ["jets", "tracks"]:
            for v in f[key].dtype.names:
                dt = np.dtype(f_old[key].dtype[v])
                dt_writer = np.dtype(f[key].dtype[v])
                if not np.issubdtype(dt, np.floating):
                    continue

                if v in full_precision_vars:
                    assert dt == np.float32
                    assert dt_writer == np.float32
                else:
                    assert dt == np.float32
                    assert dt_writer == np.float16


def test_dynamic_mode_write(tmp_path, mock_data):
    data = {"jets": mock_data[0], "tracks": mock_data[1]}

    shapes = {k: v.shape for k, v in data.items()}
    dtypes = {k: v.dtype for k, v in data.items()}

    writer = H5Writer(
        dst=Path(tmp_path) / "test_dynamic.h5",
        dtypes=dtypes,
        shapes=shapes,
        num_jets=None,  # Allow dynamic sizing
        shuffle=False,
    )

    writer.write(data)
    assert writer.num_written == len(data["jets"])

    # Should allow further writes without reshaping issues
    writer.write(data)
    assert writer.num_written == 2 * len(data["jets"])

    writer.close()
    with h5py.File(writer.dst) as f:
        assert f["jets"].shape[0] == 2 * len(data["jets"])


def test_precision_none_preserves_dtypes(tmp_path, mock_data):
    jets, tracks = mock_data
    dtypes = {"jets": jets.dtype, "tracks": tracks.dtype}
    shapes = {"jets": jets.shape, "tracks": tracks.shape}

    writer = H5Writer(
        dst=Path(tmp_path) / "test_precision_none.h5",
        dtypes=dtypes,
        shapes=shapes,
        precision=None,
        shuffle=False,
    )

    writer.write({"jets": jets, "tracks": tracks})
    writer.close()

    with h5py.File(writer.dst) as f:
        for name in ["jets", "tracks"]:
            for field in dtypes[name].names:
                expected_dtype = dtypes[name][field]
                actual_dtype = f[name].dtype[field]
                assert (
                    actual_dtype == expected_dtype
                ), f"{name}.{field} was {actual_dtype}, expected {expected_dtype}"


def test_close_raises_on_incomplete_write(tmp_path, jet_dtype):
    # Set up writer with fixed mode (num_jets set)
    writer = H5Writer(
        dst=Path(tmp_path) / "test_close_incomplete.h5",
        dtypes={"jets": jet_dtype},
        shapes={"jets": (100,)},
        num_jets=100,
        shuffle=False,
    )

    # Only write part of the data (e.g., 60 jets instead of 100)
    partial_data = {"jets": np.zeros(60, dtype=writer.dtypes["jets"])}
    writer.write(partial_data)

    # Closing should now raise ValueError
    with pytest.raises(ValueError, match="only 60 out of 100 jets have been written"):
        writer.close()
