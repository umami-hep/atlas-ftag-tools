from __future__ import annotations

import re
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import pytest

from ftag import get_mock_file
from ftag.hdf5 import H5Writer
from ftag.hdf5.h5utils import compare_groups

_PADDING_FIELD_RE = re.compile(r"^f\d+$")


def _named_fields(dtype: np.dtype) -> tuple[str, ...]:
    return tuple(name for name in dtype.names if not _PADDING_FIELD_RE.fullmatch(name))


def assert_structured_array_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    """Assert equality of two structured arrays field-by-field.

    This comparison ignores anonymous padding fields such as ``f0``, ``f1``,
    etc., which may be introduced by HDF5/h5py for compound dtypes.
    """
    assert actual.shape == expected.shape

    actual_fields = _named_fields(actual.dtype)
    expected_fields = _named_fields(expected.dtype)
    assert actual_fields == expected_fields

    for field in expected_fields:
        assert np.array_equal(actual[field], expected[field]), f"Mismatch in field '{field}'"


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
        dst=Path(tmp_path) / "test.h5",
        dtypes={"jets": jet_dtype},
        shapes={"jets": (100,)},
    )

    assert "jets" in writer.file
    assert _named_fields(writer.file["jets"].dtype) == ("pt", "eta")
    writer.close()


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
    assert_structured_array_equal(writer.file["jets"][0 : writer.num_written], data["jets"])
    writer.close()


def test_close(tmp_path, mock_data):
    writer = H5Writer(
        dst=Path(tmp_path) / "test.h5",
        dtypes={"jets": np.dtype([("pt", "f4"), ("eta", "f4")])},
        shapes={"jets": (100,)},
        num_jets=100,
        shuffle=False,
    )

    data = {"jets": mock_data[0]}
    writer.write(data)
    writer.close()

    assert not writer.file.id.valid


def test_add_attr(tmp_path, jet_dtype):
    writer = H5Writer(
        dst=Path(tmp_path) / "test.h5",
        dtypes={"jets": jet_dtype},
        shapes={"jets": (100,)},
    )

    writer.add_attr("test_attr", "test_value")
    assert "test_attr" in writer.file.attrs
    assert writer.get_attr("test_attr") == "test_value"
    writer.close()


def test_post_init_fixed_mode(tmp_path, jet_dtype):
    writer = H5Writer(
        dst=Path(tmp_path) / "test.h5",
        dtypes={"jets": jet_dtype},
        shapes={"jets": (100,)},
        num_jets=100,
    )

    assert writer.num_jets == 100
    assert writer.fixed_mode is True
    assert writer.dst == Path(tmp_path) / "test.h5"
    assert writer.rng is not None


def test_post_init_dynamic_mode(tmp_path, jet_dtype):
    writer = H5Writer(
        dst=Path(tmp_path) / "test_dynamic.h5",
        dtypes={"jets": jet_dtype},
        shapes={"jets": (100,)},
        num_jets=None,
    )

    assert writer.num_jets is None
    assert writer.fixed_mode is False
    writer.close()


def test_invalid_write(tmp_path, jet_dtype):
    writer = H5Writer(
        dst=Path(tmp_path) / "test.h5",
        dtypes={"jets": jet_dtype},
        shapes={"jets": (100,)},
        num_jets=100,
    )

    data = {"jets": np.zeros(110, dtype=writer.dtypes["jets"])}
    with pytest.raises(ValueError, match="Attempted to write more jets than expected"):
        writer.write(data)


def test_from_file(tmp_path, mock_data_path):
    f = get_mock_file()[1]
    jets = f["jets"][:]
    tracks = f["tracks"][:]
    cutbookkeeper = f["cutBookkeeper"]

    dst_path = Path(tmp_path) / "test.h5"
    writer = H5Writer.from_file(source=mock_data_path, dst=dst_path, shuffle=False)

    writer.write({"jets": jets, "tracks": tracks})
    writer.close()

    with h5py.File(dst_path) as f_out:
        assert_structured_array_equal(f_out["jets"][:], jets)
        assert_structured_array_equal(f_out["tracks"][:], tracks)
        compare_groups(f_out["cutBookkeeper"], cutbookkeeper, path="cutBookkeeper")


def test_from_file_no_groups(tmp_path, mock_data_path):
    dst_path = Path(tmp_path) / "test.h5"
    f = get_mock_file()[1]
    jets = f["jets"][:]
    tracks = f["tracks"][:]

    writer = H5Writer.from_file(
        source=mock_data_path,
        dst=dst_path,
        copy_groups=False,
        shuffle=False,
    )
    writer.write({"jets": jets, "tracks": tracks})
    writer.close()

    with h5py.File(dst_path) as f_out:
        assert "cutBookkeeper" not in f_out


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
    writer.close()

    with h5py.File(dst_path) as f_new:
        for key in ["jets", "tracks"]:
            for v in _named_fields(f_new[key].dtype):
                dt_old = np.dtype(f_old[key].dtype[v])
                dt_new = np.dtype(f_new[key].dtype[v])

                if not np.issubdtype(dt_old, np.floating):
                    continue

                if v in full_precision_vars:
                    assert dt_old == np.float32
                    assert dt_new == np.float32
                else:
                    assert dt_old == np.float32
                    assert dt_new == np.float16


def test_dynamic_mode_write(tmp_path, mock_data):
    data = {"jets": mock_data[0], "tracks": mock_data[1]}

    shapes = {k: v.shape for k, v in data.items()}
    dtypes = {k: v.dtype for k, v in data.items()}

    writer = H5Writer(
        dst=Path(tmp_path) / "test_dynamic.h5",
        dtypes=dtypes,
        shapes=shapes,
        num_jets=None,
        shuffle=False,
    )

    writer.write(data)
    assert writer.num_written == len(data["jets"])

    writer.write(data)
    assert writer.num_written == 2 * len(data["jets"])

    writer.close()

    with h5py.File(writer.dst) as f:
        assert f["jets"].shape[0] == 2 * len(data["jets"])
        assert f["tracks"].shape[0] == 2 * len(data["tracks"])


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
            for field in _named_fields(dtypes[name]):
                expected_dtype = dtypes[name][field]
                actual_dtype = f[name].dtype[field]
                assert actual_dtype == expected_dtype, (
                    f"{name}.{field} was {actual_dtype}, expected {expected_dtype}"
                )


def test_close_raises_on_incomplete_write(tmp_path, jet_dtype):
    writer = H5Writer(
        dst=Path(tmp_path) / "test_close_incomplete.h5",
        dtypes={"jets": jet_dtype},
        shapes={"jets": (100,)},
        num_jets=100,
        shuffle=False,
    )

    partial_data = {"jets": np.zeros(60, dtype=writer.dtypes["jets"])}
    writer.write(partial_data)

    with pytest.raises(ValueError, match="only 60 out of 100 jets have been written"):
        writer.close()


def test_from_file_with_variable_subset(tmp_path):
    path = tmp_path / "test_subset.h5"
    with h5py.File(path, "w") as f:
        jets = np.zeros(10, dtype=[("pt", "f4"), ("eta", "f4"), ("phi", "f4")])
        tracks = np.zeros((10, 5), dtype=[("d0", "f4"), ("z0", "f4")])
        flows = np.zeros((10, 5), dtype=[("pt", "f4")])
        f.create_dataset("jets", data=jets, compression="lzf")
        f.create_dataset("tracks", data=tracks, compression="lzf")
        f.create_dataset("flows", data=flows, compression="lzf")

    variables = {
        "jets": ["pt", "eta"],
        "tracks": ["d0"],
    }

    writer = H5Writer.from_file(
        source=path,
        dst=tmp_path / "out.h5",
        num_jets=10,
        variables=variables,
        precision=None,
    )

    assert "jets" in writer.dtypes
    assert "tracks" in writer.dtypes
    assert _named_fields(writer.dtypes["jets"]) == ("pt", "eta")
    assert _named_fields(writer.dtypes["tracks"]) == ("d0",)
    assert writer.shapes["jets"][0] == 10
    assert writer.shapes["tracks"][0] == 10

    writer.close()


@pytest.mark.parametrize(
    ("compression", "compression_opts"),
    [
        ("gzip", 1),
        ("gzip", 7),
    ],
)
def test_create_ds_with_gzip_compression_opts(
    tmp_path,
    jet_dtype,
    compression,
    compression_opts,
):
    writer = H5Writer(
        dst=Path(tmp_path) / f"test_{compression}_{compression_opts}.h5",
        dtypes={"jets": jet_dtype},
        shapes={"jets": (100,)},
        compression=compression,
        compression_opts=compression_opts,
    )

    ds = writer.file["jets"]
    assert ds.compression == "gzip"
    assert ds.compression_opts == compression_opts
    writer.close()


@pytest.mark.parametrize(
    ("compression", "compression_opts", "filter_id"),
    [
        ("lz4", None, hdf5plugin.LZ4.filter_id),
        ("zstd", None, hdf5plugin.Zstd.filter_id),
        ("zstd", 3, hdf5plugin.Zstd.filter_id),
        ("zstd", 9, hdf5plugin.Zstd.filter_id),
    ],
)
def test_create_ds_with_plugin_compression_opts(
    tmp_path,
    jet_dtype,
    compression,
    compression_opts,
    filter_id,
):
    writer = H5Writer(
        dst=Path(tmp_path) / f"test_{compression}_{compression_opts}.h5",
        dtypes={"jets": jet_dtype},
        shapes={"jets": (100,)},
        compression=compression,
        compression_opts=compression_opts,
    )

    ds = writer.file["jets"]
    plist = ds.id.get_create_plist()
    filters = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]

    assert filter_id in filters
    writer.close()


@pytest.mark.parametrize(
    "compression",
    [None, "none", "lzf", "gzip", "lz4", "zstd"],
)
def test_write_with_compression(tmp_path, mock_data, compression):
    jets, tracks = mock_data
    dtypes = {"jets": jets.dtype, "tracks": tracks.dtype}
    shapes = {"jets": jets.shape, "tracks": tracks.shape}

    writer = H5Writer(
        dst=Path(tmp_path) / f"test_write_{compression}.h5",
        dtypes=dtypes,
        shapes=shapes,
        compression=compression,
        precision=None,
        shuffle=False,
    )

    writer.write({"jets": jets, "tracks": tracks})
    assert writer.file["jets"].compression == compression
    writer.close()

    with h5py.File(writer.dst) as f:
        assert_structured_array_equal(f["jets"][:], jets)
        assert_structured_array_equal(f["tracks"][:], tracks)


@pytest.mark.parametrize("compression", ["invalid", "foo", "lz5"])
def test_invalid_compression_raises(tmp_path, jet_dtype, compression):
    with pytest.raises(ValueError, match="Unsupported compression"):
        H5Writer(
            dst=Path(tmp_path) / "test_invalid_compression.h5",
            dtypes={"jets": jet_dtype},
            shapes={"jets": (100,)},
            compression=compression,
        )


@pytest.mark.parametrize("compression", ["lzf", "gzip"])
def test_from_file_preserves_builtin_compression(tmp_path, compression):
    src_path = tmp_path / f"source_{compression}.h5"
    dst_path = tmp_path / f"dest_{compression}.h5"

    jets = np.zeros(10, dtype=[("pt", "f4"), ("eta", "f4")])
    tracks = np.zeros((10, 5), dtype=[("d0", "f4"), ("z0", "f4")])

    with h5py.File(src_path, "w") as f:
        f.create_dataset("jets", data=jets, compression=compression)
        f.create_dataset("tracks", data=tracks, compression=compression)

    writer = H5Writer.from_file(source=src_path, dst=dst_path, shuffle=False, precision=None)
    writer.write({"jets": jets, "tracks": tracks})
    writer.close()

    with h5py.File(dst_path) as f:
        assert f["jets"].compression == compression
        assert f["tracks"].compression == compression
        assert_structured_array_equal(f["jets"][:], jets)
        assert_structured_array_equal(f["tracks"][:], tracks)


def test_from_file_preserves_gzip_compression_opts(tmp_path):
    src_path = tmp_path / "source_gzip4.h5"
    dst_path = tmp_path / "dest_gzip4.h5"

    jets = np.zeros(10, dtype=[("pt", "f4"), ("eta", "f4")])
    tracks = np.zeros((10, 5), dtype=[("d0", "f4"), ("z0", "f4")])

    with h5py.File(src_path, "w") as f:
        f.create_dataset("jets", data=jets, compression="gzip", compression_opts=4)
        f.create_dataset("tracks", data=tracks, compression="gzip", compression_opts=4)

    writer = H5Writer.from_file(source=src_path, dst=dst_path, shuffle=False, precision=None)
    writer.write({"jets": jets, "tracks": tracks})
    writer.close()

    with h5py.File(dst_path) as f:
        assert f["jets"].compression == "gzip"
        assert f["tracks"].compression == "gzip"
        assert f["jets"].compression_opts == 4
        assert f["tracks"].compression_opts == 4


@pytest.mark.parametrize(
    ("compression", "filter_id"),
    [
        ("lz4", hdf5plugin.LZ4.filter_id),
        ("zstd", hdf5plugin.Zstd.filter_id),
    ],
)
def test_from_file_preserves_plugin_compression(tmp_path, compression, filter_id):
    src_path = tmp_path / f"source_{compression}.h5"
    dst_path = tmp_path / f"dest_{compression}.h5"

    jets = np.zeros(10, dtype=[("pt", "f4"), ("eta", "f4")])
    tracks = np.zeros((10, 5), dtype=[("d0", "f4"), ("z0", "f4")])

    plugin = hdf5plugin.LZ4() if compression == "lz4" else hdf5plugin.Zstd()

    with h5py.File(src_path, "w") as f:
        f.create_dataset("jets", data=jets, compression=plugin)
        f.create_dataset("tracks", data=tracks, compression=plugin)

    writer = H5Writer.from_file(
        source=src_path,
        dst=dst_path,
        shuffle=False,
        compression=compression,
        precision=None,
    )
    writer.write({"jets": jets, "tracks": tracks})
    writer.close()

    with h5py.File(dst_path) as f:
        for name in ["jets", "tracks"]:
            plist = f[name].id.get_create_plist()
            filters = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
            assert filter_id in filters


def test_from_file_override_compression(tmp_path):
    src_path = tmp_path / "source_lzf.h5"
    dst_path = tmp_path / "dest_gzip7.h5"

    jets = np.zeros(10, dtype=[("pt", "f4"), ("eta", "f4")])
    tracks = np.zeros((10, 5), dtype=[("d0", "f4"), ("z0", "f4")])

    with h5py.File(src_path, "w") as f:
        f.create_dataset("jets", data=jets, compression="lzf")
        f.create_dataset("tracks", data=tracks, compression="lzf")

    writer = H5Writer.from_file(
        source=src_path,
        dst=dst_path,
        shuffle=False,
        compression="gzip",
        compression_opts=7,
        precision=None,
    )
    writer.write({"jets": jets, "tracks": tracks})
    writer.close()

    with h5py.File(dst_path) as f:
        assert f["jets"].compression == "gzip"
        assert f["tracks"].compression == "gzip"
        assert f["jets"].compression_opts == 7
        assert f["tracks"].compression_opts == 7


def test_add_flavour_label(tmp_path, jet_dtype):
    writer = H5Writer(
        dst=Path(tmp_path) / "test_flavour_label.h5",
        dtypes={"jets": jet_dtype},
        shapes={"jets": (10,)},
        add_flavour_label=True,
    )

    assert "flavour_label" in _named_fields(writer.file["jets"].dtype)
    writer.close()


def test_add_flavour_label_does_not_duplicate(tmp_path):
    jet_dtype_with_label = np.dtype([("pt", "f4"), ("eta", "f4"), ("flavour_label", "i4")])

    writer = H5Writer(
        dst=Path(tmp_path) / "test_flavour_label_existing.h5",
        dtypes={"jets": jet_dtype_with_label},
        shapes={"jets": (10,)},
        add_flavour_label=True,
    )

    assert _named_fields(writer.file["jets"].dtype).count("flavour_label") == 1
    writer.close()


@pytest.mark.parametrize("precision", ["bad", "float32", "quarter"])
def test_invalid_precision_raises(tmp_path, jet_dtype, precision):
    with pytest.raises(ValueError, match="Invalid precision"):
        H5Writer(
            dst=Path(tmp_path) / "test_invalid_precision.h5",
            dtypes={"jets": jet_dtype},
            shapes={"jets": (10,)},
            precision=precision,
        )
