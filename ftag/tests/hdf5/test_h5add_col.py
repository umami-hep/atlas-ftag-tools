from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from ftag import get_mock_file
from ftag.hdf5.h5add_col import get_all_groups, get_shape, h5_add_column, merge_dicts


@pytest.fixture
def input_file():
    fname, _ = get_mock_file()
    return fname


@pytest.fixture
def append_func():
    def add_phi(batch):
        return {
            "jets": {
                "new_phi": batch["jets"]["pt"] * 0.1  # some dummy computation
            }
        }

    return add_phi


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        h5_add_column("nonexistent.h5", None, lambda x: x)


def test_merge_dicts_success():
    d1 = {"jets": {"pt": np.array([1, 2, 3])}}
    d2 = {"jets": {"eta": np.array([4, 5, 6])}}
    merged = merge_dicts([d1, d2])
    assert "pt" in merged["jets"]
    assert "eta" in merged["jets"]


def test_merge_dicts_conflict():
    d1 = {"jets": {"pt": np.array([1, 2, 3])}}
    d2 = {"jets": {"pt": np.array([4, 5, 6])}}
    with pytest.raises(ValueError, match="Variable pt already exists"):
        merge_dicts([d1, d2])


def test_get_shape_scalar_and_vector():
    batch = {
        "a": np.zeros((5,), dtype=np.float32),
        "b": np.zeros((5, 3), dtype=np.float32),
    }
    shape = get_shape(100, batch)
    assert shape["a"] == (100,)
    assert shape["b"] == (100, 3)


def test_get_all_groups(input_file):
    groups = get_all_groups(input_file)
    assert isinstance(groups, dict)
    assert "jets" in groups


def test_h5_add_column_appends_field(tmp_path, input_file, append_func):
    output_file = tmp_path / "out.h5"
    h5_add_column(input_file, output_file, append_func)

    with h5py.File(output_file) as f:
        assert "jets" in f
        assert "new_phi" in f["jets"].dtype.names


def test_h5_add_column_overwrite_protection(tmp_path, input_file, append_func):
    output_file = tmp_path / "overwrite.h5"
    h5_add_column(input_file, output_file, append_func)
    with pytest.raises(FileExistsError):
        h5_add_column(input_file, output_file, append_func)


def test_h5_add_column_allows_overwrite(tmp_path, input_file, append_func):
    output_file = tmp_path / "overwrite2.h5"
    h5_add_column(input_file, output_file, append_func)
    h5_add_column(input_file, output_file, append_func, overwrite=True)


def test_h5_add_column_default_output_path(input_file, append_func):
    out_path = Path(str(input_file).replace(".h5", "_additional.h5"))
    if out_path.exists():
        out_path.unlink()
    h5_add_column(input_file, None, append_func)
    assert out_path.exists()


def test_h5_add_column_batch_printing(tmp_path, input_file, append_func, capsys):
    output_file = tmp_path / "printed.h5"
    h5_add_column(
        input_file,
        output_file,
        append_func,
        reader_kwargs={"batch_size": 1},  # ensures lots of batches
    )
    out = capsys.readouterr().out
    assert "Processing batch" in out


def test_h5_add_column_rejects_existing_field(tmp_path, input_file):
    def bad_func(batch):
        return {"jets": {"pt": batch["jets"]["pt"]}}  # pt already exists

    with pytest.raises(ValueError, match="already exists in batch"):
        h5_add_column(input_file, tmp_path / "err.h5", bad_func)


def test_h5_add_column_rejects_wrong_shape(tmp_path, input_file):
    def bad_shape_func(batch):  # noqa: ARG001
        return {"jets": {"wrong": np.ones((123,))}}  # bad shape

    with pytest.raises(ValueError, match="shape is not correct"):
        h5_add_column(input_file, tmp_path / "badshape.h5", bad_shape_func)


def test_h5_add_column_rejects_wrong_output_group(tmp_path, input_file):
    def other_group(batch):
        return {"tracks": {"phi": np.ones(len(batch["jets"]))}}  # not allowed group

    with pytest.raises(ValueError, match="Trying to append phi to tracks"):
        h5_add_column(input_file, tmp_path / "wronggroup.h5", other_group)


def test_output_to_non_writen_group(tmp_path, input_file, append_func):
    with pytest.raises(ValueError, match="Trying to output to jets but only "):
        h5_add_column(
            input_file,
            tmp_path / "non_writen_group.h5",
            append_func,
            output_groups=["tracks"],  # only allow writing to tracks
            input_groups=["jets", "tracks"],  # only allow reading from jets
        )


def test_skip_tracks(tmp_path, input_file, append_func):
    h5_add_column(
        input_file,
        tmp_path / "non_writen_group.h5",
        append_func,
        output_groups=["jets"],  # only allow writing to tracks
        input_groups=["jets", "tracks"],  # only allow reading from jets
    )
