# tests/test_vds_all.py
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from ftag.vds import (
    aggregate_cutbookkeeper,
    check_subgroups,
    create_virtual_file,
    get_virtual_layout,
    glob_re,
    main,
    regex_files_from_dir,
    sum_counts_once,
)


# ---------------------------------------------------------------------
# basic fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def test_h5_files():
    """Create five tiny `.h5` files, each with a dataset ``data``.

    Yields
    ------
    list[pathlib.Path]
        Paths to the created files.  The files are deleted when the fixture
        scope ends.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        paths: list[Path] = []
        for i in range(5):
            fname = Path(tmpdir) / f"test_file_{i}.h5"
            with h5py.File(fname, "w") as f:
                f.create_dataset("data", data=[i] * 5)
            paths.append(fname)
        yield paths


@pytest.fixture
def test_h5_dirs():
    """Create three nested directories, each holding five small HDF5 files.

    Yields
    ------
    list[pathlib.Path]
        Paths to **all** created files.  The directories are kept alive
        for the whole test by retaining references to the corresponding
        ``TemporaryDirectory`` objects.
    """
    with tempfile.TemporaryDirectory(prefix="outer_tmp_", suffix="_h5dump") as top:
        file_paths: list[Path] = []

        # Keep a reference to every TemporaryDirectory so they are *not*
        # garbage-collected (and thus deleted) before the test finishes.
        _keep_alive: list[tempfile.TemporaryDirectory] = []

        for _ in range(3):
            nested = tempfile.TemporaryDirectory(
                dir=top,
                prefix="inner_tmp_",
                suffix=".h5",
            )
            _keep_alive.append(nested)  # <- retain

            for j in range(5):
                fname = Path(nested.name) / f"test_file_{j}.h5"
                with h5py.File(fname, "w") as f:
                    f.create_dataset("data", data=[j] * 5)
                file_paths.append(fname)

        yield file_paths


# ---------------------------------------------------------------------
# helpers / fixtures for cutBookkeeper tests
# ---------------------------------------------------------------------
DTYPE_COUNTS = np.dtype(
    [
        ("nEventsProcessed", "<u8"),
        ("sumOfWeights", "<f8"),
        ("sumOfWeightsSquared", "<f8"),
    ],
)


def _make_counts_array(seed: int) -> np.ndarray:
    """Return a (4,) record array whose values depend on *seed*.

    Parameters
    ----------
    seed
        Offset applied to make every file unique.

    Returns
    -------
    numpy.ndarray
        Record array with three fields and four rows.
    """
    arr = np.zeros(4, dtype=DTYPE_COUNTS)
    arr["nEventsProcessed"] = np.arange(1, 5) + seed * 10
    arr["sumOfWeights"] = 0.1 * np.arange(1, 5) + seed
    arr["sumOfWeightsSquared"] = 0.01 * np.arange(1, 5) + seed
    return arr


@pytest.fixture
def h5_with_bookkeeper(tmp_path_factory):
    """Three files; each has ``data`` plus ``cutBookkeeper/{nominal,sysUp}/counts``.

    Returns
    -------
    list[pathlib.Path]
        Paths to the created files.
    """
    root = tmp_path_factory.mktemp("bk_files")
    paths: list[Path] = []
    for i in range(3):
        fname = root / f"bk_{i}.h5"
        with h5py.File(fname, "w") as f:
            f.create_dataset("data", data=np.full(5, i, dtype="i4"))
            for sg in ("nominal", "sysUp"):
                grp = f.require_group(f"cutBookkeeper/{sg}")
                grp.create_dataset("counts", data=_make_counts_array(i))
        paths.append(fname)
    return paths


@pytest.fixture
def h5_without_bookkeeper(tmp_path_factory):
    """Three files **without** any ``cutBookkeeper`` group.

    Returns
    -------
    list[pathlib.Path]
        Paths to the created files.
    """
    root = tmp_path_factory.mktemp("plain_files")
    paths: list[Path] = []
    for i in range(3):
        fname = root / f"plain_{i}.h5"
        with h5py.File(fname, "w") as f:
            f.create_dataset("data", data=np.full(5, i, dtype="i4"))
        paths.append(fname)
    return paths


# ---------------------------------------------------------------------
# cutBookkeeper-specific tests
# ---------------------------------------------------------------------
def test_sum_counts_once():
    counts = _make_counts_array(seed=0)
    summed = sum_counts_once(counts)
    assert summed.shape == ()
    assert summed["nEventsProcessed"] == 10  # 1+2+3+4
    np.testing.assert_allclose(summed["sumOfWeights"], 1.0)


def test_check_subgroups(h5_with_bookkeeper):
    subgroups = check_subgroups(h5_with_bookkeeper)
    assert subgroups == ["nominal", "sysUp"]


def test_aggregate_cutbookkeeper(h5_with_bookkeeper):
    totals = aggregate_cutbookkeeper(h5_with_bookkeeper)
    assert set(totals) == {"nominal", "sysUp"}

    expected_nep = sum(10 + 40 * i for i in range(3))  # 10, 50, 90
    assert totals["nominal"]["nEventsProcessed"] == expected_nep
    assert totals["sysUp"]["nEventsProcessed"] == expected_nep


def test_create_virtual_file_with_bookkeeper(h5_with_bookkeeper, tmp_path):
    pattern = h5_with_bookkeeper[0].parent / "*.h5"
    out_file = tmp_path / "vds_with.h5"
    create_virtual_file(pattern, out_file, overwrite=True)

    with h5py.File(out_file) as f:
        assert f["data"].shape == (15,)
        assert f["cutBookkeeper/nominal/counts"].shape == ()
        assert "sysUp" in f["cutBookkeeper"]


def test_create_virtual_file_without_bookkeeper(h5_without_bookkeeper, tmp_path):
    pattern = h5_without_bookkeeper[0].parent / "*.h5"
    out_file = tmp_path / "vds_without.h5"
    create_virtual_file(pattern, out_file, overwrite=True)

    with h5py.File(out_file) as f:
        assert "cutBookkeeper" not in f
        assert f["data"].shape == (15,)


# ---------------------------------------------------------------------
# original tests (compound assertions split for PT018)
# ---------------------------------------------------------------------
def test_glob_re_files(test_h5_files):
    pattern = "(test_file_1|test_file_3|test_file_5).h5"
    regex_path = str(Path(test_h5_files[0].parent))
    matched_reg = glob_re(pattern, regex_path)
    assert len(matched_reg) == 2
    assert set(matched_reg) == {"test_file_3.h5", "test_file_1.h5"}


def test_glob_re_dirs(test_h5_dirs):
    pattern = "(test_file_1|test_file_3|test_file_5).h5"
    regex_path = str(Path(test_h5_dirs[0]).parent)
    matched_reg = glob_re(pattern, regex_path)
    assert len(matched_reg) == 2
    assert set(matched_reg) == {"test_file_3.h5", "test_file_1.h5"}


def test_glob_re_dirs_nopath(test_h5_dirs):
    pattern = str(Path(test_h5_dirs[0]).parent) + "(test_file_1|test_file_3|test_file_5).h5"
    assert glob_re(pattern, None) is None


def test_h5dir_files(test_h5_files):
    matched_reg = ["test_file_3.h5", "test_file_1.h5"]
    regex_path = str(Path(test_h5_files[0].parent))
    fnames = regex_files_from_dir(matched_reg, regex_path)
    assert fnames == [f"{regex_path}/test_file_3.h5", f"{regex_path}/test_file_1.h5"]


def test_h5dir_dirs(test_h5_dirs):
    to_match = str(test_h5_dirs[0].parent)
    inner_part = re.match(r"(/tmp/)(outer_tmp.*)(inner_tmp.*h5)", to_match).group(3)
    matched_reg = [inner_part]
    regex_path = str(Path(test_h5_dirs[0]).parent.parent)
    fnames = regex_files_from_dir(matched_reg, regex_path)
    assert set(fnames) == {f"{to_match}/test_file_{k}.h5" for k in range(5)}


def test_get_virtual_layout(test_h5_files):
    layout = get_virtual_layout(test_h5_files, "data")
    assert isinstance(layout, h5py.VirtualLayout)
    assert layout.shape == (25,)
    assert layout.dtype == np.dtype("int64")


def test_create_virtual_file(test_h5_files):
    with tempfile.NamedTemporaryFile() as tmpfile:
        pattern = Path(test_h5_files[0]).parent / "test_file_*"
        create_virtual_file(pattern, tmpfile.name, overwrite=True)
        with h5py.File(tmpfile.name) as f:
            assert "data" in f
            assert len(f["data"]) == 25


def test_create_virtual_file_regex(test_h5_files):
    with tempfile.NamedTemporaryFile() as tmpfile:
        regex_path = str(Path(test_h5_files[0].parent))
        pattern = "(test_file_0|test_file_1|test_file_2).h5"
        create_virtual_file(
            pattern=pattern,
            out_fname=tmpfile.name,
            use_regex=True,
            regex_path=regex_path,
            overwrite=True,
        )
        with h5py.File(tmpfile.name) as f:
            assert len(f["data"]) == 15


def test_create_virtual_file_common_groups(test_h5_files):
    with tempfile.TemporaryDirectory() as tmpdir:
        extra_paths: list[Path] = []
        for i in range(2):
            fname = Path(tmpdir) / f"extra_file_{i}.h5"
            with h5py.File(fname, "w") as f:
                f.create_dataset("extra_data", data=[i] * 5)
            extra_paths.append(fname)

        all_files = test_h5_files + extra_paths
        pattern = Path(all_files[0]).parent / "*.h5"

        with tempfile.NamedTemporaryFile() as tmpfile:
            create_virtual_file(pattern, tmpfile.name, overwrite=True)
            with h5py.File(tmpfile.name) as f:
                assert "data" in f
                assert "extra_data" not in f


def test_check_subgroups_no_bookkeeper(tmp_path):
    f = tmp_path / "a.h5"
    with h5py.File(f, "w") as h5:
        h5.create_dataset("data", data=[1])
    with pytest.raises(KeyError):
        check_subgroups([str(f)])


def test_check_subgroups_no_common(tmp_path):
    f1, f2 = (tmp_path / "f1.h5", tmp_path / "f2.h5")
    for idx, name in enumerate([f1, f2]):
        with h5py.File(name, "w") as h5:
            grp = h5.require_group(f"cutBookkeeper/sg{idx}")
            grp.create_dataset("counts", data=_make_counts_array(0))
    with pytest.raises(ValueError):
        check_subgroups([str(f1), str(f2)])


def test_regex_files_from_dir_none_args():
    assert regex_files_from_dir(None, None) is None


def test_main():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            fname = Path(tmpdir) / f"test_{i}.h5"
            with h5py.File(fname, "w") as f:
                dset = f.create_dataset("data", (10,), dtype="f")
                dset.attrs["key"] = "value"

        output_fname = Path(tmpdir) / "out.h5"
        pattern = Path(tmpdir) / "*.h5"
        main([str(pattern), str(output_fname)])

        with h5py.File(output_fname) as f:
            key = next(iter(f.keys()))
            assert key == "data"
            assert f[key].shape == (30,)
            assert f[key].attrs["key"] == "value"
