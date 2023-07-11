from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
import pytest
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from ftag.hdf5 import join_structured_arrays


class catchtime:
    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):  # noqa: A002
        self.time = perf_counter() - self.time
        self.readout = f"Time: {self.time:.3f} seconds"

    def __str__(self):
        return self.readout


n_float_vars = 20
n_int_vars = 20
save_dir = Path("/tmp")


def generate_data(n_total=1000, n_tracks=50, compression="gzip", float_type="float16"):
    f_dtype = np.dtype([(f"vf_{i}", float_type) for i in range(n_float_vars)])
    i_dtype = np.dtype([(f"vi_{i}", "uint8") for i in range(n_int_vars)])

    f_vars = np.random.rand(n_total, n_tracks, n_float_vars).astype(float_type)
    f_vars = u2s(f_vars, f_dtype)
    i_vars = (np.random.rand(n_total, n_tracks, n_int_vars) * 10).astype("uint8")
    i_vars = u2s(i_vars, i_dtype)

    tracks = join_structured_arrays([f_vars, i_vars])

    idx = np.random.choice([True, False], (n_total, n_tracks))
    tracks[idx] = 0
    tracks_u = s2u(tracks).astype(float_type)

    with h5py.File(save_dir / "test_structured.h5", "w") as f:
        f.create_dataset("tracks", data=tracks, compression=compression, chunks=(100, n_tracks))

    with h5py.File(save_dir / "test_structured2.h5", "w") as f:
        f.create_dataset("tracks", data=tracks, compression=compression, chunks=(100, n_tracks))

    with h5py.File(save_dir / "test_unstructured.h5", "w") as f:
        f.create_dataset(
            "tracks",
            data=tracks_u,
            compression=compression,
            chunks=(100, n_tracks, n_float_vars + n_int_vars),
        )


def load_structured(idx):
    with h5py.File(save_dir / "test_structured.h5", "r") as f, catchtime() as t:
        f["tracks"][idx[0] : idx[1]]
    return t


def load_unstructured(idx):
    with h5py.File(save_dir / "test_unstructured.h5", "r") as f, catchtime() as t:
        f["tracks"][idx[0] : idx[1]]
    return t


def load_structured_read_direct(idx):
    with h5py.File(save_dir / "test_structured2.h5", "r") as f:
        array = np.array(0, dtype=f["tracks"].dtype)
        jet_idx = np.s_[idx[0] : idx[1]]
        array.resize(idx[1] - idx[0], f["tracks"].shape[1], refcheck=False)
        with catchtime() as t:
            f["tracks"].read_direct(array, jet_idx)
    return t


@pytest.mark.parametrize("compression", ["gzip", "lzf"])
@pytest.mark.parametrize("float_type", ["float16", "float32"])
@pytest.mark.parametrize("n_total", [1000, 10000])
def test_timing(compression, float_type, n_total):
    idx = (0, n_total // 2)
    generate_data(compression=compression, float_type=float_type, n_total=n_total)
    load_structured_time = load_structured(idx).time
    load_unstructured_time = load_unstructured(idx).time
    load_structured_read_direct_time = load_structured_read_direct(idx).time

    print("Results:")
    print("Load structured:", load_structured_time)
    print("Load unstructured:", load_unstructured_time)
    print("Load structured read_direct:", load_structured_read_direct_time)

    # check that read_direct is fastest
    assert load_structured_read_direct_time < load_structured_time
    assert load_structured_read_direct_time < load_unstructured_time
