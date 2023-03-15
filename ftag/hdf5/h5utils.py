from tempfile import NamedTemporaryFile

import h5py
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured as u2s

__all__ = ["get_dummy_file"]


def get_dtype(ds, variables: list[str] | None = None, precision: str | None = None) -> np.dtype:
    """Return a dtype based on an existing dataset and requested variables.

    Parameters
    ----------
    ds : _type_
        _description_
    variables : list[str] | None, optional
        _description_, by default None
    precision : str | None, optional
        _description_, by default None

    Returns
    -------
    np.dtype
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    if variables is None:
        variables = ds.dtype.names

    if missing := set(variables) - set(ds.dtype.names):
        raise ValueError(
            f"Variables {missing} were not found in dataset {ds.name} in file {ds.file.filename}"
        )

    dtype = [(n, x) for n, x in ds.dtype.descr if n in variables]
    if precision:
        dtype = [(n, cast_dtype(x, precision)) for n, x in dtype]

    return np.dtype(dtype)


def cast_dtype(typestr: str, precision: str) -> np.dtype:
    """Cast float type to half or full precision.

    Parameters
    ----------
    typestr : str
        _description_
    precision : str
        _description_

    Returns
    -------
    np.dtype
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    t = np.dtype(typestr)
    if t.kind != "f" or t.itemsize != 2:
        return t
    if precision == "half":
        return np.dtype("f2")
    if precision == "full":
        return np.dtype("f4")
    raise ValueError(f"Invalid precision {precision}")


def get_dummy_file():
    jet_vars = [
        "pt",
        "eta",
        "abs_eta",
        "mass",
        "HadronConeExclTruthLabelID",
        "n_tracks",
        "n_truth_promptLepton",
    ]

    track_vars = ["pt", "deta", "dphi", "dr"]

    # settings
    n_jets = 1000
    n_tracks_per_jet = 40

    # setup jets
    shapes_jets = {
        "inputs": [n_jets, len(jet_vars)],
    }

    # setup tracks
    shapes_tracks = {
        "inputs": [n_jets, n_tracks_per_jet, len(track_vars)],
        "valid": [n_jets, n_tracks_per_jet],
    }

    # setup jets
    rng = np.random.default_rng()
    jets_dtype = np.dtype([(n, "f4") for n in jet_vars])
    jets = u2s(rng.random(shapes_jets["inputs"]), jets_dtype)
    jets["HadronConeExclTruthLabelID"] = np.random.choice([0, 4, 5], size=n_jets)
    jets["pt"] *= 400e3
    jets["eta"] = (jets["eta"] - 0.5) * 6.0
    jets["abs_eta"] = np.abs(jets["eta"])

    # setup tracks
    tracks_dtype = np.dtype([(n, "f4") for n in track_vars])
    tracks = u2s(rng.random(shapes_tracks["inputs"]), tracks_dtype)
    valid = rng.random(shapes_tracks["valid"])
    valid = valid.astype(bool).view(dtype=np.dtype([("valid", bool)]))
    tracks = join_structured_arrays([tracks, valid])

    fname = NamedTemporaryFile(suffix=".h5").name
    f = h5py.File(fname, "w")
    f.create_dataset("jets", data=jets)
    f.create_dataset("tracks", data=tracks)
    f.create_dataset("flow", data=tracks)
    return fname, f


def join_structured_arrays(arrays: list):
    """Join a list of structured numpy arrays.

    See https://github.com/numpy/numpy/issues/7811

    Parameters
    ----------
    arrays : list
        List of structured numpy arrays to join

    Returns
    -------
    np.array
        A merged structured array
    """
    dtype: list = sum((a.dtype.descr for a in arrays), [])
    newrecarray = np.empty(arrays[0].shape, dtype=dtype)
    for a in arrays:
        for name in a.dtype.names:
            newrecarray[name] = a[name]

    return newrecarray
