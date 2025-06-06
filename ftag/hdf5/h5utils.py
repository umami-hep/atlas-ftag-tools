from __future__ import annotations

import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from ftag.transform import Transform

__all__ = ["cast_dtype", "get_dtype", "join_structured_arrays"]


def get_dtype(
    ds,
    variables: list[str] | None = None,
    precision: str | None = None,
    transform: Transform | None = None,
    full_precision_vars: list[str] | None = None,
) -> np.dtype:
    """Return a dtype based on an existing dataset and requested variables.

    Parameters
    ----------
    ds : h5py.Dataset
        Input h5 dataset
    variables : list[str] | None, optional
        List of variables to include in dtype, by default None
    precision : str | None, optional
        Precision to cast floats to, "half" or "full", by default None
    transform : Transform | None, optional
        Transform to apply to variables names, by default None
    full_precision_vars : list[str] | None, optional
        List of variables to keep in full precision, by default None

    Returns
    -------
    np.dtype
        Output dtype

    Raises
    ------
    ValueError
        If variables are not found in dataset
    """
    if variables is None:
        variables = ds.dtype.names
    if full_precision_vars is None:
        full_precision_vars = []

    if (missing := set(variables) - set(ds.dtype.names)) and transform is not None:
        variables = transform.map_variable_names(ds.name, variables, inverse=True)
        missing = set(variables) - set(ds.dtype.names)
    if missing:
        raise ValueError(
            f"Variables {missing} were not found in dataset {ds.name} in file {ds.file.filename}"
        )

    dtype = [(n, x) for n, x in ds.dtype.descr if n in variables]
    if precision:
        dtype = [
            (n, cast_dtype(x, precision)) if n not in full_precision_vars else (n, x)
            for n, x in dtype
        ]

    return np.dtype(dtype)


def cast_dtype(typestr: str, precision: str) -> np.dtype:
    """Cast float type to half or full precision.

    Parameters
    ----------
    typestr : str
        Input type string
    precision : str
        Precision to cast to, "half" or "full"

    Returns
    -------
    np.dtype
        Output dtype

    Raises
    ------
    ValueError
        If precision is not "half" or "full"
    """
    t = np.dtype(typestr)
    if t.kind != "f":
        return t

    if precision == "half":
        return np.dtype("f2")
    if precision == "full":
        return np.dtype("f4")
    raise ValueError(f"Invalid precision {precision}")


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


def structured_from_dict(d: dict[str, np.ndarray]) -> np.ndarray:
    """Convert a dict to a structured array.

    Parameters
    ----------
    d : dict
        Input dict of numpy arrays

    Returns
    -------
    np.ndarray
        Structured array
    """
    arrays = np.column_stack(list(d.values()))
    dtypes = np.dtype([(k, v.dtype) for k, v in d.items()])
    return u2s(arrays, dtype=dtypes)
