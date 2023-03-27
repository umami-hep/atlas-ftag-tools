from __future__ import annotations

import numpy as np

__all__ = ["join_structured_arrays"]


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
