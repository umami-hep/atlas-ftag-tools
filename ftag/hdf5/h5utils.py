from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from ftag.transform import Transform

if TYPE_CHECKING:  # pragma: no cover
    import h5py

__all__ = [
    "cast_dtype",
    "compare_groups",
    "extract_group_full",
    "get_dtype",
    "join_structured_arrays",
    "structured_from_dict",
    "write_group_full",
]


def compare_groups(g1: h5py.Group | dict, g2: h5py.Group | dict, path: str = ""):
    """Recursively compare two h5py.Groups or in-memory dicts.

    Parameters
    ----------
    g1 : h5py.Group | dict
        First group or dict to compare
    g2 : h5py.Group | dict
        Second group or dict to compare
    path : str, optional
        Path to the current group, by default ""

    Raises
    ------
    TypeError
        If the types of the items do not match
    """
    assert set(g1.keys()) == set(g2.keys()), (
        f"{path}: Keys mismatch: {set(g1.keys())} vs {set(g2.keys())}"
    )

    for key in g1:
        item1 = g1[key]
        item2 = g2[key]
        subpath = f"{path}/{key}"

        # Check for extracted group (dict) with 'data'
        if isinstance(item1, dict) and "data" in item1:
            assert isinstance(item2, dict), f"{subpath}: One side missing 'data'"
            assert "data" in item2, f"{subpath}: One side missing 'data'"
            np.testing.assert_array_equal(
                item1["data"], item2["data"], err_msg=f"{subpath}: Data mismatch"
            )
            assert item1.get("attrs", {}) == item2.get("attrs", {}), f"{subpath}: Attr mismatch"

        # Check for full extracted group
        elif isinstance(item1, dict):
            assert isinstance(item2, dict), f"{subpath}: Expected nested dict"
            compare_groups(item1, item2, subpath)

        # h5py.Dataset or Group objects
        elif isinstance(item1, h5py.Dataset):
            np.testing.assert_array_equal(item1[()], item2[()], err_msg=f"{subpath}: Data mismatch")
            assert dict(item1.attrs) == dict(item2.attrs), f"{subpath}: Attr mismatch"

        elif isinstance(item1, h5py.Group):
            compare_groups(item1, item2, subpath)

        else:
            raise TypeError(f"{subpath}: Unexpected type: {type(item1)}")


def write_group_full(h5group: h5py.Group, data: dict):
    """Write a nested dictionary structure to an HDF5 group.

    This function recursively writes a dictionary containing datasets and subgroups
    to an HDF5 group. The dictionary should have the structure created by
    extract_group_full().

    Parameters
    ----------
    h5group : h5py.Group
        The HDF5 group to write data to
    data : dict
        Dictionary containing the data structure to write. Can contain:
        - '_group_attrs': dict of group-level attributes
        - dataset entries: dict with 'data' and 'attrs' keys
        - subgroup entries: nested dictionaries

    Raises
    ------
    TypeError
        If an unexpected value type is encountered in the data dict
    """
    # Write group-level attributes
    if "_group_attrs" in data:
        for k, v in data["_group_attrs"].items():
            h5group.attrs[k] = v

    for key, value in data.items():
        if key == "_group_attrs":
            continue
        if isinstance(value, dict) and "data" in value:
            dset = h5group.create_dataset(key, data=value["data"])
            for attr_k, attr_v in value["attrs"].items():
                dset.attrs[attr_k] = attr_v
        elif isinstance(value, dict):
            subgroup = h5group.create_group(key)
            write_group_full(subgroup, value)
        else:
            raise TypeError(f"Unexpected value type for key '{key}': {type(value)}")


def extract_group_full(group: h5py.Group) -> dict:
    """Extract the full contents of an HDF5 group into a nested dictionary.

    This function recursively extracts all datasets, subgroups, and attributes
    from an HDF5 group into an in-memory dictionary structure. Group-level
    attributes are stored under the '_group_attrs' key.

    Parameters
    ----------
    group : h5py.Group
        The HDF5 group to extract data from

    Returns
    -------
    dict
        Nested dictionary containing:
        - '_group_attrs': dict of group-level attributes (if any)
        - dataset entries: dict with 'data' (array) and 'attrs' (dict) keys
        - subgroup entries: nested dictionaries with same structure

    Raises
    ------
    TypeError
        If an unsupported HDF5 item type is encountered
    """
    result = {}
    # Save group-level attributes
    if group.attrs:
        result["_group_attrs"] = {k: group.attrs[k] for k in group.attrs}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            result[key] = {"data": item[()], "attrs": {k: item.attrs[k] for k in item.attrs}}
        elif isinstance(item, h5py.Group):
            result[key] = extract_group_full(item)
        else:
            raise TypeError(f"Unsupported item {key}: {type(item)}")
    return result


def get_dtype(
    ds: h5py.Dataset,
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
    variables = variables or ds.dtype.names
    # If we have a non structured array we just return its dtype
    if variables is None:
        return ds.dtype

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


def join_structured_arrays(arrays: list) -> np.ndarray:
    """Join a list of structured numpy arrays.

    See https://github.com/numpy/numpy/issues/7811

    Parameters
    ----------
    arrays : list
        List of structured numpy arrays to join

    Returns
    -------
    np.ndarray
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
    d : dict[str, np.ndarray]
        Input dict of numpy arrays

    Returns
    -------
    np.ndarray
        Structured array
    """
    arrays = np.column_stack(list(d.values()))
    dtypes = np.dtype([(k, v.dtype) for k, v in d.items()])
    return u2s(arrays, dtype=dtypes)
