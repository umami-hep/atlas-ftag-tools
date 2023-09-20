from __future__ import annotations

import numpy as np
import pytest

from ftag.hdf5.h5utils import cast_dtype, get_dtype, join_structured_arrays
from ftag.mock import get_mock_file
from ftag.transform import Transform


def test_get_dtype():
    f = get_mock_file()[1]
    ds = f["jets"]
    variables = ["pt", "eta"]
    expected_dtype = np.dtype([("pt", "f4"), ("eta", "f4")])
    assert get_dtype(ds, variables) == expected_dtype

    # Test with missing variables
    variables = ["foo", "baz"]
    with pytest.raises(ValueError):
        get_dtype(ds, variables)

    # Test with precision cast
    variables = ["pt", "eta"]
    expected_dtype = np.dtype([("pt", "f2"), ("eta", "f2")])
    assert get_dtype(ds, variables, "half") == expected_dtype


def test_get_dtype_with_transform():
    tf = Transform(variable_map={"jets": {"pt": "pt_new"}})
    f = get_mock_file()[1]
    ds = f["jets"]
    variables = ["pt_new", "eta"]
    expected_dtype = np.dtype([("pt", "f4"), ("eta", "f4")])
    assert get_dtype(ds, variables, transform=tf) == expected_dtype


def test_cast_dtype():
    assert cast_dtype("f4", "full") == np.dtype("f4")
    assert cast_dtype("f2", "full") == np.dtype("f4")
    assert cast_dtype("f4", "half") == np.dtype("f2")
    assert cast_dtype("f2", "half") == np.dtype("f2")
    assert cast_dtype("i4", "full") == np.dtype("i4")
    assert cast_dtype("i4", "half") == np.dtype("i4")


def test_join_structured_arrays():
    # test the function join_structured_arrays
    arr1 = np.ones((3,), dtype=[("a", int), ("b", float)])
    arr2 = np.zeros((3,), dtype=[("c", int), ("d", float)])
    arr = join_structured_arrays([arr1, arr2])
    assert arr.dtype.names == ("a", "b", "c", "d")
    assert all(arr["a"] == 1)
    assert all(arr["c"] == 0)
