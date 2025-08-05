from __future__ import annotations

import tempfile

import h5py
import numpy as np
import pytest

from ftag.hdf5.h5utils import (
    cast_dtype,
    compare_groups,
    extract_group_full,
    get_dtype,
    join_structured_arrays,
    structured_from_dict,
    write_group_full,
)
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


def test_get_dtype_unstructured():
    """Test get_dtype with unstructured array (line 184)."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as f:
            # Create unstructured dataset
            data = np.array([1, 2, 3, 4, 5])
            ds = f.create_dataset("unstructured", data=data)
            
            # Should return the original dtype (line 184)
            result_dtype = get_dtype(ds)
            assert result_dtype == data.dtype


def test_cast_dtype():
    assert cast_dtype("f4", "full") == np.dtype("f4")
    assert cast_dtype("f2", "full") == np.dtype("f4")
    assert cast_dtype("f4", "half") == np.dtype("f2")
    assert cast_dtype("f2", "half") == np.dtype("f2")
    assert cast_dtype("i4", "full") == np.dtype("i4")
    assert cast_dtype("i4", "half") == np.dtype("i4")
    
    # Test invalid precision (line 234)
    with pytest.raises(ValueError, match="Invalid precision"):
        cast_dtype("f4", "invalid")


def test_join_structured_arrays():
    # test the function join_structured_arrays
    arr1 = np.ones((3,), dtype=[("a", int), ("b", float)])
    arr2 = np.zeros((3,), dtype=[("c", int), ("d", float)])
    arr = join_structured_arrays([arr1, arr2])
    assert arr.dtype.names == ("a", "b", "c", "d")
    assert all(arr["a"] == 1)
    assert all(arr["c"] == 0)


def test_structured_from_dict():
    input_dict = {
        "field1": np.array([1, 2, 3]),
        "field2": np.array([4, 5, 6]),
        "field3": np.array([7, 8, 9]),
    }
    structured_array = structured_from_dict(input_dict)
    assert structured_array.dtype.names == ("field1", "field2", "field3")
    assert all(structured_array["field1"] == np.array([1, 2, 3]))
    assert all(structured_array["field2"] == np.array([4, 5, 6]))
    assert all(structured_array["field3"] == np.array([7, 8, 9]))


def test_compare_groups():
    """Test compare_groups function with various group structures."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as f:
            # Create test groups with datasets and attributes
            g1 = f.create_group("group1")
            g1.attrs["group_attr"] = "test_value"
            ds1 = g1.create_dataset("dataset1", data=np.array([1, 2, 3]))
            ds1.attrs["ds_attr"] = "dataset_attr"
            
            g2 = f.create_group("group2")
            g2.attrs["group_attr"] = "test_value"
            ds2 = g2.create_dataset("dataset1", data=np.array([1, 2, 3]))
            ds2.attrs["ds_attr"] = "dataset_attr"
            
            # Test identical groups - should not raise
            compare_groups(g1, g2)
            
            # Test with different data - should raise assertion error
            g3 = f.create_group("group3")
            g3.attrs["group_attr"] = "test_value"
            ds3 = g3.create_dataset("dataset1", data=np.array([1, 2, 4]))  # Different data
            ds3.attrs["ds_attr"] = "dataset_attr"
            
            with pytest.raises(AssertionError, match="Data mismatch"):
                compare_groups(g1, g3)
    
    # Test with dict structures
    dict1 = {
        "dataset1": {"data": np.array([1, 2, 3]), "attrs": {"attr1": "value1"}},
        "subgroup": {
            "dataset2": {"data": np.array([4, 5, 6]), "attrs": {}}
        }
    }
    dict2 = {
        "dataset1": {"data": np.array([1, 2, 3]), "attrs": {"attr1": "value1"}},
        "subgroup": {
            "dataset2": {"data": np.array([4, 5, 6]), "attrs": {}}
        }
    }
    
    # Should not raise
    compare_groups(dict1, dict2)
    
    # Test with mismatched keys
    dict3 = {
        "dataset1": {"data": np.array([1, 2, 3]), "attrs": {"attr1": "value1"}},
        "different_key": {
            "dataset2": {"data": np.array([4, 5, 6]), "attrs": {}}
        }
    }
    
    with pytest.raises(AssertionError, match="Keys mismatch"):
        compare_groups(dict1, dict3)


def test_compare_groups_unexpected_type():
    """Test compare_groups with unexpected type (line 66)."""
    # Create a dict with an unexpected type
    dict1 = {"key1": "string_value"}  # Unexpected type
    dict2 = {"key1": "string_value"}
    
    with pytest.raises(TypeError, match="Unexpected type"):
        compare_groups(dict1, dict2)


def test_write_group_full():
    """Test write_group_full function."""
    data = {
        "_group_attrs": {"group_level_attr": "test_value"},
        "dataset1": {
            "data": np.array([1, 2, 3, 4, 5]),
            "attrs": {"units": "GeV", "description": "test dataset"}
        },
        "subgroup": {
            "dataset2": {
                "data": np.array([[1, 2], [3, 4]]),
                "attrs": {"shape_info": "2x2 matrix"}
            },
            "_group_attrs": {"subgroup_attr": "nested_value"}
        }
    }
    
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as f:
            root_group = f.create_group("test_group")
            write_group_full(root_group, data)
            
            # Verify group attributes
            assert root_group.attrs["group_level_attr"] == "test_value"
            
            # Verify dataset
            assert "dataset1" in root_group
            ds1 = root_group["dataset1"]
            np.testing.assert_array_equal(ds1[()], np.array([1, 2, 3, 4, 5]))
            assert ds1.attrs["units"] == "GeV"
            assert ds1.attrs["description"] == "test dataset"
            
            # Verify subgroup
            assert "subgroup" in root_group
            subgroup = root_group["subgroup"]
            assert subgroup.attrs["subgroup_attr"] == "nested_value"
            
            # Verify nested dataset
            assert "dataset2" in subgroup
            ds2 = subgroup["dataset2"]
            np.testing.assert_array_equal(ds2[()], np.array([[1, 2], [3, 4]]))
            assert ds2.attrs["shape_info"] == "2x2 matrix"
    
    # Test with invalid data structure
    invalid_data = {"invalid_key": "string_value"}  # Should raise TypeError
    
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as f:
            root_group = f.create_group("test_group")
            with pytest.raises(TypeError, match="Unexpected value type"):
                write_group_full(root_group, invalid_data)


def test_extract_group_full():
    """Test extract_group_full function."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as f:
            # Create test structure
            root_group = f.create_group("test_group")
            root_group.attrs["root_attr"] = "root_value"
            
            # Add dataset
            ds1 = root_group.create_dataset("dataset1", data=np.array([10, 20, 30]))
            ds1.attrs["ds1_attr"] = "dataset1_value"
            
            # Add subgroup with nested content
            subgroup = root_group.create_group("subgroup")
            subgroup.attrs["sub_attr"] = "sub_value"
            ds2 = subgroup.create_dataset("dataset2", data=np.array([[1, 2], [3, 4]]))
            ds2.attrs["ds2_attr"] = 42
            
            # Extract the full group
            extracted = extract_group_full(root_group)
            
            # Verify structure
            assert "_group_attrs" in extracted
            assert extracted["_group_attrs"]["root_attr"] == "root_value"
            
            # Verify dataset1
            assert "dataset1" in extracted
            assert "data" in extracted["dataset1"]
            assert "attrs" in extracted["dataset1"]
            np.testing.assert_array_equal(extracted["dataset1"]["data"], np.array([10, 20, 30]))
            assert extracted["dataset1"]["attrs"]["ds1_attr"] == "dataset1_value"
            
            # Verify subgroup
            assert "subgroup" in extracted
            assert "_group_attrs" in extracted["subgroup"]
            assert extracted["subgroup"]["_group_attrs"]["sub_attr"] == "sub_value"
            
            # Verify nested dataset
            assert "dataset2" in extracted["subgroup"]
            np.testing.assert_array_equal(extracted["subgroup"]["dataset2"]["data"], np.array([[1, 2], [3, 4]]))
            assert extracted["subgroup"]["dataset2"]["attrs"]["ds2_attr"] == 42
            
            # Test round-trip: extract then write back
            with h5py.File(tmp.name.replace(".h5", "_roundtrip.h5"), "w") as f2:
                new_group = f2.create_group("recreated_group")
                write_group_full(new_group, extracted)
                
                # Verify the round-trip worked
                compare_groups(root_group, new_group)


def test_extract_group_full_unsupported_item():
    """Test extract_group_full with unsupported item type (line 145)."""
    # This is tricky to test with real HDF5 since it only allows Groups and Datasets
    # We'll test by mocking the items() method to return an unsupported type
    import unittest.mock
    
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        with h5py.File(tmp.name, "w") as f:
            group = f.create_group("test_group")
            
            # Mock an unsupported item type
            class UnsupportedItem:
                pass
            
            unsupported_item = UnsupportedItem()
            
            # Mock the items() method to return an unsupported item
            with unittest.mock.patch.object(group, 'items') as mock_items:
                mock_items.return_value = [("unsupported", unsupported_item)]
                
                with pytest.raises(TypeError, match="Unsupported item"):
                    extract_group_full(group)
