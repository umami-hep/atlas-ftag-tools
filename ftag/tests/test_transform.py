import numpy as np
import pytest

from ftag.transform import Transform


@pytest.fixture
def sample_batch():
    return {
        "group1": np.array([(1, 2), (3, 4)], dtype=[("var1", int), ("var2", int)]),
        "group2": np.array([(5, 6), (7, 8)], dtype=[("var3", int), ("var4", int)]),
    }


def test_map_ints(sample_batch):
    ints_map = {
        "group1": {
            "var1": {1: 10},
            "var2": {2: 20},
        },
        "group2": {
            "var3": {5: 50},
            "var4": {6: 60},
        },
    }
    transformer = Transform(ints_map=ints_map)
    transformed_batch = transformer.map_ints(sample_batch)
    assert transformed_batch["group1"]["var1"].tolist() == [10, 3]
    assert transformed_batch["group1"]["var2"].tolist() == [20, 4]
    assert transformed_batch["group2"]["var3"].tolist() == [50, 7]
    assert transformed_batch["group2"]["var4"].tolist() == [60, 8]


def test_rename_variables(sample_batch):
    variable_name_map = {
        "group1": {
            "var1": "new_var1",
            "var2": "new_var2",
        },
        "group2": {
            "var3": "new_var3",
        },
    }
    transformer = Transform(variable_name_map)
    transformed_batch = transformer.rename_variables(sample_batch)

    assert "new_var1" in transformed_batch["group1"].dtype.names
    assert "new_var2" in transformed_batch["group1"].dtype.names
    assert "new_var3" in transformed_batch["group2"].dtype.names
    assert "var4" in transformed_batch["group2"].dtype.names
    assert "var1" not in transformed_batch["group1"].dtype.names
    assert "var2" not in transformed_batch["group1"].dtype.names
    assert "var3" not in transformed_batch["group2"].dtype.names
    assert "new_var4" not in transformed_batch["group2"].dtype.names


def test_rename_variables_existing_variable(sample_batch):
    variable_name_map = {
        "group1": {
            "var1": "var2",
        },
    }
    transformer = Transform(variable_name_map)
    with pytest.raises(ValueError):
        transformer.rename_variables(sample_batch)


def test_transform_call(sample_batch):
    # Sample maps for testing
    variable_name_map = {
        "group1": {
            "var1": "new_var1",
            "var2": "new_var2",
        },
        "group2": {
            "var3": "new_var3",
            "var4": "new_var4",
        },
    }

    ints_map = {
        "group1": {
            "new_var1": {1: 10},
            "new_var2": {2: 20},
        },
        "group2": {
            "new_var3": {5: 50},
            "new_var4": {6: 60},
        },
    }

    transformer = Transform(variable_name_map, ints_map)
    transformed_batch = transformer(sample_batch)

    # Verify that variables are renamed and integers are mapped
    assert "new_var1" in transformed_batch["group1"].dtype.names
    assert "new_var2" in transformed_batch["group1"].dtype.names
    assert "new_var3" in transformed_batch["group2"].dtype.names
    assert "new_var4" in transformed_batch["group2"].dtype.names
    assert "var1" not in transformed_batch["group1"].dtype.names
    assert "var2" not in transformed_batch["group1"].dtype.names
    assert "var3" not in transformed_batch["group2"].dtype.names
    assert "var4" not in transformed_batch["group2"].dtype.names

    assert transformed_batch["group1"]["new_var1"].tolist() == [10, 3]
    assert transformed_batch["group1"]["new_var2"].tolist() == [20, 4]
    assert transformed_batch["group2"]["new_var3"].tolist() == [50, 7]
    assert transformed_batch["group2"]["new_var4"].tolist() == [60, 8]
