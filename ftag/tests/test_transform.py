import numpy as np
import pytest

from ftag.transform import Transform


@pytest.fixture
def sample_batch():
    return {
        "group1": np.array([(1, 2), (3, 4)], dtype=[("var1", int), ("var2", int)]),
        "group2": np.array([(5, 6), (7, 8)], dtype=[("var3", int), ("var4", int)]),
        "group3": np.array(
            [(10.0, 100.0), (100.0, 10.0)], dtype=[("var5", float), ("var6", float)]
        ),
    }


@pytest.fixture
def floats_map():
    return {
        "group3": {
            "var5": "log10",
        }
    }


@pytest.fixture
def ints_map():
    return {
        "group1": {
            "var2": {2: 20},
            "new_var2": {2: 20},
        },
        "group2": {
            "var3": {5: 50},
        },
    }


@pytest.fixture
def variable_name_map():
    return {
        "group1": {
            "var1": "new_var1",
            "var2": "new_var2",
        },
        "group2": {
            "var3": "new_var3",
        },
    }


def test_map_ints(sample_batch, ints_map):
    transform = Transform(ints_map=ints_map)
    transformed_batch = transform.map_ints(sample_batch)
    assert transformed_batch["group1"]["var2"].tolist() == [20, 4]
    assert transformed_batch["group2"]["var3"].tolist() == [50, 7]


def test_rename_variables(sample_batch, variable_name_map):
    transform = Transform(variable_name_map)
    transformed_batch = transform.rename_variables(sample_batch)

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
    transform = Transform(variable_name_map)
    with pytest.raises(ValueError):
        transform.rename_variables(sample_batch)


def test_transform_call(sample_batch, variable_name_map, ints_map):
    transform = Transform(variable_name_map, ints_map)
    transformed_batch = transform(sample_batch)

    assert "new_var2" in transformed_batch["group1"].dtype.names
    assert "new_var3" in transformed_batch["group2"].dtype.names
    assert "var3" not in transformed_batch["group2"].dtype.names
    assert transformed_batch["group1"]["new_var2"].tolist() == [20, 4]
    assert transformed_batch["group2"]["new_var3"].tolist() == [5, 7]


def test_transform_floats(sample_batch, floats_map):
    transform = Transform(floats_map=floats_map)
    transformed_batch = transform.transform_floats(sample_batch)
    assert transformed_batch["group3"]["var5"].tolist() == [1.0, 2.0]
