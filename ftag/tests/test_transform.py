import unittest
from typing import Dict

import numpy as np

from ftag.transform import Transform


def make_sample_batch() -> Dict[str, np.ndarray]:
    """Create a sample batch of structured numpy arrays.

    Returns
    -------
    dict of str to numpy.ndarray
        Dictionary with three groups containing structured arrays
        of integers and floats.
    """
    group1 = np.array(
        [(1, 2), (3, 4)],
        dtype=[("var1", int), ("var2", int)],
    )
    group2 = np.array(
        [(5, 6), (7, 8)],
        dtype=[("var3", int), ("var4", int)],
    )
    group3 = np.array(
        [(10.0, 100.0), (100.0, 10.0)],
        dtype=[("var5", float), ("var6", float)],
    )
    return {"group1": group1, "group2": group2, "group3": group3}


class TestTransformInit(unittest.TestCase):
    """Tests for the Transform class initialisation."""

    def test_init_defaults(self):
        """Test that default initialization creates empty mapping dicts.

        Notes
        -----
        This test verifies that when no mapping dictionaries are given,
        the Transform instance creates empty dictionaries for
        variable_map, ints_map, and floats_map, and that the
        inverse variable map is also an empty dictionary.
        """
        tf = Transform()
        self.assertEqual(tf.variable_map, {})
        self.assertEqual(tf.variable_map_inv, {})
        self.assertEqual(tf.ints_map, {})
        self.assertEqual(tf.floats_map, {})

    def test_init_variable_map_inverse(self):
        """Test construction of the inverse variable map.

        Notes
        -----
        This test checks that the variable_map_inv attribute is built
        as the inverse of variable_map for each group.
        """
        variable_map = {
            "group1": {"a": "b", "c": "d"},
            "group2": {"x": "y"},
        }
        tf = Transform(variable_map=variable_map)
        expected_inv = {
            "group1": {"b": "a", "d": "c"},
            "group2": {"y": "x"},
        }
        self.assertEqual(tf.variable_map_inv, expected_inv)

    def test_init_floats_map_string_to_function(self):
        """Test that float mapping strings are converted to numpy functions.

        Notes
        -----
        This test ensures that string entries in floats_map are
        converted to the corresponding numpy callables during
        initialization.
        """
        floats_map = {
            "group3": {"var5": "log10", "var6": "sqrt"},
        }
        tf = Transform(floats_map=floats_map)
        self.assertTrue(callable(tf.floats_map["group3"]["var5"]))
        self.assertIs(tf.floats_map["group3"]["var5"], np.log10)
        self.assertTrue(callable(tf.floats_map["group3"]["var6"]))
        self.assertIs(tf.floats_map["group3"]["var6"], np.sqrt)


class TestTransformMapInts(unittest.TestCase):
    """Tests for integer value mapping."""

    def test_map_ints_basic(self):
        """Test integer mapping for existing groups and variables.

        Notes
        -----
        This test verifies that specified integer values in the batch
        are replaced according to ints_map, while other values and
        variables remain unchanged.
        """
        batch = make_sample_batch()
        ints_map = {
            "group1": {
                "var2": {2: 20},
            },
            "group2": {
                "var3": {5: 50},
            },
        }
        tf = Transform(ints_map=ints_map)
        transformed = tf.map_ints(batch)
        self.assertEqual(transformed["group1"]["var2"].tolist(), [20, 4])
        self.assertEqual(transformed["group2"]["var3"].tolist(), [50, 7])
        self.assertEqual(transformed["group2"]["var4"].tolist(), [6, 8])

    def test_map_ints_missing_group_and_variable(self):
        """Test integer mapping with missing groups and variables.

        Notes
        -----
        This test checks that:
        missing groups in ints_map are ignored,
        missing variables in a present group are ignored,
        and the batch data remain unchanged in those cases.
        """
        batch = make_sample_batch()
        original_group1 = batch["group1"].copy()
        ints_map = {
            "group1": {
                "non_existing": {1: 10},
            },
            "missing_group": {
                "var1": {1: 10},
            },
        }
        tf = Transform(ints_map=ints_map)
        transformed = tf.map_ints(batch)
        self.assertTrue(np.array_equal(transformed["group1"], original_group1))
        self.assertIn("group2", transformed)
        self.assertIn("group3", transformed)

    def test_map_ints_no_mapping(self):
        """Test integer mapping when no integer map is provided.

        Notes
        -----
        This test verifies that calling map_ints on a Transform
        instance without an integer map leaves the batch unchanged.
        """
        batch = make_sample_batch()
        original = {k: v.copy() for k, v in batch.items()}
        tf = Transform()
        transformed = tf.map_ints(batch)
        for key in original:
            self.assertTrue(np.array_equal(original[key], transformed[key]))


class TestTransformMapFloats(unittest.TestCase):
    """Tests for float value transformations."""

    def test_map_floats_basic_log10(self):
        """Test float transformation using numpy log10 on a single field.

        Notes
        -----
        This test verifies that the configured float transformation
        is applied to the given variable in the specified group.
        """
        batch = make_sample_batch()
        floats_map = {
            "group3": {"var5": "log10"},
        }
        tf = Transform(floats_map=floats_map)
        transformed = tf.map_floats(batch)
        expected = np.log10(np.array([10.0, 100.0]))
        self.assertTrue(np.allclose(transformed["group3"]["var5"], expected))
        self.assertTrue(np.array_equal(transformed["group3"]["var6"], batch["group3"]["var6"]))

    def test_map_floats_missing_group(self):
        """Test float transformation with a group absent from the batch.

        Notes
        -----
        This test ensures that a group present in floats_map but
        absent from the batch does not raise an error and leaves the
        batch unchanged.
        """
        batch = make_sample_batch()
        original = {k: v.copy() for k, v in batch.items()}
        floats_map = {
            "missing_group": {"var5": "log10"},
        }
        tf = Transform(floats_map=floats_map)
        transformed = tf.map_floats(batch)
        for key in original:
            self.assertTrue(np.array_equal(original[key], transformed[key]))

    def test_map_floats_no_mapping(self):
        """Test float transformation when no float map is provided.

        Notes
        -----
        This test verifies that calling map_floats on a Transform
        instance without a float map leaves the batch unchanged.
        """
        batch = make_sample_batch()
        original = {k: v.copy() for k, v in batch.items()}
        tf = Transform()
        transformed = tf.map_floats(batch)
        for key in original:
            self.assertTrue(np.array_equal(original[key], transformed[key]))


class TestTransformMapDtypeAndVariables(unittest.TestCase):
    """Tests for dtype and variable name mapping."""

    def test_map_dtype_no_mapping(self):
        """Test that map_dtype returns the original dtype if no mapping exists.

        Notes
        -----
        This test checks that when the group name is not present in
        variable_map, the dtype returned by map_dtype equals the
        original dtype.
        """
        dtype = np.dtype([("var1", "int32"), ("var2", "float64")])
        tf = Transform(variable_map={"group1": {"var1": "new_var1"}})
        mapped = tf.map_dtype("unknown_group", dtype)
        self.assertEqual(mapped, dtype)

    def test_map_dtype_with_leading_slash_name(self):
        """Test that map_dtype handles group names with a leading slash.

        Notes
        -----
        This test verifies that the leading slash in the group name is
        ignored when looking up the mapping in variable_map.
        """
        dtype = np.dtype([("var1", "int32"), ("var2", "float64")])
        tf = Transform(variable_map={"group1": {"var1": "new_var1"}})
        mapped = tf.map_dtype("/group1", dtype)
        self.assertEqual(mapped.names, ("new_var1", "var2"))
        self.assertEqual(mapped["new_var1"], dtype["var1"])
        self.assertEqual(mapped["var2"], dtype["var2"])

    def test_map_dtype_rename_conflict(self):
        """Test that map_dtype raises when target field already exists.

        Notes
        -----
        This test checks that attempting to rename a variable to a name
        that already exists in the dtype raises a ValueError.
        """
        name = "group1"
        variable_map = {name: {"var1": "var2"}}
        tf = Transform(variable_map=variable_map)
        dtype = np.dtype([("var1", "int32"), ("var2", "float64")])
        self.assertEqual(tf.map_dtype("other_group", dtype), dtype)
        with self.assertRaises(ValueError):
            tf.map_dtype(name, dtype)

    def test_map_dtype_simple_rename(self):
        """Test a simple rename of fields in a dtype.

        Notes
        -----
        This test verifies that the dtype is correctly rebuilt when a
        simple one to one mapping is provided for some fields.
        """
        dtype = np.dtype([("var1", "int32"), ("var2", "float64")])
        tf = Transform(variable_map={"group1": {"var1": "new_var1"}})
        mapped = tf.map_dtype("group1", dtype)
        self.assertEqual(mapped.names, ("new_var1", "var2"))
        self.assertEqual(mapped["new_var1"], dtype["var1"])
        self.assertEqual(mapped["var2"], dtype["var2"])

    def test_map_variables_basic(self):
        """Test renaming of variables in a batch using map_variables.

        Notes
        -----
        This test checks that fields are renamed according to
        variable_map and that unmapped fields stay unchanged.
        """
        batch = make_sample_batch()
        variable_map = {
            "group1": {"var1": "new_var1", "var2": "new_var2"},
            "group2": {"var3": "new_var3"},
        }
        tf = Transform(variable_map=variable_map)
        transformed = tf.map_variables(batch)

        self.assertIn("new_var1", transformed["group1"].dtype.names)
        self.assertIn("new_var2", transformed["group1"].dtype.names)
        self.assertNotIn("var1", transformed["group1"].dtype.names)
        self.assertNotIn("var2", transformed["group1"].dtype.names)

        self.assertIn("new_var3", transformed["group2"].dtype.names)
        self.assertIn("var4", transformed["group2"].dtype.names)
        self.assertNotIn("var3", transformed["group2"].dtype.names)

    def test_map_variables_missing_group(self):
        """Test that map_variables ignores groups not present in the batch.

        Notes
        -----
        This test ensures that a group present in variable_map but
        absent in the batch does not cause any error and has no effect.
        """
        batch = make_sample_batch()
        original = {k: v.copy() for k, v in batch.items()}
        variable_map = {"missing_group": {"var1": "new_var1"}}
        tf = Transform(variable_map=variable_map)
        transformed = tf.map_variables(batch)
        for key in original:
            self.assertTrue(np.array_equal(original[key], transformed[key]))

    def test_map_variables_existing_target_field_raises(self):
        """Test that map_variables raises when renaming to an existing field.

        Notes
        -----
        This test verifies that attempting to rename a variable to an
        already existing field in the same dtype raises a ValueError.
        """
        batch = make_sample_batch()
        variable_map = {"group1": {"var1": "var2"}}
        tf = Transform(variable_map=variable_map)
        with self.assertRaises(ValueError):
            tf.map_variables(batch)


class TestTransformMapVariableNames(unittest.TestCase):
    """Tests for map_variable_names helper method."""

    def test_map_variable_names_forward(self):
        """Test forward mapping of variable names.

        Notes
        -----
        This test verifies that map_variable_names maps variable
        names using variable_map when called in forward mode.
        """
        variable_map = {
            "group1": {"old1": "new1", "old2": "new2"},
        }
        tf = Transform(variable_map=variable_map)
        variables = ["old1", "x", "old2"]
        mapped = tf.map_variable_names("group1", variables)
        self.assertEqual(mapped, ["new1", "x", "new2"])

    def test_map_variable_names_inverse(self):
        """Test inverse mapping of variable names.

        Notes
        -----
        This test checks that map_variable_names correctly uses
        variable_map_inv when called with inverse=True.
        """
        variable_map = {
            "group1": {"old1": "new1", "old2": "new2"},
        }
        tf = Transform(variable_map=variable_map)
        variables = ["new1", "y", "new2"]
        mapped = tf.map_variable_names("group1", variables, inverse=True)
        self.assertEqual(mapped, ["old1", "y", "old2"])

    def test_map_variable_names_missing_group(self):
        """Test mapping of variable names when the group is missing.

        Notes
        -----
        This test verifies that if the group name is not in the mapping,
        the original variables list is returned unchanged.
        """
        variable_map = {
            "group1": {"old1": "new1"},
        }
        tf = Transform(variable_map=variable_map)
        variables = ["old1", "z"]
        mapped = tf.map_variable_names("other_group", variables)
        self.assertEqual(mapped, variables)

    def test_map_variable_names_leading_slash_name(self):
        """Test mapping of variable names with a leading slash in the group.

        Notes
        -----
        This test ensures that a leading slash in the group name is
        ignored when looking up mappings in variable_map.
        """
        variable_map = {
            "group1": {"old1": "new1"},
        }
        tf = Transform(variable_map=variable_map)
        variables = ["old1"]
        mapped = tf.map_variable_names("/group1", variables)
        self.assertEqual(mapped, ["new1"])


class TestTransformCallIntegration(unittest.TestCase):
    """Integration tests for the Transform call method."""

    def test_call_applies_ints_floats_and_variables(self):
        """Test that __call__ applies integer, float, and variable mappings.

        Notes
        -----
        This test checks that the combined application of map_ints,
        map_floats, and map_variables happens in the correct
        order when calling the Transform instance.
        """
        batch = make_sample_batch()
        variable_map = {
            "group1": {"var1": "new_var1", "var2": "new_var2"},
            "group2": {"var3": "new_var3"},
        }
        ints_map = {
            "group1": {"var2": {2: 20}},
            "group2": {"var3": {5: 50}},
        }
        floats_map = {
            "group3": {"var5": "log10"},
        }
        tf = Transform(variable_map=variable_map, ints_map=ints_map, floats_map=floats_map)
        transformed = tf(batch)

        self.assertIn("new_var2", transformed["group1"].dtype.names)
        self.assertIn("new_var3", transformed["group2"].dtype.names)
        self.assertNotIn("var3", transformed["group2"].dtype.names)

        self.assertEqual(transformed["group1"]["new_var2"].tolist(), [20, 4])
        self.assertEqual(transformed["group2"]["new_var3"].tolist(), [50, 7])

        expected_log = np.log10(np.array([10.0, 100.0]))
        self.assertTrue(np.allclose(transformed["group3"]["var5"], expected_log))