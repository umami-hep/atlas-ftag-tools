from __future__ import annotations

import numpy as np
import pytest

from ftag.cuts import OPERATORS, Cut, Cuts, CutsResult

# ---------------------------------------------------------------------------
# Tests for Cut
# ---------------------------------------------------------------------------


def test_Cut_value_property_basic():
    """Literal string should be converted to int by value property."""
    c = Cut("x", "==", "1")
    assert c.value == 1


def test_Cut_value_property_special_strings():
    """Special string literals 'nan', '+inf', '-inf' must map to float equivalents."""
    c_nan = Cut("x", "==", "nan")
    c_pinf = Cut("x", "==", "inf")
    c_minf = Cut("x", "==", "-inf")
    assert np.isnan(c_nan.value)
    assert c_pinf.value == float("inf")
    assert c_minf.value == float("-inf")


@pytest.mark.parametrize(
    ("operator_str", "expected"),
    [
        ("==", np.array([False, True, False])),
        ("!=", np.array([True, False, True])),
        (">", np.array([False, False, True])),
        ("<", np.array([True, False, False])),
    ],
)
def test_Cut_call_method_comparison_ops(operator_str, expected):
    """Cut.__call__ should apply standard comparison operators correctly."""
    c = Cut("x", operator_str, 2)
    array = np.array([1, 2, 3], dtype=[("x", int)])
    np.testing.assert_array_equal(c(array), expected)


def test_Cut_modulo_operator():
    """Modulo based operators such as '%2==' must work and be auto-registered."""
    op_key = "%2=="
    assert op_key in OPERATORS
    c = Cut("x", op_key, 0)
    array = np.array([0, 1, 2, 3, 4], dtype=[("x", int)])
    np.testing.assert_array_equal(
        c(array),
        np.array([True, False, True, False, True]),
    )


def test_Cut_str_method():
    """__str__ should yield human-readable 'variable operator value'."""
    c = Cut("pt", ">=", 30)
    assert str(c) == "pt >= 30"


# ---------------------------------------------------------------------------
# Tests for Cuts
# ---------------------------------------------------------------------------


def test_Cuts_from_list_method():
    """from_list should accept list of tuples and deduplicate."""
    cuts_list = [("x", "==", "1"), ("y", ">=", "2"), ("x", "==", "1")]
    c = Cuts.from_list(cuts_list)
    assert len(c) == 2
    assert c.variables == ["x", "y"]


def test_Cuts_from_list_string_input():
    """from_list must also parse list of space-separated strings."""
    cuts_list = ["x == 1", "y >= 2"]
    c = Cuts.from_list(cuts_list)
    assert len(c) == 2
    # Ensure original ordering is preserved
    assert c.variables[0] == "x"
    assert c.variables[1] == "y"


def test_Cuts_empty_factory():
    """Cuts.empty should return a Cuts instance with length zero."""
    empty = Cuts.empty()
    assert isinstance(empty, Cuts)
    assert len(empty) == 0
    assert empty.variables == []


def test_Cuts_variables_property():
    """Variables property should list unique variables in insertion order."""
    c = Cuts.from_list(
        [("x", "==", "1"), ("y", ">=", "2"), ("x", "!=", "3")],
    )
    assert c.variables == ["x", "y"]


def test_Cuts_ignore_method():
    """Ignore must drop specified variables but keep others intact."""
    c = Cuts.from_list(
        [("x", "==", "1"), ("y", ">=", "2"), ("x", "!=", "3")],
    )
    c_ignore = c.ignore(["y"])
    assert len(c_ignore) == 2
    assert "y" not in c_ignore.variables
    assert c_ignore.variables == ["x"]


def test_Cuts_call_method():
    """Calling a Cuts instance should return correct index mask and values."""
    c = Cuts.from_list(
        [("x", "==", "1"), ("y", ">=", "2"), ("x", "!=", "3")],
    )
    array = np.array(
        [(1, 2), (2, 3), (3, 4), (1, 3)],
        dtype=[("x", int), ("y", int)],
    )
    result = c(array)
    assert isinstance(result, CutsResult)
    np.testing.assert_array_equal(result.idx, np.array([0, 3]))
    np.testing.assert_array_equal(result.values, array[[0, 3]])


def test_Cuts_call_method_nan_handling():
    """Cuts containing a nan comparison should filter using np.isnan."""
    c = Cuts.from_list([("x", "==", "nan")])
    array = np.array([np.nan, 1.0, np.nan], dtype=[("x", float)])
    result = c(array)
    np.testing.assert_array_equal(result.idx, np.array([0, 2]))


def test_Cuts_add_method():
    """Addition of two Cuts should combine and deduplicate in order."""
    c1 = Cuts.from_list([("x", "==", "1"), ("y", ">=", "2")])
    c2 = Cuts.from_list([("x", "!=", "3"), ("y", ">=", "2")])
    c3 = c1 + c2
    assert len(c3) == 3
    assert c3.variables == ["x", "y"]


def test_Cuts_len_iter():
    """Len and iter must reflect number of stored Cut instances."""
    c = Cuts.from_list([("a", "==", "0"), ("b", "==", "1")])
    assert len(c) == 2
    first_cut = next(iter(c))
    assert first_cut.variable == "a"


def test_Cuts_getitem_method():
    """Indexing by variable name should return a new Cuts with selected cuts."""
    c = Cuts.from_list(
        [("x", "==", "1"), ("y", ">=", "2"), ("x", "!=", "3")],
    )
    c_x = c["x"]
    assert len(c_x) == 2
    assert all(cut.variable == "x" for cut in c_x)


def test_Cuts_getitem_missing_variable():
    """Requesting a variable not present should yield an empty Cuts."""
    c = Cuts.from_list([("x", "==", "1")])
    c_y = c["y"]
    assert isinstance(c_y, Cuts)
    assert len(c_y) == 0


def test_Cuts_ndim_error():
    """Passing a 2-D array to Cuts.__call__ must raise ValueError."""
    c = Cuts.from_list([("x", "==", "1")])
    array = np.ones((2, 2))
    with pytest.raises(ValueError, match="only supports jet selections"):
        c(array)


def test_Cut_call_nan_not_equal():
    """Operator '!=' on a nan value should invert the np.isnan result."""
    c = Cut("x", "!=", "nan")
    array = np.array([np.nan, 1.0, np.nan], dtype=[("x", float)])
    expected = ~np.isnan(array["x"])
    np.testing.assert_array_equal(c(array), expected)


def test_Cut_call_nan_invalid_operator():
    """Operators other than '==' or '!=' with nan must raise ValueError."""
    c = Cut("x", "<", "nan")
    array = np.array([0.0, np.nan], dtype=[("x", float)])
    with pytest.raises(ValueError, match="nan.*only makes sense"):
        c(array)
