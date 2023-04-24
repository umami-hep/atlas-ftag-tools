import numpy as np

from ftag.cuts import Cut, Cuts, CutsResult


def test_Cut_value_property():
    c = Cut("x", "==", "1")
    assert c.value == 1


def test_Cut_call_method():
    c = Cut("x", "==", 1)
    array = np.array([1, 2, 3], dtype=[("x", int)])
    assert np.array_equal(c(array), np.array([True, False, False]))


def test_Cuts_from_list_method():
    cuts_list = [("x", "==", "1"), ("y", ">=", "2")]
    c = Cuts.from_list(cuts_list)
    assert len(c) == 2


def test_Cuts_variables_property():
    c = Cuts.from_list([("x", "==", "1"), ("y", ">=", "2"), ("x", "!=", "3")])
    assert c.variables == ["x", "y"]


def test_Cuts_ignore_method():
    c = Cuts.from_list([("x", "==", "1"), ("y", ">=", "2"), ("x", "!=", "3")])
    c_ignore = c.ignore(["y"])
    assert len(c_ignore) == 2
    assert "y" not in c_ignore.variables


def test_Cuts_call_method():
    c = Cuts.from_list([("x", "==", "1"), ("y", ">=", "2"), ("x", "!=", "3")])
    array = np.array([(1, 2), (2, 3), (3, 4), (1, 3)], dtype=[("x", int), ("y", int)])
    result = c(array)
    assert isinstance(result, CutsResult)
    assert np.array_equal(result.idx, np.array([0, 3]))
    assert np.array_equal(result.values, array[[0, 3]])


def test_Cuts_add_method():
    c1 = Cuts.from_list([("x", "==", "1"), ("y", ">=", "2")])
    c2 = Cuts.from_list([("x", "!=", "3")])
    c3 = c1 + c2
    assert len(c3) == 3


def test_Cuts_getitem_method():
    c = Cuts.from_list([("x", "==", "1"), ("y", ">=", "2"), ("x", "!=", "3")])
    c_x = c["x"]
    assert len(c_x) == 2
    assert all(c.variable == "x" for c in c_x)
