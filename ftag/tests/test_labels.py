from __future__ import annotations

import pytest

from ftag import Flavours
from ftag.cuts import Cuts
from ftag.labels import Label, LabelContainer, remove_suffix


def test_label_attributes():
    label = Label(
        name="test",
        label="test_label",
        cuts=[(1, 1), (2, 2)],
        colour="test_colour",
        category="test_category",
    )
    assert label.px == "ptest"
    assert label.eff_str == "test_label efficiency"
    assert label.rej_str == "test_label rejection"
    assert label.frac_str == "ftest"
    assert str(label) == "test"

    label = Label(
        name="test",
        label="test_label",
        cuts=[(1, 1), (2, 2)],
        colour="test_colour",
        category="test_category",
        _px="ptestdefined",
    )
    assert label.px == "ptestdefined"


def test_Flavours_iteration():
    for label in Flavours:
        assert isinstance(label, Label)


def test_Flavours_get_item():
    label = Flavours["bjets"]
    assert isinstance(label, Label)
    label3 = Flavours[Flavours.bjets]
    assert isinstance(label3, Label)
    assert label == label3


def test_Flavours_get_attr():
    label = Flavours.bjets
    assert isinstance(label, Label)


def test_Flavours_contains():
    assert "bjets" in Flavours
    assert Flavours.bjets in Flavours
    assert "undefined" not in Flavours


def test_Flavours_equals():
    assert Flavours == LabelContainer.from_list(Flavours)
    assert Flavours == [f.name for f in Flavours]
    assert Flavours != 1


def test_Flavours_categories():
    target = [
        "single-btag",
        "single-btag-extended",
        "single-btag-ghost",
        "xbb",
        "xbb-extended",
        "partonic",
        "lepton-decay",
        "PDGID",
        "isolation",
    ]
    assert Flavours.categories == target


def test_Flavours_by_category():
    for label in Flavours.by_category("single-btag"):
        assert label.category == "single-btag"


def test_Flavours_from_cuts():
    cuts = Cuts.from_list(["HadronConeExclTruthLabelID == 5"])
    label = Flavours.from_cuts(cuts)
    assert isinstance(label, Label)
    assert label.name == "bjets"
    with pytest.raises(KeyError):
        Flavours.from_cuts(["dummp == -1"])


def test_remove_suffix():
    assert remove_suffix("test_jets", "jets") == "test_"
    assert remove_suffix("test_jets_test", "jets") == "test_jets_test"


def test_backgrounds():
    bjet_backgrounds = Flavours.backgrounds(Flavours.bjets)
    print(bjet_backgrounds)
    assert bjet_backgrounds == LabelContainer.from_list([
        Flavours.cjets,
        Flavours.ujets,
        Flavours.taujets,
    ])

    cjet_backgrounds = Flavours.backgrounds(Flavours.cjets)
    assert cjet_backgrounds == LabelContainer.from_list([
        Flavours.bjets,
        Flavours.ujets,
        Flavours.taujets,
    ])

    bjet_background_no_light = Flavours.backgrounds(Flavours.bjets, only_signals=False)
    assert bjet_background_no_light == LabelContainer.from_list([
        Flavours.cjets,
        Flavours.taujets,
    ])
