from __future__ import annotations

import pytest

from ftag.cuts import Cuts
from ftag.flavour import (
    Flavour,
    FlavourContainer,
    Flavours,
    remove_suffix,
)


def test_flavour_attributes():
    flavour = Flavour(
        name="test",
        label="test_label",
        cuts=[(1, 1), (2, 2)],
        colour="test_colour",
        category="test_category",
    )
    assert flavour.px == "ptest"
    assert flavour.eff_str == "test_label efficiency"
    assert flavour.rej_str == "test_label rejection"
    assert flavour.frac_str == "ftest"
    assert str(flavour) == "test"


def test_Flavours_iteration():
    for flavour in Flavours:
        assert isinstance(flavour, Flavour)


def test_Flavours_get_item():
    flavour = Flavours["bjets"]
    assert isinstance(flavour, Flavour)
    flavour3 = Flavours[Flavours.bjets]
    assert isinstance(flavour3, Flavour)
    assert flavour == flavour3


def test_Flavours_get_attr():
    flavour = Flavours.bjets
    assert isinstance(flavour, Flavour)


def test_Flavours_contains():
    assert "bjets" in Flavours
    assert Flavours.bjets in Flavours
    assert "undefined" not in Flavours


def test_Flavours_equals():
    assert Flavours == FlavourContainer.from_list(Flavours)
    assert Flavours == [f.name for f in Flavours]
    assert Flavours != 1


def test_Flavours_categories():
    target = [
        "single-btag",
        "single-btag-extended",
        "xbb",
        "xbb-extended",
        "partonic",
        "lepton-decay",
        "PDGID",
        "isolation",
    ]
    assert Flavours.categories == target


def test_Flavours_by_category():
    charm_flavours = Flavours.by_category("charm")
    for flavour in charm_flavours:
        assert flavour.category == "charm"


def test_Flavours_from_cuts():
    cuts = Cuts.from_list(["HadronConeExclTruthLabelID == 5"])
    flavour = Flavours.from_cuts(cuts)
    assert isinstance(flavour, Flavour)
    assert flavour.name == "bjets"
    with pytest.raises(KeyError):
        Flavours.from_cuts(["dummp == -1"])


def test_remove_suffix():
    assert remove_suffix("test_jets", "jets") == "test_"
    assert remove_suffix("test_jets_test", "jets") == "test_jets_test"


def test_backgrounds():
    bjet_backgrounds = Flavours.backgrounds(Flavours.bjets)
    print(bjet_backgrounds)
    assert bjet_backgrounds == FlavourContainer.from_list([
        Flavours.cjets,
        Flavours.ujets,
        Flavours.taujets,
    ])

    cjet_backgrounds = Flavours.backgrounds(Flavours.cjets)
    assert cjet_backgrounds == FlavourContainer.from_list([
        Flavours.bjets,
        Flavours.ujets,
        Flavours.taujets,
    ])

    bjet_background_no_light = Flavours.backgrounds(Flavours.bjets, keep_possible_signals=False)
    assert bjet_background_no_light == FlavourContainer.from_list([
        Flavours.cjets,
        Flavours.taujets,
    ])
