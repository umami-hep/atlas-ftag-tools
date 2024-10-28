from __future__ import annotations

import numpy as np
import pytest

from ftag import Flavours
from ftag.labeller import Labeller
from ftag.labels import Label
from ftag.mock import mock_jets


# make a pytest fixture to return jets using mock data
@pytest.fixture
def jets():
    return mock_jets(1000)


def test_initialization():
    labels = Flavours.by_category("single-btag")
    labeller = Labeller(labels)
    assert len(labeller.labels) == 4
    assert all(isinstance(f, Label) for f in labeller.labels)

    labels = ["bjets", "cjets"]
    labeller = Labeller(labels)
    assert len(labeller.labels) == 2
    assert all(isinstance(f, Label) for f in labeller.labels)

    with pytest.raises(KeyError, match="Label 'nonexistent' not found"):
        Labeller(["nonexistent"])


def test_get_labels_valid_input(jets):
    labels = ["bjets", "cjets", "ujets", "taujets"]
    labeller = Labeller(labels)
    labels = labeller.get_labels(jets)

    expected = np.zeros(len(jets), dtype=int)
    expected[jets["HadronConeExclTruthLabelID"] == 5] = 0
    expected[jets["HadronConeExclTruthLabelID"] == 4] = 1
    expected[jets["HadronConeExclTruthLabelID"] == 15] = 2
    expected[jets["HadronConeExclTruthLabelID"] == 0] = 3

    assert np.array_equal(labels, expected)


def test_get_labels_unlabelled_objects(jets):
    labels = ["bjets"]
    labeller = Labeller(labels, require_labels=True)
    with pytest.raises(ValueError, match="Some objects were not labelled"):
        labeller.get_labels(jets)

    labeller = Labeller(labels, require_labels=False)
    ys = labeller.get_labels(jets)
    sel_jets = jets[jets["HadronConeExclTruthLabelID"] == 5]
    assert len(ys) == len(sel_jets)
    assert np.all(ys == 0)


def test_add_labels(jets):
    labels = ["bjets", "cjets", "ujets", "taujets"]
    labeller = Labeller(labels)
    jets = labeller.add_labels(jets)
    ys = labeller.get_labels(jets)
    assert np.array_equal(jets["labels"], ys)

    labeller = Labeller(labels, require_labels=False)
    with pytest.raises(ValueError, match="Cannot add labels if require_labels is set to False"):
        labeller.add_labels(jets)


def test_labeller_property():
    flavours = ["bjets", "cjets"]
    labeller = Labeller(flavours)
    label_vars = labeller.variables
    assert label_vars == ["HadronConeExclTruthLabelID", "HadronConeExclTruthLabelID"]

    flavours = ["qcdbb"]
    labeller = Labeller(flavours)
    label_vars = labeller.variables
    assert label_vars == ["R10TruthLabel_R22v1", "GhostBHadronsFinalCount"]
