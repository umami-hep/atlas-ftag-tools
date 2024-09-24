from __future__ import annotations

import numpy as np
import pytest

from ftag import Flavour, Flavours
from ftag.labeller import Labeller
from ftag.mock import mock_jets


# make a pytest fixture to return jets using mock data
@pytest.fixture
def jets():
    return mock_jets(1000)


def test_initialization():
    flavours = Flavours.by_category("single-btag")
    labeller = Labeller(flavours)
    assert len(labeller.labels) == 4
    assert all(isinstance(f, Flavour) for f in labeller.labels)

    flavours = ["bjets", "cjets"]
    labeller = Labeller(flavours)
    assert len(labeller.labels) == 2
    assert all(isinstance(f, Flavour) for f in labeller.labels)

    with pytest.raises(KeyError, match="Flavour 'nonexistent' not found"):
        Labeller(["nonexistent"])


def test_get_labels_valid_input(jets):
    flavours = ["bjets", "cjets", "ujets", "taujets"]
    labeller = Labeller(flavours)
    labels = labeller.get_labels(jets)

    expected = np.zeros(len(jets), dtype=int)
    expected[jets["HadronConeExclTruthLabelID"] == 5] = 0
    expected[jets["HadronConeExclTruthLabelID"] == 4] = 1
    expected[jets["HadronConeExclTruthLabelID"] == 15] = 2
    expected[jets["HadronConeExclTruthLabelID"] == 0] = 3

    assert np.array_equal(labels, expected)


def test_get_labels_unlabelled_objects(jets):
    flavours = ["bjets"]
    labeller = Labeller(flavours, require_labels=True)
    with pytest.raises(ValueError, match="Some objects were not labelled"):
        labeller.get_labels(jets)

    labeller = Labeller(flavours, require_labels=False)
    labels = labeller.get_labels(jets)
    sel_jets = jets[jets["HadronConeExclTruthLabelID"] == 5]
    assert len(labels) == len(sel_jets)
    assert np.all(labels == 0)


def test_add_labels(jets):
    flavours = ["bjets", "cjets", "ujets", "taujets"]
    labeller = Labeller(flavours)
    jets = labeller.add_labels(jets)
    labels = labeller.get_labels(jets)
    assert np.array_equal(jets["labels"], labels)

    labeller = Labeller(flavours, require_labels=False)
    with pytest.raises(ValueError, match="Cannot add labels if require_labels is set to False"):
        labeller.add_labels(jets)
