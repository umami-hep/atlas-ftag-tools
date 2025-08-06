from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np

from ftag.cuts import Cuts
from ftag.labels import Label, LabelContainer


class TestLabel(unittest.TestCase):
    """Tests for the Label dataclass."""

    def setUp(self):
        """Create a Label instance for testing."""
        self.label = Label(
            name="bjets",
            label="$b$-jets",
            cuts=Cuts.from_list(["HadronConeExclTruthLabelID == 5"]),
            colour="tab:blue",
            category="single-btag",
        )

    def test_px_property(self):
        """Test that the px property correctly infers p{flavour}."""
        self.assertEqual(self.label.px, "pb")

    def test_eff_str(self):
        """Test that the eff_str property replaces 'jets' with 'jet' and appends ' efficiency'."""
        self.assertEqual(self.label.eff_str, "$b$-jet efficiency")

    def test_rej_str(self):
        """Test that the rej_str property replaces 'jets' with 'jet' and appends ' rejection'."""
        self.assertEqual(self.label.rej_str, "$b$-jet rejection")

    def test_frac_str(self):
        """Test that frac_str prepends 'f' to the flavour name without 'jets'."""
        self.assertEqual(self.label.frac_str, "fb")

    def test_str(self):
        """Test that str(label) returns the label's name."""
        self.assertEqual(str(self.label), "bjets")

    def test_lt(self):
        """
        Test that __lt__ orders by name. 'bjets' < 'cjets' => True,
        so cjets should be considered greater than bjets.
        """
        label2 = Label(
            name="cjets",
            label="$c$-jets",
            cuts=Cuts.from_list(["HadronConeExclTruthLabelID == 4"]),
            colour="tab:orange",
            category="single-btag",
        )
        self.assertTrue(label2 > self.label)


class TestLabelContainer(unittest.TestCase):
    """Tests for the LabelContainer class."""

    def setUp(self):
        """Set up a LabelContainer with three labels (bjets, cjets, ujets)."""
        self.bjets = Label(
            name="bjets",
            label="$b$-jets",
            cuts=Cuts.from_list(["HadronConeExclTruthLabelID == 5"]),
            colour="tab:blue",
            category="single-btag",
        )
        self.cjets = Label(
            name="cjets",
            label="$c$-jets",
            cuts=Cuts.from_list(["HadronConeExclTruthLabelID == 4"]),
            colour="tab:orange",
            category="single-btag",
        )
        self.ujets = Label(
            name="ujets",
            label="Light-jets",
            cuts=Cuts.from_list(["HadronConeExclTruthLabelID == 0"]),
            colour="tab:green",
            category="single-btag",
        )
        self.container = LabelContainer.from_list([self.bjets, self.cjets, self.ujets])

    def test_len(self):
        """Test that len(container) returns the correct number of labels."""
        self.assertEqual(len(self.container), 3)

    def test_get_item(self):
        """Test retrieving labels by string name or Label object."""
        self.assertEqual(self.container["bjets"], self.bjets)
        self.assertEqual(self.container[self.bjets], self.bjets)
        with self.assertRaises(KeyError):
            _ = self.container["non_existent"]

    def test_iter(self):
        """Test that iter(container) yields Label objects."""
        labels = list(self.container)
        self.assertEqual(len(labels), 3)
        self.assertTrue(all(isinstance(iter_label, Label) for iter_label in labels))

    def test_contains(self):
        """Test that 'in' checks membership by name or Label object."""
        self.assertIn("bjets", self.container)
        self.assertIn(self.bjets, self.container)
        self.assertNotIn("non_existent", self.container)

    def test_categories(self):
        """Test that categories returns the list of distinct categories."""
        expected = ["single-btag"]
        np.testing.assert_array_equal(self.container.categories, expected)

    def test_by_category(self):
        """Test that by_category returns a container of labels filtered by the given category."""
        cat_container = self.container.by_category("single-btag")
        self.assertEqual(len(cat_container), 3)
        with self.assertRaises(KeyError):
            _ = self.container.by_category("nonexistent-category")

    def test_from_cuts(self):
        """Test that from_cuts finds the correct label or raises KeyError if not found."""
        matching_cuts = Cuts.from_list(["HadronConeExclTruthLabelID == 5"])
        self.assertEqual(self.container.from_cuts(matching_cuts), self.bjets)

        # Use an expression that does not match anything in the container
        with self.assertRaises(KeyError):
            _ = self.container.from_cuts(["HadronConeExclTruthLabelID == 999"])

    def test_eq(self):
        """
        Test that containers compare equal if contents match,
        or to a list of names if they match.
        """
        container2 = LabelContainer.from_list([self.bjets, self.cjets, self.ujets])
        self.assertTrue(self.container == container2)
        self.assertTrue(self.container == ["bjets", "cjets", "ujets"])
        self.assertFalse(self.container == ["bjets", "cjets"])
        self.assertFalse(self.container == 123)

    def test_backgrounds(self):
        """
        Test that backgrounds returns labels with the same category and excludes the given signal.
        If only_signals=False, exclude 'ujets' or 'qcd'.
        """
        bkg = self.container.backgrounds(self.cjets, only_signals=True)
        self.assertIn(self.bjets, bkg)
        self.assertIn(self.ujets, bkg)
        bkg_no_signal = self.container.backgrounds(self.cjets, only_signals=False)
        self.assertIn(self.bjets, bkg_no_signal)
        self.assertNotIn(self.ujets, bkg_no_signal)
        single_cjets = LabelContainer.from_list([self.cjets])
        with self.assertRaises(TypeError):
            single_cjets.backgrounds(self.cjets)

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="""
- name: bjets
  label: "$b$-jets"
  cuts: ["HadronConeExclTruthLabelID == 5"]
  colour: "tab:blue"
  category: "single-btag"
- name: cjets
  label: "$c$-jets"
  cuts: ["HadronConeExclTruthLabelID == 4"]
  colour: "tab:orange"
  category: "single-btag"
""",
    )
    def test_from_yaml(self, _mock_file):  # noqa: PT019
        """Test that from_yaml reads and parses the YAML data correctly."""
        container = LabelContainer.from_yaml(Path("dummy.yaml"))
        self.assertEqual(len(container), 2)
        self.assertIn("bjets", container)
        self.assertIn("cjets", container)

    def test_from_yaml_exclude(self):
        """Test the loading from yaml with exluding categories."""
        container = LabelContainer.from_yaml(
            exclude_categories=[
                "single-btag-extended",
                "single-btag-extended-ghost",
            ]
        )
        with self.assertRaises(KeyError):
            _ = container.by_category("single-btag-extended")

        self.assertEqual(
            container.by_category("single-btag"),
            LabelContainer.from_list([
                container.bjets,
                container.cjets,
                container.ujets,
                container.taujets,
            ]),
        )

    def test_from_yaml_include(self):
        """Test the loading from yaml with including categories."""
        container = LabelContainer.from_yaml(
            include_categories=[
                "single-btag",
                "single-btag-ghost",
            ]
        )
        with self.assertRaises(KeyError):
            _ = container.by_category("single-btag-extended-ghost")

        self.assertEqual(
            container.by_category("single-btag"),
            LabelContainer.from_list([
                container.bjets,
                container.cjets,
                container.ujets,
                container.taujets,
            ]),
        )

    def test_from_yaml_KeyError(self):
        """Test the KeyError if no class survived the selection."""
        with self.assertRaises(KeyError):
            LabelContainer.from_yaml(include_categories=[])
