from __future__ import annotations

import unittest

import numpy as np

from ftag import Flavours
from ftag.fraction_optimization import (
    calculate_best_fraction_values,
    calculate_rejection_sum,
    convert_dict,
    get_bkg_norm_dict,
)
from ftag.hdf5 import H5Reader
from ftag.mock import get_mock_file

# Generate a mock file for reading jets
TTBAR_FILE = get_mock_file(100_000)[0]


class TestConvertDict(unittest.TestCase):
    """Tests for the convert_dict function."""

    def setUp(self) -> None:
        """Setup test array and dict."""
        self.backgrounds = Flavours.by_category("single-btag").backgrounds(Flavours["bjets"])
        self.array = np.array([0.3, 0.4, 0.3])
        self.dict = {
            "fc": 0.3,
            "fu": 0.4,
            "ftau": 0.3,
        }

    def test_array_to_dict(self):
        """Test array to dict conversion."""
        out = convert_dict(fraction_values=self.array, backgrounds=self.backgrounds)
        self.assertEqual(out, self.dict)

    def test_dict_to_array(self):
        """Test dict to array conversion."""
        out = convert_dict(fraction_values=self.dict, backgrounds=self.backgrounds)
        np.testing.assert_array_equal(out, self.array)

    def test_wrong_input_type_value_error(self):
        """Test raising of ValueError if wrong type is given."""
        with self.assertRaises(TypeError) as ctx:
            convert_dict(fraction_values="Error", backgrounds=self.backgrounds)
        self.assertEqual(
            "Only input of type `dict` or `np.ndarray` are accepted! You gave <class 'str'>",
            str(ctx.exception),
        )


class TestGetBkgNormDict(unittest.TestCase):
    """
    Tests for the get_bkg_norm_dict function, which calculates
    a background rejection 'normalization' dictionary for each background flavor.
    """

    def setUp(self) -> None:
        """Load mock jets data once for all tests in this class."""
        self.jets = H5Reader(fname=TTBAR_FILE).load()["jets"]
        self.signal = Flavours["bjets"]
        self.flavours = Flavours.by_category("single-btag")
        self.tagger = "MockTagger"
        self.working_point = 0.70

    def test_get_bkg_norm_dict(self):
        """
        Ensure get_bkg_norm_dict returns a dict with one entry per background
        and that each entry is a finite float.
        """
        bkg_norm = get_bkg_norm_dict(
            jets=self.jets,
            tagger=self.tagger,
            signal=self.signal,
            flavours=self.flavours,
            working_point=self.working_point,
        )
        backgrounds = self.flavours.backgrounds(self.signal)

        # Check that the dict has one entry per background
        self.assertEqual(len(bkg_norm), len(backgrounds))

        # Each key is background.name, each value is a finite float
        for bkg in backgrounds:
            self.assertIn(bkg.name, bkg_norm, f"Missing background: {bkg.name}")
            val = bkg_norm[bkg.name]
            self.assertIsInstance(val, float, "bkg_norm_dict value should be float")
            self.assertTrue(np.isfinite(val), f"Non-finite value for {bkg.name}: {val}")


class TestCalculateRejectionSum(unittest.TestCase):
    """
    Tests for the calculate_rejection_sum function, which computes
    the negative sum of rejections (so that we can minimize it).
    """

    def setUp(self) -> None:
        """Load mock jets data once for all tests in this class."""
        self.jets = H5Reader(fname=TTBAR_FILE).load()["jets"]
        self.signal = Flavours["bjets"]
        self.flavours = Flavours.by_category("single-btag")
        self.tagger = "MockTagger"
        self.working_point = 0.70

        # Precompute background normalization
        self.bkg_norm = get_bkg_norm_dict(
            jets=self.jets,
            tagger=self.tagger,
            signal=self.signal,
            flavours=self.flavours,
            working_point=self.working_point,
        )
        # Uniform rejection weights
        backgrounds = self.flavours.backgrounds(self.signal)
        self.rejection_weights = {bkg.name: 1.0 for bkg in backgrounds}

    def test_calculate_rejection_sum_with_dict(self):
        """Test passing a fraction dictionary, verifying the output is a finite negative float."""
        backgrounds = self.flavours.backgrounds(self.signal)
        frac_dict = {bkg.frac_str: 1.0 / len(backgrounds) for bkg in backgrounds}

        val = calculate_rejection_sum(
            fraction_dict=frac_dict,
            jets=self.jets,
            tagger=self.tagger,
            signal=self.signal,
            flavours=self.flavours,
            working_point=self.working_point,
            bkg_norm_dict=self.bkg_norm,
            rejection_weights=self.rejection_weights,
        )
        # Should be a finite negative float
        self.assertIsInstance(val, float)
        self.assertTrue(np.isfinite(val), f"Got non-finite value: {val}")
        self.assertLess(val, 0.0, "Expected negative sum for minimization")

    def test_calculate_rejection_sum_with_array(self):
        """Test passing a fraction array, verifying the output is a finite negative float."""
        backgrounds = self.flavours.backgrounds(self.signal)
        arr_in = np.array([1.0 / len(backgrounds)] * len(backgrounds), dtype=float)

        val = calculate_rejection_sum(
            fraction_dict=arr_in,
            jets=self.jets,
            tagger=self.tagger,
            signal=self.signal,
            flavours=self.flavours,
            working_point=self.working_point,
            bkg_norm_dict=self.bkg_norm,
            rejection_weights=self.rejection_weights,
        )
        self.assertIsInstance(val, float)
        self.assertTrue(np.isfinite(val), f"Got non-finite value: {val}")
        self.assertLess(val, 0.0, "Expected negative sum for minimization")


class TestCalculateBestFractionValues(unittest.TestCase):
    """
    Tests for calculate_best_fraction_values, which performs an end-to-end
    optimization to find the best fraction dictionary.
    """

    def setUp(self) -> None:
        """Load mock jets data once for all tests in this class."""
        self.jets = H5Reader(fname=TTBAR_FILE).load()["jets"]
        self.flavours = Flavours.by_category("single-btag")
        self.tagger = "MockTagger"
        self.working_point = 0.70

    def test_calculate_best_fraction_values(self):
        """
        Run the optimizer and ensure the returned fractions form a dict
        whose values sum to ~1 and lie within [0,1].
        """
        final_dict = calculate_best_fraction_values(
            jets=self.jets,
            tagger=self.tagger,
            signal="bjets",
            flavours=self.flavours,
            working_point=self.working_point,
            rejection_weights=None,  # default = 1
            optimizer_method="Powell",
        )
        self.assertIsInstance(final_dict, dict, "Expected a dict of final fractions")

        total = sum(final_dict.values())
        self.assertAlmostEqual(total, 1.0, places=2, msg=f"Fractions sum to {total}, expected ~1.0")

        for frac_key, val in final_dict.items():
            self.assertGreaterEqual(val, 0.0, f"Fraction {frac_key} < 0.0")
            self.assertLessEqual(val, 1.0, f"Fraction {frac_key} > 1.0")
