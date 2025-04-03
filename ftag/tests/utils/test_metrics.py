"""Unit test script for the functions in metrics.py."""

from __future__ import annotations

import unittest

import numpy as np

from ftag import Flavours
from ftag.utils.logging import logger, set_log_level
from ftag.utils.metrics import (
    calculate_efficiency,
    calculate_efficiency_error,
    calculate_rejection,
    calculate_rejection_error,
    get_discriminant,
    save_divide,
)

set_log_level(logger, "DEBUG")


class SaveDivideTestCase(unittest.TestCase):
    """Test class for the save_divide function."""

    def test_divide_float_no_zero(self):
        """Test dividing two floats with no zero denominator."""
        result = save_divide(10.0, 2.0)
        self.assertEqual(result, 5.0, "Expected 10.0 / 2.0 to be 5.0")

    def test_divide_float_with_zero_denominator(self):
        """Test dividing two floats where denominator is zero."""
        # The default is 1.0
        result = save_divide(10.0, 0.0)
        self.assertEqual(result, 1.0, "Expected default value for zero denominator")

        # Custom default
        result_custom_default = save_divide(10.0, 0.0, default=999.0)
        self.assertEqual(
            result_custom_default, 999.0, "Expected custom default value for zero denominator"
        )

    def test_divide_array_no_zero(self):
        """Test dividing two arrays with no zero denominator."""
        numerator = np.array([2, 4, 6])
        denominator = np.array([1, 2, 3])
        expected = np.array([2.0, 2.0, 2.0])
        result = save_divide(numerator, denominator)
        np.testing.assert_array_equal(result, expected, "Expected element-wise division result")

    def test_divide_array_with_zero(self):
        """Test dividing where some elements of the denominator are zero."""
        numerator = np.array([10, 20, 0])
        denominator = np.array([2, 0, 5])
        expected = np.array([5.0, 1.0, 0.0])
        result = save_divide(numerator, denominator)
        np.testing.assert_array_almost_equal(
            result, expected, err_msg="Unexpected element-wise division for zero denominators"
        )

    def test_divide_float_array(self):
        """Test dividing a float by an array."""
        numerator = 10.0
        denominator = np.array([2, 0, 5])
        expected = np.array([5.0, 1.0, 2.0])
        result = save_divide(numerator, denominator)
        np.testing.assert_array_almost_equal(
            result, expected, err_msg="Unexpected division result for float / array"
        )

    def test_divide_array_float(self):
        """Test dividing an array by a float."""
        numerator = np.array([10, 20, 0])
        denominator = 5.0
        expected = np.array([2.0, 4.0, 0.0])
        result = save_divide(numerator, denominator)
        np.testing.assert_array_almost_equal(
            result, expected, err_msg="Unexpected division result for array / float"
        )

    def test_output_shape_scalar(self):
        """Test that output is a scalar float when both inputs are scalar."""
        result = save_divide(5, 10)
        self.assertIsInstance(result, float, "Expected output to be a float")
        self.assertAlmostEqual(result, 0.5, msg="Expected 0.5 for 5/10")

    def test_output_shape_array(self):
        """Test that output is an array when either of the inputs is an array."""
        numerator = np.array([1, 2, 3])
        denominator = np.array([2, 4, 6])
        result = save_divide(numerator, denominator)
        self.assertIsInstance(result, np.ndarray, "Expected output to be a numpy array")


class CalcEffTestCase(unittest.TestCase):
    """Test class for the calculate_efficiency function."""

    def setUp(self):
        rng = np.random.default_rng(seed=42)
        self.disc_sig = rng.normal(loc=3, size=100_000)
        self.disc_bkg = rng.normal(loc=0, size=100_000)

    def test_float_target(self):
        """Test efficiency and cut value calculation for one target value."""
        # we target a signal efficiency of 0.841345, which is the integral of a gaussian
        # from μ-1o to infinity
        # -->   the cut_value should be at 2, since the signal is a normal distribution
        #       with mean 3
        # https://www.wolframalpha.com/input?i=integrate+1%2Fsqrt%282+pi%29+*+exp%28-0.5*%28x-3%29**2%29+from+2+to+oo
        # -->   For the bkg efficiency this means that we integrate a normal distr.
        #       from μ+2o to infinity --> expect a value of 0.0227501
        # https://www.wolframalpha.com/input?i=integrate+1%2Fsqrt%282+pi%29+*+exp%28-0.5*x**2%29+from+2+to+oo
        bkg_eff, cut = calculate_efficiency(
            self.disc_sig,
            self.disc_bkg,
            target_eff=0.841345,
            return_cuts=True,
        )
        # the values here differ slightly from the values of the analytical integral,
        # since we use random numbers
        self.assertAlmostEqual(cut, 1.9956997)
        self.assertAlmostEqual(bkg_eff, 0.02367)

        # Test without returned cut values
        bkg_eff = calculate_efficiency(
            self.disc_sig,
            self.disc_bkg,
            target_eff=0.841345,
            return_cuts=False,
        )
        self.assertAlmostEqual(bkg_eff, 0.02367)

    def test_array_target(self):
        """Test efficiency and cut value calculation for list of target efficiencies."""
        # explanation is the same as above, now also cut the signal in the middle
        # --> target sig.efficiency 0.841345 and 0.5 --> cut at 2 and 3
        bkg_eff, cut = calculate_efficiency(
            self.disc_sig,
            self.disc_bkg,
            target_eff=[0.841345, 0.5],
            return_cuts=True,
        )
        # the values here differ slightly from the values of the analytical integral,
        # since we use random numbers
        np.testing.assert_array_almost_equal(cut, np.array([1.9956997, 2.990996]))
        np.testing.assert_array_almost_equal(bkg_eff, np.array([0.02367, 0.00144]))


class CalcRejTestCase(unittest.TestCase):
    """Test class for the calculate_rejection function."""

    def setUp(self):
        rng = np.random.default_rng(seed=42)
        self.disc_sig = rng.normal(loc=3, size=100_000)
        self.disc_bkg = rng.normal(loc=0, size=100_000)

    def test_float_target(self):
        """Test efficiency and cut value calculation for one target value."""
        # Same as for eff but this time for rejection, just use rej = 1 / eff
        bkg_rej, cut = calculate_rejection(
            self.disc_sig,
            self.disc_bkg,
            target_eff=0.841345,
            return_cuts=True,
        )
        self.assertAlmostEqual(cut, 1.9956997)
        self.assertAlmostEqual(bkg_rej, 1 / 0.02367)

        # Test without returned cut values
        bkg_rej = calculate_rejection(
            self.disc_sig,
            self.disc_bkg,
            target_eff=0.841345,
            return_cuts=False,
        )
        self.assertAlmostEqual(bkg_rej, 1 / 0.02367)

    def test_array_target(self):
        """Test efficiency and cut value calculation for list of target efficiencies."""
        # explanation is the same as above, now also cut the signal in the middle
        # --> target sig.efficiency 0.841345 and 0.5 --> cut at 2 and 3
        bkg_rej, cut = calculate_rejection(
            self.disc_sig,
            self.disc_bkg,
            target_eff=[0.841345, 0.5],
            return_cuts=True,
        )
        np.testing.assert_array_almost_equal(cut, np.array([1.9956997, 2.990996]))
        np.testing.assert_array_almost_equal(bkg_rej, 1 / np.array([0.02367, 0.00144]))

    def test_with_smooth(self):
        """Test efficiency and cut value calculation with smoothing."""
        bkg_rej, cut = calculate_rejection(
            self.disc_sig,
            self.disc_bkg,
            target_eff=[0.841345, 0.5],
            return_cuts=True,
            smooth=True,
        )
        np.testing.assert_array_almost_equal(cut, np.array([1.9956997, 2.990996]))
        np.testing.assert_array_almost_equal(bkg_rej, np.array([237.052272, 499.639743]))


class EffErrTestCase(unittest.TestCase):
    """Test class for the calculate_efficiency_error function."""

    def test_zero_n_case(self):
        """Test calculate_efficiency_error function."""
        with self.assertRaises(ValueError):
            calculate_efficiency_error(0, 0)

    def test_negative_n_case(self):
        """Test calculate_efficiency_error function."""
        with self.assertRaises(ValueError):
            calculate_efficiency_error(0, -1)

    def test_one_case(self):
        """Test calculate_efficiency_error function."""
        self.assertEqual(calculate_efficiency_error(1, 1), 0)

    def test_example_case(self):
        """Test calculate_efficiency_error function."""
        x_eff = np.array([0.25, 0.5, 0.75])
        error_eff = np.array([0.043301, 0.05, 0.043301])
        np.testing.assert_array_almost_equal(calculate_efficiency_error(x_eff, 100), error_eff)

    def test_example_case_with_norm(self):
        """Test calculate_efficiency_error function."""
        x_eff = np.array([0.25, 0.5, 0.75])
        error_eff = np.array([0.173205, 0.1, 0.057735])
        np.testing.assert_array_almost_equal(
            calculate_efficiency_error(x_eff, 100, norm=True), error_eff
        )


class RejErrTestCase(unittest.TestCase):
    """Test class for the calculate_rejection_error function."""

    def setUp(self):
        self.array = np.array([[1, 2, 3], [4, 5, 6]])

    def test_zero_n_case(self):
        """Test calculate_rejection_error function."""
        with self.assertRaises(ValueError):
            calculate_rejection_error(0, 0)

    def test_negative_n_case(self):
        """Test calculate_rejection_error function."""
        with self.assertRaises(ValueError):
            calculate_rejection_error(0, -1)

    def test_one_case(self):
        """Test calculate_rejection_error function."""
        self.assertEqual(calculate_rejection_error(1, 1), 0)

    def test_zero_x_case(self):
        """Test calculate_rejection_error function."""
        with self.assertRaises(ValueError):
            calculate_rejection_error(np.array([0, 1, 2]), 3)

    def test_example_case(self):
        """Test calculate_rejection_error function."""
        x_rej = np.array([20, 50, 100])
        error_rej = np.array([8.717798, 35.0, 99.498744])
        np.testing.assert_array_almost_equal(calculate_rejection_error(x_rej, 100), error_rej)

    def test_example_case_with_norm(self):
        """Test calculate_rejection_error function."""
        x_rej = np.array([20, 50, 100])
        error_rej = np.array([0.43589, 0.7, 0.994987])
        np.testing.assert_array_almost_equal(
            calculate_rejection_error(x_rej, 100, norm=True), error_rej
        )


class GetDiscriminantTestCase(unittest.TestCase):
    """Test class for get_discriminant function."""

    def test_get_discriminant_nominal(self):
        """
        Test the nominal case, where all probability fields exist,
        fraction values are nonzero, and the computation is straightforward.
        """
        # Get the flavours
        flavours = Flavours.by_category("single-btag")
        signal_flavour = flavours["bjets"]

        # Define a simple structured array for jets
        dtype = [
            ("mytagger_pb", np.float64),
            ("mytagger_pc", np.float64),
            ("mytagger_pu", np.float64),
            ("mytagger_ptau", np.float64),
        ]
        jets = np.array(
            [
                (0.1, 0.2, 0.6, 0.1),
                (0.05, 0.1, 0.75, 0.1),
                (0.3, 0.3, 0.3, 0.1),
            ],
            dtype=dtype,
        )

        # Fraction values for each flavour
        fraction_values = {
            "fc": 0.2,
            "fu": 0.7,
            "ftau": 0.1,
        }

        # Compute discriminant (signal = bjets)
        disc = get_discriminant(
            jets=jets,
            tagger="mytagger",
            signal=signal_flavour,
            flavours=flavours,
            fraction_values=fraction_values,
            epsilon=1e-10,
        )

        # By definition, discriminant = log( pb / (pc + pu ) ), ignoring epsilon for clarity
        # We'll check the result using np.isclose for each row
        expected = np.log([
            0.1 / (0.2 * 0.2 + 0.7 * 0.6 + 0.1 * 0.1),
            0.05 / (0.2 * 0.1 + 0.7 * 0.75 + 0.1 * 0.1),
            0.3 / (0.2 * 0.3 + 0.7 * 0.3 + 0.1 * 0.1),
        ])

        assert np.allclose(disc, expected), f"Unexpected discriminant values: {disc}"

    def test_get_discriminant_with_epsilon(self):
        """
        Test that epsilon is used correctly by comparing the result
        with a small 'pb' and zero denominator.
        """
        # Get the flavours
        flavours = Flavours.by_category("single-btag")
        signal_flavour = flavours["bjets"]

        dtype = [
            ("mytagger_pb", np.float64),
            ("mytagger_pc", np.float64),
            ("mytagger_pu", np.float64),
            ("mytagger_ptau", np.float64),
        ]
        # Very small pb, zero denominator
        jets = np.array([(1e-15, 0.0, 0.0, 0.0)], dtype=dtype)

        fraction_values = {
            "fc": 0.2,
            "fu": 0.7,
            "ftau": 0.1,
        }

        disc = get_discriminant(
            jets=jets,
            tagger="mytagger",
            signal=signal_flavour,
            flavours=flavours,
            fraction_values=fraction_values,
            epsilon=1e-10,
        )

        assert disc.shape == (1,)
        assert abs(disc[0]) < 1e-4, f"Discriminant not close to zero, got {disc[0]}"

    def test_get_discriminant_zero_fraction(self):
        """
        Test that if one of the background flavours has zero fraction,
        it is effectively skipped in the denominator.
        """
        # Get the flavours
        flavours = Flavours.by_category("single-btag")
        signal_flavour = flavours["bjets"]

        dtype = [
            ("mytagger_pb", np.float64),
            ("mytagger_pc", np.float64),
            ("mytagger_pu", np.float64),
            ("mytagger_ptau", np.float64),
        ]
        jets = np.array([(0.1, 0.2, 0.6, 0.1), (0.2, 0.2, 0.5, 0.1)], dtype=dtype)

        # Suppose we only want to consider cjets and light jets as the background
        fraction_values = {
            "fc": 0.2,
            "fu": 0.8,
            "ftau": 0.0,
        }

        disc = get_discriminant(
            jets=jets,
            tagger="mytagger",
            signal=signal_flavour,
            flavours=flavours,
            fraction_values=fraction_values,
            epsilon=1e-10,
        )

        # Now the denominator should only be pc and pu, ignoring ptau since fraction is 0
        expected = np.log([0.1 / (0.2 * 0.2 + 0.8 * 0.6), 0.2 / (0.2 * 0.2 + 0.8 * 0.5)])
        assert np.allclose(
            disc, expected
        ), f"Unexpected discriminant with zero-fraction skipping: {disc}"

    def test_get_discriminant_missing_signal_raises(self):
        """Test that if the signal field is missing in the input array, a ValueError is raised."""
        # Get the flavours
        flavours = Flavours.by_category("single-btag")
        signal_flavour = flavours["bjets"]

        # Only background probabilities
        dtype = [
            ("mytagger_pc", np.float64),
            ("mytagger_pu", np.float64),
            ("mytagger_ptau", np.float64),
        ]
        jets = np.array([(0.2, 0.7, 0.1)], dtype=dtype)

        fraction_values = {
            "fc": 0.2,
            "fu": 0.7,
            "ftau": 0.1,
        }

        with self.assertRaises(ValueError) as ctx:
            get_discriminant(
                jets=jets,
                tagger="mytagger",
                signal=signal_flavour,
                flavours=flavours,
                fraction_values=fraction_values,
            )
        self.assertEqual(
            "No signal probability value(s) found for tagger mytagger. "
            "Missing variable: mytagger_pb",
            str(ctx.exception),
        )

    def test_get_discriminant_nonzero_fraction_but_missing_prob_raises(self):
        """
        If we have a nonzero fraction for a background flavour,
        but the corresponding probability field is missing in jets,
        the function should raise a ValueError.
        """
        # Get the flavours
        flavours = Flavours.by_category("single-btag")
        signal_flavour = flavours["bjets"]

        # This array only has b, c, and u probabilities, missing 'mytagger_ptau'
        dtype = [
            ("mytagger_pb", np.float64),
            ("mytagger_pc", np.float64),
            ("mytagger_pu", np.float64),
        ]
        jets = np.array([(0.2, 0.7, 0.1)], dtype=dtype)

        # Nonzero fraction for 'ftau' but no 'mytagger_ptau' field
        fraction_values = {
            "fc": 0.2,
            "fu": 0.7,
            "ftau": 0.1,
        }

        with self.assertRaises(ValueError) as ctx:
            get_discriminant(
                jets=jets,
                tagger="mytagger",
                signal=signal_flavour,
                flavours=flavours,
                fraction_values=fraction_values,
            )

        self.assertEqual(
            "Nonzero fraction value for taujets, but 'mytagger_ptau' not found in input array.",
            str(ctx.exception),
        )
