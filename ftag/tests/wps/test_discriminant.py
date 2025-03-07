from __future__ import annotations

import numpy as np
import pytest

from ftag import Flavours
from ftag.wps.discriminant import get_discriminant


def test_get_discriminant_nominal():
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


def test_get_discriminant_with_epsilon():
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


def test_get_discriminant_zero_fraction():
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


def test_get_discriminant_missing_signal_raises():
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

    with pytest.raises(ValueError, match="Missing variable: mytagger_pb"):
        get_discriminant(
            jets=jets,
            tagger="mytagger",
            signal=signal_flavour,
            flavours=flavours,
            fraction_values=fraction_values,
        )


def test_get_discriminant_nonzero_fraction_but_missing_prob_raises():
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

    with pytest.raises(ValueError, match="Nonzero fraction value for taujets"):
        get_discriminant(
            jets=jets,
            tagger="mytagger",
            signal=signal_flavour,
            flavours=flavours,
            fraction_values=fraction_values,
        )
