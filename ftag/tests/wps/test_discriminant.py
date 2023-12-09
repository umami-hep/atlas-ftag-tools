from __future__ import annotations

import numpy as np
import pytest

from ftag.flavour import Flavours
from ftag.wps.discriminant import btag_discriminant, ctag_discriminant, get_discriminant


def test_btag_discriminant():
    jets = np.array(
        [
            (0.2, 0.3, 0.9),
            (0.8, 0.5, 0.1),
            (0.6, 0.1, 0.7),
        ],
        dtype=[("tagger_pb", "f4"), ("tagger_pc", "f4"), ("tagger_pu", "f4")],
    )
    tagger = "tagger"
    fc = 0.1
    epsilon = 1e-10
    disc = btag_discriminant(jets, tagger, fc, epsilon)
    pb, pc, pu = jets[f"{tagger}_pb"], jets[f"{tagger}_pc"], jets[f"{tagger}_pu"]
    expected = np.log((pb + epsilon) / ((1.0 - fc) * pu + fc * pc + epsilon))
    assert np.allclose(disc, expected)


def test_ctag_discriminant():
    jets = np.array(
        [
            (0.2, 0.3, 0.9),
            (0.8, 0.5, 0.1),
            (0.6, 0.1, 0.7),
        ],
        dtype=[("tagger_pb", "f4"), ("tagger_pc", "f4"), ("tagger_pu", "f4")],
    )
    tagger = "tagger"
    fb = 0.2
    epsilon = 1e-10
    disc = ctag_discriminant(jets, tagger, fb, epsilon)
    pb, pc, pu = jets[f"{tagger}_pb"], jets[f"{tagger}_pc"], jets[f"{tagger}_pu"]
    expected = np.log((pc + epsilon) / ((1.0 - fb) * pu + fb * pb + epsilon))
    assert np.allclose(disc, expected)


def test_get_discriminant():
    jets = np.array(
        [
            (0.2, 0.3, 0.5),
            (0.8, 0.5, 0.1),
            (0.6, 0.1, 0.7),
        ],
        dtype=[("tagger1_pb", "f4"), ("tagger1_pc", "f4"), ("tagger1_pu", "f4")],
    )
    tagger = "tagger1"
    signal = Flavours.bjets
    disc = get_discriminant(jets, tagger, signal, (0.1,), epsilon=1e-10)
    expected = btag_discriminant(jets, tagger, fc=0.1, epsilon=1e-10)
    assert np.allclose(disc, expected)

    signal = Flavours.cjets
    disc = get_discriminant(jets, tagger, signal, (0.2,), epsilon=1e-10)
    expected = ctag_discriminant(jets, tagger, fb=0.2, epsilon=1e-10)
    assert np.allclose(disc, expected)

    # test invalid signal flavour
    with pytest.raises(ValueError):
        get_discriminant(jets, tagger, Flavours.hbb, (0.1,), epsilon=1e-10)
