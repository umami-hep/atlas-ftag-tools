import numpy as np
import pytest

from ftag.flavour import Flavours
from ftag.wps.discriminant import btag_discriminant, ctag_discriminant, get_discriminant


def test_btag_discriminant():
    pb = np.array([0.2, 0.8, 0.6])
    pc = np.array([0.3, 0.5, 0.1])
    pu = np.array([0.9, 0.1, 0.7])
    disc = btag_discriminant(pb, pc, pu, fc=0.1, epsilon=1e-10)
    expected = np.log((pb + 1e-10) / ((1.0 - 0.1) * pu + 0.1 * pc + 1e-10))
    assert np.allclose(disc, expected)


def test_ctag_discriminant():
    pb = np.array([0.2, 0.8, 0.6])
    pc = np.array([0.3, 0.5, 0.1])
    pu = np.array([0.9, 0.1, 0.7])
    disc = ctag_discriminant(pb, pc, pu, fb=0.2, epsilon=1e-10)
    expected = np.log((pc + 1e-10) / ((1.0 - 0.2) * pu + 0.2 * pb + 1e-10))
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
    signal = Flavours.bjets
    disc = get_discriminant(jets, "tagger1", signal, fx=0.1)
    expected = btag_discriminant(
        jets["tagger1_pb"], jets["tagger1_pc"], jets["tagger1_pu"], fc=0.1, epsilon=1e-10
    )
    assert np.allclose(disc, expected)

    signal = Flavours.cjets
    disc = get_discriminant(jets, "tagger1", signal, fx=0.2)
    expected = ctag_discriminant(
        jets["tagger1_pb"], jets["tagger1_pc"], jets["tagger1_pu"], fb=0.2, epsilon=1e-10
    )
    assert np.allclose(disc, expected)

    # test invalid signal flavour
    with pytest.raises(ValueError):
        get_discriminant(jets, "tagger1", Flavours.hbb, fx=0.1)
