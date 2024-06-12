from __future__ import annotations

import numpy as np
import pytest

from ftag.flavour import Flavours
from ftag.wps.discriminant import (
    btag_discriminant,
    ctag_discriminant,
    get_discriminant,
    hbb_discriminant,
    hcc_discriminant,
)


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
    disc = btag_discriminant(jets, tagger, fc, epsilon=epsilon)
    pb, pc, pu = jets[f"{tagger}_pb"], jets[f"{tagger}_pc"], jets[f"{tagger}_pu"]
    expected = np.log((pb + epsilon) / ((1.0 - fc) * pu + fc * pc + epsilon))
    assert np.allclose(disc, expected)


def test_btag_discriminant_inc_tau():
    jets = np.array(
        [
            (0.2, 0.3, 0.9, 0.1),
            (0.8, 0.5, 0.1, 0.2),
            (0.6, 0.1, 0.7, 0.3),
        ],
        dtype=[
            ("tagger_pb", "f4"),
            ("tagger_pc", "f4"),
            ("tagger_pu", "f4"),
            ("tagger_ptau", "f4"),
        ],
    )
    tagger = "tagger"
    fc = 0.1

    epsilon = 1e-10
    for ftau in (0, 0.2):
        disc = btag_discriminant(jets, tagger, fc, ftau, epsilon=epsilon)
        pb, pc, pu, ptau = (jets[f"{tagger}_p{f}"] for f in ("b", "c", "u", "tau"))
        expected = np.log(
            (pb + epsilon) / ((1.0 - fc - ftau) * pu + fc * pc + ftau * ptau + epsilon)
        )
    assert np.allclose(disc, expected)


def test_no_tau_with_ftau():
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
    ftau = 0.2
    epsilon = 1e-10

    with pytest.raises(ValueError):
        btag_discriminant(jets, tagger, fc, ftau, epsilon=epsilon)
    with pytest.raises(ValueError):
        ctag_discriminant(jets, tagger, fc, ftau, epsilon=epsilon)


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
    disc = ctag_discriminant(jets, tagger, fb, epsilon=epsilon)
    pb, pc, pu = jets[f"{tagger}_pb"], jets[f"{tagger}_pc"], jets[f"{tagger}_pu"]
    expected = np.log((pc + epsilon) / ((1.0 - fb) * pu + fb * pb + epsilon))
    assert np.allclose(disc, expected)


def test_hbb_discriminant():
    jets = np.array(
        [
            (0.2, 0.3, 0.1, 0.4),
            (0.8, 0.5, 0.2, 0.3),
            (0.6, 0.1, 0.6, 0.7),
        ],
        dtype=[
            ("tagger_phbb", "f4"),
            ("tagger_phcc", "f4"),
            ("tagger_ptop", "f4"),
            ("tagger_pqcd", "f4"),
        ],
    )
    tagger = "tagger"
    ftop = 0.25
    fhcc = 0.02
    epsilon = 1e-10
    disc = hbb_discriminant(jets, tagger, ftop, fhcc, epsilon)
    phbb, phcc, ptop, pqcd = (
        jets[f"{tagger}_phbb"],
        jets[f"{tagger}_phcc"],
        jets[f"{tagger}_ptop"],
        jets[f"{tagger}_pqcd"],
    )
    expected = np.log(phbb / (ftop * ptop + fhcc * phcc + (1 - ftop - fhcc) * pqcd + epsilon))
    assert np.allclose(disc, expected)


def test_hcc_discriminant():
    jets = np.array(
        [
            (0.2, 0.3, 0.1, 0.4),
            (0.8, 0.5, 0.2, 0.3),
            (0.6, 0.1, 0.6, 0.7),
        ],
        dtype=[
            ("tagger_phbb", "f4"),
            ("tagger_phcc", "f4"),
            ("tagger_ptop", "f4"),
            ("tagger_pqcd", "f4"),
        ],
    )
    tagger = "tagger"
    ftop = 0.25
    fhbb = 0.3
    epsilon = 1e-10
    disc = hcc_discriminant(jets, tagger, ftop, fhbb, epsilon)
    phbb, phcc, ptop, pqcd = (
        jets[f"{tagger}_phbb"],
        jets[f"{tagger}_phcc"],
        jets[f"{tagger}_ptop"],
        jets[f"{tagger}_pqcd"],
    )
    expected = np.log(phcc / (ftop * ptop + fhbb * phbb + (1 - ftop - fhbb) * pqcd + epsilon))
    assert np.allclose(disc, expected)


def test_get_discriminant():
    jets = np.array(
        [
            (0.2, 0.3, 0.5),
            (0.8, 0.5, 0.1),
            (0.6, 0.1, 0.7),
        ],
        dtype=[("tagger_pb", "f4"), ("tagger_pc", "f4"), ("tagger_pu", "f4")],
    )
    tagger = "tagger"
    signal = Flavours.bjets
    disc = get_discriminant(jets, tagger, signal, fc=0.1)
    expected = btag_discriminant(jets, tagger, fc=0.1)
    assert np.allclose(disc, expected)

    signal = Flavours.cjets
    disc = get_discriminant(jets, tagger, signal, fb=0.2)
    expected = ctag_discriminant(jets, tagger, fb=0.2)
    assert np.allclose(disc, expected)

    jets = np.array(
        [
            (0.2, 0.3, 0.1, 0.4),
            (0.8, 0.5, 0.2, 0.3),
            (0.6, 0.1, 0.6, 0.7),
        ],
        dtype=[
            ("tagger_phbb", "f4"),
            ("tagger_phcc", "f4"),
            ("tagger_ptop", "f4"),
            ("tagger_pqcd", "f4"),
        ],
    )

    signal = Flavours.hbb
    disc = get_discriminant(jets, tagger, signal, ftop=0.2, fhcc=0.3)
    expected = hbb_discriminant(jets, tagger, ftop=0.2, fhcc=0.3)
    assert np.allclose(disc, expected)

    signal = Flavours.hcc
    disc = get_discriminant(jets, tagger, signal, ftop=0.2, fhbb=0.3)
    expected = hcc_discriminant(jets, tagger, ftop=0.2, fhbb=0.3)
    assert np.allclose(disc, expected)

    with pytest.raises(ValueError):
        get_discriminant(jets, tagger, "blah", ftop=0.2, fhcc=0.3)


def test_get_discriminant_tau():
    jets = np.array(
        [
            (0.2, 0.3, 0.5),
            (0.8, 0.5, 0.1),
            (0.6, 0.1, 0.7),
        ],
        dtype=[("tagger_pb", "f4"), ("tagger_pc", "f4"), ("tagger_pu", "f4")],
    )

    tagger = "tagger"
    signal = Flavours.bjets
    with pytest.raises(ValueError):
        get_discriminant(jets, tagger, signal, fc=0.1, ftau=0.1)

    jets = np.array(
        [
            (0.2, 0.3, 0.5, 0.1),
            (0.8, 0.5, 0.1, 0.2),
            (0.6, 0.1, 0.7, 0.3),
        ],
        dtype=[
            ("tagger_pb", "f4"),
            ("tagger_pc", "f4"),
            ("tagger_pu", "f4"),
            ("tagger_ptau", "f4"),
        ],
    )

    disc = get_discriminant(jets, tagger, Flavours.bjets, fc=0.1, ftau=0.1)
    expected = btag_discriminant(jets, tagger, fc=0.1, ftau=0.1)
    assert np.allclose(disc, expected)
