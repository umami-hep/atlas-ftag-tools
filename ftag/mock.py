from __future__ import annotations

from tempfile import NamedTemporaryFile, mkdtemp

import h5py
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from ftag.hdf5 import join_structured_arrays

__all__ = ["get_mock_file"]

JET_VARS = [
    "pt",
    "eta",
    "abs_eta",
    "mass",
    "pt_btagJes",
    "eta_btagJes",
    "n_tracks",
    "HadronConeExclTruthLabelID",
    "HadronConeExclTruthLabelPt",
    "n_truth_promptLepton",
]

TRACK_VARS = [
    "d0",
    "z0SinTheta",
    "dphi",
    "deta",
    "qOverP",
    "IP3D_signed_d0_significance",
    "IP3D_signed_z0_significance",
    "phiUncertainty",
    "thetaUncertainty",
    "qOverPUncertainty",
    "numberOfPixelHits",
    "numberOfSCTHits",
    "numberOfInnermostPixelLayerHits",
    "numberOfNextToInnermostPixelLayerHits",
    "numberOfInnermostPixelLayerSharedHits",
    "numberOfInnermostPixelLayerSplitHits",
    "numberOfPixelSharedHits",
    "numberOfPixelSplitHits",
    "numberOfSCTSharedHits",
    "numberOfPixelHoles",
    "numberOfSCTHoles",
]


def softmax(x, axis=None):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def get_mock_scores(labels: np.ndarray):
    rng = np.random.default_rng()
    scores = np.zeros((len(labels), 3))
    for label, count in zip(*np.unique(labels, return_counts=True)):
        if label == 0:
            scores[labels == label] = rng.normal(loc=[2, 0, 0], scale=1, size=(count, 3))
        elif label == 4:
            scores[labels == label] = rng.normal(loc=[0, 1, 0], scale=2.5, size=(count, 3))
        elif label == 5:
            scores[labels == label] = rng.normal(loc=[0, 0, 3.5], scale=5, size=(count, 3))
    scores = softmax(scores, axis=1)
    cols = [f"MockTagger_p{x}" for x in ["u", "c", "b"]]
    scores = u2s(scores, dtype=np.dtype([(name, "f4") for name in cols]))
    return scores


def get_mock_file(num_jets=1000, add_tagger_scores=False, tracks_name: str = "tracks"):
    # settings
    n_tracks_per_jet = 40

    # setup jets
    rng = np.random.default_rng()
    jets_dtype = np.dtype([(n, "f4") for n in JET_VARS])
    jets = u2s(rng.random((num_jets, len(JET_VARS))), jets_dtype)
    jets["HadronConeExclTruthLabelID"] = np.random.choice([0, 4, 5], size=num_jets)
    jets["pt"] *= 400e3
    jets["mass"] *= 50e3
    jets["eta"] = (jets["eta"] - 0.5) * 6.0
    jets["abs_eta"] = np.abs(jets["eta"])
    jets["n_truth_promptLepton"] = 0

    if add_tagger_scores:
        scores = get_mock_scores(jets["HadronConeExclTruthLabelID"])
        jets = join_structured_arrays([jets, scores])

    fname = NamedTemporaryFile(suffix=".h5", dir=mkdtemp()).name
    f = h5py.File(fname, "w")
    f.create_dataset("jets", data=jets)

    # setup tracks
    if tracks_name:
        tracks_dtype = np.dtype([(n, "f4") for n in TRACK_VARS])
        tracks = u2s(rng.random((num_jets, n_tracks_per_jet, len(TRACK_VARS))), tracks_dtype)
        valid = rng.choice([True, False], size=(num_jets, n_tracks_per_jet))
        valid = valid.astype(bool).view(dtype=np.dtype([("valid", bool)]))
        tracks = join_structured_arrays([tracks, valid])
        f.create_dataset(tracks_name, data=tracks)

    return fname, f
