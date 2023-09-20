from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp

import h5py
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from ftag.hdf5 import join_structured_arrays

__all__ = ["get_mock_file"]

JET_VARS = [
    ("pt", "f4"),
    ("eta", "f4"),
    ("abs_eta", "f4"),
    ("mass", "f4"),
    ("pt_btagJes", "f4"),
    ("eta_btagJes", "f4"),
    ("n_tracks", "i4"),
    ("HadronConeExclTruthLabelID", "i4"),
    ("HadronConeExclTruthLabelPt", "f4"),
    ("n_truth_promptLepton", "i4"),
    ("flavour_label", "i4"),
]

TRACK_VARS = [
    ("d0", "f4"),
    ("z0SinTheta", "f4"),
    ("dphi", "f4"),
    ("deta", "f4"),
    ("qOverP", "f4"),
    ("IP3D_signed_d0_significance", "f4"),
    ("IP3D_signed_z0_significance", "f4"),
    ("phiUncertainty", "f4"),
    ("thetaUncertainty", "f4"),
    ("qOverPUncertainty", "f4"),
    ("numberOfPixelHits", "i4"),
    ("numberOfSCTHits", "i4"),
    ("numberOfInnermostPixelLayerHits", "i4"),
    ("numberOfNextToInnermostPixelLayerHits", "i4"),
    ("numberOfInnermostPixelLayerSharedHits", "i4"),
    ("numberOfInnermostPixelLayerSplitHits", "i4"),
    ("numberOfPixelSharedHits", "i4"),
    ("numberOfPixelSplitHits", "i4"),
    ("numberOfSCTSharedHits", "i4"),
    ("numberOfPixelHoles", "i4"),
    ("numberOfSCTHoles", "i4"),
]


def softmax(x, axis=None):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def get_mock_scores(labels: np.ndarray):
    rng = np.random.default_rng(42)
    scores = np.zeros((len(labels), 3))
    for label, count in zip(*np.unique(labels, return_counts=True)):
        if label in (0, 15):
            scores[labels == label] = rng.normal(loc=[2, 0, 0], scale=1, size=(count, 3))
        elif label == 4:
            scores[labels == label] = rng.normal(loc=[0, 1, 0], scale=2.5, size=(count, 3))
        elif label == 5:
            scores[labels == label] = rng.normal(loc=[0, 0, 3.5], scale=5, size=(count, 3))
    scores = softmax(scores, axis=1)
    cols = [f"MockTagger_p{x}" for x in ["u", "c", "b"]]
    return u2s(scores, dtype=np.dtype([(name, "f4") for name in cols]))


def get_mock_file(
    num_jets=1000,
    fname: str | None = None,
    tracks_name: str = "tracks",
    num_tracks: int = 40,
) -> tuple[str, h5py.File]:
    # setup jets
    rng = np.random.default_rng(42)
    jets_dtype = np.dtype(JET_VARS)
    jets = u2s(rng.random((num_jets, len(JET_VARS))), jets_dtype)
    jets["HadronConeExclTruthLabelID"] = rng.choice([0, 4, 5, 15], size=num_jets)
    jets["flavour_label"] = rng.choice([0, 4, 5], size=num_jets)
    jets["pt"] *= 400e3
    jets["mass"] *= 50e3
    jets["eta"] = (jets["eta"] - 0.5) * 6.0
    jets["abs_eta"] = np.abs(jets["eta"])
    jets["n_truth_promptLepton"] = 0

    # add tagger scores
    scores = get_mock_scores(jets["HadronConeExclTruthLabelID"])
    jets = join_structured_arrays([jets, scores])

    # create a tempfile in a new folder
    if fname is None:
        fname = NamedTemporaryFile(suffix=".h5", dir=mkdtemp()).name
    else:
        Path(fname).parent.mkdir(exist_ok=True, parents=True)
    f = h5py.File(fname, "w")
    f.create_dataset("jets", data=jets)
    f.attrs["test"] = "test"
    f["jets"].attrs["test"] = "test"

    # setup tracks
    if tracks_name:
        tracks_dtype = np.dtype(TRACK_VARS)
        tracks = u2s(rng.random((num_jets, num_tracks, len(TRACK_VARS))), tracks_dtype)
        valid = rng.choice([True, False], size=(num_jets, num_tracks))
        valid = valid.astype(bool).view(dtype=np.dtype([("valid", bool)]))
        tracks = join_structured_arrays([tracks, valid])
        f.create_dataset(tracks_name, data=tracks)

    return fname, f
