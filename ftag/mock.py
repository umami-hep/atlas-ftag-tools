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
    ("R10TruthLabel_R22v1", "i4"),
    ("GhostBHadronsFinalCount", "i4"),
    ("GhostCHadronsFinalCount", "i4"),
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
    ("numberOfPixelHits", "u1"),
    ("numberOfSCTHits", "u1"),
    ("numberOfInnermostPixelLayerHits", "u1"),
    ("numberOfNextToInnermostPixelLayerHits", "u1"),
    ("numberOfInnermostPixelLayerSharedHits", "u1"),
    ("numberOfInnermostPixelLayerSplitHits", "u1"),
    ("numberOfPixelSharedHits", "u1"),
    ("numberOfPixelSplitHits", "u1"),
    ("numberOfSCTSharedHits", "u1"),
    ("numberOfPixelHoles", "u1"),
    ("numberOfSCTHoles", "u1"),
    ("leptonID", "i1"),
]


def softmax(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Softmax function for numpy arrays.

    Parameters
    ----------
    x : np.ndarray
        Input array for the softmax
    axis : int | None, optional
        Axis along which the softmax is calculated, by default None

    Returns
    -------
    np.ndarray
        Output array with the softmax output
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def get_mock_scores(labels: np.ndarray, is_xbb: bool = False) -> np.ndarray:
    if not is_xbb:
        label_dict = {"u": 0, "c": 4, "b": 5, "tau": 15}

    else:
        label_dict = {
            "hbb": 11,
            "hcc": 12,
            "top": 1,
            "qcd": 10,
            "htauel": 14,
            "htaumu": 15,
            "htauhad": 16,
        }

    # Set random seed
    rng = np.random.default_rng(42)

    # Set a list of possible means/scales
    mean_scale_list = [1, 2, 2.5, 3.5]

    # Get the number of classes
    n_classes = len(label_dict)

    # Init a scores array
    scores = np.zeros((len(labels), n_classes))

    # Generate means/scales
    means = []
    scales = []
    for i in range(n_classes):
        tmp_means = []
        tmp_means = [
            0 if j != i else mean_scale_list[rng.integers(0, len(mean_scale_list))]
            for j in range(n_classes)
        ]
        means.append(tmp_means)
        scales.append(mean_scale_list[rng.integers(0, len(mean_scale_list))])

    # Map the labels to the means
    label_mapping = dict(zip(label_dict.values(), means))

    # Generate random mock scores
    for i, (label, count) in enumerate(zip(*np.unique(labels, return_counts=True))):
        scores[labels == label] = rng.normal(
            loc=label_mapping[label], scale=scales[i], size=(count, n_classes)
        )

    # Pipe scores through softmax
    scores = softmax(scores, axis=1)
    name = "MockXbbTagger" if is_xbb else "MockTagger"
    cols = [f"{name}_p{x}" for x in label_dict]
    return u2s(scores, dtype=np.dtype([(name, "f4") for name in cols]))


def mock_jets(num_jets=1000) -> np.ndarray:
    # setup jets
    rng = np.random.default_rng(42)
    jets_dtype = np.dtype(JET_VARS)
    jets = u2s(rng.random((num_jets, len(JET_VARS))), jets_dtype)
    jets["flavour_label"] = rng.choice([0, 4, 5], size=num_jets)
    jets["pt"] *= 400e3
    jets["mass"] *= 50e3
    jets["eta"] = (jets["eta"] - 0.5) * 6.0
    jets["abs_eta"] = np.abs(jets["eta"])
    jets["n_truth_promptLepton"] = 0

    # add tagger scores
    jets["HadronConeExclTruthLabelID"] = rng.choice([0, 4, 5, 15], size=num_jets)
    jets["GhostBHadronsFinalCount"] = rng.choice([0, 1, 2], size=num_jets)
    jets["GhostCHadronsFinalCount"] = rng.choice([0, 1, 2], size=num_jets)
    jets["R10TruthLabel_R22v1"] = rng.choice([1, 10, 11, 12, 14, 15, 16], size=num_jets)
    scores = get_mock_scores(jets["HadronConeExclTruthLabelID"])
    xbb_scores = get_mock_scores(jets["R10TruthLabel_R22v1"], is_xbb=True)
    return join_structured_arrays([jets, scores, xbb_scores])


def mock_tracks(num_jets=1000, num_tracks=40) -> np.ndarray:
    rng = np.random.default_rng(42)
    tracks_dtype = np.dtype(TRACK_VARS)
    tracks = u2s(rng.random((num_jets, num_tracks, len(TRACK_VARS))), tracks_dtype)
    tracks["d0"] *= 5

    # for the shared hits, add some reasonable integer values
    tracks["numberOfPixelSharedHits"] = rng.integers(0, 3, size=(num_jets, num_tracks))
    tracks["numberOfSCTSharedHits"] = rng.integers(0, 3, size=(num_jets, num_tracks))

    valid = rng.choice([True, False], size=(num_jets, num_tracks))
    valid = valid.astype(bool).view(dtype=np.dtype([("valid", bool)]))
    return join_structured_arrays([tracks, valid])


def get_mock_file(
    num_jets=1000,
    fname: str | None = None,
    tracks_name: str = "tracks",
    num_tracks: int = 40,
) -> tuple[str, h5py.File]:
    jets = mock_jets(num_jets)

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
        tracks = mock_tracks(num_jets, num_tracks)
        f.create_dataset(tracks_name, data=tracks)

    return fname, f
