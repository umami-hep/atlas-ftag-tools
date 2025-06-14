from __future__ import annotations

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as s2u

from ftag.mock import JET_VARS, TRACK_VARS, get_mock_file, get_mock_scores


def test_get_mock_scores():
    labels = np.array([0, 4, 5] * 10)
    scores = get_mock_scores(labels)
    assert scores.dtype.names == (
        "MockTagger_pu",
        "MockTagger_pc",
        "MockTagger_pb",
        "MockTagger_ptau",
    )
    assert scores.shape == (len(labels),)
    assert np.allclose(np.sum(s2u(scores), axis=-1), 1)


def test_get_mock_file():
    # test jets are correctly generated
    fname, f = get_mock_file(num_jets=1000)
    jets = f["jets"]
    assert len(jets) == 1000
    assert set(jets.dtype.names) == set(
        [name for name, dtype in JET_VARS]
        + [
            "MockTagger_pu",
            "MockTagger_pc",
            "MockTagger_pb",
            "MockTagger_ptau",
            "MockXbbTagger_phbb",
            "MockXbbTagger_phcc",
            "MockXbbTagger_ptop",
            "MockXbbTagger_pqcd",
            "MockXbbTagger_phtauhad",
            "MockXbbTagger_phtauel",
            "MockXbbTagger_phtaumu",
        ]
    )
    assert all(jets["pt"] > 0)
    assert all(jets["mass"] <= 50e3)
    assert all(jets["eta"] >= -3)
    assert all(jets["eta"] <= 3)
    assert all(jets["n_truth_promptLepton"] == 0)

    # test tracks are correctly generated
    tracks_name = "tracks"
    num_tracks = 40
    fname, f = get_mock_file(num_jets=1000, tracks_name=tracks_name, num_tracks=num_tracks)
    tracks = f[tracks_name]
    assert len(tracks) == 1000
    assert set(tracks.dtype.names) == set([name for name, dtype in TRACK_VARS] + ["valid"])
    assert all(len(tracks[i]) == num_tracks for i in range(len(tracks)))

    # test tracks are not generated when tracks_name is None
    fname, f = get_mock_file(num_jets=1000, tracks_name=None)
    assert tracks_name not in f

    # test custom fname
    fname, f = get_mock_file(fname="test.h5")
    assert fname == "test.h5"


def test_get_mock_file_determinism():
    # Test that the mock file generation is deterministic
    _, f1 = get_mock_file(num_jets=1000)
    _, f2 = get_mock_file(num_jets=1000)

    np.testing.assert_array_equal(f1["jets"], f2["jets"])
    np.testing.assert_array_equal(f1["tracks"], f2["tracks"]) if "tracks" in f1 else True

    # Clean up the file
    f1.close()
