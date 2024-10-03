from __future__ import annotations

import numpy as np

from ftag import Cuts
from ftag.mock import mock_tracks
from ftag.track_selector import TrackSelector


def test_selector_no_cuts():
    tracks = mock_tracks()
    cuts = Cuts.empty()
    selector = TrackSelector(cuts)
    selected = selector(tracks.copy())
    assert np.all(selected == tracks)


def test_selector_keep_all():
    tracks = mock_tracks()
    cuts = Cuts.from_list(["d0 > 0"])
    selector = TrackSelector(cuts)
    selected = selector(tracks.copy())
    assert np.all(selected == tracks)


def test_selector_remove_all():
    tracks = mock_tracks()
    init_valid = tracks["valid"].copy()
    cuts = Cuts.from_list(["numberOfPixelHits < -999"])
    selector = TrackSelector(cuts)
    selected = selector(tracks.copy())
    selected = selected[init_valid]
    for var in tracks.dtype.names:
        if issubclass(tracks[var].dtype.type, np.floating):
            print(selected[var])
            assert np.all(np.isnan(selected[var]))
        elif issubclass(tracks[var].dtype.type, np.signedinteger):
            assert np.all(selected[var] == -1)
        elif issubclass(tracks[var].dtype.type, np.unsignedinteger):
            assert np.all(selected[var] == 0)
        elif issubclass(tracks[var].dtype.type, np.bool_):
            assert np.all(~selected[var])


def test_selector_remove_some():
    tracks = mock_tracks()
    assert np.any(tracks[tracks["valid"]]["d0"] > 3.5)
    cuts = Cuts.from_list(["d0 < 3.5"])
    init_valid = tracks["valid"].copy()
    init_d0 = tracks["d0"].copy()
    selector = TrackSelector(cuts)
    selected = selector(tracks)
    idx = init_valid & (init_d0 > 3.5)
    assert np.all(np.isnan(selected[idx]["d0"]))
    assert np.all(selected[selected["valid"]]["d0"] < 3.5)


def test_nshared_cut():
    tracks = mock_tracks()

    n_pix_shared = tracks["numberOfPixelSharedHits"]
    n_sct_shared = tracks["numberOfSCTSharedHits"]
    n_module_shared = n_pix_shared + n_sct_shared / 2
    assert np.any(n_module_shared[tracks["valid"]] > 1)

    cut = Cuts.from_list(["NSHARED < 1.1"])
    selector = TrackSelector(cut)
    selected = selector(tracks.copy())
    n_pix_shared = tracks["numberOfPixelSharedHits"]
    n_sct_shared = tracks["numberOfSCTSharedHits"]
    n_module_shared = n_pix_shared + n_sct_shared / 2

    assert not np.any(n_module_shared[selected["valid"]] > 1)

    # check an error is raised if NSHARED is already a track variable
    with np.testing.assert_raises(ValueError):
        cut = Cuts.from_list(["NSHARED < 1.1"])
        nshared_array = np.zeros((10, 40), dtype=[("NSHARED", np.int32)])
        selector = TrackSelector(cut)
        selector(nshared_array)
