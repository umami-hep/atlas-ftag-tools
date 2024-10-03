from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ftag.cuts import Cut, Cuts


@dataclass
class TrackSelector:
    """
    Apply track selections to a set of tracks stored in a structured numpy array.

    The array is assumed to have shape (n_jets, n_tracks, n_features).
    Applying cuts will NaN out the tracks that do not pass the cuts,
    but leave the shape of the array unchanged.

    Parameters
    ----------
    cuts : Cuts
        The cuts to apply to the tracks
    valid_str : str
        The name of the field in the tracks that indicates whether the track is
    """

    cuts: Cuts
    valid_str: str = "valid"

    def __call__(self, tracks: np.ndarray) -> np.ndarray:
        # get a bool array for all tracks passing before any cuts
        rm_idx = np.zeros_like(tracks[self.valid_str], dtype=bool)

        # apply the cuts
        for cut in self.cuts.cuts:
            # remove valid track indices that do not pass the selection
            keep_idx = self._nshared_cut(cut, tracks) if cut.variable == "NSHARED" else cut(tracks)
            rm_idx[tracks[self.valid_str] & ~keep_idx] = True

        # set the values of the tracks that do not pass the cuts to
        for var in tracks.dtype.names:
            if issubclass(tracks[var].dtype.type, np.floating):
                tracks[var][rm_idx] = np.nan
            elif issubclass(tracks[var].dtype.type, np.signedinteger):
                tracks[var][rm_idx] = -1
            elif issubclass(tracks[var].dtype.type, np.unsignedinteger):
                tracks[var][rm_idx] = 0
            elif issubclass(tracks[var].dtype.type, np.bool_):
                tracks[var][rm_idx] = False
            else:
                raise TypeError(f"Unknown dtype {tracks[var].dtype}")

        # specifically set the valid flag to false (even though it's already false by now)
        tracks[rm_idx][self.valid_str] = False

        return tracks

    def _nshared_cut(self, cut: Cut, tracks: np.ndarray) -> np.ndarray:
        # hack to apply the FTAG shared hit cut, which requires an intermediate step
        if cut.variable == "NSHARED" and "NSHARED" in tracks.dtype.names:
            raise ValueError("NSHARED is a reserved variable name")

        # compute
        n_pix_shared = tracks["numberOfPixelSharedHits"]
        n_sct_shared = tracks["numberOfSCTSharedHits"]
        n_module_shared = n_pix_shared + n_sct_shared / 2

        # convert n_module_shared to structured array
        n_module_shared = n_module_shared.view(dtype=[(cut.variable, n_module_shared.dtype)])

        # select
        return cut(n_module_shared)
