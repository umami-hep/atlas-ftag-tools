from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ftag import Cuts


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
            rm_idx[tracks[self.valid_str] & ~cut(tracks)] = True

        # set the values of the tracks that do not pass the cuts to
        for var in tracks.dtype.names:
            if issubclass(tracks[var].dtype.type, np.floating):
                tracks[var][rm_idx] = np.nan
            elif issubclass(tracks[var].dtype.type, np.integer):
                tracks[var][rm_idx] = -1
            elif issubclass(tracks[var].dtype.type, np.bool_):
                tracks[var][rm_idx] = False
            else:
                raise TypeError(f"Unknown dtype {tracks[var].dtype}")

        # specifically set the valid flag to false (even though it's already false by now)
        tracks[rm_idx][self.valid_str] = False

        return tracks
