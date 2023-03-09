from pathlib import Path

import yaml

from ftag.src.cuts import Cuts
from ftag.src.flavour import Flavour, FlavourContainer
from ftag.src.hdf5.h5reader import H5Reader
from ftag.src.hdf5.h5writer import H5Writer
from ftag.src.sample import Sample

# load flavours
with open(Path(__file__).parents[1] / "flavours.yaml") as f:
    flavours_yaml = yaml.safe_load(f)
flavours_dict = {f["name"]: Flavour(**f) for f in flavours_yaml}
assert len(flavours_dict) == len(flavours_yaml), "Duplicate flavour names detected"
Flavours = FlavourContainer(flavours_dict)

__all__ = ["Cuts", "Flavours", "Sample", "H5Reader", "H5Writer"]
