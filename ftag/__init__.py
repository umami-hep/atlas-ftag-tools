"""atlas-ftag-tools - Common tools for ATLAS flavour tagging software."""


__version__ = "v0.0.2"

from pathlib import Path

import yaml

import ftag.hdf5 as hdf5
from ftag.cuts import Cuts
from ftag.flavour import Flavour, FlavourContainer
from ftag.sample import Sample

# load flavours
with open(Path(__file__).parent / "flavours.yaml") as f:
    flavours_yaml = yaml.safe_load(f)
flavours_dict = {f["name"]: Flavour(cuts=Cuts.from_list(f.pop("cuts")), **f) for f in flavours_yaml}
assert len(flavours_dict) == len(flavours_yaml), "Duplicate flavour names detected"
Flavours = FlavourContainer(flavours_dict)

__all__ = ["Cuts", "Flavours", "Sample", "hdf5", "__version__"]
