"""atlas-ftag-tools - Common tools for ATLAS flavour tagging software."""


__version__ = "v0.1.4"


import ftag.hdf5 as hdf5
from ftag.cuts import Cuts
from ftag.flavour import Flavour, Flavours
from ftag.mock import get_mock_file
from ftag.sample import Sample
from ftag.wps.discriminant import get_discriminant
from ftag.wps.working_points import get_working_points

__all__ = [
    "Cuts",
    "Flavour",
    "Flavours",
    "Sample",
    "hdf5",
    "get_mock_file",
    "get_discriminant",
    "get_working_points",
    "__version__",
]
