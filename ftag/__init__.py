"""atlas-ftag-tools - Common tools for ATLAS flavour tagging software."""

from __future__ import annotations

__version__ = "v0.2.18"

from . import hdf5, utils
from .cuts import Cuts
from .flavours import Extended_Flavours, Flavours
from .fraction_optimization import calculate_best_fraction_values
from .git_check import (
    GitError,
    check_for_fork,
    check_for_uncommitted_changes,
    create_and_push_tag,
    get_git_hash,
    is_git_repo,
)
from .labeller import Labeller
from .labels import Label, LabelContainer
from .mock import get_mock_file
from .sample import Sample
from .transform import Transform
from .working_points import get_working_points

__all__ = [
    "Cuts",
    "Extended_Flavours",
    "Flavours",
    "GitError",
    "Label",
    "LabelContainer",
    "Labeller",
    "Sample",
    "Transform",
    "__version__",
    "calculate_best_fraction_values",
    "check_for_fork",
    "check_for_uncommitted_changes",
    "create_and_push_tag",
    "get_git_hash",
    "get_mock_file",
    "get_working_points",
    "hdf5",
    "is_git_repo",
    "utils",
]
