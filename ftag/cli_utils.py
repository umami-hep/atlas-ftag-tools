from __future__ import annotations

import argparse
from pathlib import Path


def valid_path(string: str) -> Path:
    """Check if the given string is a valid path.

    Parameters
    ----------
    string : str
        Input path as string

    Returns
    -------
    Path
        Output path as Path object

    Raises
    ------
    FileNotFoundError
        If the given input path as string doesn't exist
    """
    if (path := Path(string)).is_file():
        return path
    raise FileNotFoundError(string)


class HelpFormatter(  # noqa: D101
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
): ...
