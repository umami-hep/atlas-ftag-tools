from __future__ import annotations

import argparse
from pathlib import Path


def valid_path(string):
    if (path := Path(string)).is_file():
        return path
    raise FileNotFoundError(string)


class HelpFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter): ...
