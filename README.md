[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docs](https://img.shields.io/badge/info-documentation-informational)](https://umami-hep.github.io/atlas-ftag-tools/main)
[![PyPI version](https://badge.fury.io/py/atlas-ftag-tools.svg)](https://badge.fury.io/py/atlas-ftag-tools)
[![codecov](https://codecov.io/gh/umami-hep/atlas-ftag-tools/branch/main/graph/badge.svg?token=MBHLIYYQ7I)](https://codecov.io/gh/umami-hep/atlas-ftag-tools)


# ATLAS FTAG Python Tools

This is a collection of Python tools for working with files produced with the FTAG [ntuple dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/).
The code is intended to be used a [library](https://iscinumpy.dev/post/app-vs-library/) for other projects.
Please see the [example notebook](ftag/example.ipynb) for usage.

# Quickstart 

## Installation

`atlas-ftag-tools` can be installed from [PyPI](https://pypi.org/project/atlas-ftag-tools/) or using the latest code from this repository.

### Install latest release from PyPI

```bash
pip install atlas-ftag-tools
```

The installation from PyPI only allows to install tagged releases, meaning you can not
install the latest code from this repo using the above command.
If you just want to use a stable release of `atlas-ftag-tools`, this is the way to go.

### Install latest version from GitHub

```bash
pip install https://github.com/umami-hep/atlas-ftag-tools/archive/main.tar.gz
```

This will install the latest version of `atlas-ftag-tools`, i.e. the current version
from the `main` branch (no matter if it is a release/tagged commit).
If you plan on contributing to `atlas-ftag-tools` and/or want the latest version possible, this
is what you want.

### Install for development with `uv` (recommended)

For development, we recommend using [`uv`](https://docs.astral.sh/uv/), a fast Python package installer and resolver. First, install `uv`:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip (If installing from PyPI, we recommend installing uv into an isolated environment)
pip install uv
```

Then clone the repository and install `atlas-ftag-tools` with development dependencies:

```bash
git clone https://github.com/umami-hep/atlas-ftag-tools.git
cd atlas-ftag-tools
uv sync --extra dev
```

This will install `atlas-ftag-tools` in editable mode along with all development tools (testing, linting, etc.).

> [!TIP]
> In order to use locally installed version of `atlas-ftag-tools` in other `uv`-managed projects, you can add the following to the `pyproject.toml` of the other project:
> ```toml
> [tool.uv.sources]
> atlas-ftag-tools = { path = "path_to/atlas-ftag-tools" }
> ```

## Usage

Extensive examples are given in the [Examples](https://umami-hep.github.io/atlas-ftag-tools/main/examples/index.html)
