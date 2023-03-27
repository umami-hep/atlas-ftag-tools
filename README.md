# ATLAS FTAG Python Tools

This is a collection of Python tools for working with files produced with the FTAG [ntuple dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/).
The code is intended to be used a [library](https://iscinumpy.dev/post/app-vs-library/) for other projects.
Please see the [example notebook](ftag/example.ipynb) for usage.

## Installation

To install the package you can install from pip using the [release on pypi](https://pypi.org/project/atlas-ftag-tools/) via

```bash
pip install atlas-ftag-tools
```

or you can clone the repository and install in editable mode with
```bash
python -m pip install -e .
```

To install optional development dependencies (for formatting and linting) you can instead install with either from pip
```bash
pip install atlas-ftag-tools[dev]
```

or from source
```bash
python -m pip install -e ".[dev]"
```
