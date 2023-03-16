# ATLAS FTAG Python Tools

This is a collection of Python tools for working with files produce with the FTAG [ntuple dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/).
Please see the [example notebook](ftag/example.ipynb) for usage.

## Installation

To install the package you can install from pip using the [release on pypi](https://pypi.org/project/atlas-ftag-tools/) via

```
pip install atlas-ftag-tools
```

or you can clone the repository and install in editable mode with
```bash
python -m pip install -e .
```

To install optional development dependencies (for formatting and linting) you can istead install with
```bash
python -m pip install -e ".[dev]"
```
