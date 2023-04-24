[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/atlas-ftag-tools.svg)](https://badge.fury.io/py/atlas-ftag-tools)
[![codecov](https://codecov.io/gh/umami-hep/atlas-ftag-tools/branch/main/graph/badge.svg?token=MBHLIYYQ7I)](https://codecov.io/gh/umami-hep/atlas-ftag-tools)

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


## Create virtual file

This package contains a script to easily merge a set of H5 files.
A virtual file is a fast and lightweight way to wrap a set of files.
See the [h5py documentation](https://docs.h5py.org/en/stable/vds.html) for more information on virtual datasets.

The script is `vds.py` and can be run after installing this package with

```
vds <pattern> <output path>
```

The `<pattern>` argument should be a quotes enclosed [glob pattern](https://en.wikipedia.org/wiki/Glob_(programming)), for example `"dsid/path/*.h5"`


## Calculate WPs

This package contains a script to calculate tagger working points (WPs).
The script is `working_points.py` and can be run after installing this package with

```
wps \
    --ttbar "path/to/ttbar/*.h5" \
    --tagger GN120220509 \
    --fx 0.1
```

Both the `--tagger` and `--fx` options accept a list if you want to get the WPs for multiple taggers.

If you want to use the `ttbar` WPs get the efficiencies and rejections for the `zprime` sample, you can add `--zprime "path/to/zprime/*.h5"` to the command.
Note that a default selection of $p_T > 250 ~GeV$ to jets in the `zprime` sample.

By default the working points are printed to the terminal, but you can save the results to a YAML file with the `--outfile` option.

Use `--help` for more options and information.


## Tests

To run the tests you can use the `pytest` or `coverage` command, for example

```bash
coverage run --source ftag -m pytest --show-capture=stdout
```

Running `coverage report` will display the test coverage.
