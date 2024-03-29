#!/bin/bash

# install module
python -m pip install -e ".[dev]"

# install docs requirements
python -m pip install -r docs/requirements.txt

# add current working directory to PYTHONPATH such that package is found
export PYTHONPATH=$PWD:$PYTHONPATH

# build the documentation
rm -rf docs/_*
python docs/sphinx_build_multiversion.py
# copy the redirect_index.html that redirects to the main/latest version
cp docs/source/redirect_index.html docs/_build/html/index.html

# we have to create an empty .nojekyll file in order to make the html theme work
touch docs/_build/html/.nojekyll
