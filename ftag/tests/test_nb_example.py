from contextlib import suppress
from os.path import join

from pytest_notebook.nb_regression import NBRegressionError, NBRegressionFixture


# Regression test fixture for example notebook
def test_example_notebook(nb_regression: NBRegressionFixture):
    notebook_path = join("ftag", "example.ipynb")
    with suppress(NBRegressionError):
        nb_regression.check(notebook_path)
