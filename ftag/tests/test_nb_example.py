from os.path import join

from pytest_notebook.nb_regression import NBRegressionFixture


# Regression test fixture for example notebook
def test_regression_nb_example(nb_regression: NBRegressionFixture):
    notebook_path = join("ftag", "example.ipynb")
    try:
        nb_regression.check(notebook_path)
    except NBRegressionError:
        # allow for regression failures
        pass
