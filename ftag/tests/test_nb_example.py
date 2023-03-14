from os.path import join

from pytest_notebook.nb_regression import NBRegressionFixture


# Regression test fixture for example notebook
def test_regression_nb_example(nb_regression: NBRegressionFixture):
    notebook_path = join("ftag", "example.ipynb")
    nb_regression.check(notebook_path)
