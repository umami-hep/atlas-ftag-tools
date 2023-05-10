from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp

import h5py
import numpy as np
import pytest
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from ftag.hdf5.h5reader import H5Reader
from ftag.sample import Sample


# parameterise the test
@pytest.mark.parametrize("num", [1, 2, 3])
@pytest.mark.parametrize("length", [200, 301])
def test_H5Reader(num, length):
    batch_size = 100
    effective_bs = batch_size // num * num
    remainder = length % effective_bs * num

    # create test files
    tmpdirs = []
    for i in range(num):
        fname = NamedTemporaryFile(suffix=".h5", dir=mkdtemp()).name
        tmpdirs.append(Path(fname).parent)

        with h5py.File(fname, "w") as f:
            data = i * np.ones((length, 2))
            data = u2s(data, dtype=[("x", "f4"), ("y", "f4")])
            f.create_dataset("jets", data=data)

            data = i * np.ones((length, 40, 2))
            data = u2s(data, dtype=[("a", "f4"), ("b", "f4")])
            f.create_dataset("tracks", data=data)

    # create a multi-path sample
    sample = Sample([f"{x}/*.h5" for x in tmpdirs], name="test")

    # test reading from multiple paths
    reader = H5Reader(sample.path, batch_size=batch_size)
    assert reader.num_jets == num * length
    variables = {"jets": ["x", "y"], "tracks": None}
    for data in reader.stream(variables=variables):
        assert "jets" in data
        assert data["jets"].shape == (effective_bs,) or data["jets"].shape == (remainder,)
        assert len(data["jets"].dtype.names) == 2
        assert "tracks" in data
        assert data["tracks"].shape == (effective_bs, 40) or data["tracks"].shape == (remainder, 40)
        assert len(data["tracks"].dtype.names) == 2

        # check that the tracks are correctly matched to the jets
        for i in range(num):
            trk = (data["tracks"]["a"] == i).all(-1)
            jet = data["jets"]["x"] == i
            assert (jet == trk).all()

        if num > 1:
            corr = np.corrcoef(data["jets"]["x"], data["tracks"]["a"][:, 0])
            np.testing.assert_allclose(corr, 1)
