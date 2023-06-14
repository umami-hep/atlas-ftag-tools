from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp

import h5py
import numpy as np
import pytest
from numpy.lib.recfunctions import unstructured_to_structured as u2s

from ftag.cuts import Cuts
from ftag.hdf5.h5reader import H5Reader
from ftag.sample import Sample


# parameterise the test
@pytest.mark.parametrize("num", [1, 2, 3])
@pytest.mark.parametrize("length", [200, 301])
@pytest.mark.parametrize("equal_jets", [True, False])
def test_H5Reader(num, length, equal_jets):
    # calculate all possible effective batch sizes, from single file batch sizes and remainders
    batch_size = 100
    effective_bs_file = batch_size // num
    remainders = [(length * n) % effective_bs_file for n in range(num + 1)]
    effective_bs_options = [effective_bs_file * n + r for n in range(num + 1) for r in remainders]
    effective_bs_options = [x for x in effective_bs_options if x <= batch_size][1:]

    # create test files (of different lengths)
    tmpdirs = []
    for i in range(num):
        fname = NamedTemporaryFile(suffix=".h5", dir=mkdtemp()).name
        tmpdirs.append(Path(fname).parent)

        with h5py.File(fname, "w") as f:
            data = i * np.ones((length * (i + 1), 2))
            data = u2s(data, dtype=[("x", "f4"), ("y", "f4")])
            f.create_dataset("jets", data=data)

            data = i * np.ones((length * (i + 1), 40, 2))
            data = u2s(data, dtype=[("a", "f4"), ("b", "f4")])
            f.create_dataset("tracks", data=data)

    # create a multi-path sample
    sample = Sample([f"{x}/*.h5" for x in tmpdirs], name="test")

    # test reading from multiple paths
    reader = H5Reader(sample.path, batch_size=batch_size, equal_jets=equal_jets)
    assert reader.num_jets == num * (num + 1) / 2 * length
    variables = {"jets": ["x", "y"], "tracks": None}
    for data in reader.stream(variables=variables):
        assert "jets" in data
        assert data["jets"].shape in [(effective_bs,) for effective_bs in effective_bs_options]
        assert len(data["jets"].dtype.names) == 2
        assert "tracks" in data
        assert data["tracks"].shape in [(effective_bs, 40) for effective_bs in effective_bs_options]
        assert len(data["tracks"].dtype.names) == 2
        if equal_jets:  # if equal_jets is off, batches won't necessarily have data from all files
            assert (np.unique(data["jets"]["x"]) == np.array(list(range(num)))).all()

        # check that the tracks are correctly matched to the jets
        for i in range(num):
            trk = (data["tracks"]["a"] == i).all(-1)
            jet = data["jets"]["x"] == i
            assert (jet == trk).all()

        if num > 1:
            if len(np.unique(data["jets"]["x"])) == 1:
                np.testing.assert_array_equal(data["jets"]["x"], data["tracks"]["a"][:, 0])
            else:
                corr = np.corrcoef(data["jets"]["x"], data["tracks"]["a"][:, 0])
                np.testing.assert_allclose(corr, 1)

    # testing load method
    loaded_data = reader.load(num_jets=-1)

    # check if -1 is passed, all data is loaded
    if not equal_jets:
        expected_shape = (num * (num + 1) / 2 * length,)
        assert loaded_data["jets"].shape == expected_shape

    # check not passing variables explicitly uses all variables
    assert len(loaded_data["jets"].dtype.names) == 2


@pytest.mark.parametrize("equal_jets", [True, False])
def test_estimate_available_jets(equal_jets):
    # create test files (of different lengths)
    total_files = 2
    length = 100_000
    batch_size = 10_000
    tmpdirs = []
    for i in range(1, total_files + 1):
        fname = NamedTemporaryFile(suffix=".h5", dir=mkdtemp()).name
        tmpdirs.append(Path(fname).parent)

        with h5py.File(fname, "w") as f:
            data = i * np.ones((length * i, 2))
            data = u2s(data, dtype=[("x", "f4"), ("y", "f4")])
            f.create_dataset("jets", data=data)

            data = i * np.ones((length * i, 40, 2))
            data = u2s(data, dtype=[("a", "f4"), ("b", "f4")])
            f.create_dataset("tracks", data=data)

    # create a multi-path sample
    sample = Sample([f"{x}/*.h5" for x in tmpdirs], name="test")

    # test reading from multiple paths
    reader = H5Reader(sample.path, batch_size=batch_size, equal_jets=equal_jets)

    # create cuts that would not remove any jets
    cuts = Cuts.from_list([("x", "!=", "-1"), ("y", "!=", "-1")])
    estimated_num_jets = reader.estimate_available_jets(cuts, num=1000)
    if equal_jets:
        assert estimated_num_jets == total_files * length
    else:
        assert estimated_num_jets == total_files * (total_files + 1) / 2 * length
