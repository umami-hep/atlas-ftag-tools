from pathlib import Path

import pytest

from ftag.mock import get_mock_file
from ftag.sample import Sample

DSID = "user.alfroch.410470.e6337_s3681_r13144_p5169.tdd.EMPFlow.22_2_97.22-11-29-T164944_output.h5"


@pytest.fixture(scope="module")
def sample():
    fname = get_mock_file()[0]
    target_dir = Path(fname).parent / Path(DSID)
    target_dir.mkdir()
    fname = Path(fname).rename(target_dir / Path(fname).name)
    return Sample(pattern=fname, name="test_sample")


def test_sample_path(sample):
    assert isinstance(sample.path, Path)


def test_sample_files(sample):
    assert isinstance(sample.files, list)
    assert len(sample.files) > 0
    assert isinstance(sample.files[0], str)


def test_sample_num_files(sample):
    assert isinstance(sample.num_files, int)
    assert sample.num_files > 0


def test_sample_dsid(sample):
    assert isinstance(sample.dsid, list)
    assert len(sample.dsid) > 0
    assert sample.dsid[0] == DSID


def test_sample_sample_id(sample):
    assert isinstance(sample.sample_id, list)
    assert len(sample.sample_id) > 0
    assert sample.sample_id[0] == "410470"


def test_sample_tags(sample):
    assert isinstance(sample.tags, list)
    assert len(sample.tags) > 0
    assert sample.tags[0] == "e6337_s3681_r13144_p5169"


def test_sample_ptag(sample):
    assert isinstance(sample.ptag, list)
    assert len(sample.ptag) > 0
    assert sample.ptag[0] == "p5169"


def test_sample_rtag(sample):
    assert isinstance(sample.rtag, list)
    assert len(sample.rtag) > 0
    assert sample.rtag[0] == "r13144"


def test_sample_dumper_tag(sample):
    assert isinstance(sample.dumper_tag, list)
    assert len(sample.dumper_tag) > 0
    assert sample.dumper_tag[0] == "22-11-29-T164944"


def test_sample_virtual_file(sample):
    assert isinstance(sample.virtual_file(), Path)


def test_sample_str(sample):
    assert str(sample) == "test_sample"


def test_sample_lt(sample):
    fname = get_mock_file()[0]
    assert (sample < Sample(pattern=fname, name="test_sample_2")) is True
    assert (sample < Sample(pattern=fname, name="test_sample")) is False


def test_sample_eq(sample):
    fname = get_mock_file()[0]
    assert (sample == Sample(pattern=fname, name="test_sample")) is True
    assert (sample == Sample(pattern=fname, name="test_sample_2")) is False
