import pytest

from paz.inference import types


def test_sample_type_fields():
    Sample = types.SampleType(["x", "y"])
    assert Sample._fields == ("x", "y")


def test_sample_type_invalid_raises():
    with pytest.raises(ValueError):
        types.SampleType("x")
