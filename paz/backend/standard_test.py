import pytest
from paz.backend.standard import lock, merge_dicts


def function(a, b, c, d):
    return a + (2 * b) + (3 * c) + (4 * d)


@pytest.fixture
def dictionary_A():
    return {"a": 0, "b": 1}


@pytest.fixture
def dictionary_B():
    return {"c": 2, "d": 3}


def test_lock():
    locked_function = lock(function, 3, 4)
    assert locked_function(1, 2) == 30


def test_merge_dicts(dictionary_A, dictionary_B):
    dictionary = merge_dicts(dictionary_A, dictionary_B)
    assert dictionary == {"a": 0, "b": 1, "c": 2, "d": 3}
