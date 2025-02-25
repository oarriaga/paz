import pytest

from paz.abstract.sequential import Sequential


def fn_a(x):
    return x + 1


def fn_b(y):
    return y * 2


def test_simple_function():
    model = Sequential()
    model.add(fn_a)
    model.add(fn_b)
    assert model.call(1) == 4
