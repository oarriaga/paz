from paz.abstract.model import Model
from paz.abstract.node import Input, Node


def double(x):
    return 2 * x


def increment(x):
    return x + 1


def add(x, y):
    return x + y


def test_shared_input():
    x = Input("x")
    y = Node(double)(x)
    z = Node(add)(x, y)
    model = Model([x], [z], "shared_input")
    assert model(3.0).add == 9.0


def test_diamond():
    x = Input("x")
    left = Node(double, name="left")(x)
    right = Node(increment, name="right")(x)
    output = Node(add, name="output")(left, right)
    model = Model([x], [output], "diamond")
    assert model(3.0).output == 10.0


def test_multiple_outputs():
    x = Input("x")
    left = Node(double, name="left")(x)
    right = Node(increment, name="right")(x)
    outputs = Model([x], [left, right], "multiple_outputs")(3.0)
    assert (outputs.left, outputs.right) == (6.0, 4.0)
