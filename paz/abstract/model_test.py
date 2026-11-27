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


def test_deep_multi_input():
    x, w = Input("x"), Input("w")
    a = Node(increment, name="a")(x)
    b = Node(increment, name="b")(a)
    c = Node(increment, name="c")(b)
    output = Node(add, name="output")(c, w)
    model = Model([x, w], [output], "deep_multi_input")
    assert model(1.0, 10.0).output == 14.0


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


def test_node_builder_called_twice_keeps_separate_edges():
    x, w = Input("x"), Input("w")
    build_node = Node(double)
    node_x, node_w = build_node(x), build_node(w)
    assert [edge.name for edge in node_x.edges] == ["x"]
    assert [edge.name for edge in node_w.edges] == ["w"]
