from paz.abstract.node import Node


def dummy(x, y, z):
    return x + (2 * y) + (3 * z)


node = Node(dummy, 1, 2)
x = node(1.0)
