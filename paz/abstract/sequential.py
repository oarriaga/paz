from collections import namedtuple

SequentialState = namedtuple("SequentialState", ["add", "call"])


def Sequential(nodes=None):
    if nodes is None:
        nodes = []

    def add(node):
        nodes.append(node)

    def call(x):
        for node in nodes:
            x = node(x)
        return x

    return SequentialState(add, call)
