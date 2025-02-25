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


def fn_a(x):
    return x + 1


def fn_b(y):
    return y * 2


model = Sequential()
model.add(fn_a)
model.add(fn_b)

print(model.call(1.0))
