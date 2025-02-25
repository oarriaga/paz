from collections import namedtuple
from typing import Callable

import paz


NodeState = namedtuple("NodeState", ["call", "name", "edges"])


def build_type(node_names):
    if isinstance(node_names, list):
        Type = namedtuple("Type", [name for name in node_names])
    else:
        raise ValueError("'node_names' must be a list of strings")
    return Type


def Node(function, *args, name=None):
    if not isinstance(function, Callable):
        raise ValueError(f"Input {function} must be a callable")

    name = function.__name__ if name is None else name
    locked_call = paz.lock(function, *args)
    Type = build_type([name])
    edges = []

    def call(*node_args):
        [edges.append(node_arg) for node_arg in node_args]
        return NodeState(lambda *args: Type(locked_call(*args)), name, edges)

    return call


def Input(name):
    Type = build_type([name])
    return NodeState(lambda x: Type(x), name, [])
