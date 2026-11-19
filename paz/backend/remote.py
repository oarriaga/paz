"""Calls functions that run in another process or another computer.

The server only listens on localhost and the messages are pickled, so reach
it from another computer through an SSH tunnel:

    ssh -L 5000:localhost:5000 <server>

A connection carries one call at a time: finish or close a stream before
making the next call, and closing a stream early waits for it to end.
"""
from collections import namedtuple
from enum import Enum
import inspect
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener
import pickle
import socket
import traceback

import jax
import numpy as np

Reply = namedtuple("Reply", ["kind", "value"])


class Kind(Enum):
    VALUE = 0
    ERROR = 1
    END = 2


def serve(functions, port, authkey):
    """Answers calls to the named functions until the process is stopped.

    # Arguments
        functions: Dictionary from name to function. A generator function
            sends each value it yields as soon as it is ready.
        port: Int.
        authkey: Bytes. Secret that clients must also know.
    """
    # multiprocessing skips authentication when the key is empty.
    if not authkey:
        raise ValueError("authkey must not be empty")
    with Listener(("localhost", port), authkey=authkey) as listener:
        while True:
            serve_client(listener, functions)


def connect(port, authkey):
    return send_without_delay(Client(("localhost", port), authkey=authkey))


def call(connection, name, *args):
    return list(stream(connection, name, *args))[0]


def stream(connection, name, *args):
    connection.send((name, args))
    reply = connection.recv()
    try:
        while reply.kind == Kind.VALUE:
            yield reply.value
            reply = connection.recv()
    finally:
        # A caller that stops early must not leave replies for the next call.
        while reply.kind == Kind.VALUE:
            reply = connection.recv()
    if reply.kind == Kind.ERROR:
        raise RuntimeError(reply.value)


def serve_client(listener, functions):
    # A client that leaves, fails to authenticate or sends bytes that are not
    # a handshake (raised as a plain OSError) must not stop the server.
    try:
        with send_without_delay(listener.accept()) as connection:
            while True:
                answer(connection, functions)
    except (EOFError, OSError, AuthenticationError):
        pass


def answer(connection, functions):
    request = connection.recv_bytes()
    try:
        name, args = pickle.loads(request)
        for value in to_values(functions[name](*args)):
            connection.send(Reply(Kind.VALUE, to_numpy(value)))
        reply = Reply(Kind.END, None)
    except Exception:
        reply = Reply(Kind.ERROR, traceback.format_exc())
    connection.send(reply)


def to_values(result):
    if inspect.isgenerator(result):
        values = result
    else:
        values = [result]
    return values


def to_numpy(tree):
    # The client must be able to read replies without a matching JAX setup.
    return jax.tree.map(leaf_to_numpy, tree)


def leaf_to_numpy(leaf):
    if isinstance(leaf, jax.Array):
        leaf = np.asarray(leaf)
    return leaf


def send_without_delay(connection):
    # Large messages are written in two parts, and by default the second
    # part waits about 40 ms for the first one to be acknowledged.
    socket_args = (connection.fileno(), socket.AF_INET, socket.SOCK_STREAM)
    with socket.fromfd(*socket_args) as duplicate:
        duplicate.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return connection
