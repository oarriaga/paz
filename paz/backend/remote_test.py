from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener
import socket
from threading import Thread

import jax.numpy as jp
import numpy as np
import pytest

from paz.backend import remote

AUTHKEY = b"secret"


def count(stop):
    for number in range(stop):
        yield number


def fail():
    raise ValueError("broken function")


def build_reply():
    return {"boxes": jp.ones((2, 4)), "name": "cat"}


def serve_clients(listener, functions):
    while True:
        remote.serve_client(listener, functions)


@pytest.fixture
def port():
    functions = {"double": lambda x: 2 * x, "count": count, "fail": fail}
    functions["build_reply"] = build_reply
    listener = Listener(("localhost", 0), authkey=AUTHKEY)
    args = (listener, functions)
    Thread(target=serve_clients, args=args, daemon=True).start()
    return listener.address[1]


def test_call_returns_function_result(port):
    connection = remote.connect(port, AUTHKEY)
    result = remote.call(connection, "double", np.arange(3))
    assert np.array_equal(result, [0, 2, 4])


def test_call_returns_jax_arrays_as_numpy(port):
    connection = remote.connect(port, AUTHKEY)
    reply = remote.call(connection, "build_reply")
    assert type(reply["boxes"]) is np.ndarray
    assert reply["name"] == "cat"


def test_stream_yields_every_value(port):
    connection = remote.connect(port, AUTHKEY)
    assert list(remote.stream(connection, "count", 4)) == [0, 1, 2, 3]


def test_call_after_abandoned_stream(port):
    connection = remote.connect(port, AUTHKEY)
    values = remote.stream(connection, "count", 4)
    next(values)
    values.close()
    assert remote.call(connection, "double", 3) == 6


def test_call_raises_remote_error_and_keeps_serving(port):
    connection = remote.connect(port, AUTHKEY)
    with pytest.raises(RuntimeError, match="broken function"):
        remote.call(connection, "fail")
    assert remote.call(connection, "double", 3) == 6


def test_serves_next_client_after_one_leaves(port):
    remote.connect(port, AUTHKEY).close()
    connection = remote.connect(port, AUTHKEY)
    assert remote.call(connection, "double", 3) == 6


def test_wrong_authkey_is_rejected_and_keeps_serving(port):
    with pytest.raises(AuthenticationError):
        Client(("localhost", port), authkey=b"wrong")
    connection = remote.connect(port, AUTHKEY)
    assert remote.call(connection, "double", 3) == 6


def test_malformed_request_raises_and_keeps_serving(port):
    connection = remote.connect(port, AUTHKEY)
    connection.send("not a request")
    assert connection.recv().kind == remote.Kind.ERROR
    assert remote.call(connection, "double", 3) == 6


def test_keeps_serving_after_bytes_that_are_not_a_handshake(port):
    with socket.create_connection(("localhost", port)) as intruder:
        intruder.sendall(b"GET / HTTP/1.1\r\n\r\n")
        intruder.recv(1024)
    connection = remote.connect(port, AUTHKEY)
    assert remote.call(connection, "double", 3) == 6


def test_serve_rejects_empty_authkey():
    with pytest.raises(ValueError):
        remote.serve({}, 0, b"")
