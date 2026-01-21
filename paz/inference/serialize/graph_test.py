import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

from paz.inference.latent import Latent
from paz.inference.observable import Observable
from paz.inference.prior import Prior
from paz.inference.serialize.graph import (
    _build_graph,
    _build_node,
    _deserialize_distribution_fn,
    _serialize_distribution_fn,
    _serialize_graph,
    _serialize_node,
)
from paz.inference.serialize.serializable import serializable


tfd = tfp.distributions


@serializable("graph_likelihood")
def graph_likelihood(scale):
    def apply(x):
        return tfd.Normal(x, scale)
    return apply


def test_serialize_deserialize_distribution_fn():
    arrays = {}
    fn = graph_likelihood(0.3)
    spec = _serialize_distribution_fn(fn, arrays, "node")
    restored = _deserialize_distribution_fn(spec, arrays)
    dist = restored(jp.array(0.1))
    assert isinstance(dist, tfd.Normal)
    assert jp.allclose(dist.scale, 0.3)


def test_serialize_build_prior_node():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    arrays = {}
    spec = _serialize_node(prior, arrays)
    rebuilt = _build_node(spec, {}, arrays)
    assert rebuilt.name == "x"
    assert rebuilt.distribution is not None


def test_serialize_build_latent_node():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    latent = Latent(graph_likelihood(0.2), name="z")(prior)
    arrays = {}
    spec = _serialize_node(latent, arrays)
    rebuilt = _build_node(spec, {"x": prior}, arrays)
    assert rebuilt.distribution is None
    assert rebuilt.sample_inverse is not None
    assert rebuilt.edges[0].name == "x"


def test_serialize_build_observable_node():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    obs = Observable(graph_likelihood(0.2), name="y")(prior)
    arrays = {}
    spec = _serialize_node(obs, arrays)
    rebuilt = _build_node(spec, {"x": prior}, arrays)
    assert rebuilt.distribution is None
    assert rebuilt.sample_inverse is None
    assert rebuilt.edges[0].name == "x"


def test_serialize_build_graph():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    obs = Observable(graph_likelihood(0.2), name="y")(prior)
    arrays = {}
    spec = _serialize_graph([obs], arrays)
    nodes = _build_graph(spec, arrays)
    assert set(nodes.keys()) == {"x", "y"}
    assert nodes["y"].edges[0].name == "x"
