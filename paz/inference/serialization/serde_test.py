import jax
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

import paz
from paz.inference.density.gaussian import build_gaussian_density
from paz.inference.latent import Latent
from paz.inference.latent_space import build_latent_space
from paz.inference.observable import Observable
from paz.inference.pgm import PGM
from paz.inference.prior import Prior
from paz.inference.serialization.serde import _find_serde, _serde_for_type
from paz.inference.serialization.serializable import serializable


tfd = tfp.distributions


@serializable("serde_likelihood")
def serde_likelihood(scale):
    def apply(x):
        return tfd.Normal(x, scale)
    return apply


def _build_posterior():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")

    def likelihood(x_value):
        return tfd.Normal(x_value, 0.5)

    obs = Observable(likelihood, name="y")(prior)
    model = PGM([prior], [obs], "model")
    data = {"y": jp.array(0.2)}
    return paz.infer(
        jax.random.PRNGKey(0),
        data,
        model.prior,
        model.likelihood,
        method="mh",
        num_samples=10,
        num_chains=2,
        sigma=0.2,
        progress=False,
    )


def _build_density():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    latent_space = build_latent_space([prior])
    samples = latent_space.Sample(x=jp.linspace(-1.0, 1.0, 5))
    return build_gaussian_density(samples, latent_space)


def _roundtrip(obj):
    serde = _find_serde(obj)
    manifest, payload, arrays = serde.to_spec(obj)
    return serde.from_spec(manifest, payload, arrays)


def test_serde_for_unknown_type():
    try:
        _serde_for_type("missing")
    except ValueError as error:
        assert "No serializer registered" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_serde_for_known_type():
    serde = _serde_for_type("Prior")
    assert serde.type_id == "Prior"


def test_find_serde_for_prior():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    serde = _find_serde(prior)
    assert serde.type_id == "Prior"


def test_posterior_serde_roundtrip():
    posterior = _build_posterior()
    loaded = _roundtrip(posterior)
    assert jp.allclose(
        loaded.samples.position.x, posterior.samples.position.x
    )
    assert jp.allclose(
        loaded.samples.log_density, posterior.samples.log_density
    )
    assert jp.allclose(
        loaded.infos.is_accepted, posterior.infos.is_accepted
    )


def test_density_serde_roundtrip():
    density = _build_density()
    loaded = _roundtrip(density)
    sample = density.sample(jax.random.PRNGKey(1), num_samples=1, space="inv")
    log_prob = density.log_prob(sample, space="inv")
    log_prob_loaded = loaded.log_prob(sample, space="inv")
    assert jp.allclose(log_prob, log_prob_loaded)


def test_prior_serde_roundtrip():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    loaded = _roundtrip(prior)
    value = jp.array(0.1)
    log_prob = prior.apply(value).log_prob_sum
    log_prob_loaded = loaded.apply(value).log_prob_sum
    assert jp.allclose(log_prob, log_prob_loaded)


def test_latent_serde_roundtrip():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    latent = Latent(serde_likelihood(0.5), name="z")(prior)
    loaded = _roundtrip(latent)
    assert loaded.name == "z"
    assert loaded.edges[0].name == "x"


def test_observable_serde_roundtrip():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    obs = Observable(serde_likelihood(0.5), name="y")(prior)
    loaded = _roundtrip(obs)
    assert loaded.name == "y"
    assert loaded.edges[0].name == "x"


def test_pgm_serde_roundtrip():
    prior = Prior(tfd.Normal(0.0, 1.0), name="x")
    obs = Observable(serde_likelihood(0.5), name="y")(prior)
    model = PGM([prior], [obs], "model")
    loaded = _roundtrip(model)
    key = jax.random.PRNGKey(0)
    sample_inv = model.sample_inverse(key, num_samples=1)
    data = model.sample(key, num_samples=1)
    log_prob = model.likelihood.log_prob(sample_inv, data, space="inv")
    log_prob_loaded = loaded.likelihood.log_prob(sample_inv, data, space="inv")
    assert jp.allclose(log_prob, log_prob_loaded)
