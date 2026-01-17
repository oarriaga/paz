import jax
import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp

import paz
from paz.inference.observable import Observable
from paz.inference.pgm import PGM
from paz.inference.prior import Prior
from paz.inference.latent_space import to_forward_samples

tfd = tfp.distributions
tfb = tfp.bijectors


def build_bijected_model():
    low, high = 0.1, 1.0
    bijector = tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.Sigmoid()])
    x = Prior(tfd.Uniform(low, high), bijector=bijector, name="x")

    def likelihood(x_value):
        return tfd.Normal(x_value, 0.5)

    y = Observable(likelihood, name="y")(x)
    data = {"y": jp.array(0.2)}
    return PGM([x], [y], "bijected"), data


def build_simple_posterior():
    model, data = build_bijected_model()
    key = jax.random.PRNGKey(0)
    posterior = paz.infer(
        key,
        data,
        model.prior,
        model.likelihood,
        method="mh",
        num_samples=20,
        num_chains=2,
        sigma=0.2,
        progress=False,
    )
    return model, data, posterior


def test_posterior_sample_shapes():
    model, data, posterior = build_simple_posterior()
    key = jax.random.PRNGKey(1)
    samples = posterior.sample(key, num_samples=5, space="inv")
    assert samples.x.shape == (5,)
    samples_fwd = posterior.sample(key, num_samples=3, space="fwd")
    assert samples_fwd.x.shape == (3,)
    forward = posterior.samples_forward
    expected = to_forward_samples(
        posterior.latent_space, posterior.samples.position
    )
    assert jp.allclose(forward.x, expected.x)
    assert jp.allclose(posterior.forward_samples().x, expected.x)


def test_posterior_diagnostics():
    _, _, posterior = build_simple_posterior()
    diagnostics = posterior.diagnostics()
    assert "acceptance_rate" in diagnostics
    assert jp.all(jp.isfinite(diagnostics["acceptance_rate"]))


def test_posterior_as_density_gaussian():
    model, data, posterior = build_simple_posterior()
    density = posterior.as_density(method="gaussian", covariance="diag")
    key = jax.random.PRNGKey(2)
    sample = density.sample(key, num_samples=1, space="inv")
    log_prob = density.log_prob(sample, space="inv")
    assert jp.isfinite(log_prob)


def test_posterior_as_density_gmm():
    model, data, posterior = build_simple_posterior()
    key = jax.random.PRNGKey(3)
    density = posterior.as_density(
        key,
        method="gmm",
        k=2,
        num_iters=5,
        covariance="diag",
    )
    sample = density.sample(key, num_samples=2, space="inv")
    log_prob = density.log_prob(sample, space="inv")
    assert log_prob.shape == (2,)
    assert jp.all(jp.isfinite(log_prob))


def test_posterior_as_density_kde():
    model, data, posterior = build_simple_posterior()
    density = posterior.as_density(method="kde")
    key = jax.random.PRNGKey(4)
    sample = density.sample(key, num_samples=3, space="inv")
    log_prob = density.log_prob(sample, space="inv")
    assert log_prob.shape == (3,)
    assert jp.all(jp.isfinite(log_prob))
