import jax
import jax.numpy as jp
import pytest
from tensorflow_probability.substrates import jax as tfp

import paz
from paz.inference.observable import Observable
from paz.inference.pgm import PGM
from paz.inference.prior import Prior

tfd = tfp.distributions


def build_simple_model():
    x = Prior(tfd.Normal(0.0, 1.0), name="x")

    def likelihood(x_value):
        return tfd.Normal(x_value, 1.0)

    y = Observable(likelihood, name="y")(x)
    data = {"y": jp.array(0.25)}
    return PGM([x], [y], "simple"), data


def test_infer_mh_shapes():
    model, data = build_simple_model()
    key = jax.random.PRNGKey(0)
    posterior = paz.infer(
        key,
        data,
        model.prior,
        model.likelihood,
        method="mh",
        num_samples=10,
        num_chains=2,
        sigma=0.2,
        progress=False,
    )
    positions = posterior.samples.position
    assert positions.x.shape == (10, 2)
    assert posterior.samples.log_density.shape == (10, 2)


def test_infer_mh_uses_tuned_defaults():
    model, data = build_simple_model()
    key = jax.random.PRNGKey(1)
    infer_key = jax.random.split(key)[0]
    model.compile(
        num_chains=3,
        sigma=0.3,
        warmup=0,
        tuner=paz.AdaptiveStepTuner(
            sigma=0.3, num_steps=2, num_episodes=1, progress=False
        ),
    )
    posterior = model.infer(infer_key, data, num_samples=5, tune=False)
    assert bool(jp.isclose(posterior.config["sigma"], 0.3))
    assert posterior.config["num_chains"] == 3
    assert posterior.config["num_samples"] == 5


def test_infer_mh_warmup_discards_samples():
    model, data = build_simple_model()
    key = jax.random.PRNGKey(2)
    posterior = paz.infer(
        key,
        data,
        model.prior,
        model.likelihood,
        method="mh",
        num_samples=5,
        num_chains=1,
        warmup=2,
        progress=False,
    )
    positions = posterior.samples.position
    assert positions.x.shape == (5, 1)
    assert posterior.config["warmup"] == 2


def test_infer_mh_unknown_method_raises():
    model, data = build_simple_model()
    key = jax.random.PRNGKey(3)
    with pytest.raises(NotImplementedError):
        paz.infer(
            key,
            data,
            model.prior,
            model.likelihood,
            method="smc",
            num_samples=5,
        )


def test_tune_updates_default_sigma():
    model, data = build_simple_model()
    key = jax.random.PRNGKey(4)
    infer_key = jax.random.split(key)[0]
    model.compile(
        num_chains=1,
        sigma=0.4,
        warmup=0,
        tuner=paz.AdaptiveStepTuner(
            sigma=0.4, num_steps=2, num_episodes=1, progress=False
        ),
    )
    posterior = model.infer(
        infer_key,
        data,
        num_samples=5,
        num_chains=1,
        progress=False,
    )
    assert posterior.config["tune"] is True
