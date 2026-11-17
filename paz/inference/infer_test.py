import jax
import jax.numpy as jp
import pytest

import paz
from paz.inference.observable import Observable
from paz.inference.pgm import PGM
from paz.inference.prior import Prior


def build_simple_model():
    x = Prior(paz.distributions.Normal(0.0, 1.0), name="x")

    def likelihood(x_value):
        return paz.distributions.Normal(x_value, 1.0)

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
    positions = posterior.inverse_samples
    assert positions.x.shape == (10, 2)
    assert posterior.inverse_log_probs.shape == (10, 2)


def test_infer_mh_uses_tuned_defaults():
    model, data = build_simple_model()
    key = jax.random.PRNGKey(1)
    infer_key = jax.random.split(key)[0]
    model.configure(
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
    positions = posterior.inverse_samples
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
    model.configure(
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


diagonal_case = (paz.AdaptiveDiagonalTuner, (1,))
covariance_case = (paz.AdaptiveCovarianceTuner, (1, 1))


@pytest.mark.parametrize("Tuner, shape", [diagonal_case, covariance_case])
def test_infer_mh_with_shape_tuners(Tuner, shape):
    model, data = build_simple_model()
    tuner = Tuner(sigma=0.4, num_steps=50, num_episodes=4, progress=False)
    model.configure(num_chains=4, warmup=0, tuner=tuner, progress=False)
    posterior = model.infer(jax.random.PRNGKey(5), data, num_samples=2000)
    samples = posterior.samples.x
    assert posterior.config["sigma"].shape == shape
    assert jp.abs(samples.mean() - 0.125) < 0.1
    assert jp.abs(samples.std() - jp.sqrt(0.5)) < 0.1
